# L4GM-official NPU 部署报告

生成日期：2026-06-16

## 任务概述

本次部署任务是在 Ascend NPU 容器内复现 L4GM-official，并确认项目在 NPU 上的实际可运行边界。项目官方 README 给出的复现入口是推理链路：先运行 `infer_3d.py` 生成单视频的多视图与 3D 结果，再运行 `infer_4d.py` 结合 `recon.safetensors` 与 `interp.safetensors` 生成 4D 视频。仓库没有独立的 `benchmark/`、`eval.py` 或标准 benchmark 分数脚本。

仓内唯一明确的定量评估路径在训练/验证代码中，`main.py` 与 `core/models.py` 以 MSE/LPIPS 组成训练损失，并在 eval 阶段输出 PSNR。该路径依赖外部 Objaverse 渲染数据、`data_train/datalist_8fps.txt` 和 `data_train/datalist_24fps.txt`，不能仅凭仓库自带文件复现。因此本次部署验收指标定义为：NPU 可见性、权重完整加载、模型 forward 有限值、官方 3D 推理文件输出、4D 推理在可行配置下文件输出，以及对官方 `num_frames=16` 阻塞原因的定位。

部署约束为只使用后四张 Ascend 卡，通过 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 隔离；下载、权重、cache、运行结果均放在挂载盘目录，避免占用系统盘。测试过程中只管理本任务启动的进程，没有清理其他 NPU 任务。

## 环境状态

目标硬件为 Ascend950PR，`npu-smi` 版本为 `25.7.rc1`，单卡 HBM 约 114688 MB。进程内设置 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 后，`torch.npu.device_count()` 返回 4，符合只使用后四张卡的约束。测试结束后，后四张卡中本任务没有残留进程。

Python 运行环境使用挂载盘上的独立虚拟环境，关键版本如下：

| 组件 | 实测版本/状态 |
| --- | --- |
| Python 环境 | 挂载盘虚拟环境 |
| torch | `2.10.0+cpu` |
| torch_npu | `2.10.0` |
| NPU 可用性 | `torch.npu.is_available() == True` |
| 可见卡数 | `torch.npu.device_count() == 4` |
| 可见物理卡 | `4,5,6,7` |

缓存目录统一设置到挂载盘，包括 `HF_HOME`、`TRANSFORMERS_CACHE`、`DIFFUSERS_CACHE`、`TORCH_HOME`、`XDG_CACHE_HOME` 和 `PIP_CACHE_DIR`。ImageDream、L4GM recon/interp 权重均已放在项目 `pretrained/` 下，没有写入系统盘。

## 适配改动

本轮改动集中在 NPU 设备选择、CUDA 依赖降级、ImageDream 权重迁移和推理路径兼容：

| 改动范围 | 文件 | 目的 |
| --- | --- | --- |
| NPU 设备与缓存 helper | `core/device.py` | 自动选择 `npu:0`，设置挂载盘 cache，提供 NPU autocast、empty cache、CPU tensor materialize 工具。 |
| ImageDream NPU 迁移 | `mvdream/pipeline_mvdream.py` | 在迁移 `text_encoder`、`image_encoder` 到 NPU 前，先将 CPU checkpoint tensor clone 为普通 contiguous tensor，绕过 torch-npu 直接迁移 safetensors-backed tensor 卡死路径。 |
| 推理脚本设备兼容 | `infer_3d.py`、`infer_4d.py` | 替换 CUDA-only device/autocast/cache 调用，使用本项目 device helper；推理阶段禁用 LPIPS 初始化，避免额外下载 VGG 权重。 |
| 主训练入口设备兼容 | `main.py` | 使用统一 NPU device/autocast/cache helper，保留训练/评估入口的 NPU 可迁移性。 |
| ImageDream/xFormers fallback | `mvdream/mv_unet.py`、`mvdream/pipeline_mvdream.py` | 移除 xFormers 硬依赖，走 PyTorch attention fallback。 |
| Gaussian renderer fallback | `core/gs.py` | 在 CUDA-only `gsplat` 不可用时提供 NPU 可运行的简化 rasterization，用于 smoke 与功能链路验证。 |

需要强调的是，`core/gs.py` 的 fallback rasterization 不是 `gsplat` 的质量等价替代，只能证明 3D/4D 推理链路可以在 NPU 上产出文件。若要求官方视觉质量，需要实现或接入 NPU 原生 Gaussian rasterizer。

## 验证结果

本轮验证按“环境、权重、组件、官方入口”逐级推进，结果如下：

| 验证项 | 命令/场景 | 结果 |
| --- | --- | --- |
| NPU 环境 | `torch.npu.is_available()`、`torch.npu.device_count()` | 通过，可见 4 张 NPU。 |
| L4GM smoke | `scripts/npu_smoke.py` | 通过，`gaussians_shape=(1, 4096, 14)`，输出在 `npu:0`，finite 为 True。 |
| recon/interp 权重加载 | 分别加载 `recon.safetensors`、`interp.safetensors` 后做 reduced forward | 通过，`missing/unexpected=0`，输出 finite。 |
| ImageDream CPU 加载 | `MVDreamPipeline.from_pretrained(..., local_files_only=True)` | 通过，7 个组件本地加载完成。 |
| 原始 ImageDream `.to(npu)` | 未 materialize 时整体 `pipe.to(npu)` | 阻塞，停在 CLIP encoder 权重迁移阶段。 |
| CLIP materialize 后 `.to(npu)` | materialize `text_encoder`、`image_encoder` 后 `pipe.to(npu)` | 通过，约 2.99 秒完成。 |
| ImageDream 1-step 前向 | 256x256 输入图，`num_inference_steps=1` | 通过，输出 shape 为 `(5, 256, 256, 3)`，finite 为 True。 |
| 官方 3D 推理 | `infer_3d.py big --num_frames 1` | 通过，30-step diffusion 约 17 it/s，`Generate 3D takes 2.96 s`，生成 PNG/MP4。 |
| 官方 4D 推理，原 README 帧数 | `infer_4d.py big --num_frames 16` | 未通过，temporal attention softmax 申请 64 GiB HBM，NPU OOM。 |
| 4D 降帧推理 | `infer_4d.py big --num_frames 4` | 通过，生成无插值/插值 4D 视频与 fixed 视频。 |

`infer_3d.py` 输出目录包含原图、多视图 PNG、4 个 3D 渲染 PNG 和 3D mp4。`infer_4d.py --num_frames 4` 进一步生成：

| 输出 | 说明 |
| --- | --- |
| `otter-on-surfboard_fg.mp4` | 3D 推理视频 |
| `wo_interp_otter-on-surfboard_fg.mp4` | 4D 无插值视频 |
| `w_interp_otter-on-surfboard_fg.mp4` | 4D 插值视频 |
| `wo_interp_otter-on-surfboard_fg_fixed.mp4` | 4D 无插值 fixed 视角视频 |
| `w_interp_otter-on-surfboard_fg_fixed.mp4` | 4D 插值 fixed 视角视频 |

## 阻塞项与处理结果

| 阻塞项 | 实测现象 | 根因判断 | 当前处理 |
| --- | --- | --- | --- |
| ImageDream CLIP encoder 迁移卡死 | `text_encoder.to(npu)`、`image_encoder.to(npu)` 均 240 秒超时；VAE/UNet 约 2 秒完成。 | 不是权重形状或基础 dtype 不支持；synthetic 同形状 `Embedding/Linear/Conv2d` 可以迁移。直接迁移从 checkpoint loader 得到的 CLIP weight tensor 会卡住，而 `detach().clone().contiguous().to(npu)` 可以约 2 秒完成。 | 已在 NPU 迁移前 materialize CLIP encoder CPU tensor，`pipe.to(npu)` 与 1-step 前向通过。 |
| xFormers/CUDA-only attention | 当前环境没有 xFormers，原路径依赖 CUDA 扩展。 | NPU 无法使用 xFormers CUDA kernel。 | 已改为 PyTorch attention fallback；可运行，但显存压力增大。 |
| CUDA-only `gsplat` rasterizer | Ascend 环境无法使用 CUDA `gsplat`。 | 原 rasterizer 依赖 CUDA 扩展。 | 已加入简化 fallback rasterizer，能产出 smoke/功能结果；不声明为官方质量等价。 |
| 4D 官方 `num_frames=16` OOM | `core/attention.py` 的 `attn.softmax(dim=-1)` 处申请 64 GiB，报 NPU OOM；空闲后卡复测仍失败。 | PyTorch attention fallback 需要显式构造大 attention matrix；`num_frames=16` 下 temporal token 数导致二次显存放大。 | `num_frames=4` 可跑通。官方 16 帧需要后续做 memory-efficient attention、temporal chunking 或结构性降显存适配。 |
| 仓内 benchmark 缺失 | 没有标准 benchmark 脚本或分数指标。 | 项目公开入口以 demo inference 为主，训练评估依赖外部 Objaverse 数据。 | 本报告以功能复现和阻塞定位作为验收；不能生成官方 benchmark 分数。 |

## 部署结论

当前 NPU 部署已达到“功能复现可运行”的状态：L4GM recon/interp 权重能在 NPU 加载并 forward，ImageDream 多视图生成在 materialize 适配后能在 NPU 前向，官方 `infer_3d.py big` 已完整跑通并生成结果文件，`infer_4d.py big` 在 `num_frames=4` 下已生成 4D 视频。

当前不能声明完全复现 README 默认 4D 配置。README 级 `infer_4d.py big --num_frames 16` 仍被 temporal attention 显存阻塞；这不是下载、权重、环境变量或设备选择问题，而是当前 NPU 路径使用 PyTorch fallback attention 后显式 attention matrix 过大。若要交付完整 16 帧 4D，需要继续做 attention 内存优化或分块执行。

## 后续建议

1. 为 `core/attention.py` 增加 NPU 友好的 memory-efficient attention 实现，优先评估 `scaled_dot_product_attention` 或按 temporal/spatial token 分块，目标是避免一次性构造 64 GiB attention matrix。
2. 将 `infer_4d.py` 的 recon 与 interp 模型改为按阶段加载/释放，减少同时驻留的 HBM 压力；这不能单独解决 64 GiB attention matrix，但能增加余量。
3. 替换 `core/gs.py` 的简化 fallback，接入 NPU 原生或跨平台 Gaussian rasterizer；否则视觉质量不能与 CUDA `gsplat` 路径对齐。
4. 如需 benchmark 分数，补齐 Objaverse 渲染数据与 datalist 后再跑训练/验证 PSNR；当前仓库自带文件不足以支撑 PSNR 复现。
