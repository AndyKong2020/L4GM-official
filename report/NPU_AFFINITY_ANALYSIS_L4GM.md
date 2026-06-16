# L4GM-official NPU 亲和性分析

生成日期：2026-06-16

## 绑定口径

| 项 | 口径 |
| --- | --- |
| target_platform | Ascend 950PR |
| NPU 架构版本 | 3510，分离架构，AIC:AIV=1:2，支持 Regbase AIV、SIMD/SIMT 混合、NDDMA、CV Fusion |
| dtype | 推理主路径为 FP16/autocast；Gaussian 后处理、renderer fallback 和部分输出转 FP32；训练配置里的 bf16/LPIPS 未作为本轮实测口径 |
| 任务阶段 | 非 LLM prefill/decode；按 ImageDream diffusion denoise、LGM 3D recon、LGM 4D temporal/interp、Gaussian render 四段分析 |
| 运行边界 | 单进程单卡推理，后四张卡可见；没有做多卡并行、HCCL 或通信 overlap |
| 对比 GPU | 本版不做 GPU 横向对比；只按 Ascend 950PR 亲和性给结论 |

Ascend 950PR 的 FP16/BF16 Cube-only 峰值公开规格为约 432/378 TFLOPS，封装内存带宽约 1.6/1.4 TB/s，因此矩阵路径 FP16 粗平衡点约 270 FLOP/Byte。Vector FP16/BF16 峰值约 54/47 TFLOPS，向量路径平衡点约 34 FLOP/Byte。平衡点只作为 roofline 拐点，不是性能承诺；所有未上板 profile 的效率项均标 `待测`。

## 总体结论

L4GM-official 对 Ascend 950PR 不是全量原生亲和，但已经具备明确的可运行适配路径。基础 Conv2d、Linear、GroupNorm/LayerNorm、reshape、interpolate、matmul、softmax、VAE/UNet 和 scheduler tensor 操作可以支撑 `infer_3d.py big --num_frames 1` 完整跑通，也可以支撑 `infer_4d.py big --num_frames 4` 生成 4D 视频。

当前不亲和点集中在三处：

1. ImageDream CLIP text/image encoder 的 checkpoint-backed CPU tensor 直接 `.to(npu)` 会卡死；经 CPU materialize 后可迁移并前向。
2. `core/attention.py` 在 xFormers 不可用时走显式 PyTorch attention，16 帧 4D 的 temporal attention 在 `softmax` 处申请 64 GiB HBM，默认 README 4D 配置 OOM。
3. CUDA `gsplat` 不可用，当前 NPU fallback 用 scatter/alpha 简化 rasterization，只能证明功能链路，不代表官方视觉质量。

## 五路径拆解

| 子段 | Cube 压力 | Vector 压力 | MTE/FixPipe 压力 | communication | host/head | 亲和判定 | 证据与边界 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ImageDream CLIP text/image encoder 权重迁移 | 无 | 无 | 高，checkpoint-backed CPU tensor 到 NPU copy 路径异常 | 无 | 中，卡在迁移阶段阻塞后续 launch | 原流程差；materialize 后中 | 原始 `text_encoder.to(npu)`、`image_encoder.to(npu)` 240s 超时；`detach().clone().contiguous()` 后约 2s 迁移。该问题是 MTE/host transfer 路径，不是 CLIP 算子本身不支持。 |
| ImageDream diffusion denoise | 中，UNet 中 Linear/conv 与 attention matmul 可运行 | 中，LayerNorm/GroupNorm、softmax、scheduler step、guidance | 中，latents/context/camera 多次 cat/repeat 与 VAE encode/decode | 无 | 中，30-step diffusion 有循环 launch/head | 中 | 1-step 前向输出 finite；3D 入口 30-step diffusion 约 17 it/s。未做逐层 profile，Cube/Vector/MTE 效率 `待测`。 |
| LGM 3D recon forward | 中高，Conv2d/Linear/attention matmul 是主计算 | 中，GroupNorm、SiLU、softplus/sigmoid/tanh/normalize、softmax | 中，view/reshape/permute 多但多数为 layout 元操作或 contiguous copy | 无 | 低到中，单样本推理 kernel 多 | 中高 | recon checkpoint missing/unexpected=0，forward finite；`infer_3d.py big --num_frames 1` 已生成 PNG/MP4。视觉质量受 renderer fallback 限制。 |
| LGM 4D temporal/interp，`num_frames=4` | 中，Conv/attention matmul 可运行 | 中高，temporal attention softmax 和后处理 | 中，帧/视角 reshape、cat、视频输出搬运 | 无 | 中，循环处理 150 帧输入 | 中 | `infer_4d.py big --num_frames 4` 完整跑通并生成无插值/插值视频。 |
| LGM 4D temporal，`num_frames=16` 原配置 | Cube 并非首要瓶颈 | 极高，显式 attention softmax workspace 峰值 | 高，attention matrix materialize 和 workspace 占 HBM | 无 | 中 | 差 | OOM 点在 `core/attention.py` 的 `attn.softmax(dim=-1)`，单次申请 64 GiB；换到空闲后卡仍失败。 |
| Gaussian renderer fallback | 低，Cube 基本闲置 | 中，投影、比较、clamp、alpha blend | 中高，scatter_add、小包/离散访问、stack 输出 | 无 | 中，Python loop over view/batch | 功能中，质量低 | fallback 可生成结果，但 scatter/排序/alpha blending 属不规则负载；无 950 实测归档，吞吐和质量均 `待测`。 |
| 训练/PSNR eval | 待测 | 待测 | 待测，dataloader/Objaverse tar 读放大 | 可能有，Accelerate gather | 高 | 未验证 | 仓库缺外部 Objaverse 数据与 datalist；不能报告 PSNR benchmark。 |

## 算子级亲和性

| 算子/组合 | 主导路径 | NPU 状态 | 证据 | 待测/风险 |
| --- | --- | --- | --- | --- |
| `Tensor.to("npu")` 普通 contiguous tensor | MTE/FixPipe | 支持 | float16/float32/int64 synthetic tensor、synthetic `Embedding/Linear/Conv2d` weight 均可迁移 | checkpoint-backed CLIP tensor 直接迁移会超时 |
| `detach().clone().contiguous()` materialize | MTE/FixPipe + host/head | 支持 | CLIP weight clone 后 `.to(npu)` 约 1.9s，完整 `pipe.to(npu)` 约 3s | 额外 CPU 内存与一次性 copy 开销；建议作为加载期处理 |
| `Conv2d` 1x1/3x3/stride conv | Cube + Vector | 支持 | LGM smoke、3D、4D 降帧通过；ImageDream VAE/UNet 可迁移和前向 | tile 利用率、L0/L1 驻留未 profile |
| `nn.Linear` / qkv projection | Cube | 条件支持 | LGM/CLIP/ImageDream 前向通过；synthetic Linear 迁移通过 | CLIP checkpoint-backed weight 需 materialize |
| `nn.Embedding` | MTE + Vector | 条件支持 | synthetic `Embedding(49408,1024)` 可迁移；CLIP materialize 后前向通过 | checkpoint-backed embedding 直接迁移超时 |
| `GroupNorm`、`LayerNorm` | Vector | 支持 | 3D/4D 成功链路覆盖；小型 LayerNorm 单独迁移通过 | repeat/mask 密度和 aiv_vec_time `待测` |
| `SiLU`、`softplus`、`sigmoid`、`tanh`、`clamp`、`F.normalize` | Vector | 支持 | Gaussian 参数后处理输出 finite | Vector SFU/elementwise 效率 `待测` |
| `view/reshape/permute/transpose/contiguous` | MTE/FixPipe 或元操作 | 支持但需折叠判断 | 3D/4D 多处 layout 变换成功 | view 不计 GM 搬运；contiguous copy 才计真实字节，需 profile 区分 |
| `cat/stack/repeat/repeat_interleave/chunk` | MTE/FixPipe | 支持 | diffusion latent/context、rays embedding、4D 输出拼接通过 | 高帧数下会放大 HBM 和 copy 次数 |
| `F.interpolate` bilinear/nearest | Vector + MTE | 支持 | 输入 resize、上采样、VAE latent resize、视频输出路径通过 | 大 batch/训练增强未测 |
| `q @ k.T`、`attn @ v` | Cube | 条件支持 | 3D、4 帧 4D、ImageDream 30-step 跑通 | 算术密度高，但显式矩阵 HBM 峰值会掩盖 Cube 上限 |
| `softmax(dim=-1)` attention | Vector + MTE | 条件支持/16 帧阻塞 | 3D 和 4 帧 4D 通过；16 帧 4D OOM | 需要 online/chunk/memory-efficient attention；不能继续显式 materialize 大矩阵 |
| DDIM scheduler `set_timesteps/scale_model_input/step`、`randn_tensor` | Vector + MTE + head | 支持 | ImageDream 1-step 和 30-step 通过 | scheduler 小算子 launch/head 占比 `待测` |
| VAE encode/decode | Cube + Vector + MTE | 支持 | ImageDream 1-step、3D 推理通过 | 质量 parity 未测 |
| `torch.inverse` 小型 4x4、矩阵乘 `@` | Cube/Vector | 支持 | camera view/proj 生成与渲染路径通过 | 仅小矩阵，不外推大 linalg |
| 比较、boolean mask、高级索引、`round/long` | Vector + MTE | 支持 | Gaussian fallback renderer 可出图 | 不规则访问效率 `待测` |
| `scatter_add_` | Vector/SIMT 候选 + MTE | 功能支持，亲和偏低 | fallback renderer 生成文件 | 950 支持 SIMD/SIMT 不规则负载候选，但 scatter/atomic 冲突吞吐无实测归档 |
| `mse_loss/mean/log10` PSNR | Vector/reduce | 未验证 | 训练 eval 未跑 | 需 Objaverse 数据后测 |
| `grid_sample` | Vector + MTE | 未验证 | 只在训练增强路径 | 推理未覆盖，训练需单独 microbench |
| LPIPS/VGG | Cube + Vector + MTE | 未验证/推理禁用 | 推理设置 `lambda_lpips=0` | 启用训练 LPIPS 需要权重、算子和 HBM 验证 |
| xFormers `memory_efficient_attention` | Cube + Vector | 不可用 | Ascend 环境无 CUDA xFormers | 需 NPU 等价实现 |
| CUDA `gsplat.rendering.rasterization` | Vector/SIMT + MTE | 不可用 | 当前使用 fallback | 需要 NPU 原生/等价 renderer |

## Roofline 粗估

### Attention

显式 attention 的 matmul 部分通常有较高算术密度，理论上更接近 Cube 路径；但本项目当前失败点不是 Cube FLOPs，而是 `attn = q @ k.T` 后保留完整 `[B, heads, N, N]` attention matrix，并在 `softmax` 处需要大 HBM workspace。

以 LGM temporal attention 为例，令 `N = num_frames * H * W`，FP16 attention matrix 至少约 `groups * heads * N^2 * 2B`。`num_frames=16` 时，中间分辨率若达到 `H=W=32`，仅 FP16 attention matrix 就是数十 GiB；softmax 若使用 FP32/额外 workspace，峰值可翻倍到与实测 64 GiB OOM 同量级。因此该段亲和性不是“矩阵算术密度高所以好”，而是 Cube 算力可用但 Vector softmax + MTE/HBM materialization 共同主导。

结论：`num_frames=4` 的 attention 峰值仍能落入单卡可承受范围，所以 4D 降帧跑通；`num_frames=16` 必须改为 online softmax、chunk attention、scaled-dot-product/memory-efficient attention 或降低 token 分辨率，才能恢复 NPU 亲和。

### Conv/Linear 主干

LGM recon/interp 的 Conv2d/Linear 主干在 FP16 下应使用 Cube 路径。与 950PR FP16 Cube 平衡点约 270 FLOP/Byte 相比，3x3 Conv 和大 Linear 在合理 tile 下具备较高算术密度，初判不是主要瓶颈。风险在于 U-Net 多尺度 feature、skip concat、view/temporal reshape 可能引入额外 MTE copy，实际效率需通过 profile 拆出 Cube time、Vector time 和 GM bytes。

结论：主干 Conv/Linear 为中高亲和，待测项是 tile 驻留、double buffer 和 layout copy 占比。

### Renderer fallback

Gaussian fallback renderer 主要是投影、小矩阵乘、mask、scatter_add 和 alpha blend，Cube 基本闲置，更多落在 Vector/MTE 或 950 的 SIMT/mixed 候选模型。3DGS 类负载中的排序、tile binning、scatter、atomic、早退分支属于不规则高风险项；当前 skill 的 measured 归档为空，因此不能编造 scatter/atomic 吞吐。

结论：fallback renderer 功能亲和中等、性能/质量亲和偏低；若目标是官方视觉质量，应单独设计 NPU 版 Gaussian rasterizer 并上板 microbench。

## NPU 特有亲和项检查

| 检查项 | 本项目结论 |
| --- | --- |
| tile 驻留与 double buffer | Conv/Linear 主干需要 profile 验证 L0/L1/UB 驻留；当前功能已跑通但没有 tile 利用率数据。16 帧 attention 的主问题是 `N^2` matrix 不可驻留，不能靠扩大融合解决。 |
| 32B/512B/128B sector 对齐 | 950PR 采用 512B L2 cacheline + 4x128B sector；本轮未做 packet/alignment microbench。报告不沿用 A2 的 512B GM 对齐曲线。 |
| repeat/mask 密度 | Norm/activation/softmax、renderer mask/scatter 都需要 aiv_vec_time 或 microbench。当前只确认功能。 |
| layout 折叠 | `view/reshape/permute` 优先按元操作或地址变换处理；只有 `contiguous`、`cat/stack/repeat` 等真实 materialization 计入 GM bytes。16 帧 attention 的 `N^2` matrix 不能视作可折叠 layout。 |
| reduce/state layout | 训练 PSNR、MSE、LPIPS 和 reducer 未进入本轮测试；需单独验证 reduce 轴和 state layout。 |
| 同步边界 | 当前单进程单卡，无 HCCL/CCU/URMA；4D 循环存在 host/head 和多 kernel launch，但没有通信同步。 |
| host/head | ImageDream 30-step diffusion 和 4D 视频循环有明显多 step launch/head；功能通过，性能占比待 profile。 |
| 不规则负载 | renderer fallback 的 scatter_add/mask 属不规则路径，3510 可考虑 SIMD/SIMT mixed，但原语效率无归档实测，必须标 `待测`。 |

## 数学等价/流程重写建议

当前判低亲和的段必须尝试重写，不能停留在原流程：

| 原流程问题 | 重写方向 | 五路径变化 | 精度/风险 |
| --- | --- | --- | --- |
| 16 帧 temporal attention 显式 materialize `[N,N]` | online softmax / chunk attention / blockwise attention，保持数学等价的 softmax 累积 | 降低 MTE/HBM 峰值，Cube matmul 分块执行，Vector softmax 变为分块 reduce | 需与原 PyTorch attention 做误差对齐；FP32 accumulator 后回写 FP16 |
| xFormers CUDA kernel 不可用 | 替换为 NPU 可用 SDPA 或自定义 AscendC memory-efficient attention | 恢复 Cube/Vector/MTE 流水，避免显式大矩阵 | API 支持和性能 `待测` |
| CLIP checkpoint-backed tensor 迁移卡死 | 加载后 CPU materialize，再一次性迁移；或在 loader 层禁用 mmap/backing 特性 | 从卡死的 MTE/host path 变为普通 contiguous copy | 已验证功能；额外加载期内存开销可接受 |
| Gaussian fallback 质量不等价 | 设计 tile binning + depth sort + alpha blend 的 NPU renderer，或持久化可排序 tile layout | 从 Python loop + scatter fallback 转为 SIMD/SIMT mixed；减少 host/head | 排序、atomic、scatter 冲突吞吐必须上板测 |
| 4D recon/interp 同时驻留 | 分阶段加载/释放或模块 offload；forward 后释放不再使用的模型和激活 | 降低 HBM baseline，为 attention workspace 留空间 | 只能增加余量，不能单独解决 64 GiB attention matrix |

## 待测清单

1. 使用 profiler 拆 `infer_3d.py big` 的 Cube time、Vector time、MTE bytes、kernel launch/head，确认 Conv/Linear 主干是否接近预期 Cube 路径。
2. 对 `core/attention.py` 做 `num_frames=4/8/12/16` 梯度测试，记录 attention matrix/workspace 峰值与 OOM 阈值。
3. 上板 microbench `softmax`、`qk matmul + softmax + av matmul` 的 chunk/online 实现，比较显式 attention fallback。
4. 对 CLIP materialize 路径记录加载期 CPU 内存峰值和迁移耗时，确认不同 safetensors 文件均稳定。
5. 对 renderer fallback 的 `scatter_add_`、mask、boolean indexing 做 3510 microbench；当前无 measured 归档，效率全部 `待测`。
6. 若需要训练/PSNR，补齐 Objaverse 数据后单独验证 `mse_loss/mean/log10`、LPIPS/VGG、`grid_sample` 和 dataloader I/O。
7. 若要多卡，单独验证 HCCL/CCU/框架 collective；本轮没有通信路径结论。

## 结论表

| 子段 | 主导路径 | 算术密度 vs 平衡点 | 亲和结论 | 基于流程 | 待测 |
| --- | --- | --- | --- | --- | --- |
| ImageDream CLIP 权重迁移 | MTE/FixPipe + host/head | 不适用 | 原流程差，materialize 后中 | 基于等价加载流程重写 | loader/backing 根因、加载期内存 |
| ImageDream diffusion | Cube + Vector + MTE | Conv/attention 部分高；scheduler/elementwise 低 | 中 | 当前 NPU fallback 流程 | profile 分轨、质量 parity |
| LGM 3D recon | Cube 主导，Vector/MTE 辅助 | Conv/Linear 初判高于 Cube 平衡点，layout bytes 待测 | 中高 | 当前流程 | tile 驻留、renderer 替换 |
| LGM 4D `num_frames=4` | Cube + Vector + MTE | 可承受范围内 | 中 | 当前流程 | 性能 profile |
| LGM 4D `num_frames=16` | Vector softmax + MTE/HBM 主导 | 显式 attention matrix 使 HBM 峰值压倒 Cube 算力 | 差 | 原流程 | memory-efficient/chunk attention |
| Gaussian renderer fallback | Vector/MTE/SIMT 候选 | 不规则负载，不用 Cube 平衡点判断 | 功能中，质量/性能低 | 当前 fallback | scatter/atomic/sort microbench |
| 训练/PSNR | Vector/reduce + I/O | 待测 | 未验证 | 原训练流程 | 数据、LPIPS、grid_sample、reduce |

本轮亲和性结论不替代上板 profile。当前可对外陈述的是：Ascend 950PR 上 3D 推理和 4 帧 4D 推理已功能跑通；完整 README 默认 16 帧 4D 的主要阻塞是显式 attention 的 HBM 峰值与 CUDA renderer 替代，不是基础 Conv/Linear/Norm 算子缺失。
