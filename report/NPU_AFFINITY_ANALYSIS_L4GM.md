# L4GM-official NPU 亲和性分析

生成日期：2026-06-16

## 分析结论

L4GM-official 在 Ascend950PR 与 `torch_npu 2.10.0` 环境下具备可适配运行基础，但不是开箱即用的全量 NPU 原生项目。项目核心 LGM recon/interp 网络主要由 PyTorch conv、linear、attention、interpolation、normalization 和 tensor reshape/reduce 组成，经过 CUDA device 假设清理后，checkpoint 加载、reduced forward、3D 推理和降帧 4D 推理均已在 NPU 上跑通。

当前 NPU 亲和风险集中在三条链路：第一，ImageDream 的 CLIP text/image encoder 从 safetensors/diffusers loader 得到的 CPU tensor 直接 `.to(npu)` 会卡死，需要先在 CPU 侧 clone materialize；第二，项目原依赖 xFormers 与 `gsplat` 这类 CUDA 扩展，NPU 只能走 PyTorch fallback 或简化 renderer；第三，4D 默认 `num_frames=16` 在 PyTorch attention fallback 下会一次性构造约 64 GiB attention matrix，导致 HBM OOM。

因此，本次结论是：3D demo inference 已具备 NPU 功能可运行性；4D inference 在 `num_frames=4` 下可运行；README 默认 16 帧 4D 尚未完成 NPU 亲和，需要专门做 attention 内存优化和 renderer 替换后，才能声明完整配置适配。

## 已验证的 NPU 友好路径

本轮实测通过的路径说明项目主体并非被 NPU 算子全面阻塞。LGM recon/interp 模型可以加载 safetensors 权重并迁移到 NPU，`forward_gaussians` 在 reduced 输入下输出有限值。ImageDream 的 VAE、UNet 迁移到 NPU 正常，CLIP encoder 经 materialize 后也可以迁移并完成 1-step 多视图生成。

| NPU 功能/模块路径 | 实测状态 | 验证方式 | 边界说明 |
| --- | --- | --- | --- |
| NPU 设备发现与后四卡隔离 | 支持 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 后 `torch.npu.device_count()==4` | 进程内 `npu:0..3` 映射到物理后四张卡；本轮主要使用单进程单卡。 |
| LGM recon 模型 | 支持 | `recon.safetensors` 加载 `missing/unexpected=0`，reduced forward finite | 官方 3D 推理已跑通；视觉质量受 renderer fallback 影响。 |
| LGM interp 模型 | 支持 | `interp.safetensors` 加载 `missing/unexpected=0`，reduced forward finite | 4D 降帧推理已跑通；16 帧受 attention HBM 阻塞。 |
| ImageDream VAE/UNet | 支持 | `vae.to(npu)` 约 1.89 秒，`unet.to(npu)` 约 2.25 秒 | xFormers 不可用时走 PyTorch attention fallback。 |
| ImageDream CLIP encoder materialize 后迁移 | 支持 | materialize 后 `text_encoder.to(npu)` 约 2.03 秒，`image_encoder.to(npu)` 约 2.04 秒 | 必须避开原始 checkpoint tensor 的直接 NPU copy 路径。 |
| ImageDream 1-step 前向 | 支持 | 256x256 输入，`num_inference_steps=1` 输出 `(5, 256, 256, 3)`，finite True | 该测试验证 CLIP/VAE/UNet 调用链，不代表最终生成质量。 |
| 官方 3D 推理入口 | 支持 | `infer_3d.py big --num_frames 1` 生成 PNG/MP4 | 当前 renderer fallback 可产出结果，但不等价于 CUDA `gsplat`。 |
| 官方 4D 推理入口降帧 | 支持 | `infer_4d.py big --num_frames 4` 生成 4D 视频 | 说明 4D 代码链路可跑；默认 16 帧仍需降显存适配。 |

## 算子级亲和性矩阵

下表按实际代码路径和本轮运行结果归纳算子亲和性。这里的“通过”表示该算子或算子组合已经出现在 NPU smoke、ImageDream 1-step、`infer_3d.py` 或 `infer_4d.py --num_frames 4` 成功链路中；“条件支持”表示功能可用但有显存、质量或前置 materialize 条件；“未验证”表示源码中存在但本轮没有进入该路径。

| 算子/算子组合 | 代码路径 | NPU 状态 | 实测依据 | 风险边界 |
| --- | --- | --- | --- | --- |
| CPU tensor -> NPU tensor copy：`Tensor.to("npu")` | 权重迁移、输入迁移 | 条件支持 | 普通 float16/float32/int64 tensor、synthetic `Embedding/Linear/Conv2d` 权重可在约 2 秒迁移。 | safetensors/diffusers loader 得到的 CLIP weight 直接 `.to(npu)` 会超时，必须先 `detach().clone().contiguous()` materialize。 |
| `clone`、`detach`、`contiguous`、`nn.Parameter` 重建 | `core/device.py` materialize helper | 支持 | CLIP text/image encoder materialize 后 `.to(npu)` 通过，完整 `pipe.to(npu)` 约 2.99 秒完成。 | 仅作为 NPU 迁移前处理；不是性能优化算子。 |
| `Conv2d`、1x1/3x3 conv、stride conv | `core/unet.py`、`core/models.py`、ImageDream VAE/UNet | 支持 | LGM reduced forward、3D 推理、4 帧 4D 推理均通过；ImageDream 30-step diffusion 通过。 | `gsplat` 不是 Conv2d 路径，不能由此推断 renderer 质量。 |
| `GroupNorm`、`LayerNorm` | LGM UNet、CLIP、ImageDream UNet/VAE | 支持 | 小型 LayerNorm 单独迁移通过；3D/4D 成功链路覆盖 GroupNorm；CLIP materialize 后 1-step 前向通过。 | 未做 CPU/NPU 数值逐层 parity，只做功能可运行验证。 |
| 激活函数：`silu`、`softplus`、`sigmoid`、`tanh`、`clamp`、`normalize` | `core/unet.py`、`core/models.py`、`core/gs.py` | 支持 | LGM gaussian 输出后处理和渲染链路通过，输出 finite。 | `F.normalize` 用于 quaternion/rotation 归一化，未单独做数值误差评估。 |
| shape/layout：`view`、`reshape`、`permute`、`transpose`、`contiguous`、`squeeze`、`unsqueeze` | LGM/4D temporal-view reshape、ImageDream pipeline | 支持 | 3D 与 4 帧 4D 完整通过，覆盖频繁 layout 变换。 | 主要是元数据/拷贝路径，风险低于 attention 和 renderer。 |
| 拼接/堆叠：`cat`、`stack`、`repeat`、`repeat_interleave`、`chunk` | rays embedding、prompt/image embeds、latent batch 构造、4D 输出拼接 | 支持 | ImageDream 1-step、30-step 3D 推理和 4 帧 4D 推理通过。 | 大 batch/高帧数时会放大 HBM 占用。 |
| resize：`F.interpolate` bilinear/nearest | 输入预处理、上采样、VAE latent image resize、视频输出 resize | 支持 | 3D/4D 推理通过，覆盖 bilinear 与 nearest。 | 训练数据增强里的更大输入未验证。 |
| pooling：`AvgPool2d` | LGM DownBlock downsample 可选路径 | 支持但覆盖有限 | `big` 配置 forward 可进入下采样路径，3D/4D 推理通过。 | 未单独压测不同 dtype/shape。 |
| Linear projection：`nn.Linear` | LGM attention qkv/proj、CLIP、ImageDream attention/MLP | 条件支持 | synthetic Linear 可迁移；CLIP materialize 后 1-step 前向通过；LGM forward 通过。 | CLIP checkpoint-backed Linear weight 直接 `.to(npu)` 会超时；需要 materialize。 |
| embedding lookup：`nn.Embedding` | CLIP token embedding、ImageDream label/camera embedding | 条件支持 | synthetic `Embedding(49408,1024)` 可迁移；CLIP materialize 后前向通过。 | CLIP checkpoint-backed embedding weight 直接 `.to(npu)` 超时。 |
| attention matmul：`q @ k.transpose`、`attn @ v`、`torch.matmul` | `core/attention.py`、`mvdream/mv_unet.py`、CLIP/ImageDream attention | 条件支持 | 3D、4 帧 4D、ImageDream 30-step diffusion 均通过。 | 随 token 数二次增长；16 帧 4D 的显存峰值来自 attention matrix。 |
| `softmax(dim=-1)` | LGM MV/Temp attention、ImageDream attention | 条件支持/默认 16 帧阻塞 | 3D 和 4 帧 4D 通过；16 帧 4D 在 `attn.softmax(dim=-1)` 处申请 64 GiB 并 OOM。 | 需要 memory-efficient attention 或 chunk attention，不能继续依赖显式大矩阵。 |
| random/scheduler tensor：`randn_tensor`、DDIM scheduler `set_timesteps/step/scale_model_input` | `mvdream/pipeline_mvdream.py` | 支持 | ImageDream 1-step 与 30-step 3D 推理通过。 | 未对随机一致性和速度做专项评估。 |
| VAE encode/decode 分布采样 | ImageDream image latent encode/decode | 支持 | 1-step ImageDream 前向和 3D 推理通过。 | 仅功能验证，未做图像质量 parity。 |
| 相机矩阵：`torch.inverse`、矩阵乘 `@`、`transpose` | `infer_3d.py`、`infer_4d.py`、dataset provider | 支持 | 3D 渲染对齐和 4D 输出路径通过。 | 仅小型 4x4 矩阵；不能代表大规模 linalg 亲和。 |
| masking/indexing：比较、boolean mask、`round`、`long`、高级索引 | Gaussian fallback renderer | 支持 | fallback rasterizer 生成 PNG/MP4。 | 该路径功能可用但不是质量等价 renderer。 |
| `scatter_add_` | Gaussian fallback renderer alpha/color splat | 支持但质量受限 | 3D 与 4D 降帧结果文件已生成。 | 当前 scatter splat 是简化实现，不包含官方 `gsplat` 的排序、覆盖和反走样。 |
| `torch.zeros_like`、`torch.zeros`、`torch.ones` | ImageDream negative embeds、renderer buffers、背景色 | 支持 | 多条成功链路覆盖；仅观察到 base format warning，不影响功能。 | base format warning 可能影响性能，但未导致失败。 |
| `mse_loss`、`mean`、`log10` | 训练/eval PSNR 路径 | 未验证 | 仓内 PSNR 路径需要外部 Objaverse 数据，本轮未进入。 | 不能报告训练指标或 PSNR benchmark。 |
| `grid_sample` | `core/utils.py` grid distortion 数据增强 | 未验证 | 推理入口没有进入该训练增强路径。 | 训练适配时需要单独测试。 |
| LPIPS/VGG | `core/models.py` 训练 loss | 未验证/推理禁用 | 推理中设置 `lambda_lpips=0`，避免额外权重下载。 | 训练若启用 LPIPS，需要验证 VGG 权重、算子和内存。 |
| xFormers `memory_efficient_attention` | `core/attention.py` 原优先路径 | 阻塞/不可用 | 当前 Ascend 环境无 CUDA xFormers，已 fallback 到 PyTorch attention。 | 需要替换为 NPU 可用的 memory-efficient attention。 |
| `gsplat.rendering.rasterization` | `core/gs.py` 原 renderer | 阻塞/不可用 | Ascend 环境无法加载 CUDA `gsplat`，当前使用 fallback。 | 完整视觉质量复现必须替换 renderer。 |

从算子角度看，项目不是缺少基础 Conv/Linear/Norm/MatMul 能力，而是被两个“组合路径”限制：一是 checkpoint-backed tensor 的 NPU copy path，二是 attention 显式矩阵在 16 帧 4D 下的 HBM 峰值。前者已经通过 materialize 规避；后者仍是完整默认配置的主阻塞。

## 阻塞项分析

### ImageDream CLIP tensor 迁移卡死

最初的端到端阻塞表现为 `MVDreamPipeline.from_pretrained(...)` 能在本地权重上约 1 秒完成，但 `pipe.to(npu)` 多分钟不返回，`infer_3d.py` 因此无法进入 `[INFO] Processing` 和文件输出阶段。组件拆分后，`vae.to(npu)` 与 `unet.to(npu)` 都在约 2 秒内完成；`text_encoder.to(npu)` 和 `image_encoder.to(npu)` 均 240 秒超时。

进一步拆分显示，CLIP 的大型 embedding 和 transformer block 子模块会超时，小型 LayerNorm 可以迁移。基础 dtype 与形状测试排除了“Ascend 不支持该形状”的解释：同形状 synthetic `Embedding(49408, 1024)`、`Linear(1024, 4096)`、`Linear(1280, 5120)`、patch `Conv2d` 都能约 2 秒迁移到 NPU。真正卡住的是从 checkpoint loader 得到的 CLIP 权重 tensor 直接 `.to(npu)`。

关键对照结果如下：

| 测试对象 | 直接 `.to(npu)` | `detach().clone().contiguous().to(npu)` |
| --- | --- | --- |
| `text_encoder.text_model.embeddings.token_embedding.weight` | 90 秒超时 | 约 1.93 秒完成 |
| `text_encoder` layer 0 `q_proj.weight` | 90 秒超时 | 约 1.94 秒完成 |
| `image_encoder` patch embedding weight | 90 秒超时 | 约 1.90 秒完成 |
| `image_encoder` layer 0 `q_proj.weight` | 90 秒超时 | 约 1.90 秒完成 |

因此判断为 safetensors/diffusers/Transformers loader 产生的 CPU tensor backing 与 torch-npu copy path 不兼容。当前适配是在 `core/device.py` 中提供 `materialize_cpu_module_tensors()`，并在 `mvdream/pipeline_mvdream.py` 中对 `text_encoder` 和 `image_encoder` 的 NPU 迁移做预处理。适配后，完整 `pipe.to(npu)` 约 2.99 秒完成，1-step ImageDream 前向通过。

### xFormers 缺失与 PyTorch attention fallback

项目原 attention 路径优先使用 xFormers memory efficient attention。Ascend NPU 环境不能使用 CUDA xFormers，因此当前走 `core/attention.py` 中的 PyTorch fallback：先计算 `attn = q @ k.transpose(-2, -1)`，再执行 `attn.softmax(dim=-1)`。这条路径功能上可运行，但显式 materialize attention matrix，显存复杂度为 token 数的二次方。

3D 与 4 帧 4D 可以接受这条 fallback；16 帧 4D 则会在 temporal attention 上触发 HBM OOM。OOM 点明确位于 `core/attention.py` 的 softmax，错误信息为申请 64 GiB HBM；在空闲后卡上复测时，模型已占约 36.93 GiB active，当前空闲约 45.31 GiB，无法满足单次 64 GiB 分配。

这说明完整 16 帧 4D 的关键不是简单换卡或清 cache，而是必须避免一次性构造大 attention matrix。可选方向包括实现 NPU 可用的 memory-efficient attention、按 temporal/spatial token chunk、降低 token 分辨率，或重构 4D 处理流程让 recon/interp 分阶段释放模型与激活。

### CUDA-only Gaussian rasterizer

`core/gs.py` 原始路径依赖 `gsplat.rendering.rasterization`。该扩展在当前 Ascend 环境不可用，因此已加入简化 fallback rasterization，使用投影、scatter 和 alpha blending 在 NPU 上产出图像。这解决了功能链路阻塞，使 `infer_3d.py` 与降帧 `infer_4d.py` 能生成文件。

该 fallback 不是质量等价实现。它缺少 `gsplat` 的完整 splat 覆盖、排序、可微/反走样和高质量 alpha 合成能力，不能作为官方视觉结果对齐依据。若后续目标是论文/README 级视觉复现，需要 NPU 原生 Gaussian rasterizer 或跨平台等价 renderer。

### 官方 4D 默认帧数显存阻塞

`infer_4d.py big --num_frames 16` 在物理后四卡内映射到空闲 NPU 后仍报 OOM，错误为在 attention softmax 处申请 64 GiB。把 `num_frames` 降为 4 后，完整 `infer_4d.py` 可以跑通并生成无插值、插值与 fixed 视角视频。这验证了 4D 链路本身可迁移，阻塞集中在默认帧数下 attention matrix 的显存峰值。

当前 16 帧配置不能通过环境变量或更换后卡根治。即使单卡 HBM 约 112 GiB，模型、激活、runtime reserve 与单个 64 GiB attention 分配叠加后仍超出可用连续空间。后续适配应以降低 attention 峰值为主，而不是继续尝试下载、重装或清理系统盘。

## 模块级 NPU 亲和矩阵

| 项目能力 | NPU 亲和状态 | 依据 | 后续工作 |
| --- | --- | --- | --- |
| 环境与缓存 | 高 | cache/权重/输出均可放挂载盘，后四卡可见。 | 保持 `scripts/npu_env.sh` 作为统一入口。 |
| LGM recon forward | 高 | checkpoint load 与 NPU forward 通过，3D 推理完成。 | 若追求质量，需要 renderer 对齐。 |
| LGM interp forward | 中 | reduced forward 与 4 帧 4D 通过。 | 16 帧需 attention 优化。 |
| ImageDream VAE/UNet | 中高 | 迁移和 1-step 前向通过。 | 进一步做 30-step 质量/性能采样。 |
| ImageDream CLIP encoders | 中 | 原 tensor 迁移卡死，materialize 后通过。 | 保留 materialize 补丁；后续可定位 torch-npu 与 safetensors backing 的底层兼容问题。 |
| Attention | 中低 | PyTorch fallback 功能可用，但 16 帧 OOM。 | 实现 memory-efficient attention 或 chunk attention。 |
| Gaussian rasterization | 低 | `gsplat` 不可用，当前 fallback 只保功能。 | 替换为 NPU 原生/等价 renderer。 |
| 训练与 PSNR eval | 未验证 | 需要外部 Objaverse 数据与 datalist。 | 补数据后单独验证 dataloader、loss、PSNR 和多卡训练。 |
| 多卡并行 | 未验证 | 当前推理单进程单卡；只验证后四卡可见。 | 若要利用 4 张卡，需要新增并行策略，当前代码不会自动分摊单样本显存。 |

## 能否适配跑通

可以适配跑通，但要区分“功能跑通”和“官方默认完整配置跑通”。

当前已经跑通的范围是：ImageDream 多视图生成、LGM 3D 推理、LGM 4D 降帧推理，均在 Ascend NPU 上完成并生成文件。关键适配点是 CLIP encoder materialize、CUDA device 假设替换、xFormers fallback 和 renderer fallback。

尚未跑通的是 README 默认 16 帧 4D 配置。这个阻塞已经定位到 temporal attention 的 64 GiB 单次分配，而不是权重缺失、网络下载、系统盘空间、NPU 可见性或某个 checkpoint 损坏。只要仍使用当前 PyTorch fallback attention，16 帧配置大概率继续 OOM。

要把 16 帧配置推进到可交付状态，建议按以下优先级处理：

1. 在 `core/attention.py` 为 NPU 增加 memory-efficient attention 路径，避免显式创建完整 `[B, heads, N, N]` attention matrix。
2. 如果后端 attention 能力不足，按 token 或时间维做 chunk，先保证数值可接受，再评估速度。
3. 将 `infer_4d.py` 中 recon 与 interp 模型分阶段驻留，forward 后释放无关模型和中间 tensor，降低非 attention HBM 占用。
4. 保留 `num_frames=4` 作为当前 NPU smoke/回归配置，后续每次改 attention 后先跑 4 帧，再逐步拉到 8、12、16 帧。
5. 替换 renderer fallback，建立 CUDA `gsplat` 与 NPU renderer 的图像/视频对齐检查，否则即使 16 帧跑通，也不能声明视觉质量等价。

## 结论

L4GM-official 的 NPU 适配不是被整体框架或模型权重阻断。经过针对性 patch，3D 和降帧 4D 已经可以在 NPU 上产出结果，说明主干模型和 ImageDream 组件具备可迁移性。剩余核心问题是 16 帧 4D 的 attention 显存峰值和 CUDA rasterizer 替代；这两个问题解决前，不应对外宣称完整 README 配置和官方视觉质量已经在 NPU 上复现。
