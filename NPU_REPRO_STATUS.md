# L4GM Ascend NPU Reproduction Status

## Scope

This note records the NPU reproduction status for L4GM-official without
including host aliases, container IDs, proxy endpoints, local user paths, or
other deployment-private details. The target runtime is an Ascend950PR
container with only the last four physical cards exposed through
`ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`.

All model weights, caches, and generated outputs should stay under the mounted
project filesystem. The helper script `scripts/npu_env.sh` defaults cache
directories to `${L4GM_REPO_ROOT}/.cache`, and callers may override this with
`L4GM_CACHE_ROOT`.

## Reproduction Method And Metrics

The repository does not ship a standalone `benchmark/`, `eval.py`, or benchmark
score script. Official README reproduction is inference-oriented:

1. `infer_3d.py`: input video -> ImageDream multiview generation -> L4GM recon
   model -> 3D PNG/MP4 outputs.
2. `infer_4d.py`: generated multiview PNGs plus `recon.safetensors` and
   `interp.safetensors` -> 4D videos.

The only quantitative metric implemented in the repository is training/eval
PSNR in `main.py` and `core/models.py`, with training loss composed of MSE plus
optional LPIPS. That path requires external Objaverse render data and datalist
files, so it is not reproducible from the repository alone.

## Completed Adaptation

- Added `core/device.py` for NPU device selection, cache configuration,
  autocast, cache cleanup, and CPU tensor materialization before NPU transfer.
- Updated `infer_3d.py`, `infer_4d.py`, and `main.py` to use the shared device
  helper instead of CUDA-only assumptions.
- Disabled inference-time LPIPS initialization in `infer_3d.py` and
  `infer_4d.py` to avoid unrelated VGG weight downloads during demo inference.
- Updated ImageDream attention paths so xFormers is optional and PyTorch
  attention fallback is used when CUDA xFormers is unavailable.
- Added materialization before moving ImageDream `text_encoder` and
  `image_encoder` to NPU. This avoids the torch-npu hang observed when copying
  checkpoint-backed CLIP tensors directly to NPU.
- Added a simplified NPU fallback rasterization path for environments where
  CUDA `gsplat` is unavailable. This is a functional fallback, not a
  quality-equivalent replacement for official Gaussian rasterization.
- Added `scripts/npu_env.sh` and `scripts/npu_smoke.py` for repeatable NPU
  setup and reduced model smoke testing.

## Verified Results

The target environment was verified with:

```text
torch 2.10.0+cpu
torch_npu 2.10.0
torch.npu.is_available() == True
torch.npu.device_count() == 4
```

L4GM checkpoint load and reduced NPU forward smoke passed for both recon and
interp checkpoints:

```text
missing=0 unexpected=0
forward output shape: (1, 4096, 14)
output device: npu:0
finite: True
```

ImageDream component testing showed:

```text
vae.to(npu)  -> passed
unet.to(npu) -> passed
text_encoder.to(npu) without materialize -> timeout
image_encoder.to(npu) without materialize -> timeout
text_encoder/image_encoder with materialize -> passed
full pipe.to(npu) after materialize -> about 3 seconds
```

ImageDream 1-step forward passed with finite output:

```text
output shape: (5, 256, 256, 3)
finite: True
```

Official 3D inference passed:

```bash
python infer_3d.py big \
  --workspace results_npu_materialized \
  --resume pretrained/recon.safetensors \
  --num_frames 1 \
  --test_path data_test/otter-on-surfboard_fg.mp4
```

The run generated the expected original frame, multiview PNGs, rendered PNGs,
and 3D MP4.

4D inference with the README default `num_frames=16` currently fails with NPU
OOM in temporal attention. The failing path is `core/attention.py` at
`attn.softmax(dim=-1)`, where the PyTorch attention fallback attempts a 64 GiB
allocation for the explicit attention matrix.

4D inference with reduced frame count passed:

```bash
python infer_4d.py big \
  --workspace results_npu_materialized \
  --resume pretrained/recon.safetensors \
  --interpresume pretrained/interp.safetensors \
  --num_frames 4 \
  --test_path data_test/otter-on-surfboard_fg.mp4
```

The run generated both non-interpolated and interpolated 4D videos, including
fixed-view outputs.

## Current Boundary

The project is functionally runnable on NPU for 3D inference and reduced-frame
4D inference. It should not yet be reported as a full README-default
reproduction because `infer_4d.py --num_frames 16` is blocked by attention HBM
usage, and the current Gaussian renderer fallback is not visually equivalent to
CUDA `gsplat`.

The next adaptation target is an NPU-friendly memory-efficient attention path
or chunked temporal attention in `core/attention.py`, followed by a proper
NPU-compatible Gaussian rasterizer if visual quality parity is required.
