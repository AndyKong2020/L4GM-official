import torch

from core.device import configure_cache_dirs, device_autocast, get_torch_device
from core.models import LGM
from core.options import Options


def main():
    configure_cache_dirs()

    device = get_torch_device()
    if device.type != "npu":
        raise RuntimeError(f"Expected NPU device, got {device}")
    if torch.npu.device_count() != 4:
        raise RuntimeError(f"Expected 4 visible NPU devices, got {torch.npu.device_count()}")

    opt = Options(
        input_size=32,
        down_channels=(32, 64),
        down_attention=(False, False),
        mid_attention=False,
        up_channels=(64, 32),
        up_attention=(False, False),
        splat_size=32,
        output_size=64,
        batch_size=1,
        num_frames=1,
        num_views=4,
        num_input_views=4,
        lambda_lpips=0,
        use_temp_attn=False,
    )

    model = LGM(opt).half().to(device).eval()
    sample = torch.randn(1, opt.num_frames * opt.num_input_views, 9, opt.input_size, opt.input_size, device=device, dtype=torch.float16)
    with torch.no_grad(), device_autocast(device, dtype=torch.float16):
        gaussians = model.forward_gaussians(sample)

    print("device", device)
    print("visible_npu_count", torch.npu.device_count())
    print("gaussians_shape", tuple(gaussians.shape))
    print("gaussians_device", gaussians.device)
    print("gaussians_dtype", gaussians.dtype)
    print("gaussians_finite", bool(torch.isfinite(gaussians).all().item()))


if __name__ == "__main__":
    main()
