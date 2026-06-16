import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_CACHE_ROOT = str(Path(__file__).resolve().parents[1] / ".cache")


def configure_cache_dirs():
    cache_root = os.environ.setdefault("L4GM_CACHE_ROOT", DEFAULT_CACHE_ROOT)
    for name in ("huggingface", "torch", "xdg", "pip"):
        os.makedirs(os.path.join(cache_root, name), exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("DIFFUSERS_CACHE", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_root, "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cache_root, "xdg"))
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(cache_root, "pip"))


def _enable_torch_npu():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu")


def get_torch_device():
    if _enable_torch_npu() and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_autocast(device, dtype=torch.float16):
    if device.type in {"cuda", "npu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def empty_cache():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_info():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.mem_get_info()
    if torch.cuda.is_available():
        return torch.cuda.mem_get_info()
    return None


def is_npu_device(device):
    if device is None:
        return False
    return torch.device(device).type == "npu"


def materialize_cpu_module_tensors(module: nn.Module):
    """Clone CPU-backed parameters/buffers into ordinary contiguous tensors.

    Some safetensors-backed Transformers tensors can hang when torch-npu copies
    them directly to NPU. Materializing them on CPU first avoids that transfer
    path while preserving dtype and requires_grad.
    """
    if module is None:
        return None

    with torch.no_grad():
        for submodule in module.modules():
            for name, param in list(submodule._parameters.items()):
                if param is None or param.device.type != "cpu":
                    continue
                data = param.detach().clone().contiguous()
                submodule._parameters[name] = nn.Parameter(
                    data, requires_grad=param.requires_grad
                )

            for name, buffer in list(submodule._buffers.items()):
                if buffer is None or buffer.device.type != "cpu":
                    continue
                submodule._buffers[name] = buffer.detach().clone().contiguous()

    return module
