"""Ascend NPU runtime helpers used by every REAP execution path.

The project intentionally supports Ascend and CPU offload only.  Importing this
module registers the ``npu`` PyTorch backend when ``torch_npu`` is installed,
while keeping CPU-only unit tests and safetensors plan pruning importable.
"""

from __future__ import annotations

from contextlib import nullcontext
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # CPU-only validation remains supported.
    _TORCH_NPU_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def configure_npu_environment() -> None:
    """Apply safe defaults shared by calibration, pruning and evaluation."""
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1800")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "1800")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _npu_api() -> Any | None:
    return getattr(torch, "npu", None)


def is_available() -> bool:
    """Return whether a usable Ascend NPU runtime is available."""
    api = _npu_api()
    return bool(api is not None and api.is_available())


def require_npu() -> torch.device:
    """Validate the Ascend runtime and return the first visible NPU device."""
    configure_npu_environment()
    if _TORCH_NPU_IMPORT_ERROR is not None:
        raise RuntimeError(
            "torch_npu is required for REAP observation and model execution. "
            "Run inside the supported Ascend image or install requirements-ascend.txt."
        ) from _TORCH_NPU_IMPORT_ERROR
    if not is_available():
        raise RuntimeError(
            "No Ascend NPU is visible. Check the CANN driver mounts and "
            "ASCEND_RT_VISIBLE_DEVICES."
        )
    return torch.device("npu:0")


def device_count(require: bool = True) -> int:
    """Return the number of visible NPUs, optionally failing when none exist."""
    if require:
        require_npu()
    api = _npu_api()
    return int(api.device_count()) if api is not None and api.is_available() else 0


def default_device(require: bool = True) -> torch.device:
    """Return ``npu:0`` or CPU when an explicit CPU-only path is allowed."""
    if require:
        return require_npu()
    return torch.device("npu:0" if is_available() else "cpu")


def synchronize() -> None:
    """Synchronize outstanding NPU work when the runtime is active."""
    if is_available():
        _npu_api().synchronize()


def empty_cache(synchronize_first: bool = False) -> None:
    """Release cached NPU allocations through the torch-npu runtime."""
    if not is_available():
        return
    if synchronize_first:
        synchronize()
    _npu_api().empty_cache()


def memory_gib() -> tuple[float, float]:
    """Return allocated and reserved NPU memory in GiB."""
    if not is_available():
        return 0.0, 0.0
    api = _npu_api()
    scale = float(1024**3)
    return api.memory_allocated() / scale, api.memory_reserved() / scale


def disabled_autocast(device: torch.device | str):
    """Return a disabled autocast context appropriate for NPU or CPU replay."""
    device_type = torch.device(device).type
    if device_type == "npu":
        return torch.autocast(device_type="npu", enabled=False)
    return nullcontext()


configure_npu_environment()
