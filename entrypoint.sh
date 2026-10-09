#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
import torch_npu  # noqa: F401

if not torch.npu.is_available():
    raise SystemExit(
        "Ascend NPU unavailable: check driver mounts and ASCEND_RT_VISIBLE_DEVICES"
    )
print(f"Ascend runtime ready: {torch.npu.device_count()} visible NPU(s)")
PY

exec "$@"
