"""Runtime diagnostics for Python, PyTorch, and ONNX Runtime."""

import platform
import sys
import sysconfig
from typing import Any

import onnxruntime as ort
import torch


def doctor_report() -> dict[str, Any]:
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    jit = getattr(sys, "_jit", None)
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": bool(gil_enabled),
        "jit_available": bool(jit and jit.is_available()),
        "jit_enabled": bool(jit and jit.is_enabled()),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "onnxruntime": ort.__version__,
        "onnxruntime_providers": ort.get_available_providers(),
    }
