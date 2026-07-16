"""Shared ONNX Runtime session policy for paced tracking inference."""

from pathlib import Path
from typing import Protocol, cast

import onnxruntime as ort


class _OrtSessionOptions(Protocol):
    execution_mode: object
    intra_op_num_threads: int
    inter_op_num_threads: int

    def add_session_config_entry(self, key: str, value: str) -> None: ...


def create_ort_session(
    model_path: Path,
    *,
    providers: list[str] | None,
    tensorrt_fp16: bool,
    intra_threads: int,
    allow_spinning: bool,
) -> ort.InferenceSession:
    """Create one validated sequential ORT session with explicit idle-thread behavior."""

    if intra_threads < 1:
        raise ValueError("ORT intra-op thread count must be positive")
    requested = providers or ["CPUExecutionProvider"]
    if tensorrt_fp16 and "TensorrtExecutionProvider" not in requested:
        raise ValueError("TensorRT FP16 requires TensorrtExecutionProvider")
    available = cast(list[str], ort.get_available_providers())
    unavailable = set(requested).difference(available)
    if unavailable:
        raise RuntimeError(f"requested ONNX Runtime providers are unavailable: {unavailable}")
    provider_specs: list[str | tuple[str, dict[str, str]]] = [
        (provider, {"trt_fp16_enable": "1"})
        if provider == "TensorrtExecutionProvider" and tensorrt_fp16
        else provider
        for provider in requested
    ]
    options = cast(_OrtSessionOptions, ort.SessionOptions())
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = 1
    spinning = "1" if allow_spinning else "0"
    options.add_session_config_entry("session.intra_op.allow_spinning", spinning)
    options.add_session_config_entry("session.inter_op.allow_spinning", spinning)
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=provider_specs,
    )
