# Evaluation contract

- Core quality: per-signal error/correlation, neutral jitter, dynamic delay, peak attenuation,
  left/right asymmetry, identity bias, geometry consistency, and confidence calibration.
- Failure quality: occlusion, out-of-frame, tracking lost, recovery time, and state classification.
- Runtime quality: capture-to-result and result-age P50/P95/P99, CPU, hottest thread, GPU, VRAM,
  copies, queue depth, and long-run growth.
- ONNX and optimized runtimes compare each named output against PyTorch fixed vectors with declared
  tolerances.
