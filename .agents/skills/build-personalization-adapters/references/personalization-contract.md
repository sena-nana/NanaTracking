# Personalization contract

- Level A: offset, robust range, scale, deadzone, clamp, left/right balance, affine correction, EMA,
  least squares, or RLS without a training framework.
- Level B: freeze the encoder, train a small affine/MLP/TCN/low-rank residual adapter in PyTorch,
  export it separately to ONNX, and validate overfitting and version compatibility.
- Level C: optional bounded online adapter with explicit user action, hard compute limits, high-
  confidence samples, cancellation, reset, rollback, and drift protection.
- A feature revision change invalidates learned adapters by default. Deleting a profile never
  deletes or modifies the base model package.
