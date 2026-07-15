---
name: train-pytorch-models
description: Build and operate reproducible NanaTracking PyTorch model training, including model heads, losses, AMP, deterministic seeds, checkpoint and resume behavior, experiment metadata, and failure handling. Use for model architecture, optimizer, training loop, or checkpoint changes.
---

# Train PyTorch models

1. Read `references/training-contract.md` before editing the model or training engine.
2. Start from a validated manifest and resolved typed configuration.
3. Keep expression, identity, and pose objectives explicit and use named output heads.
4. Record seed, revisions, Git state, lock digest, device, precision, and resolved configuration.
5. Save enough optimizer and RNG state to resume without restarting the schedule.
6. Keep synthetic smoke evidence clearly separate from real-data accuracy evidence.
7. Run deterministic, resume, evaluation, and export tests after training changes.
