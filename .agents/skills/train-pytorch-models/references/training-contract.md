# Training contract

- A run owns `config.resolved.yaml`, append-only metrics, checkpoints, and immutable provenance.
- Checkpoints include model, optimizer, RNG, step/epoch, config digest, data revision, NTP and Signal
  Registry revisions, Git state, and dependency lock digest.
- AMP and accelerator choices belong in configuration and metadata; never silently change them.
- Multi-task losses report components separately once real heads are introduced.
- Resume tests must prove monotonic step progression and restored optimizer/RNG state.
