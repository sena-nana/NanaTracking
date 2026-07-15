# Training data governance and failure feedback v1

## License admission

Every teacher source references one immutable license review. Admission requires an approved review
and explicit permission for collection, distillation, pseudo-labeling, and commercial model
training. A missing, pending, rejected, or partially permitted review fails manifest validation.
SDK terms, dataset terms, model weights, code licenses, and output-use restrictions are reviewed
separately; technical availability does not imply training permission. Evidence is retained outside
the manifest when confidential, with a stable review reference in the manifest.

## Privacy and retention

- Collect only consented modalities and required metadata. Use random de-identified identity and
  session IDs; keep contact and consent records in a separate access-controlled system.
- Encrypt raw RGB/depth at rest and in transit, restrict access by role, audit exports, and define a
  retention deadline per collection. Do not place raw recordings, biometric templates, private
  metadata, or access credentials in Git.
- Support withdrawal and deletion by identity ID across raw records, derived labels, caches, and
  future training revisions. A deletion produces a new dataset revision/digest and invalidates
  affected checkpoints for future release.
- Do not use recordings for unrelated identity recognition. Report aggregate evaluation and
  de-identified failure samples; external sharing requires a separate consent/license decision.

## Failure-sample feedback

1. Evaluation emits a stable failure-sample ID with data revision, split, identity/session/device
   groups, fixed-sequence ID, signal family, state, and failure code.
2. Triage classifies capture failure, synchronization failure, teacher disagreement, label mapping,
   observability, model error, calibration, or runtime error. Raw media is accessed only by an
   authorized reviewer.
3. Capture/teacher defects are marked unavailable or recollected; mapping defects require a catalog
   revision; model errors enter the next training candidate set without changing their identity
   split.
4. A second reviewer resolves teacher disagreements and records the decision. Unresolved cases stay
   unavailable and cannot silently become confident pseudo-labels.
5. The next dataset revision records added/removed failure IDs and a new digest. Regression suites
   pin reviewed samples, while privacy deletion takes precedence over reproducibility.
6. Closure requires the original fixed sequence plus adjacent sequences to pass, no regression in
   other output families, and a report linked to the repaired data/model revisions.

Synthetic failure fixtures are smoke-only. They verify routing and state semantics but do not count
as production failure coverage.
