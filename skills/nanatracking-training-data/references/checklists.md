# Execution checklists

## New source or asset

- [ ] Authoritative dataset and item/content terms located
- [ ] Commercial training and model distribution explicitly decided
- [ ] Raw redistribution and attribution/share-alike duties recorded
- [ ] Likeness/biometric consent basis recorded
- [ ] SDK/teacher distillation, pseudo-label, and derivative-output rights separately decided
- [ ] License text snapshot/digest pinned
- [ ] Registry record added before download/training
- [ ] Requested pipeline stage passes admission

## F training

- [ ] No CREMA-D or emotion-class supervision
- [ ] Dataset/asset/teacher records pass commercial admission
- [ ] Identity and session isolated; explicit devices held out for test
- [ ] No frame-random video split; augmentation inherits split
- [ ] Label source/model/version/sync/confidence/mapping/role complete
- [ ] Dataset, file, license, mapping, config, lock, Git, NTP, Signal, feature digests pinned
- [ ] Direct F report uses parameter/geometry-labeled holdouts only

## G or CREMA-D training

- [ ] CREMA-D registry record approved for expression training and release obligations understood
- [ ] F package/digest/revisions fixed and `frozen=true`
- [ ] Cache contains ordered BasicSet 1..36 plus confidence, visibility, head pose, frame quality,
      timestamps, distribution/intensity labels, provenance, and shard digests
- [ ] Actors and clips isolated; cache inherits split
- [ ] G optimizer contains no F parameters
- [ ] All/single/velocity/acceleration/mouth-only/no-mouth/head-only/no-head/shuffled/RGB ablations run
- [ ] Report says downstream expression evidence, not parameter truth

## Model release

- [ ] `model-release` license admission passes for every data/asset/teacher record
- [ ] Withdrawn, expired, unapproved, non-commercial, and unknown sources absent
- [ ] F direct report and G downstream report attached separately
- [ ] NTP, Signal Registry, mapping, feature, data/cache, config, checkpoint/model digests recorded
- [ ] Synthetic evidence marked smoke-only and excluded from production claims
- [ ] F semantic and regression gates pass even if G metrics improve

