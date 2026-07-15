# NanaFaceID migration disposition

- Checked: 2026-07-15
- Expected source: `https://github.com/sena-nana/NanaFaceID`
- Intended destination: `apps/nana-capture-ios/`
- NanaTracking tracking issue: #15
- Disposition: no source repository exists in the authenticated GitHub or local workspace view, so
  there is no repository content or history to migrate.

## Inventory evidence

The migration audit checked all source locations available to this workspace:

- `gh repo view sena-nana/NanaFaceID` returned repository not found;
- the authenticated `sena-nana` repository inventory contained no repository whose name included
  `face` or `capture`;
- GitHub repository search returned no `NanaFaceID` repository;
- the local `/Users/zt-c604184/Documents/workspace` inventory contained no `NanaFaceID` checkout;
- GitHub issue search found `NanaFaceID` only in NanaTracking planning issues, with no old-repository
  issues to map;
- NanaTracking history contained no prior `apps/nana-capture-ios/` import.

Consequently, an original URL can be recorded, but there is no verifiable last source commit,
default branch, README, issue inventory, archive flag, source asset, build configuration, or Git
history. Creating an empty repository, synthetic commit, redirect README, or fabricated archive
state would make the migration record less truthful and is explicitly rejected.

## Effective task map

No old issue can be updated because no old issue tracker exists. The valid engineering work is
already separated in NanaTracking:

| Responsibility | Current issue | Migration decision |
| --- | --- | --- |
| iOS TrueDepth/ARKit capture application | #16 | Greenfield implementation under `apps/nana-capture-ios/`; no legacy source assumed. |
| Windows Capture Studio and reliable sync | #16 | Greenfield implementation; no Windows code belongs in migration #15. |
| Spatial producer and TrueDepth/RGB fusion contract | #8 | Contract/model work remains independent from an app repository. |
| Commercial data, consent, split, and F/G boundaries | #12 and #13 | Existing NanaTracking data governance remains authoritative. |
| Training-data schema and evaluation standard | #6 | Completed contract work remains in NanaTracking. |

The destination path is reserved for #16. This disposition does not add an empty application
shell or imply that capture functionality exists.

## Sensitive-content review

Because zero bytes were imported, this migration introduces no participant recording, biometric
sample, device key, provisioning secret, or training data. As a defense-in-depth check of the
destination repository at this disposition commit:

- tracked-file extension scan found no movie, HEIC, depth dump, NumPy array, checkpoint, ONNX
  package, or archive payload;
- tracked-content scan found no PEM/OpenSSH private-key header, AWS access-key form, GitHub token
  form, or Slack token form;
- generated runs, datasets, checkpoints, packages, caches, and benchmark working output remain
  excluded by repository ignore rules.

This is a repository-presence and tracked-content audit, not proof about a source that is absent or
about untracked data outside NanaTracking.

## Closure rule

Issue #15 is closed as not planned because its only subject, the old repository, is unavailable.
If a verifiable NanaFaceID repository is later recovered, reopen #15 and replace this disposition
with a migration record containing the actual source URL, last commit SHA, issue map, filtered
history import, excluded-content rationale, sensitive-data scan, README redirect, and archive
evidence. Do not mix that recovery with #16 feature implementation.
