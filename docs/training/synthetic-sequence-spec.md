# Synthetic F sequence generation specification v1

Every admitted renderer and asset has its own approved license-registry record. The current ICT
record is pending, so this document is a generation contract rather than authorization to render or
train.

Each frame stores RGB, BasicSet truth, original source coefficients, mapping revision, 2D/3D
landmarks or mesh, head pose, eye direction and eyelid state, depth, normals, segmentation,
visibility, occlusion, timestamp, parameter velocity/acceleration, camera/exposure/light/background
settings, asset IDs, and license-record IDs. Files are sharded outside Git; manifests pin digests.

Sequences use neutral, onset, peak, and recovery phases with bounded velocity and acceleration.
The sampler contains isolated single-parameter actions, reviewed physiological combinations,
left/right asymmetric actions, expression plus viseme, rare boundary cases, and explicit invalid
cases kept outside training truth. It must not sample every coefficient independently and uniformly.

Domain randomization covers identity shape, camera intrinsics/distance/yaw/pitch, exposure, light
direction/intensity/color, motion and defocus blur, codec damage, backgrounds, hair, glasses,
makeup, hands, clothing, self-occlusion, and partial out-of-frame states. Randomization seeds,
renderer version, asset digests, and mapping revision are recorded. Identity assets stay in one
split; cameras and render assets used for held-out tests are declared before generation.

`configs/data/ict-facekit-light-to-ntp-v1.json` freezes the mapping interface. Its coefficient matrix
remains deliberately unavailable until the source coefficient ordering and commercial grant are
reviewed; generating plausible-looking coefficients in their absence is prohibited.

