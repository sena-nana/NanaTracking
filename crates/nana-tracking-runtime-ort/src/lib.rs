#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use std::{
    collections::BTreeMap,
    f32::consts::PI,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use nana_tracking_runtime_api::{
    ActiveProvider, ModelPose, ModelQuaternion, ModelRegionQuality, ModelScalar, ModelTracked,
    ModelTrackingState, ModelVector3, TrackingModelInput, TrackingModelMetadata,
    TrackingModelOutput, TrackingModelSession, TrackingRuntimeError, verify_model_package,
};
use ndarray::{Array4, ArrayD, ArrayViewD, Ix4};
use ndarray_npy::NpzReader;
use ort::{
    execution_providers::CoreMLExecutionProvider,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
};
use serde_json::Value;

const BASIC_SIGNAL_COUNT: usize = 36;
const SPATIAL_SIGNAL_COUNT: usize = 41;
const FULL_SIGNAL_COUNT: usize = 35;
const UNSIGNED_SPATIAL_SLOTS: [usize; 14] = [8, 9, 12, 13, 14, 15, 16, 27, 29, 32, 33, 34, 35, 40];

const BASIC_OUTPUTS: [&str; 5] = ["rig", "pose", "landmarks", "visibility", "confidence"];
const SPATIAL_OUTPUTS: [&str; 9] = [
    "rig",
    "pose",
    "eye_origins",
    "eye_directions",
    "look_at_head",
    "face_geometry",
    "visibility",
    "tongue_visibility",
    "confidence",
];
const FULL_OUTPUTS: [&str; 9] = [
    "rig",
    "torso_pose",
    "joint_positions",
    "joint_rotations",
    "limb_directions",
    "limb_twists",
    "bone_lengths",
    "visibility",
    "confidence",
];
const CORE_ML_PROVIDER: &str = "CoreMLExecutionProvider";
const CPU_PROVIDER: &str = "CPUExecutionProvider";
static PROFILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Initialize `ort` from an application-provided ONNX Runtime dynamic library.
///
/// This must be called once, before constructing any ORT session. The backend never searches for
/// Python environments or downloads a runtime at startup.
///
/// # Errors
///
/// Returns a structured backend error when the library cannot be loaded or initialized.
pub fn initialize_from_dylib(path: &Path) -> Result<(), TrackingRuntimeError> {
    ort::init_from(path.to_string_lossy())
        .with_name("NanaTracking")
        .commit()
        .map(|_| ())
        .map_err(backend_error)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrtCpuOptions {
    pub intra_threads: usize,
}

impl Default for OrtCpuOptions {
    fn default() -> Self {
        Self { intra_threads: 1 }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreMlComputePolicy {
    All,
    CpuOnly,
    RequireAneCapableDevice,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrtCoreMlOptions {
    pub intra_threads: usize,
    pub compute_policy: CoreMlComputePolicy,
    pub validation_profile_directory: PathBuf,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
}

impl OrtCoreMlOptions {
    #[must_use]
    pub fn new(validation_profile_directory: PathBuf) -> Self {
        Self {
            intra_threads: 1,
            compute_policy: CoreMlComputePolicy::All,
            validation_profile_directory,
            absolute_tolerance: 1.0e-3,
            relative_tolerance: 1.0e-3,
        }
    }

    fn validate(&self) -> Result<(), TrackingRuntimeError> {
        if self.intra_threads == 0
            || !self.validation_profile_directory.is_dir()
            || !self.absolute_tolerance.is_finite()
            || !self.relative_tolerance.is_finite()
            || self.absolute_tolerance < 0.0
            || self.relative_tolerance < 0.0
        {
            return Err(TrackingRuntimeError::InvalidInput);
        }
        if !cfg!(any(target_os = "macos", target_os = "ios")) {
            return Err(TrackingRuntimeError::UnsupportedProvider(
                CORE_ML_PROVIDER.to_owned(),
            ));
        }
        Ok(())
    }
}

enum OrtSessionOptions {
    Cpu(OrtCpuOptions),
    CoreMl(OrtCoreMlOptions),
}

impl OrtSessionOptions {
    const fn intra_threads(&self) -> usize {
        match self {
            Self::Cpu(options) => options.intra_threads,
            Self::CoreMl(options) => options.intra_threads,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputParity {
    pub mean_absolute_error: f32,
    pub maximum_absolute_error: f32,
}

struct PackageContract<'a> {
    profile: &'a str,
    supported_signals: std::ops::RangeInclusive<u16>,
    structures: &'a [&'a str],
    outputs: &'a [&'a str],
    requires_topology: bool,
}

struct OrtPackageSession {
    metadata: TrackingModelMetadata,
    session: Session,
    input: Array4<f32>,
    package_root: PathBuf,
    active_provider: ActiveProvider,
}

impl OrtPackageSession {
    fn load(
        root: &Path,
        options: OrtSessionOptions,
        contract: PackageContract<'_>,
    ) -> Result<Self, TrackingRuntimeError> {
        match &options {
            OrtSessionOptions::Cpu(options) if options.intra_threads == 0 => {
                return Err(TrackingRuntimeError::InvalidInput);
            }
            OrtSessionOptions::CoreMl(options) => options.validate()?,
            OrtSessionOptions::Cpu(_) => {}
        }
        let package = verify_model_package(root)?;
        let metadata = package.metadata;
        let signals = contract.supported_signals.collect::<Vec<_>>();
        let structures = contract
            .structures
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let outputs = contract
            .outputs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        if metadata.guaranteed_profile != contract.profile
            || metadata.supported_signals != signals
            || metadata.supported_structures != structures
            || metadata.output_names != outputs
            || contract.requires_topology
                && metadata
                    .geometry_topology_revision
                    .as_deref()
                    .is_none_or(str::is_empty)
            || !metadata
                .allowed_backends
                .iter()
                .any(|backend| backend == "onnxruntime")
        {
            return Err(TrackingRuntimeError::InvalidMetadata);
        }
        let [batch, channels, height, width]: [usize; 4] = metadata
            .input_shape
            .clone()
            .try_into()
            .map_err(|_| TrackingRuntimeError::InvalidMetadata)?;
        if batch != 1
            || channels != 3
            || metadata.input_layout != "NCHW"
            || metadata.input_color != "RGB"
            || metadata.input_range != (0.0, 1.0)
        {
            return Err(TrackingRuntimeError::InvalidMetadata);
        }
        let mut builder = Session::builder()
            .map_err(backend_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(backend_error)?
            .with_intra_threads(options.intra_threads())
            .map_err(backend_error)?;
        if let OrtSessionOptions::CoreMl(core_ml) = &options {
            let execution_provider = match core_ml.compute_policy {
                CoreMlComputePolicy::All => CoreMLExecutionProvider::default(),
                CoreMlComputePolicy::CpuOnly => CoreMLExecutionProvider::default().with_cpu_only(),
                CoreMlComputePolicy::RequireAneCapableDevice => {
                    CoreMLExecutionProvider::default().with_ane_only()
                }
            };
            builder = builder
                .with_execution_providers([execution_provider.build().error_on_failure()])
                .map_err(backend_error)?
                .with_profiling(core_ml_profile_prefix(core_ml))
                .map_err(backend_error)?;
        }
        let session = builder
            .commit_from_file(&package.model_path)
            .map_err(backend_error)?;
        if session.inputs.len() != 1 || session.inputs[0].name != "image" {
            return Err(TrackingRuntimeError::InvalidMetadata);
        }
        let mut loaded = Self {
            metadata,
            session,
            input: Array4::zeros((1, 3, height, width)),
            package_root: package.root,
            active_provider: ActiveProvider::OnnxRuntimeCpu,
        };
        if let OrtSessionOptions::CoreMl(core_ml) = options {
            let cpu_fallback = loaded.validate_core_ml_provider(&core_ml)?;
            loaded.active_provider = ActiveProvider::OnnxRuntimeCoreMl { cpu_fallback };
        }
        Ok(loaded)
    }

    fn verify_fixed_vector(
        &mut self,
        absolute_tolerance: f32,
        relative_tolerance: f32,
    ) -> Result<BTreeMap<String, OutputParity>, TrackingRuntimeError> {
        if !absolute_tolerance.is_finite()
            || !relative_tolerance.is_finite()
            || absolute_tolerance < 0.0
            || relative_tolerance < 0.0
        {
            return Err(TrackingRuntimeError::InvalidInput);
        }
        let mut inputs = NpzReader::new(
            File::open(self.package_root.join("test-vectors/input.npz"))
                .map_err(TrackingRuntimeError::Io)?,
        )
        .map_err(backend_error)?;
        let image: ArrayD<f32> = inputs.by_name("image").map_err(backend_error)?;
        let image = image.into_dimensionality::<Ix4>().map_err(backend_error)?;
        if image.shape() != self.input.shape() {
            return Err(TrackingRuntimeError::InvalidMetadata);
        }
        let values = self
            .session
            .run(ort::inputs!["image" => image.view()].map_err(backend_error)?)
            .map_err(backend_error)?;
        let mut expected = NpzReader::new(
            File::open(self.package_root.join("test-vectors/expected.npz"))
                .map_err(TrackingRuntimeError::Io)?,
        )
        .map_err(backend_error)?;
        let mut report = BTreeMap::new();
        for name in &self.metadata.output_names {
            let reference: ArrayD<f32> = expected.by_name(name).map_err(backend_error)?;
            let actual = values
                .get(name)
                .ok_or_else(|| missing_output(name))?
                .try_extract_tensor::<f32>()
                .map_err(backend_error)?;
            if actual.shape() != reference.shape() || actual.is_empty() {
                return Err(TrackingRuntimeError::Backend(format!(
                    "fixed-vector output {name} has the wrong shape"
                )));
            }
            let mut absolute_sum = 0.0_f32;
            let mut maximum = 0.0_f32;
            let mut count = 0.0_f32;
            for (&observed, &wanted) in actual.iter().zip(reference.iter()) {
                let difference = (observed - wanted).abs();
                if !difference.is_finite()
                    || difference > absolute_tolerance + relative_tolerance * wanted.abs()
                {
                    return Err(TrackingRuntimeError::Backend(format!(
                        "fixed-vector output {name} exceeds parity tolerance"
                    )));
                }
                absolute_sum += difference;
                maximum = maximum.max(difference);
                count += 1.0;
            }
            report.insert(
                name.clone(),
                OutputParity {
                    mean_absolute_error: absolute_sum / count,
                    maximum_absolute_error: maximum,
                },
            );
        }
        Ok(report)
    }

    fn preprocess(&mut self, source: &TrackingModelInput<'_>) {
        let target_height = self.input.shape()[2];
        let target_width = self.input.shape()[3];
        let plane = target_height * target_width;
        let target = self
            .input
            .as_slice_mut()
            .expect("owned input is contiguous");
        for output_y in 0..target_height {
            let source_y = output_y * source.height / target_height;
            for output_x in 0..target_width {
                let source_x = output_x * source.width / target_width;
                let source_offset = source_y * source.row_stride + source_x * 3;
                let target_offset = output_y * target_width + output_x;
                for channel in 0..3 {
                    target[channel * plane + target_offset] =
                        f32::from(source.rgb[source_offset + channel]) / 255.0;
                }
            }
        }
    }

    fn validate_core_ml_provider(
        &mut self,
        options: &OrtCoreMlOptions,
    ) -> Result<bool, TrackingRuntimeError> {
        let parity =
            self.verify_fixed_vector(options.absolute_tolerance, options.relative_tolerance);
        let profile_path = self.session.end_profiling().map_err(backend_error)?;
        let profile = fs::read_to_string(&profile_path).map_err(TrackingRuntimeError::Io);
        let cleanup = fs::remove_file(&profile_path).map_err(TrackingRuntimeError::Io);
        parity?;
        let cpu_fallback = core_ml_cpu_fallback(&profile?)?;
        cleanup?;
        Ok(cpu_fallback)
    }
}

fn core_ml_profile_prefix(options: &OrtCoreMlOptions) -> PathBuf {
    let sequence = PROFILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    options.validation_profile_directory.join(format!(
        "nana-tracking-coreml-{}-{sequence}",
        std::process::id()
    ))
}

fn core_ml_cpu_fallback(profile: &str) -> Result<bool, TrackingRuntimeError> {
    let events: Vec<Value> = serde_json::from_str(profile).map_err(TrackingRuntimeError::Json)?;
    let mut core_ml_nodes = 0_usize;
    let mut cpu_nodes = 0_usize;
    for event in events {
        if event.get("cat").and_then(Value::as_str) != Some("Node") {
            continue;
        }
        match event
            .get("args")
            .and_then(|args| args.get("provider"))
            .and_then(Value::as_str)
        {
            Some(CORE_ML_PROVIDER) => core_ml_nodes = core_ml_nodes.saturating_add(1),
            Some(CPU_PROVIDER) => cpu_nodes = cpu_nodes.saturating_add(1),
            _ => {}
        }
    }
    if core_ml_nodes == 0 {
        return Err(TrackingRuntimeError::UnsupportedProvider(
            "CoreMLExecutionProvider did not execute any graph nodes".to_owned(),
        ));
    }
    Ok(cpu_nodes > 0)
}

pub struct OrtFaceBasicSession {
    inner: OrtPackageSession,
}

impl OrtFaceBasicSession {
    /// Verify a portable `FaceBasic` package and create a real ORT CPU inference session.
    ///
    /// # Errors
    ///
    /// Fails closed on package, capability, graph I/O, provider, or session errors.
    pub fn load(root: &Path, options: OrtCpuOptions) -> Result<Self, TrackingRuntimeError> {
        Self::load_with_options(root, OrtSessionOptions::Cpu(options))
    }

    /// Create a Core ML execution-provider session and prove fixed-vector parity and actual node
    /// assignment before returning it to the application.
    ///
    /// # Errors
    ///
    /// Fails closed when Core ML registration, parity, profiling, or node assignment fails.
    pub fn load_core_ml(
        root: &Path,
        options: OrtCoreMlOptions,
    ) -> Result<Self, TrackingRuntimeError> {
        Self::load_with_options(root, OrtSessionOptions::CoreMl(options))
    }

    fn load_with_options(
        root: &Path,
        options: OrtSessionOptions,
    ) -> Result<Self, TrackingRuntimeError> {
        Ok(Self {
            inner: OrtPackageSession::load(
                root,
                options,
                PackageContract {
                    profile: "Basic",
                    supported_signals: 1..=36,
                    structures: &["head_geometry"],
                    outputs: &BASIC_OUTPUTS,
                    requires_topology: false,
                },
            )?,
        })
    }

    /// Compare every package output against its fixed `PyTorch` vector.
    ///
    /// # Errors
    ///
    /// Fails closed on malformed vectors, shape drift, non-finite values, or tolerance failures.
    pub fn verify_fixed_vector(
        &mut self,
        absolute_tolerance: f32,
        relative_tolerance: f32,
    ) -> Result<BTreeMap<String, OutputParity>, TrackingRuntimeError> {
        self.inner
            .verify_fixed_vector(absolute_tolerance, relative_tolerance)
    }
}

impl TrackingModelSession for OrtFaceBasicSession {
    fn metadata(&self) -> &TrackingModelMetadata {
        &self.inner.metadata
    }

    fn infer(
        &mut self,
        input: TrackingModelInput<'_>,
        output: &mut TrackingModelOutput,
    ) -> Result<(), TrackingRuntimeError> {
        input.validate()?;
        output.clear();
        output.provider.clone_from(&self.inner.active_provider);

        let preprocess_started = Instant::now();
        self.inner.preprocess(&input);
        output.preprocess_ns = elapsed_ns(preprocess_started);

        let inference_started = Instant::now();
        let values = self
            .inner
            .session
            .run(ort::inputs!["image" => self.inner.input.view()].map_err(backend_error)?)
            .map_err(backend_error)?;
        output.inference_ns = elapsed_ns(inference_started);

        let readback_started = Instant::now();
        populate_basic(&extract_basic(&values)?, input.capture_timestamp_ns, output)?;
        finish_output(
            input.processing_started_timestamp_ns,
            readback_started,
            output,
        );
        Ok(())
    }

    fn reset_temporal_state(&mut self) {}
}

pub struct OrtFaceSpatialSession {
    inner: OrtPackageSession,
}

impl OrtFaceSpatialSession {
    /// Verify a `FaceSpatial` package and create a reusable ORT CPU session.
    ///
    /// # Errors
    ///
    /// Fails closed unless all Spatial signals, structures, topology, and outputs are declared.
    pub fn load(root: &Path, options: OrtCpuOptions) -> Result<Self, TrackingRuntimeError> {
        Self::load_with_options(root, OrtSessionOptions::Cpu(options))
    }

    /// Create a Core ML execution-provider session and prove fixed-vector parity and actual node
    /// assignment before returning it to the application.
    ///
    /// # Errors
    ///
    /// Fails closed when Core ML registration, parity, profiling, or node assignment fails.
    pub fn load_core_ml(
        root: &Path,
        options: OrtCoreMlOptions,
    ) -> Result<Self, TrackingRuntimeError> {
        Self::load_with_options(root, OrtSessionOptions::CoreMl(options))
    }

    fn load_with_options(
        root: &Path,
        options: OrtSessionOptions,
    ) -> Result<Self, TrackingRuntimeError> {
        Ok(Self {
            inner: OrtPackageSession::load(
                root,
                options,
                PackageContract {
                    profile: "Spatial",
                    supported_signals: 1..=41,
                    structures: &[
                        "head_geometry",
                        "eye_geometry",
                        "look_at_point",
                        "face_geometry",
                    ],
                    outputs: &SPATIAL_OUTPUTS,
                    requires_topology: true,
                },
            )?,
        })
    }

    /// Compare every package output against its fixed `PyTorch` vector.
    ///
    /// # Errors
    ///
    /// Fails closed on malformed vectors, shape drift, non-finite values, or tolerance failures.
    pub fn verify_fixed_vector(
        &mut self,
        absolute_tolerance: f32,
        relative_tolerance: f32,
    ) -> Result<BTreeMap<String, OutputParity>, TrackingRuntimeError> {
        self.inner
            .verify_fixed_vector(absolute_tolerance, relative_tolerance)
    }
}

impl TrackingModelSession for OrtFaceSpatialSession {
    fn metadata(&self) -> &TrackingModelMetadata {
        &self.inner.metadata
    }

    fn infer(
        &mut self,
        input: TrackingModelInput<'_>,
        output: &mut TrackingModelOutput,
    ) -> Result<(), TrackingRuntimeError> {
        input.validate()?;
        output.clear();
        output.provider.clone_from(&self.inner.active_provider);

        let preprocess_started = Instant::now();
        self.inner.preprocess(&input);
        output.preprocess_ns = elapsed_ns(preprocess_started);

        let inference_started = Instant::now();
        let values = self
            .inner
            .session
            .run(ort::inputs!["image" => self.inner.input.view()].map_err(backend_error)?)
            .map_err(backend_error)?;
        output.inference_ns = elapsed_ns(inference_started);

        let readback_started = Instant::now();
        populate_spatial(
            &extract_spatial(&values)?,
            input.capture_timestamp_ns,
            output,
        )?;
        finish_output(
            input.processing_started_timestamp_ns,
            readback_started,
            output,
        );
        Ok(())
    }

    fn reset_temporal_state(&mut self) {}
}

/// ORT session for the Full-only extension package.
///
/// A Full package does not produce signals 1..41 or a head pose. Use [`Self::infer_fused`] with a
/// same-capture Spatial result; emitting the extension as a standalone Full result would make the
/// head-relative and tongue fields semantically invalid.
pub struct OrtFullSetSession {
    inner: OrtPackageSession,
}

impl OrtFullSetSession {
    /// Verify a Full-only package and create a reusable ORT CPU session.
    ///
    /// # Errors
    ///
    /// Fails closed unless the package declares exactly signals 42..76 and body skeleton outputs.
    pub fn load(root: &Path, options: OrtCpuOptions) -> Result<Self, TrackingRuntimeError> {
        Self::load_with_options(root, OrtSessionOptions::Cpu(options))
    }

    /// Create a Core ML execution-provider session and prove fixed-vector parity and actual node
    /// assignment before returning it to the application.
    ///
    /// # Errors
    ///
    /// Fails closed when Core ML registration, parity, profiling, or node assignment fails.
    pub fn load_core_ml(
        root: &Path,
        options: OrtCoreMlOptions,
    ) -> Result<Self, TrackingRuntimeError> {
        Self::load_with_options(root, OrtSessionOptions::CoreMl(options))
    }

    fn load_with_options(
        root: &Path,
        options: OrtSessionOptions,
    ) -> Result<Self, TrackingRuntimeError> {
        Ok(Self {
            inner: OrtPackageSession::load(
                root,
                options,
                PackageContract {
                    profile: "Partial",
                    supported_signals: 42..=76,
                    structures: &["body_skeleton"],
                    outputs: &FULL_OUTPUTS,
                    requires_topology: false,
                },
            )?,
        })
    }

    #[must_use]
    pub fn metadata(&self) -> &TrackingModelMetadata {
        &self.inner.metadata
    }

    /// Compare every package output against its fixed `PyTorch` vector.
    ///
    /// # Errors
    ///
    /// Fails closed on malformed vectors, shape drift, non-finite values, or tolerance failures.
    pub fn verify_fixed_vector(
        &mut self,
        absolute_tolerance: f32,
        relative_tolerance: f32,
    ) -> Result<BTreeMap<String, OutputParity>, TrackingRuntimeError> {
        self.inner
            .verify_fixed_vector(absolute_tolerance, relative_tolerance)
    }

    /// Run the Full-only model and fuse it into an already populated same-capture Spatial output.
    ///
    /// The method preserves Spatial signals and geometry, derives the torso/head and articulated
    /// scalar views from their authoritative structured geometry, and reuses the session input.
    ///
    /// # Errors
    ///
    /// Fails closed when the supplied output is not a usable same-capture Spatial result or when
    /// any Full tensor has a wrong shape, non-finite value, or degenerate bone.
    #[allow(clippy::needless_pass_by_value)]
    pub fn infer_fused(
        &mut self,
        input: TrackingModelInput<'_>,
        output: &mut TrackingModelOutput,
    ) -> Result<(), TrackingRuntimeError> {
        input.validate()?;
        validate_spatial_fusion_input(input.capture_timestamp_ns, output)?;
        if output.provider != self.inner.active_provider {
            return Err(TrackingRuntimeError::UnsupportedProvider(
                "Spatial and Full stages must use the same active provider".to_owned(),
            ));
        }
        let prior_preprocess_ns = output.preprocess_ns;
        let prior_inference_ns = output.inference_ns;
        let prior_readback_ns = output.readback_ns;

        let preprocess_started = Instant::now();
        self.inner.preprocess(&input);
        output.preprocess_ns = output
            .preprocess_ns
            .saturating_add(elapsed_ns(preprocess_started));

        let inference_started = Instant::now();
        let values = self
            .inner
            .session
            .run(ort::inputs!["image" => self.inner.input.view()].map_err(backend_error)?)
            .map_err(backend_error)?;
        output.inference_ns = output
            .inference_ns
            .saturating_add(elapsed_ns(inference_started));

        let readback_started = Instant::now();
        populate_full(&extract_full(&values)?, input.capture_timestamp_ns, output)?;
        output.readback_ns = output
            .readback_ns
            .saturating_add(elapsed_ns(readback_started));
        let extension_elapsed_ns = output
            .preprocess_ns
            .saturating_sub(prior_preprocess_ns)
            .saturating_add(output.inference_ns.saturating_sub(prior_inference_ns))
            .saturating_add(output.readback_ns.saturating_sub(prior_readback_ns));
        output.produced_timestamp_ns = output.produced_timestamp_ns.max(completion_timestamp(
            input.processing_started_timestamp_ns,
            extension_elapsed_ns,
        ));
        Ok(())
    }
}

struct FaceBasicViews<'a> {
    rig: ArrayViewD<'a, f32>,
    pose: ArrayViewD<'a, f32>,
    visibility: ArrayViewD<'a, f32>,
    confidence: ArrayViewD<'a, f32>,
}

struct FaceSpatialViews<'a> {
    rig: ArrayViewD<'a, f32>,
    pose: ArrayViewD<'a, f32>,
    eye_origins: ArrayViewD<'a, f32>,
    eye_directions: ArrayViewD<'a, f32>,
    look_at_head: ArrayViewD<'a, f32>,
    face_geometry: ArrayViewD<'a, f32>,
    visibility: ArrayViewD<'a, f32>,
    tongue_visibility: ArrayViewD<'a, f32>,
    confidence: ArrayViewD<'a, f32>,
}

struct FullSetViews<'a> {
    rig: ArrayViewD<'a, f32>,
    torso_pose: ArrayViewD<'a, f32>,
    joint_positions: ArrayViewD<'a, f32>,
    joint_rotations: ArrayViewD<'a, f32>,
    limb_directions: ArrayViewD<'a, f32>,
    limb_twists: ArrayViewD<'a, f32>,
    bone_lengths: ArrayViewD<'a, f32>,
    visibility: ArrayViewD<'a, f32>,
    confidence: ArrayViewD<'a, f32>,
}

fn tensor<'a>(
    values: &'a SessionOutputs<'_, '_>,
    name: &str,
) -> Result<ArrayViewD<'a, f32>, TrackingRuntimeError> {
    values
        .get(name)
        .ok_or_else(|| missing_output(name))?
        .try_extract_tensor::<f32>()
        .map_err(backend_error)
}

fn extract_basic<'a>(
    values: &'a SessionOutputs<'_, '_>,
) -> Result<FaceBasicViews<'a>, TrackingRuntimeError> {
    let result = FaceBasicViews {
        rig: tensor(values, "rig")?,
        pose: tensor(values, "pose")?,
        visibility: tensor(values, "visibility")?,
        confidence: tensor(values, "confidence")?,
    };
    if result.rig.len() != BASIC_SIGNAL_COUNT
        || result.confidence.len() != BASIC_SIGNAL_COUNT
        || result.pose.len() != 7
        || result.visibility.len() != 3
    {
        return Err(shape_error("FaceBasic"));
    }
    Ok(result)
}

fn extract_spatial<'a>(
    values: &'a SessionOutputs<'_, '_>,
) -> Result<FaceSpatialViews<'a>, TrackingRuntimeError> {
    let result = FaceSpatialViews {
        rig: tensor(values, "rig")?,
        pose: tensor(values, "pose")?,
        eye_origins: tensor(values, "eye_origins")?,
        eye_directions: tensor(values, "eye_directions")?,
        look_at_head: tensor(values, "look_at_head")?,
        face_geometry: tensor(values, "face_geometry")?,
        visibility: tensor(values, "visibility")?,
        tongue_visibility: tensor(values, "tongue_visibility")?,
        confidence: tensor(values, "confidence")?,
    };
    if result.rig.len() != SPATIAL_SIGNAL_COUNT
        || result.confidence.len() != SPATIAL_SIGNAL_COUNT
        || result.pose.len() != 7
        || result.eye_origins.len() != 6
        || result.eye_directions.len() != 6
        || result.look_at_head.len() != 3
        || result.face_geometry.is_empty()
        || result.face_geometry.len() % 3 != 0
        || result.visibility.len() != 3
        || result.tongue_visibility.len() != 2
    {
        return Err(shape_error("FaceSpatial"));
    }
    Ok(result)
}

fn extract_full<'a>(
    values: &'a SessionOutputs<'_, '_>,
) -> Result<FullSetViews<'a>, TrackingRuntimeError> {
    let result = FullSetViews {
        rig: tensor(values, "rig")?,
        torso_pose: tensor(values, "torso_pose")?,
        joint_positions: tensor(values, "joint_positions")?,
        joint_rotations: tensor(values, "joint_rotations")?,
        limb_directions: tensor(values, "limb_directions")?,
        limb_twists: tensor(values, "limb_twists")?,
        bone_lengths: tensor(values, "bone_lengths")?,
        visibility: tensor(values, "visibility")?,
        confidence: tensor(values, "confidence")?,
    };
    if result.rig.len() != FULL_SIGNAL_COUNT
        || result.confidence.len() != FULL_SIGNAL_COUNT
        || result.torso_pose.len() != 7
        || result.joint_positions.len() != 18
        || result.joint_rotations.len() != 24
        || result.limb_directions.len() != 12
        || result.limb_twists.len() != 4
        || result.bone_lengths.len() != 4
        || result.visibility.len() != 30
    {
        return Err(shape_error("FullSet"));
    }
    Ok(result)
}

fn populate_basic(
    values: &FaceBasicViews<'_>,
    capture_timestamp_ns: u64,
    output: &mut TrackingModelOutput,
) -> Result<(), TrackingRuntimeError> {
    let state = face_state(argmax(values.visibility.iter().copied())?);
    let confidence = finite_values(&values.confidence, "FaceBasic confidence")?;
    let mean_confidence = confidence
        .iter()
        .copied()
        .map(clamp_confidence)
        .sum::<f32>()
        / 36.0;
    let rig = finite_values(&values.rig, "FaceBasic rig")?;
    for (slot, (&value, &sample_confidence)) in rig.iter().zip(confidence).enumerate() {
        let value = clamp_spatial_signal(slot, value);
        output.signals[slot] = Some(tracked_value(
            value,
            clamp_confidence(sample_confidence),
            state,
            capture_timestamp_ns,
        ));
    }
    let pose = finite_values(&values.pose, "FaceBasic pose")?;
    output.geometry.head_camera_pose = tracked_pose(
        pose_slice(pose)?,
        mean_confidence,
        state,
        capture_timestamp_ns,
    );
    output.quality.overall_confidence = mean_confidence;
    output.quality.face = ModelRegionQuality {
        confidence: mean_confidence,
        state,
    };
    output.quality.eyes = output.quality.face;
    Ok(())
}

fn populate_spatial(
    values: &FaceSpatialViews<'_>,
    capture_timestamp_ns: u64,
    output: &mut TrackingModelOutput,
) -> Result<(), TrackingRuntimeError> {
    let state = face_state(argmax(values.visibility.iter().copied())?);
    let tongue_visible = argmax(values.tongue_visibility.iter().copied())? == 1;
    let tongue_state = if state == ModelTrackingState::Observed && !tongue_visible {
        ModelTrackingState::Occluded
    } else {
        state
    };
    let rig = finite_values(&values.rig, "FaceSpatial rig")?;
    let confidence = finite_values(&values.confidence, "FaceSpatial confidence")?;
    for (slot, (&value, &sample_confidence)) in rig.iter().zip(confidence).enumerate() {
        output.signals[slot] = Some(tracked_value(
            clamp_spatial_signal(slot, value),
            clamp_confidence(sample_confidence),
            if slot == 40 { tongue_state } else { state },
            capture_timestamp_ns,
        ));
    }
    let pose = finite_values(&values.pose, "FaceSpatial pose")?;
    let pose = pose_slice(pose)?;
    let quaternion = normalize_quaternion([pose[3], pose[4], pose[5], pose[6]])?;
    let face_confidence = confidence
        .iter()
        .copied()
        .map(clamp_confidence)
        .sum::<f32>()
        / 41.0;
    output.geometry.head_camera_pose =
        tracked_pose(pose, face_confidence, state, capture_timestamp_ns);

    let origins = finite_values(&values.eye_origins, "FaceSpatial eye origins")?;
    let directions = finite_values(&values.eye_directions, "FaceSpatial eye directions")?;
    let eye_confidence = confidence[36..40]
        .iter()
        .copied()
        .map(clamp_confidence)
        .sum::<f32>()
        / 4.0;
    for eye in 0..2 {
        let origin = vector_from_slice(&origins[eye * 3..eye * 3 + 3])?;
        let direction =
            normalize_eye_direction(vector_from_slice(&directions[eye * 3..eye * 3 + 3])?);
        output.geometry.eye_origins_head[eye] =
            tracked_value(origin, eye_confidence, state, capture_timestamp_ns);
        output.geometry.eye_directions_head[eye] =
            tracked_value(direction, eye_confidence, state, capture_timestamp_ns);
    }
    let look_at = finite_values(&values.look_at_head, "FaceSpatial look-at")?;
    let rotated = rotate_vector(vector_from_slice(look_at)?, quaternion);
    output.geometry.look_at_camera = tracked_value(
        ModelVector3 {
            x: pose[0] + rotated.x,
            y: pose[1] + rotated.y,
            z: pose[2] + rotated.z,
        },
        eye_confidence,
        state,
        capture_timestamp_ns,
    );
    finite_values(&values.face_geometry, "FaceSpatial face geometry")?;
    output.geometry.face_geometry_state = state;
    output.quality.overall_confidence = face_confidence;
    output.quality.face = ModelRegionQuality {
        confidence: face_confidence,
        state,
    };
    output.quality.eyes = ModelRegionQuality {
        confidence: eye_confidence,
        state,
    };
    Ok(())
}

fn validate_spatial_fusion_input(
    capture_timestamp_ns: u64,
    output: &TrackingModelOutput,
) -> Result<(), TrackingRuntimeError> {
    let signals_ready = output.signals[..SPATIAL_SIGNAL_COUNT]
        .iter()
        .all(Option::is_some);
    let head = &output.geometry.head_camera_pose;
    if !signals_ready || head.sample_capture_timestamp_ns != capture_timestamp_ns {
        return Err(TrackingRuntimeError::InvalidInput);
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn populate_full(
    values: &FullSetViews<'_>,
    capture_timestamp_ns: u64,
    output: &mut TrackingModelOutput,
) -> Result<(), TrackingRuntimeError> {
    let mut rig: [f32; FULL_SIGNAL_COUNT] = finite_values(&values.rig, "FullSet rig")?
        .try_into()
        .map_err(|_| shape_error("FullSet rig"))?;
    let confidence = finite_values(&values.confidence, "FullSet confidence")?;
    let torso = finite_values(&values.torso_pose, "FullSet torso pose")?;
    let joints = finite_values(&values.joint_positions, "FullSet joint positions")?;
    let rotations = finite_values(&values.joint_rotations, "FullSet joint rotations")?;
    let _directions = finite_values(&values.limb_directions, "FullSet limb directions")?;
    let twists = finite_values(&values.limb_twists, "FullSet limb twists")?;
    finite_values(&values.bone_lengths, "FullSet bone lengths")?;
    let visibility = finite_values(&values.visibility, "FullSet visibility")?;
    let mut states = [ModelTrackingState::Unsupported; 5];
    for (region, row) in visibility.chunks_exact(6).enumerate() {
        states[region] = full_state(argmax(row.iter().copied())?)?;
    }
    let torso_quaternion = normalize_quaternion([torso[3], torso[4], torso[5], torso[6]])?;
    let torso_angles = euler_pitch_yaw_roll(torso_quaternion);
    rig[..6].copy_from_slice(&[
        torso[0].clamp(-1.0, 1.0),
        torso[1].clamp(-1.0, 1.0),
        torso[2].clamp(-1.0, 1.0),
        torso_angles[0],
        torso_angles[1],
        torso_angles[2],
    ]);

    let mut derived_directions = [[ModelVector3::default(); 2]; 2];
    for side in 0..2 {
        let shoulder = joint_vector(joints, side, 0)?;
        let elbow = joint_vector(joints, side, 1)?;
        let wrist = joint_vector(joints, side, 2)?;
        derived_directions[side][0] = normalize_vector(subtract(elbow, shoulder))?;
        derived_directions[side][1] = normalize_vector(subtract(wrist, elbow))?;
        let upper = derived_directions[side][0];
        let forearm = derived_directions[side][1];
        let start = if side == 0 { 25 } else { 30 };
        rig[start] = upper.z;
        rig[start + 1] = upper.x * if side == 0 { -1.0 } else { 1.0 };
        rig[start + 2] = twists[side * 2];
        rig[start + 3] = ((1.0 - dot(upper, forearm)) * 0.5).clamp(0.0, 1.0);
        rig[start + 4] = twists[side * 2 + 1];
    }

    let face_state = output.quality.face.state;
    let face_confidence = output.quality.face.confidence;
    let mut signal_states = [ModelTrackingState::Unsupported; FULL_SIGNAL_COUNT];
    signal_states[..12].fill(states[0]);
    signal_states[12..15].fill(face_state);
    signal_states[15..21].copy_from_slice(&[
        states[3], states[4], states[3], states[4], states[3], states[4],
    ]);
    signal_states[21..25].copy_from_slice(&[states[1], states[2], states[1], states[2]]);
    signal_states[25..30].fill(states[1]);
    signal_states[30..35].fill(states[2]);

    populate_head_relative(
        &mut rig,
        confidence,
        torso,
        torso_quaternion,
        states[0],
        capture_timestamp_ns,
        output,
    )?;
    for offset in 0..FULL_SIGNAL_COUNT {
        if (6..=11).contains(&offset) {
            continue;
        }
        let state = signal_states[offset];
        let sample_confidence = if (12..=14).contains(&offset) {
            clamp_confidence(confidence[offset]).min(face_confidence)
        } else {
            clamp_confidence(confidence[offset])
        };
        output.signals[41 + offset] = Some(tracked_value(
            clamp_full_signal(offset, rig[offset]),
            sample_confidence,
            state,
            capture_timestamp_ns,
        ));
    }

    let torso_confidence = mean(&confidence[..6]);
    output.geometry.torso_camera_pose = tracked_value(
        ModelPose {
            position: vector_from_slice(&torso[..3])?,
            rotation: torso_quaternion,
        },
        torso_confidence,
        states[0],
        capture_timestamp_ns,
    );
    let geometry_confidence = mean(confidence);
    for side in 0..2 {
        let state = states[side + 1];
        for joint in 0..3 {
            let flat = side * 3 + joint;
            let rotation_offset = flat * 4;
            output.geometry.upper_body_joint_positions[flat] = tracked_value(
                joint_vector(joints, side, joint)?,
                geometry_confidence,
                state,
                capture_timestamp_ns,
            );
            output.geometry.upper_body_joint_rotations[flat] = tracked_value(
                normalize_quaternion([
                    rotations[rotation_offset],
                    rotations[rotation_offset + 1],
                    rotations[rotation_offset + 2],
                    rotations[rotation_offset + 3],
                ])?,
                geometry_confidence,
                state,
                capture_timestamp_ns,
            );
        }
        output.geometry.upper_arm_directions[side] = tracked_value(
            derived_directions[side][0],
            geometry_confidence,
            state,
            capture_timestamp_ns,
        );
        output.geometry.forearm_directions[side] = tracked_value(
            derived_directions[side][1],
            geometry_confidence,
            state,
            capture_timestamp_ns,
        );
        output.geometry.upper_arm_twists[side] = tracked_value(
            twists[side * 2] * PI,
            geometry_confidence,
            state,
            capture_timestamp_ns,
        );
        output.geometry.forearm_twists[side] = tracked_value(
            twists[side * 2 + 1] * PI,
            geometry_confidence,
            state,
            capture_timestamp_ns,
        );
    }
    output.quality.torso = ModelRegionQuality {
        confidence: mean(&confidence[..12]),
        state: states[0],
    };
    output.quality.arm[0] = ModelRegionQuality {
        confidence: mean(&confidence[21..30]),
        state: states[1],
    };
    output.quality.arm[1] = ModelRegionQuality {
        confidence: indexed_mean(confidence, &[22, 24, 30, 31, 32, 33, 34]),
        state: states[2],
    };
    output.quality.auricle[0] = ModelRegionQuality {
        confidence: indexed_mean(confidence, &[15, 17, 19]),
        state: states[3],
    };
    output.quality.auricle[1] = ModelRegionQuality {
        confidence: indexed_mean(confidence, &[16, 18, 20]),
        state: states[4],
    };
    output.quality.overall_confidence = output.quality.overall_confidence.min(mean(confidence));
    Ok(())
}

fn populate_head_relative(
    rig: &mut [f32],
    confidence: &[f32],
    torso: &[f32],
    torso_quaternion: ModelQuaternion,
    torso_state: ModelTrackingState,
    capture_timestamp_ns: u64,
    output: &mut TrackingModelOutput,
) -> Result<(), TrackingRuntimeError> {
    let head = output.geometry.head_camera_pose;
    let state = if carries_value(torso_state) && carries_value(head.state) {
        ModelTrackingState::Fused
    } else if !carries_value(torso_state) {
        torso_state
    } else {
        head.state
    };
    let Some(head_pose) = head.value.filter(|_| carries_value(state)) else {
        for offset in 6..=11 {
            output.signals[41 + offset] =
                Some(ModelScalar::unavailable(0.0, state, capture_timestamp_ns));
        }
        return Ok(());
    };
    let inverse_torso = quaternion_conjugate(torso_quaternion);
    let relative_position = rotate_vector(
        subtract(head_pose.position, vector_from_slice(&torso[..3])?),
        inverse_torso,
    );
    let relative_quaternion = quaternion_multiply(inverse_torso, head_pose.rotation)?;
    let relative_angles = euler_pitch_yaw_roll(relative_quaternion);
    let relative = [
        relative_position.x.clamp(-1.0, 1.0),
        relative_position.y.clamp(-1.0, 1.0),
        relative_position.z.clamp(-1.0, 1.0),
        relative_angles[0],
        relative_angles[1],
        relative_angles[2],
    ];
    rig[6..12].copy_from_slice(&relative);
    let combined_confidence = mean(&confidence[6..12]).min(head.confidence);
    let timestamp_ns = capture_timestamp_ns.min(head.sample_capture_timestamp_ns);
    for (index, value) in relative.into_iter().enumerate() {
        output.signals[47 + index] = Some(tracked_value(
            value,
            combined_confidence,
            ModelTrackingState::Fused,
            timestamp_ns,
        ));
    }
    Ok(())
}

fn tracked_pose(
    pose: &[f32; 7],
    confidence: f32,
    state: ModelTrackingState,
    capture_timestamp_ns: u64,
) -> ModelTracked<ModelPose> {
    let value =
        normalize_quaternion([pose[3], pose[4], pose[5], pose[6]]).map(|rotation| ModelPose {
            position: ModelVector3 {
                x: pose[0],
                y: pose[1],
                z: pose[2],
            },
            rotation,
        });
    match value {
        Ok(value) => tracked_value(value, confidence, state, capture_timestamp_ns),
        Err(_) => ModelTracked::unavailable(confidence, state, capture_timestamp_ns),
    }
}

fn tracked_value<T>(
    value: T,
    confidence: f32,
    state: ModelTrackingState,
    capture_timestamp_ns: u64,
) -> ModelTracked<T> {
    ModelTracked {
        value: carries_value(state).then_some(value),
        confidence: clamp_confidence(confidence),
        state,
        sample_capture_timestamp_ns: capture_timestamp_ns,
        prediction_horizon_ns: 0,
    }
}

fn carries_value(state: ModelTrackingState) -> bool {
    matches!(
        state,
        ModelTrackingState::Observed | ModelTrackingState::Fused | ModelTrackingState::Predicted
    )
}

fn face_state(index: usize) -> ModelTrackingState {
    match index {
        0 => ModelTrackingState::Observed,
        1 => ModelTrackingState::Occluded,
        2 => ModelTrackingState::OutOfFrame,
        _ => unreachable!("three visibility logits"),
    }
}

fn full_state(index: usize) -> Result<ModelTrackingState, TrackingRuntimeError> {
    match index {
        0 => Ok(ModelTrackingState::Observed),
        1 | 2 => Ok(ModelTrackingState::Occluded),
        3 => Ok(ModelTrackingState::OutOfFrame),
        4 => Ok(ModelTrackingState::Predicted),
        5 => Ok(ModelTrackingState::TrackingLost),
        _ => Err(shape_error("FullSet visibility")),
    }
}

fn finite_values<'a>(
    values: &'a ArrayViewD<'_, f32>,
    label: &str,
) -> Result<&'a [f32], TrackingRuntimeError> {
    let result = values.as_slice().ok_or_else(|| {
        TrackingRuntimeError::Backend(format!("{label} is not a contiguous tensor"))
    })?;
    if result.iter().any(|value| !value.is_finite()) {
        return Err(TrackingRuntimeError::Backend(format!(
            "{label} contains a non-finite value"
        )));
    }
    Ok(result)
}

fn pose_slice(values: &[f32]) -> Result<&[f32; 7], TrackingRuntimeError> {
    values.try_into().map_err(|_| shape_error("pose"))
}

fn vector_from_slice(values: &[f32]) -> Result<ModelVector3, TrackingRuntimeError> {
    let [x, y, z]: [f32; 3] = values.try_into().map_err(|_| shape_error("vector"))?;
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(TrackingRuntimeError::Backend(
            "geometry vector contains a non-finite value".into(),
        ));
    }
    Ok(ModelVector3 { x, y, z })
}

fn normalize_vector(value: ModelVector3) -> Result<ModelVector3, TrackingRuntimeError> {
    let norm = (value.x * value.x + value.y * value.y + value.z * value.z).sqrt();
    if !norm.is_finite() || norm < 1.0e-6 {
        return Err(TrackingRuntimeError::Backend(
            "geometry vector is degenerate".into(),
        ));
    }
    Ok(ModelVector3 {
        x: value.x / norm,
        y: value.y / norm,
        z: value.z / norm,
    })
}

fn normalize_eye_direction(value: ModelVector3) -> ModelVector3 {
    normalize_vector(value).unwrap_or(ModelVector3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    })
}

fn normalize_quaternion(values: [f32; 4]) -> Result<ModelQuaternion, TrackingRuntimeError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(TrackingRuntimeError::Backend(
            "quaternion contains a non-finite value".into(),
        ));
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let mut result = if norm < 1.0e-6 {
        ModelQuaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    } else {
        ModelQuaternion {
            x: values[0] / norm,
            y: values[1] / norm,
            z: values[2] / norm,
            w: values[3] / norm,
        }
    };
    if result.w < 0.0 {
        result = ModelQuaternion {
            x: -result.x,
            y: -result.y,
            z: -result.z,
            w: -result.w,
        };
    }
    Ok(result)
}

fn quaternion_conjugate(value: ModelQuaternion) -> ModelQuaternion {
    ModelQuaternion {
        x: -value.x,
        y: -value.y,
        z: -value.z,
        w: value.w,
    }
}

fn quaternion_multiply(
    left: ModelQuaternion,
    right: ModelQuaternion,
) -> Result<ModelQuaternion, TrackingRuntimeError> {
    normalize_quaternion([
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
    ])
}

fn rotate_vector(vector: ModelVector3, quaternion: ModelQuaternion) -> ModelVector3 {
    let q = ModelVector3 {
        x: quaternion.x,
        y: quaternion.y,
        z: quaternion.z,
    };
    let twice_cross = scale(cross(q, vector), 2.0);
    add(
        add(vector, scale(twice_cross, quaternion.w)),
        cross(q, twice_cross),
    )
}

fn euler_pitch_yaw_roll(quaternion: ModelQuaternion) -> [f32; 3] {
    let ModelQuaternion { x, y, z, w } = quaternion;
    [
        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin(),
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
    ]
}

fn joint_vector(
    joints: &[f32],
    side: usize,
    joint: usize,
) -> Result<ModelVector3, TrackingRuntimeError> {
    let offset = (side * 3 + joint) * 3;
    vector_from_slice(&joints[offset..offset + 3])
}

fn add(left: ModelVector3, right: ModelVector3) -> ModelVector3 {
    ModelVector3 {
        x: left.x + right.x,
        y: left.y + right.y,
        z: left.z + right.z,
    }
}

fn subtract(left: ModelVector3, right: ModelVector3) -> ModelVector3 {
    ModelVector3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn scale(value: ModelVector3, factor: f32) -> ModelVector3 {
    ModelVector3 {
        x: value.x * factor,
        y: value.y * factor,
        z: value.z * factor,
    }
}

fn cross(left: ModelVector3, right: ModelVector3) -> ModelVector3 {
    ModelVector3 {
        x: left.y * right.z - left.z * right.y,
        y: left.z * right.x - left.x * right.z,
        z: left.x * right.y - left.y * right.x,
    }
}

fn dot(left: ModelVector3, right: ModelVector3) -> f32 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn mean(values: &[f32]) -> f32 {
    let (sum, count) = values
        .iter()
        .fold((0.0_f32, 0.0_f32), |(sum, count), value| {
            (sum + clamp_confidence(*value), count + 1.0)
        });
    sum / count
}

fn indexed_mean(values: &[f32], indices: &[usize]) -> f32 {
    let (sum, count) = indices
        .iter()
        .fold((0.0_f32, 0.0_f32), |(sum, count), index| {
            (sum + clamp_confidence(values[*index]), count + 1.0)
        });
    sum / count
}

fn clamp_spatial_signal(slot: usize, value: f32) -> f32 {
    match slot {
        36 | 38 => value.clamp(-1.2, 1.2),
        37 | 39 => value.clamp(-0.8, 0.8),
        _ if UNSIGNED_SPATIAL_SLOTS.contains(&slot) => value.clamp(0.0, 1.0),
        _ => value.clamp(-1.0, 1.0),
    }
}

fn clamp_full_signal(slot: usize, value: f32) -> f32 {
    match slot {
        3..=5 | 9..=11 => value.clamp(-PI, PI - f32::EPSILON),
        28 | 33 => value.clamp(0.0, 1.0),
        _ => value.clamp(-1.0, 1.0),
    }
}

fn argmax(values: impl Iterator<Item = f32>) -> Result<usize, TrackingRuntimeError> {
    values
        .enumerate()
        .try_fold(None, |best, (index, value)| {
            if !value.is_finite() {
                return Err(TrackingRuntimeError::Backend(
                    "visibility logits contain a non-finite value".into(),
                ));
            }
            Ok(match best {
                Some((_, best_value)) if best_value >= value => best,
                _ => Some((index, value)),
            })
        })?
        .map(|(index, _)| index)
        .ok_or_else(|| TrackingRuntimeError::Backend("visibility logits are empty".into()))
}

fn clamp_confidence(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn finish_output(
    processing_started_timestamp_ns: u64,
    readback_started: Instant,
    output: &mut TrackingModelOutput,
) {
    output.readback_ns = elapsed_ns(readback_started);
    let backend_elapsed_ns = output
        .preprocess_ns
        .saturating_add(output.inference_ns)
        .saturating_add(output.readback_ns);
    output.produced_timestamp_ns =
        completion_timestamp(processing_started_timestamp_ns, backend_elapsed_ns);
}

fn completion_timestamp(processing_started_timestamp_ns: u64, backend_elapsed_ns: u64) -> u64 {
    processing_started_timestamp_ns.saturating_add(backend_elapsed_ns)
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn backend_error(error: impl std::fmt::Display) -> TrackingRuntimeError {
    TrackingRuntimeError::Backend(error.to_string())
}

fn missing_output(name: &str) -> TrackingRuntimeError {
    TrackingRuntimeError::Backend(format!("model did not return required output {name}"))
}

fn shape_error(model: &str) -> TrackingRuntimeError {
    TrackingRuntimeError::Backend(format!(
        "{model} output shape differs from its package contract"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, ArrayD};

    #[test]
    fn core_ml_profile_requires_executed_nodes_and_reports_cpu_fallback() {
        let mixed = r#"[
            {"cat":"Node","args":{"provider":"CoreMLExecutionProvider"}},
            {"cat":"Node","args":{"provider":"CPUExecutionProvider"}}
        ]"#;
        assert!(core_ml_cpu_fallback(mixed).expect("Core ML node is present"));

        let cpu_only = r#"[
            {"cat":"Node","args":{"provider":"CPUExecutionProvider"}}
        ]"#;
        assert!(matches!(
            core_ml_cpu_fallback(cpu_only),
            Err(TrackingRuntimeError::UnsupportedProvider(_))
        ));
    }

    #[test]
    fn completion_time_includes_queue_delay_without_double_counting_prior_stage() {
        let capture_timestamp_ns = 1_000;
        let spatial_processing_started_ns = 1_050;
        let spatial_completed_ns = completion_timestamp(spatial_processing_started_ns, 30);
        assert_eq!(spatial_completed_ns - capture_timestamp_ns, 80);

        let full_completed_ns = spatial_completed_ns.max(completion_timestamp(1_100, 40));
        assert_eq!(full_completed_ns, 1_140);
    }

    #[test]
    fn spatial_mapping_preserves_state_and_normalizes_geometry() {
        let rig = array(vec![0.25; SPATIAL_SIGNAL_COUNT]);
        let pose = array(vec![0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]);
        let eye_origins = array(vec![-0.2, 0.0, 0.0, 0.2, 0.0, 0.0]);
        let eye_directions = array(vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
        let look_at_head = array(vec![0.0, 0.0, 1.0]);
        let face_geometry = array(vec![0.0; 6]);
        let visibility = array(vec![1.0, 0.0, 0.0]);
        let tongue_visibility = array(vec![1.0, 0.0]);
        let confidence = array(vec![0.8; SPATIAL_SIGNAL_COUNT]);
        let views = FaceSpatialViews {
            rig: rig.view(),
            pose: pose.view(),
            eye_origins: eye_origins.view(),
            eye_directions: eye_directions.view(),
            look_at_head: look_at_head.view(),
            face_geometry: face_geometry.view(),
            visibility: visibility.view(),
            tongue_visibility: tongue_visibility.view(),
            confidence: confidence.view(),
        };
        let mut output = TrackingModelOutput::preallocated(ActiveProvider::OnnxRuntimeCpu);

        populate_spatial(&views, 123, &mut output).expect("valid Spatial mapping");

        assert_eq!(output.signals[0].expect("basic signal").value, Some(0.25));
        let tongue = output.signals[40].expect("tongue extension");
        assert_eq!(tongue.state, ModelTrackingState::Occluded);
        assert_eq!(tongue.value, None);
        assert_eq!(
            output.geometry.eye_directions_head[0].value,
            Some(ModelVector3 {
                x: 0.0,
                y: 0.0,
                z: 1.0
            })
        );
        assert_eq!(
            output.geometry.look_at_camera.value,
            Some(ModelVector3 {
                x: 0.1,
                y: 0.2,
                z: 1.3
            })
        );
    }

    #[test]
    fn full_mapping_fuses_same_capture_and_derives_arm_views_from_joints() {
        let rig = array(vec![0.0; FULL_SIGNAL_COUNT]);
        let torso_pose = array(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let joint_positions = array(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0,
            0.0,
        ]);
        let mut rotations = Vec::with_capacity(24);
        for _ in 0..6 {
            rotations.extend([0.0, 0.0, 0.0, 1.0]);
        }
        let joint_rotations = array(rotations);
        let limb_directions = array(vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        let limb_twists = array(vec![0.0; 4]);
        let bone_lengths = array(vec![0.5; 4]);
        let mut visibility_logits = Vec::with_capacity(30);
        for _ in 0..5 {
            visibility_logits.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        let visibility = array(visibility_logits);
        let confidence = array(vec![0.8; FULL_SIGNAL_COUNT]);
        let views = FullSetViews {
            rig: rig.view(),
            torso_pose: torso_pose.view(),
            joint_positions: joint_positions.view(),
            joint_rotations: joint_rotations.view(),
            limb_directions: limb_directions.view(),
            limb_twists: limb_twists.view(),
            bone_lengths: bone_lengths.view(),
            visibility: visibility.view(),
            confidence: confidence.view(),
        };
        let mut output = TrackingModelOutput::preallocated(ActiveProvider::OnnxRuntimeCpu);
        for signal in &mut output.signals[..SPATIAL_SIGNAL_COUNT] {
            *signal = Some(ModelScalar::observed(0.0, 0.9, 123));
        }
        output.geometry.head_camera_pose = ModelTracked::observed(
            ModelPose {
                position: ModelVector3::default(),
                rotation: ModelQuaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
            },
            0.9,
            123,
        );
        output.quality.face = ModelRegionQuality {
            confidence: 0.9,
            state: ModelTrackingState::Observed,
        };
        output.quality.overall_confidence = 0.9;

        populate_full(&views, 123, &mut output).expect("valid Full fusion");

        assert!(output.signals[..76].iter().all(Option::is_some));
        assert_eq!(
            output.signals[47].expect("head relative x").state,
            ModelTrackingState::Fused
        );
        assert_eq!(
            output.geometry.upper_arm_directions[0].value,
            Some(ModelVector3 {
                x: 0.0,
                y: 1.0,
                z: 0.0
            })
        );
        assert_eq!(
            output.signals[69].expect("left elbow flexion").value,
            Some(0.0)
        );
    }

    fn array(values: Vec<f32>) -> ArrayD<f32> {
        Array1::from(values).into_dyn()
    }
}
