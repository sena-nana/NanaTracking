#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use std::{collections::BTreeMap, fs::File, path::Path, path::PathBuf, time::Instant};

use nana_tracking_runtime_api::{
    ActiveProvider, ModelPose, ModelQuaternion, ModelRegionQuality, ModelScalar,
    ModelTrackingState, ModelVector3, TrackingModelInput, TrackingModelMetadata,
    TrackingModelOutput, TrackingModelSession, TrackingRuntimeError, verify_model_package,
};
use ndarray::{Array4, ArrayD, ArrayViewD, Ix4};
use ndarray_npy::NpzReader;
use ort::session::{Session, SessionOutputs, builder::GraphOptimizationLevel};

const BASIC_SIGNAL_COUNT: usize = 36;
const UNSIGNED_SLOTS: [usize; 14] = [8, 9, 12, 13, 14, 15, 16, 27, 29, 32, 33, 34, 35, 40];

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

pub struct OrtFaceBasicSession {
    metadata: TrackingModelMetadata,
    session: Session,
    input: Array4<f32>,
    package_root: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputParity {
    pub mean_absolute_error: f32,
    pub maximum_absolute_error: f32,
}

impl OrtFaceBasicSession {
    /// Verify a portable package and create a real ORT CPU inference session.
    ///
    /// # Errors
    ///
    /// Fails closed on package, capability, graph I/O, provider, or session errors.
    pub fn load(root: &Path, options: OrtCpuOptions) -> Result<Self, TrackingRuntimeError> {
        if options.intra_threads == 0 {
            return Err(TrackingRuntimeError::InvalidInput);
        }
        let package = verify_model_package(root)?;
        let metadata = package.metadata;
        if metadata.supported_signals != (1..=36).collect::<Vec<_>>()
            || metadata.supported_structures != ["head_geometry"]
            || !metadata
                .allowed_backends
                .iter()
                .any(|backend| backend == "onnxruntime")
        {
            return Err(TrackingRuntimeError::InvalidMetadata);
        }
        for required in ["rig", "pose", "visibility", "confidence"] {
            if !metadata.output_names.iter().any(|name| name == required) {
                return Err(TrackingRuntimeError::InvalidMetadata);
            }
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
        let session = Session::builder()
            .map_err(backend_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(backend_error)?
            .with_intra_threads(options.intra_threads)
            .map_err(backend_error)?
            .commit_from_file(&package.model_path)
            .map_err(backend_error)?;
        if session.inputs.len() != 1 || session.inputs[0].name != "image" {
            return Err(TrackingRuntimeError::InvalidMetadata);
        }
        Ok(Self {
            metadata,
            session,
            input: Array4::zeros((1, 3, height, width)),
            package_root: package.root,
        })
    }

    /// Run the package's interoperable NPZ vector through the Rust ORT session and compare every
    /// named deployment output with the packaged `PyTorch` reference.
    ///
    /// # Errors
    ///
    /// Fails on malformed vectors, missing or mismatched outputs, non-finite values, shape changes,
    /// or any element outside `absolute_tolerance + relative_tolerance * abs(expected)`.
    pub fn verify_fixed_vector(
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
}

impl TrackingModelSession for OrtFaceBasicSession {
    fn metadata(&self) -> &TrackingModelMetadata {
        &self.metadata
    }

    fn infer(
        &mut self,
        input: TrackingModelInput<'_>,
        output: &mut TrackingModelOutput,
    ) -> Result<(), TrackingRuntimeError> {
        input.validate()?;
        output.clear();
        output.provider = ActiveProvider::OnnxRuntimeCpu;

        let preprocess_started = Instant::now();
        self.preprocess(&input);
        output.preprocess_ns = elapsed_ns(preprocess_started);

        let inference_started = Instant::now();
        let values = self
            .session
            .run(ort::inputs!["image" => self.input.view()].map_err(backend_error)?)
            .map_err(backend_error)?;
        output.inference_ns = elapsed_ns(inference_started);

        let readback_started = Instant::now();
        let extracted = extract_outputs(&values)?;
        populate_output(&extracted, input.capture_timestamp_ns, output)?;
        output.readback_ns = elapsed_ns(readback_started);
        output.produced_timestamp_ns = input
            .capture_timestamp_ns
            .saturating_add(output.preprocess_ns)
            .saturating_add(output.inference_ns)
            .saturating_add(output.readback_ns);
        Ok(())
    }

    fn reset_temporal_state(&mut self) {}
}

struct FaceBasicViews<'a> {
    rig: ArrayViewD<'a, f32>,
    pose: ArrayViewD<'a, f32>,
    visibility: ArrayViewD<'a, f32>,
    confidence: ArrayViewD<'a, f32>,
}

fn extract_outputs<'a>(
    values: &'a SessionOutputs<'_, '_>,
) -> Result<FaceBasicViews<'a>, TrackingRuntimeError> {
    let tensor = |name| {
        values
            .get(name)
            .ok_or_else(|| missing_output(name))?
            .try_extract_tensor::<f32>()
            .map_err(backend_error)
    };
    let result = FaceBasicViews {
        rig: tensor("rig")?,
        pose: tensor("pose")?,
        visibility: tensor("visibility")?,
        confidence: tensor("confidence")?,
    };
    if result.rig.len() != BASIC_SIGNAL_COUNT
        || result.confidence.len() != BASIC_SIGNAL_COUNT
        || result.pose.len() != 7
        || result.visibility.len() != 3
    {
        return Err(TrackingRuntimeError::Backend(
            "FaceBasic output shape differs from its package contract".into(),
        ));
    }
    Ok(result)
}

fn populate_output(
    values: &FaceBasicViews<'_>,
    capture_timestamp_ns: u64,
    output: &mut TrackingModelOutput,
) -> Result<(), TrackingRuntimeError> {
    let state = match argmax(values.visibility.iter().copied())? {
        0 => ModelTrackingState::Observed,
        1 => ModelTrackingState::Occluded,
        2 => ModelTrackingState::OutOfFrame,
        _ => unreachable!("three visibility logits"),
    };
    let mean_confidence = values
        .confidence
        .iter()
        .copied()
        .map(clamp_confidence)
        .sum::<f32>()
        / 36.0;
    for (slot, (&value, &sample_confidence)) in
        values.rig.iter().zip(values.confidence.iter()).enumerate()
    {
        if !value.is_finite() || !sample_confidence.is_finite() {
            return Err(TrackingRuntimeError::Backend(
                "FaceBasic produced a non-finite scalar".into(),
            ));
        }
        let minimum = if UNSIGNED_SLOTS.contains(&slot) {
            0.0
        } else {
            -1.0
        };
        let value = value.clamp(minimum, 1.0);
        output.signals[slot] = Some(if state == ModelTrackingState::Observed {
            ModelScalar::observed(
                value,
                clamp_confidence(sample_confidence),
                capture_timestamp_ns,
            )
        } else {
            ModelScalar::unavailable(
                clamp_confidence(sample_confidence),
                state,
                capture_timestamp_ns,
            )
        });
    }
    let pose = values.pose.iter().copied().collect::<Vec<_>>();
    if pose.iter().any(|value| !value.is_finite()) {
        return Err(TrackingRuntimeError::Backend(
            "FaceBasic produced a non-finite pose".into(),
        ));
    }
    output.geometry.head_camera_pose = if state == ModelTrackingState::Observed {
        observed_pose(&pose, mean_confidence, capture_timestamp_ns)
    } else {
        nana_tracking_runtime_api::ModelTracked::unavailable(
            mean_confidence,
            state,
            capture_timestamp_ns,
        )
    };
    output.quality.overall_confidence = mean_confidence;
    output.quality.face = ModelRegionQuality {
        confidence: mean_confidence,
        state,
    };
    output.quality.eyes = ModelRegionQuality {
        confidence: mean_confidence,
        state,
    };
    Ok(())
}

fn observed_pose(
    pose: &[f32],
    confidence: f32,
    capture_timestamp_ns: u64,
) -> nana_tracking_runtime_api::ModelTracked<ModelPose> {
    let norm = pose[3..7]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let mut quaternion = if norm < 1.0e-6 {
        ModelQuaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    } else {
        ModelQuaternion {
            x: pose[3] / norm,
            y: pose[4] / norm,
            z: pose[5] / norm,
            w: pose[6] / norm,
        }
    };
    if quaternion.w < 0.0 {
        quaternion = ModelQuaternion {
            x: -quaternion.x,
            y: -quaternion.y,
            z: -quaternion.z,
            w: -quaternion.w,
        };
    }
    nana_tracking_runtime_api::ModelTracked::observed(
        ModelPose {
            position: ModelVector3 {
                x: pose[0],
                y: pose[1],
                z: pose[2],
            },
            rotation: quaternion,
        },
        confidence,
        capture_timestamp_ns,
    )
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

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn backend_error(error: impl std::fmt::Display) -> TrackingRuntimeError {
    TrackingRuntimeError::Backend(error.to_string())
}

fn missing_output(name: &str) -> TrackingRuntimeError {
    TrackingRuntimeError::Backend(format!("model did not return required output {name}"))
}
