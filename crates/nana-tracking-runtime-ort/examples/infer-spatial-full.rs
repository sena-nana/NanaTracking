use std::{env, path::Path};

use nana_tracking_runtime_api::{
    ActiveProvider, TrackingModelInput, TrackingModelOutput, TrackingModelSession,
};
use nana_tracking_runtime_ort::{
    OrtCoreMlOptions, OrtCpuOptions, OrtFaceSpatialSession, OrtFullSetSession,
    initialize_from_dylib,
};

const WARMUP_ITERATIONS: usize = 100;
const MEASURED_ITERATIONS: usize = 2_000;
const CAPTURE_TIMESTAMP_NS: u64 = 1_000_000_000;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let dylib = arguments
        .next()
        .ok_or("usage: infer-spatial-full <libonnxruntime> <spatial-package> <full-package> [coreml-profile-directory]")?;
    let spatial_package = arguments
        .next()
        .ok_or("usage: infer-spatial-full <libonnxruntime> <spatial-package> <full-package> [coreml-profile-directory]")?;
    let full_package = arguments
        .next()
        .ok_or("usage: infer-spatial-full <libonnxruntime> <spatial-package> <full-package> [coreml-profile-directory]")?;
    let core_ml_profile_directory = arguments.next();
    if arguments.next().is_some() {
        return Err("usage: infer-spatial-full <libonnxruntime> <spatial-package> <full-package> [coreml-profile-directory]".into());
    }
    initialize_from_dylib(Path::new(&dylib))?;
    let core_ml = core_ml_profile_directory.is_some();
    let (mut spatial, mut full) = if let Some(directory) = core_ml_profile_directory {
        let directory = Path::new(&directory);
        (
            OrtFaceSpatialSession::load_core_ml(
                Path::new(&spatial_package),
                OrtCoreMlOptions::new(directory.to_path_buf()),
            )?,
            OrtFullSetSession::load_core_ml(
                Path::new(&full_package),
                OrtCoreMlOptions::new(directory.to_path_buf()),
            )?,
        )
    } else {
        let options = OrtCpuOptions::default();
        (
            OrtFaceSpatialSession::load(Path::new(&spatial_package), options)?,
            OrtFullSetSession::load(Path::new(&full_package), options)?,
        )
    };
    let (absolute_tolerance, relative_tolerance) = if core_ml {
        (1.0e-3, 1.0e-3)
    } else {
        (1.0e-5, 1.0e-4)
    };
    let spatial_parity = spatial.verify_fixed_vector(absolute_tolerance, relative_tolerance)?;
    let full_parity = full.verify_fixed_vector(absolute_tolerance, relative_tolerance)?;

    let spatial_height = spatial.metadata().input_shape[2];
    let spatial_width = spatial.metadata().input_shape[3];
    let full_height = full.metadata().input_shape[2];
    let full_width = full.metadata().input_shape[3];
    let spatial_rgb = vec![127_u8; spatial_width * spatial_height * 3];
    let full_rgb = vec![127_u8; full_width * full_height * 3];
    let mut output = TrackingModelOutput::preallocated(ActiveProvider::OnnxRuntimeCpu);

    for _ in 0..WARMUP_ITERATIONS {
        infer_spatial(
            &mut spatial,
            &spatial_rgb,
            spatial_width,
            spatial_height,
            &mut output,
        )?;
    }
    let mut spatial_preprocess = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut spatial_inference = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut spatial_readback = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut spatial_total = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut spatial_result_age = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..MEASURED_ITERATIONS {
        infer_spatial(
            &mut spatial,
            &spatial_rgb,
            spatial_width,
            spatial_height,
            &mut output,
        )?;
        spatial_preprocess.push(output.preprocess_ns);
        spatial_inference.push(output.inference_ns);
        spatial_readback.push(output.readback_ns);
        spatial_total.push(output.preprocess_ns + output.inference_ns + output.readback_ns);
        spatial_result_age.push(output.produced_timestamp_ns - CAPTURE_TIMESTAMP_NS);
    }
    let spatial_template = output.clone();

    for _ in 0..WARMUP_ITERATIONS {
        output.clone_from(&spatial_template);
        infer_full(&mut full, &full_rgb, full_width, full_height, &mut output)?;
    }
    let mut full_preprocess = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut full_inference = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut full_readback = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut full_total = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut fused_result_age = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..MEASURED_ITERATIONS {
        output.clone_from(&spatial_template);
        let prior_preprocess_ns = output.preprocess_ns;
        let prior_inference_ns = output.inference_ns;
        let prior_readback_ns = output.readback_ns;
        infer_full(&mut full, &full_rgb, full_width, full_height, &mut output)?;
        let preprocess_ns = output.preprocess_ns - prior_preprocess_ns;
        let inference_ns = output.inference_ns - prior_inference_ns;
        let readback_ns = output.readback_ns - prior_readback_ns;
        full_preprocess.push(preprocess_ns);
        full_inference.push(inference_ns);
        full_readback.push(readback_ns);
        full_total.push(preprocess_ns + inference_ns + readback_ns);
        fused_result_age.push(output.produced_timestamp_ns - CAPTURE_TIMESTAMP_NS);
    }
    let supported = output.signals.iter().take(76).flatten().count();
    if supported != 76
        || output.geometry.head_camera_pose.state.is_unsupported()
        || output.geometry.torso_camera_pose.state.is_unsupported()
        || output
            .geometry
            .upper_body_joint_positions
            .iter()
            .any(|joint| joint.state.is_unsupported())
    {
        return Err("fused runtime output did not contain signals 1..76 and geometry state".into());
    }
    println!(
        "provider={:?} signals={supported} spatial_parity_outputs={} spatial_parity_max_abs={} full_parity_outputs={} full_parity_max_abs={} spatial_preprocess_ns={:?} spatial_inference_ns={:?} spatial_readback_ns={:?} spatial_total_ns={:?} spatial_result_age_ns={:?} full_preprocess_ns={:?} full_inference_ns={:?} full_readback_ns={:?} full_total_ns={:?} fused_result_age_ns={:?}",
        output.provider,
        spatial_parity.len(),
        spatial_parity
            .values()
            .map(|output| output.maximum_absolute_error)
            .fold(0.0_f32, f32::max),
        full_parity.len(),
        full_parity
            .values()
            .map(|output| output.maximum_absolute_error)
            .fold(0.0_f32, f32::max),
        percentiles(spatial_preprocess),
        percentiles(spatial_inference),
        percentiles(spatial_readback),
        percentiles(spatial_total),
        percentiles(spatial_result_age),
        percentiles(full_preprocess),
        percentiles(full_inference),
        percentiles(full_readback),
        percentiles(full_total),
        percentiles(fused_result_age),
    );
    Ok(())
}

fn infer_spatial(
    session: &mut OrtFaceSpatialSession,
    rgb: &[u8],
    width: usize,
    height: usize,
    output: &mut TrackingModelOutput,
) -> Result<(), nana_tracking_runtime_api::TrackingRuntimeError> {
    session.infer(
        model_input(rgb, width, height, CAPTURE_TIMESTAMP_NS + 50_000_000),
        output,
    )
}

fn infer_full(
    session: &mut OrtFullSetSession,
    rgb: &[u8],
    width: usize,
    height: usize,
    output: &mut TrackingModelOutput,
) -> Result<(), nana_tracking_runtime_api::TrackingRuntimeError> {
    session.infer_fused(
        model_input(rgb, width, height, output.produced_timestamp_ns),
        output,
    )
}

fn model_input(
    rgb: &[u8],
    width: usize,
    height: usize,
    processing_started_timestamp_ns: u64,
) -> TrackingModelInput<'_> {
    TrackingModelInput {
        rgb,
        width,
        height,
        row_stride: width * 3,
        capture_timestamp_ns: CAPTURE_TIMESTAMP_NS,
        processing_started_timestamp_ns,
        generation: 0,
    }
}

fn percentiles(mut values: Vec<u64>) -> (u64, u64, u64) {
    values.sort_unstable();
    (values[999], values[1_899], values[1_979])
}
