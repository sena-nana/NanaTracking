use std::{env, path::Path};

use nana_tracking_runtime_api::{
    ActiveProvider, TrackingModelInput, TrackingModelOutput, TrackingModelSession,
};
use nana_tracking_runtime_ort::{OrtCpuOptions, OrtFaceBasicSession, initialize_from_dylib};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let dylib = arguments
        .next()
        .ok_or("usage: infer-face-basic <libonnxruntime> <model-package>")?;
    let package = arguments
        .next()
        .ok_or("usage: infer-face-basic <libonnxruntime> <model-package>")?;
    if arguments.next().is_some() {
        return Err("usage: infer-face-basic <libonnxruntime> <model-package>".into());
    }
    initialize_from_dylib(Path::new(&dylib))?;
    let mut session = OrtFaceBasicSession::load(Path::new(&package), OrtCpuOptions::default())?;
    let parity = session.verify_fixed_vector(1.0e-5, 1.0e-4)?;
    let height = session.metadata().input_shape[2];
    let width = session.metadata().input_shape[3];
    let rgb = vec![127_u8; width * height * 3];
    let mut output = TrackingModelOutput::preallocated(ActiveProvider::OnnxRuntimeCpu);
    for _ in 0..100 {
        infer_once(&mut session, &rgb, width, height, &mut output)?;
    }
    let mut preprocess = Vec::with_capacity(2_000);
    let mut inference = Vec::with_capacity(2_000);
    let mut readback = Vec::with_capacity(2_000);
    for _ in 0..2_000 {
        infer_once(&mut session, &rgb, width, height, &mut output)?;
        preprocess.push(output.preprocess_ns);
        inference.push(output.inference_ns);
        readback.push(output.readback_ns);
    }
    let supported = output.signals.iter().flatten().count();
    if supported != 36 || output.geometry.head_camera_pose.state.is_unsupported() {
        return Err("runtime output did not contain a complete FaceBasic result".into());
    }
    println!(
        "provider={:?} signals={supported} parity_outputs={} preprocess_ns={:?} inference_ns={:?} readback_ns={:?}",
        output.provider,
        parity.len(),
        percentiles(preprocess),
        percentiles(inference),
        percentiles(readback)
    );
    Ok(())
}

fn infer_once(
    session: &mut OrtFaceBasicSession,
    rgb: &[u8],
    width: usize,
    height: usize,
    output: &mut TrackingModelOutput,
) -> Result<(), nana_tracking_runtime_api::TrackingRuntimeError> {
    session.infer(
        TrackingModelInput {
            rgb,
            width,
            height,
            row_stride: width * 3,
            capture_timestamp_ns: 1_000_000_000,
            generation: 0,
        },
        output,
    )
}

fn percentiles(mut values: Vec<u64>) -> (u64, u64, u64) {
    values.sort_unstable();
    (values[999], values[1_899], values[1_979])
}
