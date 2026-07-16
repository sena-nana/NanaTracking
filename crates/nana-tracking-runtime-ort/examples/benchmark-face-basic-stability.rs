use std::{
    collections::VecDeque,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

use nana_tracking_runtime_api::{
    ActiveProvider, TrackingModelInput, TrackingModelOutput, TrackingModelSession,
};
use nana_tracking_runtime_ort::{OrtCpuOptions, OrtFaceBasicSession, initialize_from_dylib};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const WARMUP_ITERATIONS: usize = 100;
const RESERVOIR_CAPACITY: usize = 65_536;
const EDGE_WINDOW_CAPACITY: usize = 4_096;
const RESERVOIR_SEED: u64 = 47;
const MAXIMUM_RESULT_AGE_P95_DRIFT_MS: f64 = 2.0;
const MAXIMUM_RSS_GROWTH_BYTES: i64 = 32 * 1024 * 1024;
const MAXIMUM_THREAD_GROWTH: i64 = 2;
const MAXIMUM_CPU_CORE_EQUIVALENTS: f64 = 1.0;
const MAXIMUM_STOPPED_CPU_CORE_EQUIVALENTS: f64 = 0.05;
const STOP_OBSERVATION_SECONDS: f64 = 1.0;

#[derive(Clone, Copy)]
struct LatencySample {
    capture_to_result_ns: u64,
    result_age_ns: u64,
}

struct BoundedLatencySamples {
    seen: u64,
    state: u64,
    reservoir: Vec<LatencySample>,
    first: Vec<LatencySample>,
    last: VecDeque<LatencySample>,
}

impl BoundedLatencySamples {
    fn new() -> Self {
        Self {
            seen: 0,
            state: RESERVOIR_SEED,
            reservoir: Vec::with_capacity(RESERVOIR_CAPACITY),
            first: Vec::with_capacity(EDGE_WINDOW_CAPACITY),
            last: VecDeque::with_capacity(EDGE_WINDOW_CAPACITY),
        }
    }

    fn add(&mut self, sample: LatencySample) {
        self.seen = self.seen.saturating_add(1);
        if self.first.len() < EDGE_WINDOW_CAPACITY {
            self.first.push(sample);
        }
        if self.last.len() == EDGE_WINDOW_CAPACITY {
            self.last.pop_front();
        }
        self.last.push_back(sample);
        if self.reservoir.len() < RESERVOIR_CAPACITY {
            self.reservoir.push(sample);
            return;
        }
        let candidate = self.next_random() % self.seen;
        let capacity = u64::try_from(RESERVOIR_CAPACITY).expect("reservoir capacity fits u64");
        if candidate < capacity {
            let index = usize::try_from(candidate).expect("candidate is below reservoir capacity");
            self.reservoir[index] = sample;
        }
    }

    fn next_random(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn retained(&self) -> usize {
        self.reservoir.len() + self.first.len() + self.last.len()
    }

    fn summary(&self, result_age: bool) -> Value {
        json!({
            "all_reservoir": latency_summary(self.reservoir.iter().copied(), result_age),
            "first_window": latency_summary(self.first.iter().copied(), result_age),
            "last_window": latency_summary(self.last.iter().copied(), result_age),
        })
    }
}

#[derive(Clone)]
struct ResourceSample {
    elapsed_seconds: f64,
    rss_bytes: u64,
    thread_count: u64,
    process_cpu_seconds: f64,
    hottest_thread_cpu_percent: f64,
}

impl ResourceSample {
    fn as_json(&self) -> Value {
        json!({
            "elapsed_seconds": self.elapsed_seconds,
            "rss_bytes": self.rss_bytes,
            "thread_count": self.thread_count,
            "process_cpu_seconds": self.process_cpu_seconds,
            "hottest_thread_cpu_percent": self.hottest_thread_cpu_percent,
        })
    }
}

struct Configuration {
    dylib: PathBuf,
    package: PathBuf,
    output: PathBuf,
    duration: Duration,
    target_fps: f64,
    resource_interval: Duration,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let configuration = parse_configuration()?;
    initialize_from_dylib(&configuration.dylib)?;
    let options = OrtCpuOptions::default();
    let mut session = OrtFaceBasicSession::load(&configuration.package, options)?;
    let parity = session.verify_fixed_vector(1.0e-5, 1.0e-4)?;
    let metadata = session.metadata().clone();
    let height = metadata.input_shape[2];
    let width = metadata.input_shape[3];
    let rgb = vec![127_u8; width * height * 3];
    let input_digest = sha256_bytes(&rgb);
    let dylib_digest = sha256_file(&configuration.dylib)?;
    let mut output = TrackingModelOutput::preallocated(ActiveProvider::OnnxRuntimeCpu);
    let clock_origin = Instant::now();
    for _ in 0..WARMUP_ITERATIONS {
        infer_at_capture(
            &mut session,
            &rgb,
            width,
            height,
            elapsed_ns(clock_origin),
            &mut output,
        )?;
    }
    let provider = format!("{:?}", output.provider);

    let benchmark_started = Instant::now();
    let initial_resources = process_resources(benchmark_started, benchmark_started)?;
    let mut resource_samples = vec![initial_resources.clone()];
    let period = Duration::from_secs_f64(1.0 / configuration.target_fps);
    let mut deadline = benchmark_started;
    let end = benchmark_started + configuration.duration;
    let mut next_resource_sample = benchmark_started + configuration.resource_interval;
    let mut samples = BoundedLatencySamples::new();
    let mut skipped_capture_periods = 0_u64;

    loop {
        let mut now = Instant::now();
        if now < deadline {
            thread::sleep(deadline - now);
            now = Instant::now();
        }
        if now >= end && samples.seen > 0 {
            break;
        }
        if now > deadline {
            let skipped = now.duration_since(deadline).as_nanos() / period.as_nanos();
            let skipped = u32::try_from(skipped).unwrap_or(u32::MAX);
            skipped_capture_periods = skipped_capture_periods.saturating_add(u64::from(skipped));
            deadline += period * skipped;
        }
        let capture_timestamp_ns = elapsed_ns(clock_origin);
        infer_at_capture(
            &mut session,
            &rgb,
            width,
            height,
            capture_timestamp_ns,
            &mut output,
        )?;
        let consumed_timestamp_ns = elapsed_ns(clock_origin);
        samples.add(LatencySample {
            capture_to_result_ns: output
                .produced_timestamp_ns
                .saturating_sub(capture_timestamp_ns),
            result_age_ns: consumed_timestamp_ns.saturating_sub(capture_timestamp_ns),
        });
        deadline += period;
        if now >= next_resource_sample {
            resource_samples.push(process_resources(benchmark_started, Instant::now())?);
            next_resource_sample += configuration.resource_interval;
        }
    }

    let duration_measured = benchmark_started.elapsed().as_secs_f64();
    let active_final_resources = process_resources(benchmark_started, Instant::now())?;
    resource_samples.push(active_final_resources.clone());
    drop(session);
    thread::sleep(Duration::from_secs_f64(STOP_OBSERVATION_SECONDS));
    let stopped_resources = process_resources(benchmark_started, Instant::now())?;

    let report = build_report(&ReportInputs {
        configuration: &configuration,
        metadata: &metadata,
        options: &options,
        provider: &provider,
        parity_max_abs: parity
            .values()
            .map(|value| value.maximum_absolute_error)
            .fold(0.0_f32, f32::max),
        input_digest: &input_digest,
        dylib_digest: &dylib_digest,
        duration_measured,
        samples: &samples,
        skipped_capture_periods,
        resource_samples: &resource_samples,
        stopped_resources: &stopped_resources,
    })?;
    write_report(&configuration.output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report["stability"]["passed"] != Value::Bool(true) {
        return Err("Rust ORT stability gates failed".into());
    }
    Ok(())
}

fn write_report(output: &Path, report: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn infer_at_capture(
    session: &mut OrtFaceBasicSession,
    rgb: &[u8],
    width: usize,
    height: usize,
    capture_timestamp_ns: u64,
    output: &mut TrackingModelOutput,
) -> Result<(), nana_tracking_runtime_api::TrackingRuntimeError> {
    session.infer(
        TrackingModelInput {
            rgb,
            width,
            height,
            row_stride: width * 3,
            capture_timestamp_ns,
            processing_started_timestamp_ns: capture_timestamp_ns,
            generation: 0,
        },
        output,
    )
}

fn parse_configuration() -> Result<Configuration, Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let usage = "usage: benchmark-face-basic-stability <libonnxruntime> <model-package> <output-json> [duration-seconds] [target-fps] [resource-sample-seconds]";
    let dylib = arguments.next().ok_or(usage)?;
    let package = arguments.next().ok_or(usage)?;
    let output = arguments.next().ok_or(usage)?;
    let duration_seconds = parse_optional(&mut arguments, 1_800.0, "duration")?;
    let target_fps = parse_optional(&mut arguments, 60.0, "target FPS")?;
    let resource_seconds = parse_optional(&mut arguments, 60.0, "resource interval")?;
    if arguments.next().is_some()
        || !(0.01..=7_200.0).contains(&duration_seconds)
        || !(1.0..=240.0).contains(&target_fps)
        || !(0.01..=600.0).contains(&resource_seconds)
    {
        return Err(usage.into());
    }
    Ok(Configuration {
        dylib: dylib.into(),
        package: package.into(),
        output: output.into(),
        duration: Duration::from_secs_f64(duration_seconds),
        target_fps,
        resource_interval: Duration::from_secs_f64(resource_seconds),
    })
}

fn parse_optional(
    arguments: &mut impl Iterator<Item = String>,
    default: f64,
    name: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    arguments.next().map_or(Ok(default), |value| {
        value.parse().map_err(|_| format!("invalid {name}").into())
    })
}

struct ReportInputs<'a> {
    configuration: &'a Configuration,
    metadata: &'a nana_tracking_runtime_api::TrackingModelMetadata,
    options: &'a OrtCpuOptions,
    provider: &'a str,
    parity_max_abs: f32,
    input_digest: &'a str,
    dylib_digest: &'a str,
    duration_measured: f64,
    samples: &'a BoundedLatencySamples,
    skipped_capture_periods: u64,
    resource_samples: &'a [ResourceSample],
    stopped_resources: &'a ResourceSample,
}

// Keeping gate derivation beside the emitted schema prevents report fields and pass/fail logic
// from drifting apart during review.
#[allow(clippy::too_many_lines)]
fn build_report(inputs: &ReportInputs<'_>) -> Result<Value, Box<dyn std::error::Error>> {
    let capture_to_result = inputs.samples.summary(false);
    let result_age = inputs.samples.summary(true);
    let first_p95 = result_age["first_window"]["p95_ms"]
        .as_f64()
        .ok_or("missing first-window P95")?;
    let last_p95 = result_age["last_window"]["p95_ms"]
        .as_f64()
        .ok_or("missing last-window P95")?;
    let result_age_p95_drift_ms = last_p95 - first_p95;
    let initial = inputs
        .resource_samples
        .first()
        .ok_or("missing initial resources")?;
    let active_final = inputs
        .resource_samples
        .last()
        .ok_or("missing final resources")?;
    let active_cpu_seconds = active_final.process_cpu_seconds - initial.process_cpu_seconds;
    let cpu_core_equivalents = active_cpu_seconds / inputs.duration_measured.max(f64::EPSILON);
    let stopped_wall_seconds =
        inputs.stopped_resources.elapsed_seconds - active_final.elapsed_seconds;
    let stopped_cpu_core_equivalents = (inputs.stopped_resources.process_cpu_seconds
        - active_final.process_cpu_seconds)
        / stopped_wall_seconds.max(f64::EPSILON);
    let rss_growth_bytes = signed_difference(active_final.rss_bytes, initial.rss_bytes);
    let thread_growth = signed_difference(active_final.thread_count, initial.thread_count);
    let delivered_fps = u64_to_f64(inputs.samples.seen) / inputs.duration_measured;
    let stopped_cpu_limit =
        MAXIMUM_STOPPED_CPU_CORE_EQUIVALENTS.min((cpu_core_equivalents * 0.25).max(0.01));
    let gates = json!({
        "duration_reached": inputs.duration_measured >= inputs.configuration.duration.as_secs_f64() * 0.99,
        "target_cadence_reached": delivered_fps >= inputs.configuration.target_fps * 0.95,
        "result_age_p95_drift_within_limit": result_age_p95_drift_ms <= MAXIMUM_RESULT_AGE_P95_DRIFT_MS,
        "rss_growth_within_limit": rss_growth_bytes <= MAXIMUM_RSS_GROWTH_BYTES,
        "thread_growth_within_limit": thread_growth <= MAXIMUM_THREAD_GROWTH,
        "cpu_core_equivalents_within_limit": cpu_core_equivalents <= MAXIMUM_CPU_CORE_EQUIVALENTS,
        "stopped_cpu_reduced_within_limit": stopped_cpu_core_equivalents <= stopped_cpu_limit,
        "stopped_thread_count_within_limit": inputs.stopped_resources.thread_count <= initial.thread_count + 1,
    });
    let passed = gates
        .as_object()
        .is_some_and(|values| values.values().all(|value| value == &Value::Bool(true)));
    let peak_rss = inputs
        .resource_samples
        .iter()
        .map(|sample| sample.rss_bytes)
        .max()
        .ok_or("missing RSS samples")?;
    let hottest_thread = inputs
        .resource_samples
        .iter()
        .map(|sample| sample.hottest_thread_cpu_percent)
        .fold(0.0_f64, f64::max);
    let (git_commit, git_dirty) = git_state()?;
    Ok(json!({
        "schema_version": "rust-face-basic-runtime-stability/1.0.0",
        "smoke_only": inputs.metadata.smoke_only,
        "model_digest": inputs.metadata.model_digest,
        "source_checkpoint_digest": inputs.metadata.source_checkpoint_digest,
        "ntp_schema_revision": inputs.metadata.ntp_schema_revision,
        "signal_registry_revision": inputs.metadata.signal_registry_revision,
        "normalization_revision": inputs.metadata.normalization_revision,
        "calibration_revision": inputs.metadata.calibration_revision,
        "feature_revision": inputs.metadata.feature_revision,
        "hardware": {
            "os": env::consts::OS,
            "architecture": env::consts::ARCH,
            "cpu": cpu_description(),
        },
        "runtime": {
            "rust": command_line("rustc", &["--version"]).unwrap_or_else(|_| "unavailable".into()),
            "runtime_ort_crate": env!("CARGO_PKG_VERSION"),
            "ort_wrapper": "2.0.0-rc.9",
            "onnxruntime_dylib": inputs.configuration.dylib,
            "onnxruntime_dylib_sha256": inputs.dylib_digest,
            "active_provider": inputs.provider,
            "intra_threads": inputs.options.intra_threads,
            "inter_threads": 1,
            "execution_mode": "sequential",
            "allow_spinning": inputs.options.allow_spinning,
            "input_shape": inputs.metadata.input_shape,
            "target_fps": inputs.configuration.target_fps,
            "delivered_fps": delivered_fps,
            "completed_frames": inputs.samples.seen,
            "skipped_capture_periods": inputs.skipped_capture_periods,
            "warmup": WARMUP_ITERATIONS,
            "duration_seconds_requested": inputs.configuration.duration.as_secs_f64(),
            "duration_seconds_measured": inputs.duration_measured,
            "scheduling": "paced latest-capture; overdue capture periods are skipped, never queued",
            "queue_capacity": 0,
            "maximum_observed_queue_depth": 0,
            "known_copy_boundary": "borrowed RGB is resized directly into one reused owned NCHW input; ORT-internal copies are not observable",
        },
        "fixed_vector": {
            "input_sha256": inputs.input_digest,
            "maximum_absolute_error": inputs.parity_max_abs,
            "absolute_tolerance": 1.0e-5,
            "relative_tolerance": 1.0e-4,
        },
        "bounded_sampling": {
            "algorithm": "deterministic Algorithm R reservoir plus fixed first/last windows",
            "seed": RESERVOIR_SEED,
            "observed_samples": inputs.samples.seen,
            "retained_samples_including_windows": inputs.samples.retained(),
            "reservoir_capacity": RESERVOIR_CAPACITY,
            "edge_window_capacity": EDGE_WINDOW_CAPACITY,
        },
        "capture_to_result": capture_to_result,
        "result_age_at_consume": result_age,
        "resources": {
            "cpu_core_equivalents": cpu_core_equivalents,
            "peak_sampled_rss_bytes": peak_rss,
            "rss_growth_bytes": rss_growth_bytes,
            "thread_growth": thread_growth,
            "peak_sampled_hottest_thread_cpu_percent": hottest_thread,
            "samples": inputs.resource_samples.iter().map(ResourceSample::as_json).collect::<Vec<_>>(),
            "after_session_drop": inputs.stopped_resources.as_json(),
            "stopped_cpu_core_equivalents": stopped_cpu_core_equivalents,
            "gpu": null,
            "vram": null,
            "gpu_note": "GPU and VRAM are unavailable for the CPU provider and are not inferred",
        },
        "stability": {
            "passed": passed,
            "gates": gates,
            "result_age_p95_drift_ms": result_age_p95_drift_ms,
            "maximum_result_age_p95_drift_ms": MAXIMUM_RESULT_AGE_P95_DRIFT_MS,
            "maximum_rss_growth_bytes": MAXIMUM_RSS_GROWTH_BYTES,
            "maximum_thread_growth": MAXIMUM_THREAD_GROWTH,
            "maximum_cpu_core_equivalents": MAXIMUM_CPU_CORE_EQUIVALENTS,
            "maximum_stopped_cpu_core_equivalents": stopped_cpu_limit,
            "stop_observation_seconds": STOP_OBSERVATION_SECONDS,
        },
        "provenance": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "cargo_lock_sha256": sha256_file(Path::new("Cargo.lock"))?,
        },
        "limitations": "Fixed package test-vector RGB smoke only. This proves the Rust ORT CPU consumer scheduling, result freshness, bounded sampling, and sampled process resources on this host. It does not prove camera I/O, tracking quality, GPU execution, other platforms, or production readiness.",
    }))
}

fn process_resources(
    benchmark_started: Instant,
    sampled_at: Instant,
) -> Result<ResourceSample, Box<dyn std::error::Error>> {
    let pid = std::process::id().to_string();
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    return Err("resource sampling is implemented for macOS and Linux; Windows is skipped".into());
    let output = Command::new("ps")
        .args(["-o", "rss=", "-o", "time=", "-p", &pid])
        .output()?;
    if !output.status.success() {
        return Err("ps resource sample failed".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let mut fields = text.split_whitespace();
    let rss_kib: u64 = fields.next().ok_or("missing RSS")?.parse()?;
    let process_cpu_seconds = parse_cpu_time(fields.next().ok_or("missing CPU time")?)?;
    if fields.next().is_some() {
        return Err("unexpected ps resource fields".into());
    }
    let (thread_count, hottest_thread_cpu_percent) = thread_resources(&pid)?;
    Ok(ResourceSample {
        elapsed_seconds: sampled_at.duration_since(benchmark_started).as_secs_f64(),
        rss_bytes: rss_kib.saturating_mul(1_024),
        thread_count,
        process_cpu_seconds,
        hottest_thread_cpu_percent,
    })
}

fn thread_resources(pid: &str) -> Result<(u64, f64), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    let arguments = ["-M", "-p", pid, "-o", "%cpu"];
    #[cfg(target_os = "linux")]
    let arguments = ["-L", "-p", pid, "-o", "pcpu"];
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    return Err("per-thread CPU sampling is implemented for macOS and Linux".into());
    let output = Command::new("ps").args(arguments).output()?;
    if !output.status.success() {
        return Err("ps per-thread CPU sample failed".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let mut thread_count = 0_u64;
    let mut hottest = 0.0_f64;
    for line in text.lines().skip(1).filter(|line| !line.trim().is_empty()) {
        let cpu: f64 = line
            .split_whitespace()
            .last()
            .ok_or("missing thread CPU")?
            .parse()?;
        thread_count = thread_count.saturating_add(1);
        hottest = hottest.max(cpu);
    }
    if thread_count == 0 {
        return Err("missing per-thread CPU samples".into());
    }
    Ok((thread_count, hottest))
}

fn parse_cpu_time(value: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let (days, clock) = value
        .split_once('-')
        .map_or(Ok((0_u64, value)), |(days, clock)| {
            days.parse().map(|days| (days, clock))
        })?;
    let fields = clock.split(':').collect::<Vec<_>>();
    let (hours, minutes, seconds): (u64, u64, f64) = match fields.as_slice() {
        [minutes, seconds] => (0_u64, minutes.parse()?, seconds.parse::<f64>()?),
        [hours, minutes, seconds] => (hours.parse()?, minutes.parse()?, seconds.parse::<f64>()?),
        _ => return Err("unsupported ps CPU time".into()),
    };
    Ok(u64_to_f64(days.saturating_mul(86_400))
        + u64_to_f64(hours.saturating_mul(3_600))
        + u64_to_f64(minutes.saturating_mul(60))
        + seconds)
}

fn latency_summary(samples: impl Iterator<Item = LatencySample>, result_age: bool) -> Value {
    let mut values = samples
        .map(|sample| {
            if result_age {
                sample.result_age_ns
            } else {
                sample.capture_to_result_ns
            }
        })
        .collect::<Vec<_>>();
    values.sort_unstable();
    let mean =
        values.iter().copied().map(u64_to_f64).sum::<f64>() / usize_to_f64(values.len()).max(1.0);
    json!({
        "p50_ms": percentile_ms(&values, 50, 100),
        "p95_ms": percentile_ms(&values, 95, 100),
        "p99_ms": percentile_ms(&values, 99, 100),
        "mean_ms": mean / 1_000_000.0,
    })
}

fn percentile_ms(values: &[u64], numerator: usize, denominator: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let maximum_index = values.len() - 1;
    let scaled = maximum_index.saturating_mul(numerator);
    let lower = scaled / denominator;
    let remainder = scaled % denominator;
    let upper = (lower + usize::from(remainder > 0)).min(maximum_index);
    let fraction = usize_to_f64(remainder) / usize_to_f64(denominator);
    (u64_to_f64(values[lower]) * (1.0 - fraction) + u64_to_f64(values[upper]) * fraction)
        / 1_000_000.0
}

fn elapsed_ns(origin: Instant) -> u64 {
    u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn signed_difference(after: u64, before: u64) -> i64 {
    let difference = i128::from(after)
        .saturating_sub(i128::from(before))
        .clamp(i128::from(i64::MIN), i128::from(i64::MAX));
    i64::try_from(difference).expect("difference was clamped to i64")
}

#[allow(clippy::cast_precision_loss)]
fn u64_to_f64(value: u64) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    fs::read(path).map(|bytes| sha256_bytes(&bytes))
}

fn git_state() -> Result<(String, bool), Box<dyn std::error::Error>> {
    let commit = command_line("git", &["rev-parse", "HEAD"])?;
    let dirty = !command_line("git", &["status", "--porcelain"])?.is_empty();
    Ok((commit, dirty))
}

fn cpu_description() -> String {
    #[cfg(target_os = "macos")]
    return command_line("sysctl", &["-n", "machdep.cpu.brand_string"])
        .unwrap_or_else(|_| "unavailable".into());
    #[cfg(target_os = "linux")]
    return fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("model name\t: "))
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unavailable".into());
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    "unavailable".into()
}

fn command_line(program: &str, arguments: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!("{program} failed").into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_samples_preserve_edges_and_capacity() {
        let mut samples = BoundedLatencySamples::new();
        for value in 0..100_000_u64 {
            samples.add(LatencySample {
                capture_to_result_ns: value,
                result_age_ns: value + 1,
            });
        }
        assert_eq!(samples.seen, 100_000);
        assert_eq!(samples.first.first().unwrap().capture_to_result_ns, 0);
        assert_eq!(samples.last.back().unwrap().capture_to_result_ns, 99_999);
        assert_eq!(
            samples.retained(),
            RESERVOIR_CAPACITY + 2 * EDGE_WINDOW_CAPACITY
        );
    }

    #[test]
    fn parses_ps_cpu_time_with_days_and_fractional_seconds() {
        assert_eq!(parse_cpu_time("01:02.50").unwrap(), 62.5);
        assert_eq!(parse_cpu_time("2:01:02.50").unwrap(), 7_262.5);
        assert_eq!(parse_cpu_time("1-02:01:02.50").unwrap(), 93_662.5);
    }
}
