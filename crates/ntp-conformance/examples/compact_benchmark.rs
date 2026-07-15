use std::{
    alloc::System,
    hint::black_box,
    time::{Duration, Instant},
};

use nana_tracking_protocol::{
    ActiveLayout, CompactFrameCodec, CompactFrameInput, CompactSample, LayoutLimits,
    LayoutProposal, NanaTrackingDescriptor, NanaTrackingResult, SessionId, SignalBitSet, SignalId,
    SignalMetadata, SignalSample, SignalState, StructureFeatures, TrackingFeatures,
    TrackingProfile, WireDecode, WireEncode,
};
use serde::Serialize;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

const COMPACT_ITERATIONS: u32 = 100_000;
const CANONICAL_ITERATIONS: u32 = 5_000;
const HEADER_LEN: usize = 56;

#[derive(Serialize)]
struct Measurement {
    iterations: u32,
    ns_per_frame: f64,
    allocations_total: usize,
    bytes_allocated_total: usize,
}

#[derive(Serialize)]
struct Representation {
    bytes_per_frame: usize,
    bytes_per_second_60_fps: usize,
    bytes_per_second_120_fps: usize,
    encode: Measurement,
    decode: Measurement,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    smoke_only: bool,
    host_os: &'static str,
    host_arch: &'static str,
    profile: &'static str,
    signal_count: usize,
    layout_hash_hex: String,
    compact_dense_i16: Representation,
    per_frame_id_value: Representation,
    canonical_nullable_result: Representation,
    interpretation: [&'static str; 3],
}

fn main() {
    let descriptor = NanaTrackingDescriptor::from_capabilities(
        SignalBitSet::stable_through(41),
        StructureFeatures::HEAD_GEOMETRY,
        TrackingFeatures::empty(),
    );
    let mut proposal = LayoutProposal::for_profile(TrackingProfile::Basic, 120);
    proposal.extra_signals = vec![SignalId::new(37).unwrap(), SignalId::new(41).unwrap()];
    let layout =
        ActiveLayout::negotiate(7, proposal, &descriptor, LayoutLimits::default()).unwrap();
    let samples = layout
        .signals()
        .iter()
        .map(|&signal| {
            let (minimum, maximum) = SignalMetadata::get(signal)
                .unwrap()
                .scalar_type
                .valid_range();
            CompactSample::available((minimum + maximum) * 0.5, 0.8, SignalState::Observed)
        })
        .collect::<Vec<_>>();
    let input = CompactFrameInput {
        session_id: SessionId([1; 16]),
        generation: 2,
        sequence: 3,
        capture_timestamp_ns: 1_000,
        produced_timestamp_ns: 1_100,
        samples: &samples,
    };
    let compact = CompactFrameCodec::encode(&layout, &input).unwrap();

    let mut id_value = vec![0_u8; HEADER_LEN + layout.signals().len() * 6];
    encode_id_value(layout.signals(), &samples, &mut id_value);

    let mut canonical = NanaTrackingResult::unsupported(SessionId([1; 16]), 2, 3, 1_000, 1_100);
    for &signal in layout.signals() {
        canonical.rig.set(
            signal,
            SignalSample::available(0.0, 0.8, SignalState::Observed, 1_000, 0),
        );
    }
    let canonical_bytes = canonical.encode_wire().unwrap();

    let mut compact_scratch = vec![0_u8; layout.frame_len()];
    let compact_encode = measure(COMPACT_ITERATIONS, || {
        CompactFrameCodec::encode_into(&layout, &input, black_box(&mut compact_scratch)).unwrap();
    });
    let compact_decode = measure(COMPACT_ITERATIONS, || {
        black_box(CompactFrameCodec::decode(&layout, black_box(&compact)).unwrap());
    });

    let mut id_value_scratch = vec![0_u8; id_value.len()];
    let id_value_encode = measure(COMPACT_ITERATIONS, || {
        encode_id_value(layout.signals(), &samples, black_box(&mut id_value_scratch));
    });
    let id_value_decode = measure(COMPACT_ITERATIONS, || {
        black_box(decode_id_value(black_box(&id_value)).unwrap());
    });

    let canonical_encode = measure(CANONICAL_ITERATIONS, || {
        black_box(canonical.encode_wire().unwrap());
    });
    let canonical_decode = measure(CANONICAL_ITERATIONS, || {
        black_box(NanaTrackingResult::decode_wire(black_box(&canonical_bytes)).unwrap());
    });

    let report = Report {
        schema: "nanatracking.issue14.compact-benchmark/1",
        smoke_only: true,
        host_os: std::env::consts::OS,
        host_arch: std::env::consts::ARCH,
        profile: "BasicV1+gaze.left.yaw+tongue.extension",
        signal_count: layout.signals().len(),
        layout_hash_hex: hex(&layout.hash()),
        compact_dense_i16: representation(compact.len(), compact_encode, compact_decode),
        per_frame_id_value: representation(id_value.len(), id_value_encode, id_value_decode),
        canonical_nullable_result: representation(
            canonical_bytes.len(),
            canonical_encode,
            canonical_decode,
        ),
        interpretation: [
            "Synthetic protocol microbenchmark only; it does not prove tracking quality or production readiness.",
            "Compact encode_into and decode reuse a validated layout and caller-owned storage.",
            "Canonical nullable results remain the owned/debug contract, not the live compact transport format.",
        ],
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn representation(
    bytes_per_frame: usize,
    encode: Measurement,
    decode: Measurement,
) -> Representation {
    Representation {
        bytes_per_frame,
        bytes_per_second_60_fps: bytes_per_frame * 60,
        bytes_per_second_120_fps: bytes_per_frame * 120,
        encode,
        decode,
    }
}

fn measure(iterations: u32, mut operation: impl FnMut()) -> Measurement {
    for _ in 0..1_000 {
        operation();
    }
    let region = Region::new(GLOBAL);
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    let elapsed = started.elapsed();
    let allocation = region.change();
    Measurement {
        iterations,
        ns_per_frame: nanos_per_frame(elapsed, iterations),
        allocations_total: allocation.allocations,
        bytes_allocated_total: allocation.bytes_allocated,
    }
}

fn nanos_per_frame(duration: Duration, iterations: u32) -> f64 {
    duration.as_secs_f64() * 1_000_000_000.0 / f64::from(iterations)
}

fn encode_id_value(signals: &[SignalId], samples: &[CompactSample], output: &mut [u8]) {
    output[..HEADER_LEN].fill(0);
    for (index, (signal, sample)) in signals.iter().zip(samples).enumerate() {
        let offset = HEADER_LEN + index * 6;
        output[offset..offset + 2].copy_from_slice(&signal.get().to_le_bytes());
        let scalar = SignalMetadata::get(*signal).unwrap().scalar_type;
        let (minimum, maximum) = scalar.valid_range();
        let normalized = (sample.value.unwrap() - minimum) / (maximum - minimum);
        #[allow(clippy::cast_possible_truncation)]
        let value = (normalized * 65_534.0 - 32_767.0).round() as i16;
        output[offset + 2..offset + 4].copy_from_slice(&value.to_le_bytes());
        output[offset + 4] = sample.state as u8;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let confidence = (sample.confidence * 255.0).round() as u8;
        output[offset + 5] = confidence;
    }
}

fn decode_id_value(bytes: &[u8]) -> Option<Vec<(SignalId, f32, SignalState, f32)>> {
    let mut seen = [false; 88];
    let mut decoded = Vec::with_capacity((bytes.len() - HEADER_LEN) / 6);
    for item in bytes[HEADER_LEN..].chunks_exact(6) {
        let id = SignalId::new(u16::from_le_bytes([item[0], item[1]]))?;
        let slot = id.stable_slot()?;
        if seen[slot] {
            return None;
        }
        seen[slot] = true;
        let scalar = SignalMetadata::get(id)?.scalar_type;
        let raw = i16::from_le_bytes([item[2], item[3]]);
        if raw == i16::MIN {
            return None;
        }
        let (minimum, maximum) = scalar.valid_range();
        #[allow(clippy::cast_precision_loss)]
        let normalized = ((i32::from(raw) + 32_767) as f32) / 65_534.0;
        let value = normalized * (maximum - minimum) + minimum;
        if !scalar.contains(value) {
            return None;
        }
        let state = match item[4] {
            0 => SignalState::Observed,
            1 => SignalState::Fused,
            2 => SignalState::Predicted,
            _ => return None,
        };
        decoded.push((id, value, state, f32::from(item[5]) / 255.0));
    }
    Some(decoded)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, byte| {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").unwrap();
        output
    })
}
