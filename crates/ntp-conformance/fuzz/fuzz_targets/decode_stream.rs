#![no_main]

use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use nana_tracking_protocol::{
    ActiveLayout, CanonicalCodec, CompactFrameCodec, CompactFrameInput, CompactSample, LayoutLimits,
    LayoutProposal, NanaTrackingDescriptor, NanaTrackingResult, SessionId, SignalBitSet,
    SignalState, StructureFeatures, TrackingFeatures, TrackingProfile,
};

static COMPACT_SEED: OnceLock<(ActiveLayout, Vec<u8>)> = OnceLock::new();

fuzz_target!(|data: &[u8]| {
    let _ = CanonicalCodec::decode::<NanaTrackingDescriptor>(data);
    let _ = CanonicalCodec::decode::<NanaTrackingResult>(data);
    let (layout, seed) = COMPACT_SEED.get_or_init(|| {
        let descriptor = NanaTrackingDescriptor::from_capabilities(
            SignalBitSet::stable_through(36),
            StructureFeatures::HEAD_GEOMETRY,
            TrackingFeatures::empty(),
        );
        let layout = ActiveLayout::negotiate(
            1,
            LayoutProposal::for_profile(TrackingProfile::Basic, 60),
            &descriptor,
            LayoutLimits::default(),
        )
        .expect("fixed fuzz layout is valid");
        let samples = vec![
            CompactSample::available(0.0, 0.5, SignalState::Observed);
            layout.signals().len()
        ];
        let seed = CompactFrameCodec::encode(
            &layout,
            &CompactFrameInput {
                session_id: SessionId([1; 16]),
                generation: 1,
                sequence: 1,
                capture_timestamp_ns: 1,
                produced_timestamp_ns: 2,
                samples: &samples,
            },
        )
        .expect("fixed fuzz frame is valid");
        (layout, seed)
    });
    let _ = CompactFrameCodec::decode(&layout, data);
    let mut mutated = seed.clone();
    for (target, mutation) in mutated.iter_mut().zip(data) {
        *target ^= mutation;
    }
    let _ = CompactFrameCodec::decode(layout, &mutated);
});
