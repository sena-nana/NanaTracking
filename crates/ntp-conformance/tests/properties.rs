use nana_tracking_protocol::{
    CanonicalCodec, NanaTrackingDescriptor, NanaTrackingResult, SignalBitSet, StructureFeatures,
    TrackingFeatures,
};
use ntp_conformance::assess_profile;

#[test]
fn profile_assessment_matches_the_protocol_for_all_prefixes_and_structure_sets() {
    for last in 0..=88 {
        for bits in 0..32 {
            let descriptor = NanaTrackingDescriptor::from_capabilities(
                SignalBitSet::stable_through(last),
                StructureFeatures(bits),
                TrackingFeatures::empty(),
            );
            let assessment = assess_profile(&descriptor);
            assert_eq!(assessment.declared_profile, assessment.computed_profile);
            assert_eq!(
                assessment.computed_profile,
                NanaTrackingDescriptor::highest_profile(
                    &descriptor.supported_signals,
                    descriptor.supported_structures,
                )
            );
        }
    }
}

#[test]
fn arbitrary_wire_bytes_are_rejected_without_panicking() {
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    for length in 0..512 {
        let mut bytes = Vec::with_capacity(length);
        for _ in 0..length {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            bytes.push((state >> 56) as u8);
        }
        let _ = CanonicalCodec::decode::<NanaTrackingDescriptor>(&bytes);
        let _ = CanonicalCodec::decode::<NanaTrackingResult>(&bytes);
    }
}
