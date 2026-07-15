use nana_tracking_protocol::{
    NanaTrackingDescriptor, SignalId, StructureFeatures, TrackingProfile,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProfileAssessment {
    pub declared_profile: TrackingProfile,
    pub computed_profile: TrackingProfile,
    pub missing_basic_signals: Vec<u16>,
    pub missing_spatial_signals: Vec<u16>,
    pub missing_full_signals: Vec<u16>,
    pub missing_basic_structures: u64,
    pub missing_spatial_structures: u64,
    pub missing_full_structures: u64,
}

/// Computes profile certification exclusively from complete signal and structure sets.
#[must_use]
pub fn assess_profile(descriptor: &NanaTrackingDescriptor) -> ProfileAssessment {
    let structures = descriptor.supported_structures;
    ProfileAssessment {
        declared_profile: descriptor.guaranteed_profile,
        computed_profile: NanaTrackingDescriptor::highest_profile(
            &descriptor.supported_signals,
            structures,
        ),
        missing_basic_signals: missing_signals(descriptor, 1, 36),
        missing_spatial_signals: missing_signals(descriptor, 1, 41),
        missing_full_signals: missing_signals(descriptor, 1, 76),
        missing_basic_structures: missing_structure_bits(
            structures,
            StructureFeatures::BASIC_REQUIRED,
        ),
        missing_spatial_structures: missing_structure_bits(
            structures,
            StructureFeatures::SPATIAL_REQUIRED,
        ),
        missing_full_structures: missing_structure_bits(
            structures,
            StructureFeatures::FULL_REQUIRED,
        ),
    }
}

fn missing_signals(descriptor: &NanaTrackingDescriptor, first: u16, last: u16) -> Vec<u16> {
    (first..=last)
        .filter(|raw| {
            !descriptor
                .supported_signals
                .contains(SignalId::new(*raw).expect("profile IDs are non-zero"))
        })
        .collect()
}

const fn missing_structure_bits(actual: StructureFeatures, required: StructureFeatures) -> u64 {
    required.bits() & !actual.bits()
}
