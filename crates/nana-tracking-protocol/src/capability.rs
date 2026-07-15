use serde::{Deserialize, Serialize};

use crate::{revision::ContractRevisions, signal::SignalBitSet};

#[derive(
    Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum TrackingProfile {
    #[default]
    Partial = 0,
    Basic = 1,
    Spatial = 2,
    Full = 3,
}

macro_rules! feature_bits {
    ($name:ident { $($constant:ident = $value:expr),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        #[repr(transparent)]
        pub struct $name(pub u64);

        impl $name {
            $(pub const $constant: Self = Self($value);)+

            #[must_use]
            pub const fn empty() -> Self { Self(0) }

            #[must_use]
            pub const fn bits(self) -> u64 { self.0 }

            #[must_use]
            pub const fn contains(self, required: Self) -> bool {
                self.0 & required.0 == required.0
            }

            pub fn insert(&mut self, feature: Self) { self.0 |= feature.0; }
        }

        impl core::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self::Output { Self(self.0 | rhs.0) }
        }
    };
}

feature_bits!(StructureFeatures {
    HEAD_GEOMETRY = 1 << 0,
    EYE_GEOMETRY = 1 << 1,
    LOOK_AT_POINT = 1 << 2,
    FACE_GEOMETRY = 1 << 3,
    BODY_SKELETON = 1 << 4,
});

feature_bits!(TrackingFeatures {
    METRIC_COORDINATES = 1 << 0,
    DENSE_FACE_MESH = 1 << 1,
    AURICLE_LOCAL_GEOMETRY = 1 << 2,
    WRIST_POSE = 1 << 3,
});

impl StructureFeatures {
    pub const BASIC_REQUIRED: Self = Self::HEAD_GEOMETRY;
    pub const SPATIAL_REQUIRED: Self = Self(
        Self::HEAD_GEOMETRY.0
            | Self::EYE_GEOMETRY.0
            | Self::LOOK_AT_POINT.0
            | Self::FACE_GEOMETRY.0,
    );
    pub const FULL_REQUIRED: Self = Self(Self::SPATIAL_REQUIRED.0 | Self::BODY_SKELETON.0);
}

/// Session-level NTP capability declaration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanaTrackingDescriptor {
    pub revisions: ContractRevisions,
    pub guaranteed_profile: TrackingProfile,
    pub supported_signals: SignalBitSet,
    pub supported_structures: StructureFeatures,
    pub features: TrackingFeatures,
}

impl NanaTrackingDescriptor {
    #[must_use]
    pub fn from_capabilities(
        supported_signals: SignalBitSet,
        supported_structures: StructureFeatures,
        features: TrackingFeatures,
    ) -> Self {
        let guaranteed_profile = Self::highest_profile(&supported_signals, supported_structures);
        Self {
            revisions: ContractRevisions::NTP_V1,
            guaranteed_profile,
            supported_signals,
            supported_structures,
            features,
        }
    }

    #[must_use]
    pub fn highest_profile(
        signals: &SignalBitSet,
        structures: StructureFeatures,
    ) -> TrackingProfile {
        if signals.contains_stable_range(76)
            && structures.contains(StructureFeatures::FULL_REQUIRED)
        {
            TrackingProfile::Full
        } else if signals.contains_stable_range(41)
            && structures.contains(StructureFeatures::SPATIAL_REQUIRED)
        {
            TrackingProfile::Spatial
        } else if signals.contains_stable_range(36)
            && structures.contains(StructureFeatures::BASIC_REQUIRED)
        {
            TrackingProfile::Basic
        } else {
            TrackingProfile::Partial
        }
    }
}
