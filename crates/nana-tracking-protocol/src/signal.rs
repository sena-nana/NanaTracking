use alloc::{boxed::Box, vec::Vec};
use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

pub const STABLE_SIGNAL_COUNT: usize = 88;
const STABLE_SIGNAL_COUNT_U16: u16 = 88;
const SIGNAL_WORD_COUNT: usize = (u16::MAX as usize + 1) / 64;

/// A non-zero stable or additive NTP Signal ID.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct SignalId(u16);

impl SignalId {
    #[must_use]
    pub const fn new(raw: u16) -> Option<Self> {
        if raw == 0 { None } else { Some(Self(raw)) }
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }

    #[must_use]
    pub const fn stable_slot(self) -> Option<usize> {
        if self.0 >= 1 && self.0 <= STABLE_SIGNAL_COUNT_U16 {
            Some(self.0 as usize - 1)
        } else {
            None
        }
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn from_stable_slot(slot: usize) -> Option<Self> {
        if slot < STABLE_SIGNAL_COUNT {
            Some(Self(slot as u16 + 1))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn is_experimental(self) -> bool {
        self.0 >= 0x8000
    }
}

/// Descriptor capability bitmap covering the complete non-zero `u16` ID domain.
///
/// Its wire representation is a compact, sorted ID list; the 8 KiB dense storage is never copied
/// into every frame and makes membership deterministic for known and future additive IDs.
#[derive(Clone, Eq, PartialEq)]
pub struct SignalBitSet {
    words: Box<[u64; SIGNAL_WORD_COUNT]>,
}

impl SignalBitSet {
    #[must_use]
    pub fn new() -> Self {
        Self {
            words: Box::new([0; SIGNAL_WORD_COUNT]),
        }
    }

    #[must_use]
    pub fn stable_through(last: u16) -> Self {
        let mut value = Self::new();
        for raw in 1..=last.min(STABLE_SIGNAL_COUNT_U16) {
            value.insert(SignalId(raw));
        }
        value
    }

    pub fn insert(&mut self, id: SignalId) -> bool {
        let raw = usize::from(id.0);
        let mask = 1_u64 << (raw % 64);
        let word = &mut self.words[raw / 64];
        let was_absent = *word & mask == 0;
        *word |= mask;
        was_absent
    }

    pub fn remove(&mut self, id: SignalId) -> bool {
        let raw = usize::from(id.0);
        let mask = 1_u64 << (raw % 64);
        let word = &mut self.words[raw / 64];
        let was_present = *word & mask != 0;
        *word &= !mask;
        was_present
    }

    #[must_use]
    pub fn contains(&self, id: SignalId) -> bool {
        let raw = usize::from(id.0);
        self.words[raw / 64] & (1_u64 << (raw % 64)) != 0
    }

    #[must_use]
    pub fn contains_stable_range(&self, last: u16) -> bool {
        (1..=last).all(|raw| self.contains(SignalId(raw)))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    pub fn iter(&self) -> impl Iterator<Item = SignalId> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, word)| {
                let word = *word;
                (0..64).filter_map(move |bit| {
                    let raw = u16::try_from(word_index * 64 + bit).ok()?;
                    (raw != 0 && word & (1_u64 << bit) != 0).then_some(SignalId(raw))
                })
            })
    }
}

impl Default for SignalBitSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SignalBitSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_set().entries(self.iter()).finish()
    }
}

impl Serialize for SignalBitSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.iter()
            .map(SignalId::get)
            .collect::<Vec<_>>()
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SignalBitSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let ids = Vec::<u16>::deserialize(deserializer)?;
        let mut result = Self::new();
        for raw in ids {
            let id =
                SignalId::new(raw).ok_or_else(|| D::Error::custom("Signal ID 0 is reserved"))?;
            result.insert(id);
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum StableSet {
    Basic,
    Spatial,
    Full,
    Optional,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    NormalizedSigned,
    NormalizedUnsigned,
    GazeYaw,
    GazePitch,
    TorsoTranslation,
    HeadTranslation,
    Angle,
}

impl ScalarType {
    #[must_use]
    pub fn valid_range(self) -> (f32, f32) {
        match self {
            Self::NormalizedSigned | Self::TorsoTranslation | Self::HeadTranslation => (-1.0, 1.0),
            Self::NormalizedUnsigned => (0.0, 1.0),
            Self::GazeYaw => (-1.2, 1.2),
            Self::GazePitch => (-0.8, 0.8),
            Self::Angle => (-core::f32::consts::PI, core::f32::consts::PI),
        }
    }

    #[must_use]
    pub fn contains(self, value: f32) -> bool {
        if !value.is_finite() {
            return false;
        }
        let (minimum, maximum) = self.valid_range();
        value >= minimum
            && if self == Self::Angle {
                value < maximum
            } else {
                value <= maximum
            }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SignalMetadata {
    pub id: SignalId,
    pub stable_name: &'static str,
    pub scalar_type: ScalarType,
    pub set: StableSet,
}

impl SignalMetadata {
    /// Calibrated neutral for every NTP v1 scalar type. A producer represents unavailable state
    /// through state/value fields and must never substitute this number for missing data.
    #[must_use]
    pub const fn neutral_value(self) -> f32 {
        0.0
    }

    #[must_use]
    pub fn get(id: SignalId) -> Option<Self> {
        let slot = id.stable_slot()?;
        let scalar_type = match id.get() {
            9 | 10 | 13..=17 | 28 | 30 | 33..=36 | 41 | 70 | 75 | 79 | 80 => {
                ScalarType::NormalizedUnsigned
            }
            37 | 39 => ScalarType::GazeYaw,
            38 | 40 => ScalarType::GazePitch,
            42..=44 => ScalarType::TorsoTranslation,
            45..=47 | 51..=53 => ScalarType::Angle,
            48..=50 => ScalarType::HeadTranslation,
            _ => ScalarType::NormalizedSigned,
        };
        let set = match id.get() {
            1..=36 => StableSet::Basic,
            37..=41 => StableSet::Spatial,
            42..=76 => StableSet::Full,
            77..=88 => StableSet::Optional,
            _ => return None,
        };
        Some(Self {
            id,
            stable_name: STABLE_SIGNAL_NAMES[slot],
            scalar_type,
            set,
        })
    }

    pub fn all() -> impl Iterator<Item = Self> {
        (1..=STABLE_SIGNAL_COUNT_U16).filter_map(|raw| Self::get(SignalId(raw)))
    }
}

const STABLE_SIGNAL_NAMES: [&str; STABLE_SIGNAL_COUNT] = [
    "brow.left.inner_vertical",
    "brow.right.inner_vertical",
    "brow.left.outer_vertical",
    "brow.right.outer_vertical",
    "brow.left.medial",
    "brow.right.medial",
    "eye.left.aperture",
    "eye.right.aperture",
    "eye.left.squint",
    "eye.right.squint",
    "cheek.left.inflation",
    "cheek.right.inflation",
    "cheek.left.raise",
    "cheek.right.raise",
    "nose.left.sneer",
    "nose.right.sneer",
    "jaw.open",
    "jaw.lateral",
    "jaw.protraction",
    "mouth.corner.left.vertical",
    "mouth.corner.right.vertical",
    "mouth.corner.left.horizontal",
    "mouth.corner.right.horizontal",
    "mouth.lip.upper_left.vertical",
    "mouth.lip.upper_right.vertical",
    "mouth.lip.lower_left.vertical",
    "mouth.lip.lower_right.vertical",
    "mouth.seal",
    "mouth.protrusion",
    "mouth.roundness",
    "mouth.lip.upper_roll",
    "mouth.lip.lower_roll",
    "mouth.press.left",
    "mouth.press.right",
    "mouth.dimple.left",
    "mouth.dimple.right",
    "gaze.left.yaw",
    "gaze.left.pitch",
    "gaze.right.yaw",
    "gaze.right.pitch",
    "tongue.extension",
    "torso.translation.x",
    "torso.translation.y",
    "torso.translation.z",
    "torso.rotation.pitch",
    "torso.rotation.yaw",
    "torso.rotation.roll",
    "head.relative_translation.x",
    "head.relative_translation.y",
    "head.relative_translation.z",
    "head.relative_rotation.pitch",
    "head.relative_rotation.yaw",
    "head.relative_rotation.roll",
    "tongue.horizontal",
    "tongue.vertical",
    "tongue.curl",
    "auricle.left.elevation",
    "auricle.right.elevation",
    "auricle.left.protraction",
    "auricle.right.protraction",
    "auricle.left.flattening",
    "auricle.right.flattening",
    "shoulder_girdle.left.elevation",
    "shoulder_girdle.right.elevation",
    "shoulder_girdle.left.protraction",
    "shoulder_girdle.right.protraction",
    "arm.left.shoulder.flexion",
    "arm.left.shoulder.abduction",
    "arm.left.shoulder.twist",
    "arm.left.elbow.flexion",
    "arm.left.forearm.twist",
    "arm.right.shoulder.flexion",
    "arm.right.shoulder.abduction",
    "arm.right.shoulder.twist",
    "arm.right.elbow.flexion",
    "arm.right.forearm.twist",
    "nose.left.alar_flare",
    "nose.right.alar_flare",
    "mouth.bite.upper_lip",
    "mouth.bite.lower_lip",
    "auricle.left.twist",
    "auricle.right.twist",
    "wrist.left.flexion",
    "wrist.left.deviation",
    "wrist.left.twist",
    "wrist.right.flexion",
    "wrist.right.deviation",
    "wrist.right.twist",
];
