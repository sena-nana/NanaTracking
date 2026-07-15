use serde::{Deserialize, Serialize};

/// Semantic protocol version. A major change may reinterpret existing fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const V1_0: Self = Self { major: 1, minor: 0 };
}

/// Revision of a registry or processing contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct Revision {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl Revision {
    pub const V1_0_0: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };
}

/// Every revision that affects the meaning of an NTP artifact boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ContractRevisions {
    pub protocol: ProtocolVersion,
    pub schema_revision: u32,
    pub signal_registry: Revision,
    pub normalization: Revision,
    pub calibration: Revision,
    pub features: Revision,
}

impl ContractRevisions {
    pub const NTP_V1: Self = Self {
        protocol: ProtocolVersion::V1_0,
        schema_revision: 1,
        signal_registry: Revision::V1_0_0,
        normalization: Revision::V1_0_0,
        calibration: Revision::V1_0_0,
        features: Revision::V1_0_0,
    };
}

impl Default for ContractRevisions {
    fn default() -> Self {
        Self::NTP_V1
    }
}
