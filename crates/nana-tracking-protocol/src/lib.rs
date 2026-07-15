#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

extern crate alloc;

pub mod capability;
pub mod codec;
#[cfg(feature = "diagnostic-json")]
pub mod diagnostic;
pub mod ffi;
pub mod revision;
pub mod signal;
pub mod stream;
pub mod types;
pub mod validate;

pub use capability::{
    NanaTrackingDescriptor, StructureFeatures, TrackingFeatures, TrackingProfile,
};
pub use codec::{CanonicalCodec, CodecError, WireDecode, WireEncode};
pub use revision::{ContractRevisions, ProtocolVersion, Revision};
pub use signal::{
    STABLE_SIGNAL_COUNT, ScalarType, SignalBitSet, SignalId, SignalMetadata, StableSet,
};
pub use stream::{AcceptedFrame, ResultStreamGuard, StreamError};
pub use types::*;
pub use validate::{ContractError, Validate};
