//! Negotiated dense frame layout for latency-sensitive NTP transports.
//!
//! Layout construction is deliberately separate from the frame hot path. A session confirms one
//! immutable [`ActiveLayout`] before streaming; frames then contain only a fixed header, dense
//! little-endian `i16` values, and the negotiated fixed-width quality block. Transport adapters
//! carry these bytes but do not reinterpret them.

use alloc::{boxed::Box, vec, vec::Vec};
use core::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    ContractRevisions, NanaTrackingDescriptor, SessionId, SignalId, SignalMetadata, SignalState,
    TrackingProfile, Validate,
};

const COMPACT_MAGIC: [u8; 4] = *b"NTC1";
const COMPACT_WIRE_VERSION: u8 = 1;
const COMPACT_HEADER_LEN: usize = 56;
const LAYOUT_HASH_DOMAIN: &[u8] = b"NTP-COMPACT-LAYOUT\0";
const MISSING_VALUE: i16 = i16::MIN;
const QUANTIZED_MIN: i16 = -32_767;
const QUANTIZED_MAX: i16 = 32_767;

pub const BASE_LAYOUT_VERSION: u16 = 1;
pub const MAX_LAYOUT_SIGNALS: usize = 88;
pub const MAX_COMPACT_LAYOUT_BYTES: usize = 512;
pub const MAX_COMPACT_FRAME_BYTES: usize = 4_096;
pub const MAX_TARGET_FPS: u16 = 240;
pub const MAX_PENDING_LAYOUTS: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ValueEncoding {
    I16Normalized = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum QualityEncoding {
    None = 0,
    StateAndConfidenceU8 = 1,
}

impl QualityEncoding {
    const fn bytes_per_signal(self) -> usize {
        match self {
            Self::None => 0,
            Self::StateAndConfidenceU8 => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayoutProposal {
    pub revisions: ContractRevisions,
    pub profile: TrackingProfile,
    pub base_layout_version: u16,
    pub extra_signals: Vec<SignalId>,
    pub value_encoding: ValueEncoding,
    pub quality_encoding: QualityEncoding,
    pub target_fps: u16,
}

impl LayoutProposal {
    #[must_use]
    pub fn for_profile(profile: TrackingProfile, target_fps: u16) -> Self {
        Self {
            revisions: ContractRevisions::NTP_V1,
            profile,
            base_layout_version: BASE_LAYOUT_VERSION,
            extra_signals: Vec::new(),
            value_encoding: ValueEncoding::I16Normalized,
            quality_encoding: QualityEncoding::StateAndConfidenceU8,
            target_fps,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayoutLimits {
    pub max_signals: usize,
    pub max_layout_bytes: usize,
    pub max_frame_bytes: usize,
    pub max_target_fps: u16,
}

impl Default for LayoutLimits {
    fn default() -> Self {
        Self {
            max_signals: MAX_LAYOUT_SIGNALS,
            max_layout_bytes: MAX_COMPACT_LAYOUT_BYTES,
            max_frame_bytes: MAX_COMPACT_FRAME_BYTES,
            max_target_fps: MAX_TARGET_FPS,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutError {
    InvalidDescriptor,
    IncompatibleRevisions,
    ProfileMismatch,
    UnsupportedProfile,
    UnsupportedBaseLayout(u16),
    InvalidTargetFps,
    InvalidLimits,
    UnregisteredSignal(SignalId),
    UnsupportedSignal(SignalId),
    DuplicateSignal(SignalId),
    BaseSignalListedAsExtra(SignalId),
    TooManySignals,
    LayoutTooLarge,
    FrameTooLarge,
    SizeOverflow,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor => formatter.write_str("producer descriptor is invalid"),
            Self::IncompatibleRevisions => formatter.write_str("layout contract revisions differ"),
            Self::ProfileMismatch => {
                formatter.write_str("layout profile exceeds the descriptor guarantee")
            }
            Self::UnsupportedProfile => {
                formatter.write_str("partial profile has no dense base layout")
            }
            Self::UnsupportedBaseLayout(version) => {
                write!(formatter, "unsupported base layout version {version}")
            }
            Self::InvalidTargetFps => formatter.write_str("target frame rate is outside limits"),
            Self::InvalidLimits => formatter.write_str("invalid layout resource limits"),
            Self::UnregisteredSignal(id) => {
                write!(formatter, "unregistered Signal ID {}", id.get())
            }
            Self::UnsupportedSignal(id) => write!(formatter, "unsupported Signal ID {}", id.get()),
            Self::DuplicateSignal(id) => write!(formatter, "duplicate Signal ID {}", id.get()),
            Self::BaseSignalListedAsExtra(id) => {
                write!(formatter, "base Signal ID {} listed as extra", id.get())
            }
            Self::TooManySignals => formatter.write_str("layout signal count exceeds hard limit"),
            Self::LayoutTooLarge => formatter.write_str("layout proposal exceeds byte limit"),
            Self::FrameTooLarge => formatter.write_str("compact frame exceeds byte limit"),
            Self::SizeOverflow => formatter.write_str("compact layout size overflow"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LayoutError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveLayout {
    layout_id: u32,
    proposal: LayoutProposal,
    signals: Box<[SignalId]>,
    wire_rules: Box<[WireRule]>,
    hash: [u8; 32],
    frame_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WireRule {
    rejects_quantized_max: bool,
}

impl ActiveLayout {
    /// Validate a proposal against the producer descriptor and local hard limits.
    ///
    /// Signal ordering is consensus state: extras remain in proposal order and therefore affect
    /// the canonical layout hash.
    ///
    /// # Errors
    ///
    /// Rejects incompatible contracts, inconsistent capabilities, illegal signal lists, invalid
    /// rate limits, and any checked size calculation that exceeds the negotiated hard limits.
    pub fn negotiate(
        layout_id: u32,
        proposal: LayoutProposal,
        producer: &NanaTrackingDescriptor,
        limits: LayoutLimits,
    ) -> Result<Self, LayoutError> {
        producer
            .validate()
            .map_err(|_| LayoutError::InvalidDescriptor)?;
        validate_layout_limits(limits)?;
        if proposal.revisions != producer.revisions
            || proposal.revisions != ContractRevisions::NTP_V1
        {
            return Err(LayoutError::IncompatibleRevisions);
        }
        if proposal.profile > producer.guaranteed_profile {
            return Err(LayoutError::ProfileMismatch);
        }
        if proposal.base_layout_version != BASE_LAYOUT_VERSION {
            return Err(LayoutError::UnsupportedBaseLayout(
                proposal.base_layout_version,
            ));
        }
        if proposal.target_fps == 0 || proposal.target_fps > limits.max_target_fps {
            return Err(LayoutError::InvalidTargetFps);
        }

        let base_count = base_signal_count(proposal.profile)?;
        let signal_count = base_count
            .checked_add(proposal.extra_signals.len())
            .ok_or(LayoutError::SizeOverflow)?;
        if signal_count > limits.max_signals || signal_count > MAX_LAYOUT_SIGNALS {
            return Err(LayoutError::TooManySignals);
        }
        let layout_bytes = 64_usize
            .checked_add(
                proposal
                    .extra_signals
                    .len()
                    .checked_mul(2)
                    .ok_or(LayoutError::SizeOverflow)?,
            )
            .ok_or(LayoutError::SizeOverflow)?;
        if layout_bytes > limits.max_layout_bytes || layout_bytes > MAX_COMPACT_LAYOUT_BYTES {
            return Err(LayoutError::LayoutTooLarge);
        }

        let mut seen = [false; MAX_LAYOUT_SIGNALS];
        let mut signals = Vec::with_capacity(signal_count);
        for raw in 1..=u16::try_from(base_count).map_err(|_| LayoutError::SizeOverflow)? {
            let id = SignalId::new(raw).ok_or(LayoutError::SizeOverflow)?;
            if !producer.supported_signals.contains(id) {
                return Err(LayoutError::UnsupportedSignal(id));
            }
            seen[usize::from(raw - 1)] = true;
            signals.push(id);
        }
        for &id in &proposal.extra_signals {
            let slot = id
                .stable_slot()
                .ok_or(LayoutError::UnregisteredSignal(id))?;
            if slot < base_count {
                return Err(LayoutError::BaseSignalListedAsExtra(id));
            }
            if seen[slot] {
                return Err(LayoutError::DuplicateSignal(id));
            }
            if SignalMetadata::get(id).is_none() {
                return Err(LayoutError::UnregisteredSignal(id));
            }
            if !producer.supported_signals.contains(id) {
                return Err(LayoutError::UnsupportedSignal(id));
            }
            seen[slot] = true;
            signals.push(id);
        }

        let bytes_per_signal = 2_usize
            .checked_add(proposal.quality_encoding.bytes_per_signal())
            .ok_or(LayoutError::SizeOverflow)?;
        let frame_len = signals
            .len()
            .checked_mul(bytes_per_signal)
            .and_then(|length| length.checked_add(COMPACT_HEADER_LEN))
            .ok_or(LayoutError::SizeOverflow)?;
        if frame_len > limits.max_frame_bytes || frame_len > MAX_COMPACT_FRAME_BYTES {
            return Err(LayoutError::FrameTooLarge);
        }
        let wire_rules = signals
            .iter()
            .map(|&signal| WireRule {
                rejects_quantized_max: SignalMetadata::get(signal)
                    .is_some_and(|metadata| metadata.scalar_type == crate::ScalarType::Angle),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let hash = layout_hash(&proposal, &signals)?;
        Ok(Self {
            layout_id,
            proposal,
            signals: signals.into_boxed_slice(),
            wire_rules,
            hash,
            frame_len,
        })
    }

    #[must_use]
    pub const fn layout_id(&self) -> u32 {
        self.layout_id
    }

    #[must_use]
    pub fn signals(&self) -> &[SignalId] {
        &self.signals
    }

    #[must_use]
    pub const fn hash(&self) -> [u8; 32] {
        self.hash
    }

    #[must_use]
    pub const fn frame_len(&self) -> usize {
        self.frame_len
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn parameter_count(&self) -> u16 {
        self.signals.len() as u16
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn expected_payload_len(&self) -> u32 {
        self.frame_len as u32
    }

    #[must_use]
    pub const fn proposal(&self) -> &LayoutProposal {
        &self.proposal
    }

    #[must_use]
    pub const fn confirmation(&self) -> LayoutConfirm {
        LayoutConfirm {
            layout_id: self.layout_id,
            layout_hash: self.hash,
        }
    }

    #[must_use]
    pub fn recording_header(&self, session_id: SessionId, generation: u32) -> LayoutRecord {
        LayoutRecord {
            session_id,
            generation,
            layout_id: self.layout_id,
            layout_hash: self.hash,
            proposal: self.proposal.clone(),
            ordered_signals: self.signals.to_vec(),
            frame_len: self.expected_payload_len(),
        }
    }
}

fn base_signal_count(profile: TrackingProfile) -> Result<usize, LayoutError> {
    match profile {
        TrackingProfile::Basic => Ok(36),
        TrackingProfile::Spatial => Ok(41),
        TrackingProfile::Full => Ok(76),
        TrackingProfile::Partial => Err(LayoutError::UnsupportedProfile),
    }
}

fn validate_layout_limits(limits: LayoutLimits) -> Result<(), LayoutError> {
    if limits.max_signals == 0
        || limits.max_signals > MAX_LAYOUT_SIGNALS
        || limits.max_layout_bytes < 64
        || limits.max_layout_bytes > MAX_COMPACT_LAYOUT_BYTES
        || limits.max_frame_bytes < COMPACT_HEADER_LEN
        || limits.max_frame_bytes > MAX_COMPACT_FRAME_BYTES
        || limits.max_target_fps == 0
        || limits.max_target_fps > MAX_TARGET_FPS
    {
        Err(LayoutError::InvalidLimits)
    } else {
        Ok(())
    }
}

fn layout_hash(proposal: &LayoutProposal, signals: &[SignalId]) -> Result<[u8; 32], LayoutError> {
    let signal_count = u16::try_from(signals.len()).map_err(|_| LayoutError::TooManySignals)?;
    let mut hash = Sha256::new();
    hash.update(LAYOUT_HASH_DOMAIN);
    hash.update(proposal.revisions.protocol.major.to_le_bytes());
    hash.update(proposal.revisions.protocol.minor.to_le_bytes());
    hash.update(proposal.revisions.schema_revision.to_le_bytes());
    for revision in [
        proposal.revisions.signal_registry,
        proposal.revisions.normalization,
        proposal.revisions.calibration,
        proposal.revisions.features,
    ] {
        hash.update(revision.major.to_le_bytes());
        hash.update(revision.minor.to_le_bytes());
        hash.update(revision.patch.to_le_bytes());
    }
    hash.update([proposal.profile as u8]);
    hash.update(proposal.base_layout_version.to_le_bytes());
    hash.update([proposal.value_encoding as u8]);
    hash.update([proposal.quality_encoding as u8]);
    hash.update(proposal.target_fps.to_le_bytes());
    hash.update(signal_count.to_le_bytes());
    for id in signals {
        hash.update(id.get().to_le_bytes());
    }
    Ok(hash.finalize().into())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayoutRecord {
    pub session_id: SessionId,
    pub generation: u32,
    pub layout_id: u32,
    pub layout_hash: [u8; 32],
    pub proposal: LayoutProposal,
    pub ordered_signals: Vec<SignalId>,
    pub frame_len: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayoutAccept {
    pub layout_id: u32,
    pub layout_hash: [u8; 32],
    pub parameter_count: u16,
    pub expected_payload_len: u32,
}

impl LayoutAccept {
    #[must_use]
    pub const fn confirmation(self) -> LayoutConfirm {
        LayoutConfirm {
            layout_id: self.layout_id,
            layout_hash: self.layout_hash,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayoutConfirm {
    pub layout_id: u32,
    pub layout_hash: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandshakeLimits {
    pub layout: LayoutLimits,
    pub max_pending_layouts: usize,
    pub min_proposal_interval_ns: u64,
}

impl Default for HandshakeLimits {
    fn default() -> Self {
        Self {
            layout: LayoutLimits::default(),
            max_pending_layouts: MAX_PENDING_LAYOUTS,
            min_proposal_interval_ns: 100_000_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HandshakeError {
    InvalidLimits,
    RateLimited,
    TooManyPendingLayouts,
    DuplicateLayoutId(u32),
    UnknownLayout(u32),
    ConfirmationMismatch,
    Layout(LayoutError),
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("invalid handshake limits"),
            Self::RateLimited => formatter.write_str("layout proposal is rate limited"),
            Self::TooManyPendingLayouts => formatter.write_str("too many pending layouts"),
            Self::DuplicateLayoutId(id) => write!(formatter, "layout ID {id} is already pending"),
            Self::UnknownLayout(id) => write!(formatter, "layout ID {id} is not pending"),
            Self::ConfirmationMismatch => formatter.write_str("layout confirmation does not match"),
            Self::Layout(error) => write!(formatter, "invalid layout proposal: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HandshakeError {}

/// Bounded consumer-side `Proposal -> Accept -> Confirm` state machine.
#[derive(Clone, Debug)]
pub struct LayoutNegotiator {
    limits: HandshakeLimits,
    pending: Vec<ActiveLayout>,
    last_proposal_ns: Option<u64>,
}

impl LayoutNegotiator {
    /// Create a negotiator with an explicit pending-layout and renegotiation budget.
    ///
    /// # Errors
    ///
    /// Rejects zero or compile-time-exceeding pending-layout limits.
    pub fn new(limits: HandshakeLimits) -> Result<Self, HandshakeError> {
        if limits.max_pending_layouts == 0 || limits.max_pending_layouts > MAX_PENDING_LAYOUTS {
            return Err(HandshakeError::InvalidLimits);
        }
        Ok(Self {
            limits,
            pending: Vec::with_capacity(limits.max_pending_layouts),
            last_proposal_ns: None,
        })
    }

    /// Validate and retain one proposal until the remote peer confirms the returned hash.
    ///
    /// `now_ns` belongs to the connection adapter's monotonic clock. Even invalid proposals consume
    /// the rate budget so malformed renegotiation cannot become an unbounded CPU path.
    ///
    /// # Errors
    ///
    /// Rejects rate, pending-count, duplicate-ID, and underlying layout validation failures.
    pub fn receive_proposal(
        &mut self,
        layout_id: u32,
        proposal: LayoutProposal,
        producer: &NanaTrackingDescriptor,
        now_ns: u64,
    ) -> Result<LayoutAccept, HandshakeError> {
        if let Some(last) = self.last_proposal_ns
            && (now_ns < last || now_ns.saturating_sub(last) < self.limits.min_proposal_interval_ns)
        {
            return Err(HandshakeError::RateLimited);
        }
        self.last_proposal_ns = Some(now_ns);
        if self.pending.len() >= self.limits.max_pending_layouts {
            return Err(HandshakeError::TooManyPendingLayouts);
        }
        if self
            .pending
            .iter()
            .any(|layout| layout.layout_id == layout_id)
        {
            return Err(HandshakeError::DuplicateLayoutId(layout_id));
        }
        let layout = ActiveLayout::negotiate(layout_id, proposal, producer, self.limits.layout)
            .map_err(HandshakeError::Layout)?;
        let accept = LayoutAccept {
            layout_id,
            layout_hash: layout.hash,
            parameter_count: layout.parameter_count(),
            expected_payload_len: layout.expected_payload_len(),
        };
        self.pending.push(layout);
        Ok(accept)
    }

    /// Activate a pending layout only when the peer echoes its canonical ID and hash.
    ///
    /// # Errors
    ///
    /// Rejects unknown layout IDs and hash mismatches without removing the pending layout.
    pub fn confirm(&mut self, confirm: LayoutConfirm) -> Result<ActiveLayout, HandshakeError> {
        let index = self
            .pending
            .iter()
            .position(|layout| layout.layout_id == confirm.layout_id)
            .ok_or(HandshakeError::UnknownLayout(confirm.layout_id))?;
        if self.pending[index].hash != confirm.layout_hash {
            return Err(HandshakeError::ConfirmationMismatch);
        }
        Ok(self.pending.remove(index))
    }

    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactSample {
    pub value: Option<f32>,
    pub confidence: f32,
    pub state: SignalState,
}

impl CompactSample {
    #[must_use]
    pub const fn available(value: f32, confidence: f32, state: SignalState) -> Self {
        Self {
            value: Some(value),
            confidence,
            state,
        }
    }

    #[must_use]
    pub const fn unavailable(confidence: f32, state: SignalState) -> Self {
        Self {
            value: None,
            confidence,
            state,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompactFrameInput<'a> {
    pub session_id: SessionId,
    pub generation: u32,
    pub sequence: u64,
    pub capture_timestamp_ns: u64,
    pub produced_timestamp_ns: u64,
    pub samples: &'a [CompactSample],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompactFrameError {
    WrongLength { expected: usize, actual: usize },
    InvalidMagic,
    IncompatibleVersion,
    NonZeroReserved,
    WrongQualityEncoding,
    WrongSession,
    WrongGeneration { expected: u32, actual: u32 },
    WrongLayout { expected: u32, actual: u32 },
    InvalidTimestamp,
    InvalidState(u8),
    InvalidStateValue { signal: SignalId },
    UnsupportedState { signal: SignalId },
    InvalidConfidence { signal: SignalId },
    InvalidValue { signal: SignalId },
}

impl fmt::Display for CompactFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength { expected, actual } => {
                write!(formatter, "expected {expected} frame bytes, got {actual}")
            }
            Self::InvalidMagic => formatter.write_str("invalid compact-frame magic"),
            Self::IncompatibleVersion => formatter.write_str("incompatible compact-frame version"),
            Self::NonZeroReserved => {
                formatter.write_str("compact-frame reserved bytes are non-zero")
            }
            Self::WrongQualityEncoding => {
                formatter.write_str("frame quality encoding differs from layout")
            }
            Self::WrongSession => formatter.write_str("compact frame belongs to another session"),
            Self::WrongGeneration { expected, actual } => {
                write!(formatter, "expected generation {expected}, got {actual}")
            }
            Self::WrongLayout { expected, actual } => {
                write!(formatter, "expected layout {expected}, got {actual}")
            }
            Self::InvalidTimestamp => formatter.write_str("produced timestamp precedes capture"),
            Self::InvalidState(value) => write!(formatter, "invalid signal state {value}"),
            Self::InvalidStateValue { signal } => {
                write!(
                    formatter,
                    "state/value mismatch for Signal ID {}",
                    signal.get()
                )
            }
            Self::UnsupportedState { signal } => {
                write!(
                    formatter,
                    "active Signal ID {} cannot be unsupported",
                    signal.get()
                )
            }
            Self::InvalidConfidence { signal } => {
                write!(
                    formatter,
                    "invalid confidence for Signal ID {}",
                    signal.get()
                )
            }
            Self::InvalidValue { signal } => {
                write!(formatter, "invalid value for Signal ID {}", signal.get())
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CompactFrameError {}

pub struct CompactFrameCodec;

impl CompactFrameCodec {
    /// Encode into an exactly sized caller-owned buffer; no allocation occurs in this method.
    ///
    /// # Errors
    ///
    /// Rejects size, timestamp, state/value, confidence, and signal-range violations.
    pub fn encode_into(
        layout: &ActiveLayout,
        input: &CompactFrameInput<'_>,
        output: &mut [u8],
    ) -> Result<(), CompactFrameError> {
        if output.len() != layout.frame_len {
            return Err(CompactFrameError::WrongLength {
                expected: layout.frame_len,
                actual: output.len(),
            });
        }
        if input.samples.len() != layout.signals.len() {
            return Err(CompactFrameError::WrongLength {
                expected: layout.signals.len(),
                actual: input.samples.len(),
            });
        }
        if input.produced_timestamp_ns < input.capture_timestamp_ns {
            return Err(CompactFrameError::InvalidTimestamp);
        }

        output[0..4].copy_from_slice(&COMPACT_MAGIC);
        output[4] = COMPACT_WIRE_VERSION;
        output[5] = layout.proposal.quality_encoding as u8;
        output[6..8].fill(0);
        output[8..24].copy_from_slice(&input.session_id.0);
        write_u32(output, 24, input.generation);
        write_u32(output, 28, layout.layout_id);
        write_u64(output, 32, input.sequence);
        write_u64(output, 40, input.capture_timestamp_ns);
        write_u64(output, 48, input.produced_timestamp_ns);

        let quality_offset = COMPACT_HEADER_LEN + layout.signals.len() * 2;
        for (index, (&signal, sample)) in layout.signals.iter().zip(input.samples).enumerate() {
            let (quantized, confidence) =
                encode_sample(signal, *sample, layout.proposal.quality_encoding)?;
            let value_offset = COMPACT_HEADER_LEN + index * 2;
            output[value_offset..value_offset + 2].copy_from_slice(&quantized.to_le_bytes());
            if layout.proposal.quality_encoding == QualityEncoding::StateAndConfidenceU8 {
                let offset = quality_offset + index * 2;
                output[offset] = sample.state as u8;
                output[offset + 1] = confidence;
            }
        }
        Ok(())
    }

    /// Convenience allocation for cold paths and tests. Reuse a `frame_len()` buffer in live use.
    ///
    /// # Errors
    ///
    /// Returns the same validation failures as [`Self::encode_into`].
    pub fn encode(
        layout: &ActiveLayout,
        input: &CompactFrameInput<'_>,
    ) -> Result<Vec<u8>, CompactFrameError> {
        let mut output = vec![0; layout.frame_len];
        Self::encode_into(layout, input, &mut output)?;
        Ok(output)
    }

    /// Decode and structurally validate without allocating or copying the frame payload.
    ///
    /// # Errors
    ///
    /// Rejects frames with the wrong exact size, header, layout, timestamp order, signal state,
    /// or normalized value representation.
    pub fn decode<'a>(
        layout: &'a ActiveLayout,
        bytes: &'a [u8],
    ) -> Result<CompactFrameRef<'a>, CompactFrameError> {
        if bytes.len() != layout.frame_len {
            return Err(CompactFrameError::WrongLength {
                expected: layout.frame_len,
                actual: bytes.len(),
            });
        }
        if bytes[0..4] != COMPACT_MAGIC {
            return Err(CompactFrameError::InvalidMagic);
        }
        if bytes[4] != COMPACT_WIRE_VERSION {
            return Err(CompactFrameError::IncompatibleVersion);
        }
        if bytes[5] != layout.proposal.quality_encoding as u8 {
            return Err(CompactFrameError::WrongQualityEncoding);
        }
        if bytes[6] != 0 || bytes[7] != 0 {
            return Err(CompactFrameError::NonZeroReserved);
        }
        let session_id =
            SessionId(
                bytes[8..24]
                    .try_into()
                    .map_err(|_| CompactFrameError::WrongLength {
                        expected: layout.frame_len,
                        actual: bytes.len(),
                    })?,
            );
        let generation = read_u32(bytes, 24)?;
        let layout_id = read_u32(bytes, 28)?;
        let sequence = read_u64(bytes, 32)?;
        let capture_timestamp_ns = read_u64(bytes, 40)?;
        let produced_timestamp_ns = read_u64(bytes, 48)?;
        if layout_id != layout.layout_id {
            return Err(CompactFrameError::WrongLayout {
                expected: layout.layout_id,
                actual: layout_id,
            });
        }
        if produced_timestamp_ns < capture_timestamp_ns {
            return Err(CompactFrameError::InvalidTimestamp);
        }

        let frame = CompactFrameRef {
            session_id,
            generation,
            layout_id,
            sequence,
            capture_timestamp_ns,
            produced_timestamp_ns,
            layout,
            bytes,
        };
        for index in 0..layout.signals.len() {
            frame.validate_wire_sample(index)?;
        }
        Ok(frame)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodedCompactSample {
    pub signal: SignalId,
    pub value: Option<f32>,
    pub confidence: f32,
    pub state: SignalState,
}

#[derive(Clone, Copy, Debug)]
pub struct CompactFrameRef<'a> {
    pub session_id: SessionId,
    pub generation: u32,
    pub layout_id: u32,
    pub sequence: u64,
    pub capture_timestamp_ns: u64,
    pub produced_timestamp_ns: u64,
    layout: &'a ActiveLayout,
    bytes: &'a [u8],
}

impl CompactFrameRef<'_> {
    #[must_use]
    pub const fn bytes(&self) -> &[u8] {
        self.bytes
    }

    #[must_use]
    pub fn sample(&self, index: usize) -> Option<Result<DecodedCompactSample, CompactFrameError>> {
        let signal = *self.layout.signals.get(index)?;
        let value_offset = COMPACT_HEADER_LEN + index * 2;
        let raw = i16::from_le_bytes(
            self.bytes
                .get(value_offset..value_offset + 2)?
                .try_into()
                .ok()?,
        );
        let (state, confidence) = match self.layout.proposal.quality_encoding {
            QualityEncoding::None => (SignalState::Observed, 1.0),
            QualityEncoding::StateAndConfidenceU8 => {
                let quality_offset = COMPACT_HEADER_LEN + self.layout.signals.len() * 2 + index * 2;
                let state = match decode_state(*self.bytes.get(quality_offset)?) {
                    Ok(state) => state,
                    Err(error) => return Some(Err(error)),
                };
                let confidence = f32::from(*self.bytes.get(quality_offset + 1)?) / 255.0;
                (state, confidence)
            }
        };
        if state == SignalState::Unsupported {
            return Some(Err(CompactFrameError::UnsupportedState { signal }));
        }
        let value = if raw == MISSING_VALUE {
            None
        } else {
            match dequantize(signal, raw) {
                Ok(value) => Some(value),
                Err(error) => return Some(Err(error)),
            }
        };
        if state.carries_value() != value.is_some() {
            return Some(Err(CompactFrameError::InvalidStateValue { signal }));
        }
        Some(Ok(DecodedCompactSample {
            signal,
            value,
            confidence,
            state,
        }))
    }

    fn validate_wire_sample(&self, index: usize) -> Result<(), CompactFrameError> {
        let signal = *self
            .layout
            .signals
            .get(index)
            .ok_or(CompactFrameError::WrongLength {
                expected: self.layout.signals.len(),
                actual: index,
            })?;
        let value_offset = COMPACT_HEADER_LEN + index * 2;
        let raw = i16::from_le_bytes(
            self.bytes[value_offset..value_offset + 2]
                .try_into()
                .map_err(|_| CompactFrameError::WrongLength {
                    expected: self.layout.frame_len,
                    actual: self.bytes.len(),
                })?,
        );
        let state = match self.layout.proposal.quality_encoding {
            QualityEncoding::None => SignalState::Observed,
            QualityEncoding::StateAndConfidenceU8 => {
                let quality_offset = COMPACT_HEADER_LEN + self.layout.signals.len() * 2 + index * 2;
                decode_state(self.bytes[quality_offset])?
            }
        };
        if state == SignalState::Unsupported {
            return Err(CompactFrameError::UnsupportedState { signal });
        }
        if state.carries_value() != (raw != MISSING_VALUE) {
            return Err(CompactFrameError::InvalidStateValue { signal });
        }
        if raw != MISSING_VALUE {
            let wire_rule =
                self.layout
                    .wire_rules
                    .get(index)
                    .ok_or(CompactFrameError::WrongLength {
                        expected: self.layout.signals.len(),
                        actual: index,
                    })?;
            if wire_rule.rejects_quantized_max && raw == QUANTIZED_MAX {
                return Err(CompactFrameError::InvalidValue { signal });
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn samples(
        &self,
    ) -> impl ExactSizeIterator<Item = Result<DecodedCompactSample, CompactFrameError>> + '_ {
        (0..self.layout.signals.len()).map(|index| {
            self.sample(index)
                .unwrap_or(Err(CompactFrameError::WrongLength {
                    expected: self.layout.signals.len(),
                    actual: index,
                }))
        })
    }
}

fn encode_sample(
    signal: SignalId,
    sample: CompactSample,
    quality: QualityEncoding,
) -> Result<(i16, u8), CompactFrameError> {
    if sample.state == SignalState::Unsupported {
        return Err(CompactFrameError::UnsupportedState { signal });
    }
    if sample.state.carries_value() != sample.value.is_some()
        || (quality == QualityEncoding::None && sample.value.is_none())
        || (quality == QualityEncoding::None && sample.state != SignalState::Observed)
    {
        return Err(CompactFrameError::InvalidStateValue { signal });
    }
    if quality == QualityEncoding::None && sample.confidence.to_bits() != 1.0_f32.to_bits() {
        return Err(CompactFrameError::InvalidConfidence { signal });
    }
    let confidence = quantize_confidence(signal, sample.confidence)?;
    let value = match sample.value {
        Some(value) => quantize(signal, value)?,
        None => MISSING_VALUE,
    };
    Ok((value, confidence))
}

fn quantize(signal: SignalId, value: f32) -> Result<i16, CompactFrameError> {
    let metadata = SignalMetadata::get(signal).ok_or(CompactFrameError::InvalidValue { signal })?;
    if !metadata.scalar_type.contains(value) {
        return Err(CompactFrameError::InvalidValue { signal });
    }
    let (minimum, maximum) = metadata.scalar_type.valid_range();
    let normalized = (value - minimum) / (maximum - minimum);
    let scaled = normalized * 65_534.0 - 32_767.0;
    #[allow(clippy::cast_possible_truncation)]
    let quantized = if scaled >= 0.0 {
        (scaled + 0.5) as i16
    } else {
        (scaled - 0.5) as i16
    };
    Ok(if metadata.scalar_type == crate::ScalarType::Angle {
        quantized.min(QUANTIZED_MAX - 1)
    } else {
        quantized
    })
}

fn dequantize(signal: SignalId, raw: i16) -> Result<f32, CompactFrameError> {
    if !(QUANTIZED_MIN..=QUANTIZED_MAX).contains(&raw) {
        return Err(CompactFrameError::InvalidValue { signal });
    }
    let metadata = SignalMetadata::get(signal).ok_or(CompactFrameError::InvalidValue { signal })?;
    let (minimum, maximum) = metadata.scalar_type.valid_range();
    #[allow(clippy::cast_precision_loss)]
    let normalized = ((i32::from(raw) - i32::from(QUANTIZED_MIN)) as f32) / 65_534.0;
    let value = normalized * (maximum - minimum) + minimum;
    if metadata.scalar_type.contains(value) {
        Ok(value)
    } else {
        Err(CompactFrameError::InvalidValue { signal })
    }
}

fn quantize_confidence(signal: SignalId, value: f32) -> Result<u8, CompactFrameError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(CompactFrameError::InvalidConfidence { signal });
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok((value * 255.0 + 0.5) as u8)
}

fn decode_state(value: u8) -> Result<SignalState, CompactFrameError> {
    match value {
        0 => Ok(SignalState::Observed),
        1 => Ok(SignalState::Fused),
        2 => Ok(SignalState::Predicted),
        3 => Ok(SignalState::Occluded),
        4 => Ok(SignalState::OutOfFrame),
        5 => Ok(SignalState::TrackingLost),
        6 => Ok(SignalState::Unsupported),
        _ => Err(CompactFrameError::InvalidState(value)),
    }
}

fn write_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CompactFrameError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .and_then(|value| value.try_into().ok())
            .ok_or(CompactFrameError::WrongLength {
                expected: offset + 4,
                actual: bytes.len(),
            })?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, CompactFrameError> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .and_then(|value| value.try_into().ok())
            .ok_or(CompactFrameError::WrongLength {
                expected: offset + 8,
                actual: bytes.len(),
            })?,
    ))
}

/// A transport-provided estimate of current time in the producer's monotonic clock domain.
///
/// Receiver-local monotonic timestamps must never be compared directly with compact frame
/// timestamps. A connection adapter obtains this estimate from its bounded clock synchronization
/// and carries the remaining uncertainty explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerClockEstimate {
    now_ns: u64,
    uncertainty_ns: u64,
}

impl ProducerClockEstimate {
    #[must_use]
    pub const fn synchronized(now_ns: u64, uncertainty_ns: u64) -> Self {
        Self {
            now_ns,
            uncertainty_ns,
        }
    }

    #[must_use]
    pub const fn now_ns(self) -> u64 {
        self.now_ns
    }

    #[must_use]
    pub const fn uncertainty_ns(self) -> u64 {
        self.uncertainty_ns
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactStreamPolicy {
    pub max_frame_age_ns: u64,
    pub max_future_skew_ns: u64,
    pub max_clock_uncertainty_ns: u64,
    pub max_sequence_gap: u64,
    pub max_capture_jump_ns: u64,
}

impl Default for CompactStreamPolicy {
    fn default() -> Self {
        Self {
            max_frame_age_ns: u64::MAX,
            max_future_skew_ns: 0,
            max_clock_uncertainty_ns: 0,
            max_sequence_gap: u64::MAX,
            max_capture_jump_ns: u64::MAX,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompactStreamError {
    ConfirmationMismatch,
    GenerationDidNotAdvance,
    Frame(CompactFrameError),
    DuplicateOrReplay { last: u64, actual: u64 },
    SequenceGapTooLarge,
    ClockUncertaintyExceeded { maximum: u64, actual: u64 },
    StaleFrame,
    FutureFrame,
    CaptureTimestampRegressed,
    CaptureTimestampJump,
}

impl fmt::Display for CompactStreamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfirmationMismatch => formatter.write_str("layout confirmation does not match"),
            Self::GenerationDidNotAdvance => {
                formatter.write_str("layout switch must advance generation")
            }
            Self::Frame(error) => write!(formatter, "invalid compact frame: {error}"),
            Self::DuplicateOrReplay { last, actual } => {
                write!(formatter, "sequence {actual} is not newer than {last}")
            }
            Self::SequenceGapTooLarge => formatter.write_str("sequence gap exceeds policy"),
            Self::ClockUncertaintyExceeded { maximum, actual } => write!(
                formatter,
                "producer clock uncertainty {actual} ns exceeds policy maximum {maximum} ns"
            ),
            Self::StaleFrame => formatter.write_str("compact frame is stale"),
            Self::FutureFrame => formatter.write_str("compact frame timestamp is in the future"),
            Self::CaptureTimestampRegressed => {
                formatter.write_str("compact frame capture timestamp regressed")
            }
            Self::CaptureTimestampJump => {
                formatter.write_str("compact frame capture timestamp jumped beyond policy")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CompactStreamError {}

#[derive(Clone, Debug)]
pub struct CompactStreamGuard {
    session_id: SessionId,
    generation: u32,
    layout: ActiveLayout,
    policy: CompactStreamPolicy,
    last_sequence: Option<u64>,
    last_capture_timestamp_ns: Option<u64>,
}

impl CompactStreamGuard {
    /// Create a streaming guard only after the remote peer confirms the canonical layout hash.
    ///
    /// # Errors
    ///
    /// Returns [`CompactStreamError::ConfirmationMismatch`] for an unconfirmed layout.
    pub fn confirmed(
        session_id: SessionId,
        generation: u32,
        layout: ActiveLayout,
        remote: LayoutConfirm,
        policy: CompactStreamPolicy,
    ) -> Result<Self, CompactStreamError> {
        if remote != layout.confirmation() {
            return Err(CompactStreamError::ConfirmationMismatch);
        }
        Ok(Self {
            session_id,
            generation,
            layout,
            policy,
            last_sequence: None,
            last_capture_timestamp_ns: None,
        })
    }

    /// Install a confirmed layout in a strictly newer generation.
    ///
    /// # Errors
    ///
    /// Rejects generation rollback/reuse and mismatched layout confirmations.
    pub fn switch_layout(
        &mut self,
        generation: u32,
        layout: ActiveLayout,
        remote: LayoutConfirm,
    ) -> Result<(), CompactStreamError> {
        if generation <= self.generation {
            return Err(CompactStreamError::GenerationDidNotAdvance);
        }
        if remote != layout.confirmation() {
            return Err(CompactStreamError::ConfirmationMismatch);
        }
        self.generation = generation;
        self.layout = layout;
        self.last_sequence = None;
        self.last_capture_timestamp_ns = None;
        Ok(())
    }

    #[must_use]
    pub const fn layout(&self) -> &ActiveLayout {
        &self.layout
    }

    /// Decode and accept one current, non-replayed frame under the configured time limits.
    /// `clock` must be synchronized into the producer timestamp domain by the transport adapter;
    /// its bounded uncertainty is included in age/skew limits rather than comparing unrelated
    /// receiver and producer monotonic epochs.
    ///
    /// # Errors
    ///
    /// Rejects malformed frames, session/layout/generation mismatches, replay, excessive gaps,
    /// and stale or future timestamps without advancing stream state.
    pub fn accept<'a>(
        &'a mut self,
        bytes: &'a [u8],
        clock: ProducerClockEstimate,
    ) -> Result<CompactFrameRef<'a>, CompactStreamError> {
        let frame =
            CompactFrameCodec::decode(&self.layout, bytes).map_err(CompactStreamError::Frame)?;
        if frame.session_id != self.session_id {
            return Err(CompactStreamError::Frame(CompactFrameError::WrongSession));
        }
        if frame.generation != self.generation {
            return Err(CompactStreamError::Frame(
                CompactFrameError::WrongGeneration {
                    expected: self.generation,
                    actual: frame.generation,
                },
            ));
        }
        if clock.uncertainty_ns > self.policy.max_clock_uncertainty_ns {
            return Err(CompactStreamError::ClockUncertaintyExceeded {
                maximum: self.policy.max_clock_uncertainty_ns,
                actual: clock.uncertainty_ns,
            });
        }
        let future_limit = clock
            .now_ns
            .saturating_add(self.policy.max_future_skew_ns)
            .saturating_add(clock.uncertainty_ns);
        if frame.capture_timestamp_ns > future_limit || frame.produced_timestamp_ns > future_limit {
            return Err(CompactStreamError::FutureFrame);
        }
        let age_limit = self
            .policy
            .max_frame_age_ns
            .saturating_add(clock.uncertainty_ns);
        if clock.now_ns.saturating_sub(frame.capture_timestamp_ns) > age_limit {
            return Err(CompactStreamError::StaleFrame);
        }
        if let Some(last) = self.last_sequence {
            if frame.sequence <= last {
                return Err(CompactStreamError::DuplicateOrReplay {
                    last,
                    actual: frame.sequence,
                });
            }
            if frame.sequence.saturating_sub(last).saturating_sub(1) > self.policy.max_sequence_gap
            {
                return Err(CompactStreamError::SequenceGapTooLarge);
            }
        }
        if let Some(last) = self.last_capture_timestamp_ns {
            if frame.capture_timestamp_ns < last {
                return Err(CompactStreamError::CaptureTimestampRegressed);
            }
            if frame.capture_timestamp_ns.saturating_sub(last) > self.policy.max_capture_jump_ns {
                return Err(CompactStreamError::CaptureTimestampJump);
            }
        }
        self.last_sequence = Some(frame.sequence);
        self.last_capture_timestamp_ns = Some(frame.capture_timestamp_ns);
        Ok(frame)
    }
}

/// A bounded latest-frame handoff. Replacing an unread value is explicit and measurable.
#[derive(Clone, Debug, Default)]
pub struct LatestFrame<T> {
    pending: Option<T>,
    dropped: u64,
}

impl<T> LatestFrame<T> {
    pub fn publish(&mut self, frame: T) {
        if self.pending.replace(frame).is_some() {
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    pub fn take(&mut self) -> Option<T> {
        self.pending.take()
    }

    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// Transport-neutral recording sink. A recording must persist the layout header before its frames.
pub trait CompactRecordingSink {
    type Error;

    /// Persist a layout record before any frame that references its generation and layout ID.
    ///
    /// # Errors
    ///
    /// Returns a sink-specific persistence error.
    fn record_layout(&mut self, layout: &LayoutRecord) -> Result<(), Self::Error>;

    /// Persist one already validated compact frame.
    ///
    /// # Errors
    ///
    /// Returns a sink-specific persistence error.
    fn record_frame(&mut self, compact_frame: &[u8]) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecordingError<E> {
    NoActiveLayout,
    WrongSession,
    WrongGeneration,
    Frame(CompactFrameError),
    Sink(E),
}

/// Enforces `LayoutRecord` before frame ordering for a transport-independent recording sink.
pub struct CompactRecorder<S> {
    sink: S,
    active: Option<(SessionId, u32, ActiveLayout)>,
}

impl<S: CompactRecordingSink> CompactRecorder<S> {
    #[must_use]
    pub const fn new(sink: S) -> Self {
        Self { sink, active: None }
    }

    /// Persist and activate a confirmed layout record for subsequent frames.
    ///
    /// # Errors
    ///
    /// Returns the sink error without changing the active recording layout.
    pub fn begin_layout(
        &mut self,
        session_id: SessionId,
        generation: u32,
        layout: ActiveLayout,
    ) -> Result<(), RecordingError<S::Error>> {
        let record = layout.recording_header(session_id, generation);
        self.sink
            .record_layout(&record)
            .map_err(RecordingError::Sink)?;
        self.active = Some((session_id, generation, layout));
        Ok(())
    }

    /// Validate and persist a frame only after its referenced layout record exists.
    ///
    /// # Errors
    ///
    /// Rejects missing layout records, malformed frames, session/generation mismatches, and
    /// sink-specific persistence failures.
    pub fn record_frame(&mut self, compact_frame: &[u8]) -> Result<(), RecordingError<S::Error>> {
        let (session_id, generation, layout) =
            self.active.as_ref().ok_or(RecordingError::NoActiveLayout)?;
        let frame =
            CompactFrameCodec::decode(layout, compact_frame).map_err(RecordingError::Frame)?;
        if frame.session_id != *session_id {
            return Err(RecordingError::WrongSession);
        }
        if frame.generation != *generation {
            return Err(RecordingError::WrongGeneration);
        }
        self.sink
            .record_frame(compact_frame)
            .map_err(RecordingError::Sink)
    }

    #[must_use]
    pub fn into_inner(self) -> S {
        self.sink
    }
}

/// Minimal transport surface for a reliable control channel plus compact-frame data channel.
/// Implementations own encryption, authentication, discovery, flow control, and reconnect policy.
pub trait CompactSessionTransport {
    type Error;

    /// Send a validated layout acceptance on the reliable control channel.
    ///
    /// # Errors
    ///
    /// Returns a transport-specific send error.
    fn send_layout_accept(&mut self, accept: LayoutAccept) -> Result<(), Self::Error>;

    /// Send the peer's layout confirmation on the reliable control channel.
    ///
    /// # Errors
    ///
    /// Returns a transport-specific send error.
    fn send_layout_confirm(&mut self, confirm: LayoutConfirm) -> Result<(), Self::Error>;

    /// Send one exact-size compact frame on the adapter's latest-frame-first data path.
    ///
    /// # Errors
    ///
    /// Returns a transport-specific send error.
    fn send_compact_frame(&mut self, compact_frame: &[u8]) -> Result<(), Self::Error>;
}
