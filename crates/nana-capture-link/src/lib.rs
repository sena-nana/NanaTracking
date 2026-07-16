//! Authenticated, runtime-neutral `MutsukiLink` adapter for `NanaTracking` capture.
//!
//! This crate owns transport envelopes and resource limits only. Capture schemas,
//! NTP label mappings, local durability, and dataset freezing remain in their
//! authoritative `NanaTracking` layers.

#![forbid(unsafe_code)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::too_many_lines
)]

use mutsuki_link_core::{
    AuthenticatedSession, Connection, PeerId, ProtocolId, RealtimeDatagram, RealtimeFlowId,
    RealtimePriority, SendOutcome, SessionId as LinkSessionId, TransportError, TransportErrorKind,
};
use mutsuki_link_pairing::{KeyState, LinkPermission, TrustRecord};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

pub const CONTROL_PROTOCOL: &str = "nana.capture.control.v1";
pub const PREVIEW_NAMESPACE: &str = "nana.capture.preview.v1";
pub const SYNC_NAMESPACE: &str = "nana.capture.sync.v1";
pub const PREVIEW_FLOW: RealtimeFlowId = RealtimeFlowId(0xca16);

const CONTROL_MAGIC: [u8; 4] = *b"NCCT";
const DATA_MAGIC: [u8; 4] = *b"NCDA";
const WIRE_VERSION: u8 = 1;
const CONTROL_HEADER_BYTES: usize = 26;
const DATA_HEADER_BYTES: usize = 26;
const PREVIEW_HEADER_BYTES: usize = 20;
const SYNC_SEGMENT_HEADER_BYTES: usize = 68;
const SYNC_ACK_BYTES: usize = 56;
const SYNC_RANGE_BYTES: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureRole {
    Device,
    Studio,
}

impl CaptureRole {
    const fn code(self) -> u8 {
        match self {
            Self::Device => 1,
            Self::Studio => 2,
        }
    }

    fn from_code(code: u8) -> Result<Self, CaptureLinkError> {
        match code {
            1 => Ok(Self::Device),
            2 => Ok(Self::Studio),
            _ => Err(CaptureLinkError::InvalidEnvelope),
        }
    }
}

/// One-use capability. It is intentionally neither `Clone` nor `Copy`.
#[derive(Eq, PartialEq)]
pub struct CaptureAuthorization {
    peer_id: PeerId,
    link_session_id: LinkSessionId,
    role: CaptureRole,
    trust_revision: u64,
    datagram_allowed: bool,
}

impl fmt::Debug for CaptureAuthorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CaptureAuthorization {{ peer_id: {}, link_session_id: {}, role: {:?}, trust_revision: {}, datagram_allowed: {} }}",
            self.peer_id,
            self.link_session_id,
            self.role,
            self.trust_revision,
            self.datagram_allowed
        )
    }
}

impl CaptureAuthorization {
    #[must_use]
    pub const fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    #[must_use]
    pub const fn link_session_id(&self) -> LinkSessionId {
        self.link_session_id
    }

    #[must_use]
    pub const fn role(&self) -> CaptureRole {
        self.role
    }

    #[must_use]
    pub const fn trust_revision(&self) -> u64 {
        self.trust_revision
    }

    #[must_use]
    pub const fn datagram_allowed(&self) -> bool {
        self.datagram_allowed
    }
}

pub fn authorize_capture_session(
    session: AuthenticatedSession<'_>,
    role: CaptureRole,
    record: &TrustRecord,
    trust_revision: u64,
) -> Result<CaptureAuthorization, CaptureLinkError> {
    match &record.key_state {
        KeyState::Active => {}
        KeyState::Revoked { .. } => return Err(CaptureLinkError::PeerRevoked),
        KeyState::Rotated { .. } => return Err(CaptureLinkError::PeerRotated),
    }
    if record.peer_id != session.info().peer_id
        || record.public_key_fingerprint() != session.security().identity.public_key_fingerprint
    {
        return Err(CaptureLinkError::Unauthorized);
    }
    for namespace in [CONTROL_PROTOCOL, PREVIEW_NAMESPACE, SYNC_NAMESPACE] {
        if !record
            .permissions
            .contains(&LinkPermission::OpenNamespace(namespace.to_owned()))
        {
            return Err(CaptureLinkError::Unauthorized);
        }
    }
    Ok(CaptureAuthorization {
        peer_id: session.info().peer_id,
        link_session_id: session.info().session_id,
        role,
        trust_revision,
        datagram_allowed: record.permissions.contains(&LinkPermission::Datagram),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreviewTransportMode {
    Datagram,
    ReliableLatestOnly,
}

impl PreviewTransportMode {
    const fn code(self) -> u8 {
        match self {
            Self::Datagram => 1,
            Self::ReliableLatestOnly => 2,
        }
    }

    fn from_code(code: u8) -> Result<Self, CaptureLinkError> {
        match code {
            1 => Ok(Self::Datagram),
            2 => Ok(Self::ReliableLatestOnly),
            _ => Err(CaptureLinkError::InvalidEnvelope),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CaptureLinkConfig {
    pub max_control_bytes: usize,
    pub max_preview_bytes: usize,
    pub max_sync_segment_bytes: usize,
    pub max_pending_control: usize,
    pub max_pending_sync: usize,
    pub preview_deadline: Duration,
}

impl Default for CaptureLinkConfig {
    fn default() -> Self {
        Self {
            max_control_bytes: 64 * 1024,
            max_preview_bytes: 1024 * 1024,
            max_sync_segment_bytes: 1024 * 1024,
            max_pending_control: 16,
            max_pending_sync: 3,
            preview_deadline: Duration::from_millis(100),
        }
    }
}

impl CaptureLinkConfig {
    pub fn validate(self) -> Result<Self, CaptureLinkError> {
        if self.max_control_bytes == 0
            || self.max_preview_bytes == 0
            || self.max_sync_segment_bytes == 0
            || self.max_pending_control == 0
            || self.max_pending_sync == 0
            || self.preview_deadline.is_zero()
            || self.max_control_bytes > u32::MAX as usize
            || self.max_preview_bytes > u32::MAX as usize
            || self.max_sync_segment_bytes > u32::MAX as usize
        {
            return Err(CaptureLinkError::InvalidConfig);
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaptureLinkError {
    Transport(TransportError),
    InvalidConfig,
    Unauthorized,
    PeerRevoked,
    PeerRotated,
    SessionBindingMismatch,
    InvalidState,
    InvalidEnvelope,
    PayloadLimit,
    QueueFull,
    DigestMismatch,
    ReplayOrDuplicate,
}

impl fmt::Display for CaptureLinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(error) => write!(formatter, "capture transport error: {error}"),
            Self::InvalidConfig => formatter.write_str("capture Link configuration is invalid"),
            Self::Unauthorized => formatter.write_str("capture Link permission is not granted"),
            Self::PeerRevoked => formatter.write_str("capture Link peer trust is revoked"),
            Self::PeerRotated => formatter.write_str("capture Link peer trust was rotated"),
            Self::SessionBindingMismatch => {
                formatter.write_str("capture Link session binding does not match")
            }
            Self::InvalidState => {
                formatter.write_str("capture Link state disallows this operation")
            }
            Self::InvalidEnvelope => formatter.write_str("capture Link envelope is invalid"),
            Self::PayloadLimit => formatter.write_str("capture Link payload limit is exceeded"),
            Self::QueueFull => formatter.write_str("capture Link bounded queue is full"),
            Self::DigestMismatch => {
                formatter.write_str("capture sync segment digest does not match")
            }
            Self::ReplayOrDuplicate => {
                formatter.write_str("capture preview is replayed or duplicated")
            }
        }
    }
}

impl std::error::Error for CaptureLinkError {}

impl From<TransportError> for CaptureLinkError {
    fn from(value: TransportError) -> Self {
        Self::Transport(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewFrame {
    pub generation: u32,
    pub sequence: u64,
    pub capture_timestamp_ns: u64,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyncSegment {
    pub transfer_id: [u8; 16],
    pub offset: u64,
    pub total_bytes: u64,
    pub payload_sha256: [u8; 32],
    pub payload: Vec<u8>,
}

impl SyncSegment {
    pub fn new(
        transfer_id: [u8; 16],
        offset: u64,
        total_bytes: u64,
        payload: Vec<u8>,
    ) -> Result<Self, CaptureLinkError> {
        let payload_bytes =
            u64::try_from(payload.len()).map_err(|_| CaptureLinkError::PayloadLimit)?;
        if payload.is_empty()
            || offset >= total_bytes
            || offset.saturating_add(payload_bytes) > total_bytes
        {
            return Err(CaptureLinkError::InvalidEnvelope);
        }
        let payload_sha256 = Sha256::digest(&payload).into();
        Ok(Self {
            transfer_id,
            offset,
            total_bytes,
            payload_sha256,
            payload,
        })
    }

    fn validate(&self) -> Result<(), CaptureLinkError> {
        let payload_bytes =
            u64::try_from(self.payload.len()).map_err(|_| CaptureLinkError::PayloadLimit)?;
        if self.payload.is_empty()
            || self.offset >= self.total_bytes
            || self.offset.saturating_add(payload_bytes) > self.total_bytes
        {
            return Err(CaptureLinkError::InvalidEnvelope);
        }
        if <[u8; 32]>::from(Sha256::digest(&self.payload)) != self.payload_sha256 {
            return Err(CaptureLinkError::DigestMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SyncAcknowledgement {
    pub transfer_id: [u8; 16],
    pub persisted_through: u64,
    pub full_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MissingRange {
    pub transfer_id: [u8; 16],
    pub start: u64,
    pub end: u64,
}

impl MissingRange {
    fn validate(self) -> Result<Self, CaptureLinkError> {
        if self.start >= self.end {
            return Err(CaptureLinkError::InvalidEnvelope);
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControlEvent {
    PeerReady,
    Application(Vec<u8>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DataEvent {
    Preview(PreviewFrame),
    Segment(SyncSegment),
    Acknowledgement(SyncAcknowledgement),
    MissingRange(MissingRange),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CaptureLinkTelemetry {
    pub pending_control: usize,
    pub pending_sync: usize,
    pub preview_pending: bool,
    pub preview_replacements: u64,
    pub preview_replays_dropped: u64,
    pub sent_sync_messages: u64,
    pub received_sync_messages: u64,
}

struct PendingReliablePreview {
    frame: PreviewFrame,
    wire: Option<Vec<u8>>,
}

pub struct CaptureLink<C: Connection> {
    connection: C,
    authorization: CaptureAuthorization,
    config: CaptureLinkConfig,
    mode: PreviewTransportMode,
    remote_ready: bool,
    control_outbox: VecDeque<Vec<u8>>,
    sync_outbox: VecDeque<Vec<u8>>,
    reliable_preview: Option<PendingReliablePreview>,
    preview_replacements: u64,
    preview_replays_dropped: u64,
    last_preview_sent: Option<(u32, u64)>,
    last_preview_received: Option<(u32, u64)>,
    sent_sync_messages: u64,
    received_sync_messages: u64,
}

impl<C: Connection> CaptureLink<C> {
    pub fn new(
        connection: C,
        authorization: CaptureAuthorization,
        config: CaptureLinkConfig,
    ) -> Result<Self, CaptureLinkError> {
        let config = config.validate()?;
        if !connection.metadata().reliable {
            return Err(CaptureLinkError::InvalidConfig);
        }
        if connection
            .metadata()
            .peer_hint
            .is_some_and(|peer| peer != authorization.peer_id())
        {
            return Err(CaptureLinkError::SessionBindingMismatch);
        }
        let mode =
            if authorization.datagram_allowed() && connection.max_datagram_payload().is_some() {
                PreviewTransportMode::Datagram
            } else {
                PreviewTransportMode::ReliableLatestOnly
            };
        Ok(Self {
            connection,
            authorization,
            config,
            mode,
            remote_ready: false,
            control_outbox: VecDeque::with_capacity(config.max_pending_control),
            sync_outbox: VecDeque::with_capacity(config.max_pending_sync),
            reliable_preview: None,
            preview_replacements: 0,
            preview_replays_dropped: 0,
            last_preview_sent: None,
            last_preview_received: None,
            sent_sync_messages: 0,
            received_sync_messages: 0,
        })
    }

    /// Replaces a disconnected transport with a newly authenticated Link session.
    ///
    /// Reusing the previous session binding is rejected so replay state cannot be
    /// silently reset on the same authenticated session.
    pub fn reconnect(
        &mut self,
        connection: C,
        authorization: CaptureAuthorization,
    ) -> Result<(), CaptureLinkError> {
        if authorization.link_session_id() == self.authorization.link_session_id()
            || authorization.role() != self.authorization.role()
            || authorization.peer_id() != self.authorization.peer_id()
        {
            return Err(CaptureLinkError::SessionBindingMismatch);
        }
        let replacement = Self::new(connection, authorization, self.config)?;
        *self = replacement;
        Ok(())
    }

    #[must_use]
    pub const fn preview_mode(&self) -> PreviewTransportMode {
        self.mode
    }

    #[must_use]
    pub const fn remote_ready(&self) -> bool {
        self.remote_ready
    }

    #[must_use]
    pub fn telemetry(&self) -> CaptureLinkTelemetry {
        CaptureLinkTelemetry {
            pending_control: self.control_outbox.len(),
            pending_sync: self.sync_outbox.len(),
            preview_pending: self.reliable_preview.is_some(),
            preview_replacements: self.preview_replacements,
            preview_replays_dropped: self.preview_replays_dropped,
            sent_sync_messages: self.sent_sync_messages,
            received_sync_messages: self.received_sync_messages,
        }
    }

    pub fn try_send_hello(&mut self) -> Result<(), CaptureLinkError> {
        let hello = [self.authorization.role().code(), self.mode.code()];
        self.queue_control(encode_control(
            self.authorization.link_session_id(),
            1,
            &hello,
            self.config,
        )?)?;
        self.flush_control()
    }

    pub fn try_send_control(&mut self, payload: &[u8]) -> Result<(), CaptureLinkError> {
        if !self.remote_ready {
            return Err(CaptureLinkError::InvalidState);
        }
        self.queue_control(encode_control(
            self.authorization.link_session_id(),
            2,
            payload,
            self.config,
        )?)?;
        self.flush_control()
    }

    pub fn poll_control(&mut self) -> Result<Option<ControlEvent>, CaptureLinkError> {
        self.flush_control()?;
        let bytes = match self
            .connection
            .open_control_stream(protocol())?
            .try_receive()
        {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Ok(None),
            Err(error) if error.kind == TransportErrorKind::WouldBlock => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let (kind, payload) =
            decode_control(&bytes, self.authorization.link_session_id(), self.config)?;
        match kind {
            1 if payload.len() == 2 => {
                let remote_role = CaptureRole::from_code(payload[0])?;
                if remote_role == self.authorization.role() {
                    return Err(CaptureLinkError::InvalidEnvelope);
                }
                let remote_mode = PreviewTransportMode::from_code(payload[1])?;
                if remote_mode != PreviewTransportMode::Datagram {
                    self.mode = PreviewTransportMode::ReliableLatestOnly;
                }
                self.remote_ready = true;
                Ok(Some(ControlEvent::PeerReady))
            }
            2 if self.remote_ready => Ok(Some(ControlEvent::Application(payload.to_vec()))),
            _ => Err(CaptureLinkError::InvalidState),
        }
    }

    pub fn try_send_preview(
        &mut self,
        frame: PreviewFrame,
    ) -> Result<SendOutcome, CaptureLinkError> {
        if !self.remote_ready {
            return Err(CaptureLinkError::InvalidState);
        }
        if frame.payload.is_empty() || frame.payload.len() > self.config.max_preview_bytes {
            return Err(CaptureLinkError::PayloadLimit);
        }
        if self
            .last_preview_sent
            .is_some_and(|(generation, sequence)| {
                frame.generation < generation
                    || (frame.generation == generation && frame.sequence <= sequence)
            })
        {
            return Err(CaptureLinkError::ReplayOrDuplicate);
        }
        let frame_identity = (frame.generation, frame.sequence);
        let outcome = if self.mode == PreviewTransportMode::Datagram {
            let payload =
                encode_preview(self.authorization.link_session_id(), &frame, self.config)?;
            let max_payload = self
                .connection
                .max_datagram_payload()
                .ok_or(CaptureLinkError::PayloadLimit)?;
            if payload.len() > max_payload {
                return Err(CaptureLinkError::PayloadLimit);
            }
            self.connection.try_send_latest(RealtimeDatagram {
                flow: PREVIEW_FLOW,
                generation: frame.generation,
                sequence: frame.sequence,
                deadline: Instant::now() + self.config.preview_deadline,
                priority: RealtimePriority::Disposable,
                payload: &payload,
            })?
        } else {
            let replaced = self
                .reliable_preview
                .replace(PendingReliablePreview { frame, wire: None })
                .is_some();
            if replaced {
                self.preview_replacements = self.preview_replacements.saturating_add(1);
                SendOutcome::ReplacedOlder
            } else {
                self.flush_data()?;
                SendOutcome::Enqueued
            }
        };
        self.last_preview_sent = Some(frame_identity);
        Ok(outcome)
    }

    pub fn try_send_sync_segment(&mut self, segment: &SyncSegment) -> Result<(), CaptureLinkError> {
        segment.validate()?;
        if segment.payload.len() > self.config.max_sync_segment_bytes {
            return Err(CaptureLinkError::PayloadLimit);
        }
        self.queue_sync(encode_sync_segment(
            self.authorization.link_session_id(),
            segment,
            self.config,
        )?)?;
        self.flush_data()
    }

    pub fn try_send_acknowledgement(
        &mut self,
        acknowledgement: SyncAcknowledgement,
    ) -> Result<(), CaptureLinkError> {
        self.queue_sync(encode_sync_ack(
            self.authorization.link_session_id(),
            acknowledgement,
        )?)?;
        self.flush_data()
    }

    pub fn try_send_missing_range(&mut self, range: MissingRange) -> Result<(), CaptureLinkError> {
        self.queue_sync(encode_sync_range(
            self.authorization.link_session_id(),
            range.validate()?,
        )?)?;
        self.flush_data()
    }

    pub fn poll_data(&mut self) -> Result<Option<DataEvent>, CaptureLinkError> {
        self.flush_data()?;
        match self.connection.try_receive() {
            Ok(Some(bytes)) => {
                let event = decode_data(&bytes, self.authorization.link_session_id(), self.config)?;
                return Ok(self.accept_data_event(event));
            }
            Ok(None) => {}
            Err(error) if error.kind == TransportErrorKind::WouldBlock => {}
            Err(error) => return Err(error.into()),
        }
        if self.mode == PreviewTransportMode::Datagram {
            match self.connection.try_receive_realtime() {
                Ok(Some(message)) => {
                    if message.flow != PREVIEW_FLOW {
                        return Err(CaptureLinkError::InvalidEnvelope);
                    }
                    let frame = decode_preview(
                        &message.payload,
                        self.authorization.link_session_id(),
                        self.config,
                    )?;
                    if frame.generation != message.generation || frame.sequence != message.sequence
                    {
                        return Err(CaptureLinkError::InvalidEnvelope);
                    }
                    return Ok(self.accept_preview(frame));
                }
                Ok(None) => {}
                Err(error) if error.kind == TransportErrorKind::WouldBlock => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(None)
    }

    pub fn poll_outbound(&mut self) -> Result<(), CaptureLinkError> {
        self.flush_control()?;
        self.flush_data()
    }

    pub fn into_inner(self) -> C {
        self.connection
    }

    fn accept_preview(&mut self, frame: PreviewFrame) -> Option<DataEvent> {
        if self
            .last_preview_received
            .is_some_and(|(generation, sequence)| {
                frame.generation < generation
                    || (frame.generation == generation && frame.sequence <= sequence)
            })
        {
            self.preview_replays_dropped = self.preview_replays_dropped.saturating_add(1);
            return None;
        }
        self.last_preview_received = Some((frame.generation, frame.sequence));
        Some(DataEvent::Preview(frame))
    }

    fn accept_data_event(&mut self, event: DataEvent) -> Option<DataEvent> {
        if let DataEvent::Preview(frame) = event {
            self.accept_preview(frame)
        } else {
            self.received_sync_messages = self.received_sync_messages.saturating_add(1);
            Some(event)
        }
    }

    fn queue_control(&mut self, message: Vec<u8>) -> Result<(), CaptureLinkError> {
        if self.control_outbox.len() >= self.config.max_pending_control {
            return Err(CaptureLinkError::QueueFull);
        }
        self.control_outbox.push_back(message);
        Ok(())
    }

    fn queue_sync(&mut self, message: Vec<u8>) -> Result<(), CaptureLinkError> {
        if !self.remote_ready {
            return Err(CaptureLinkError::InvalidState);
        }
        if self.sync_outbox.len() >= self.config.max_pending_sync {
            return Err(CaptureLinkError::QueueFull);
        }
        self.sync_outbox.push_back(message);
        Ok(())
    }

    fn flush_control(&mut self) -> Result<(), CaptureLinkError> {
        while let Some(message) = self.control_outbox.front() {
            match self
                .connection
                .open_control_stream(protocol())?
                .try_send(message)
            {
                Ok(()) => {
                    self.control_outbox.pop_front();
                }
                Err(error) if error.kind == TransportErrorKind::WouldBlock => return Ok(()),
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    fn flush_data(&mut self) -> Result<(), CaptureLinkError> {
        if let Some(sync) = self.sync_outbox.front() {
            match self.connection.try_send(sync) {
                Ok(()) => {
                    self.sync_outbox.pop_front();
                    self.sent_sync_messages = self.sent_sync_messages.saturating_add(1);
                    return Ok(());
                }
                Err(error) if error.kind == TransportErrorKind::WouldBlock => return Ok(()),
                Err(error) => return Err(error.into()),
            }
        }
        if let Some(preview) = self.reliable_preview.as_mut() {
            if preview.wire.is_none() {
                preview.wire = Some(encode_preview(
                    self.authorization.link_session_id(),
                    &preview.frame,
                    self.config,
                )?);
            }
            let wire = preview
                .wire
                .as_deref()
                .ok_or(CaptureLinkError::InvalidState)?;
            match self.connection.try_send(wire) {
                Ok(()) => {
                    self.reliable_preview = None;
                    return Ok(());
                }
                Err(error) if error.kind == TransportErrorKind::WouldBlock => return Ok(()),
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

fn protocol() -> ProtocolId {
    ProtocolId::new(CONTROL_PROTOCOL).expect("static capture control protocol is valid")
}

fn encode_control(
    session_id: LinkSessionId,
    kind: u8,
    payload: &[u8],
    config: CaptureLinkConfig,
) -> Result<Vec<u8>, CaptureLinkError> {
    if payload.len() > config.max_control_bytes || payload.len() > u32::MAX as usize {
        return Err(CaptureLinkError::PayloadLimit);
    }
    let mut wire = Vec::with_capacity(CONTROL_HEADER_BYTES + payload.len());
    wire.extend_from_slice(&CONTROL_MAGIC);
    wire.push(WIRE_VERSION);
    wire.push(kind);
    wire.extend_from_slice(session_id.as_bytes());
    let payload_len = u32::try_from(payload.len()).map_err(|_| CaptureLinkError::PayloadLimit)?;
    wire.extend_from_slice(&payload_len.to_be_bytes());
    wire.extend_from_slice(payload);
    Ok(wire)
}

fn decode_control(
    wire: &[u8],
    session_id: LinkSessionId,
    config: CaptureLinkConfig,
) -> Result<(u8, &[u8]), CaptureLinkError> {
    if wire.len() < CONTROL_HEADER_BYTES
        || wire.len() > CONTROL_HEADER_BYTES + config.max_control_bytes
        || wire[..4] != CONTROL_MAGIC
        || wire[4] != WIRE_VERSION
    {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    if wire[6..22] != *session_id.as_bytes() {
        return Err(CaptureLinkError::SessionBindingMismatch);
    }
    let payload_len =
        usize::try_from(read_u32(wire, 22)?).map_err(|_| CaptureLinkError::InvalidEnvelope)?;
    if payload_len != wire.len() - CONTROL_HEADER_BYTES {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    Ok((wire[5], &wire[CONTROL_HEADER_BYTES..]))
}

fn data_prefix(
    session_id: LinkSessionId,
    kind: u8,
    payload_len: usize,
) -> Result<Vec<u8>, CaptureLinkError> {
    let wire_payload_len =
        u32::try_from(payload_len).map_err(|_| CaptureLinkError::PayloadLimit)?;
    let mut wire = Vec::with_capacity(DATA_HEADER_BYTES + payload_len);
    wire.extend_from_slice(&DATA_MAGIC);
    wire.push(WIRE_VERSION);
    wire.push(kind);
    wire.extend_from_slice(session_id.as_bytes());
    wire.extend_from_slice(&wire_payload_len.to_be_bytes());
    Ok(wire)
}

fn encode_preview(
    session_id: LinkSessionId,
    frame: &PreviewFrame,
    config: CaptureLinkConfig,
) -> Result<Vec<u8>, CaptureLinkError> {
    if frame.payload.is_empty() || frame.payload.len() > config.max_preview_bytes {
        return Err(CaptureLinkError::PayloadLimit);
    }
    let payload_len = PREVIEW_HEADER_BYTES + frame.payload.len();
    let mut wire = data_prefix(session_id, 1, payload_len)?;
    wire.extend_from_slice(&frame.generation.to_be_bytes());
    wire.extend_from_slice(&frame.sequence.to_be_bytes());
    wire.extend_from_slice(&frame.capture_timestamp_ns.to_be_bytes());
    wire.extend_from_slice(&frame.payload);
    Ok(wire)
}

fn encode_sync_segment(
    session_id: LinkSessionId,
    segment: &SyncSegment,
    config: CaptureLinkConfig,
) -> Result<Vec<u8>, CaptureLinkError> {
    if segment.payload.len() > config.max_sync_segment_bytes {
        return Err(CaptureLinkError::PayloadLimit);
    }
    let payload_len = SYNC_SEGMENT_HEADER_BYTES + segment.payload.len();
    let mut wire = data_prefix(session_id, 2, payload_len)?;
    wire.extend_from_slice(&segment.transfer_id);
    wire.extend_from_slice(&segment.offset.to_be_bytes());
    wire.extend_from_slice(&segment.total_bytes.to_be_bytes());
    wire.extend_from_slice(&segment.payload_sha256);
    let segment_payload_len =
        u32::try_from(segment.payload.len()).map_err(|_| CaptureLinkError::PayloadLimit)?;
    wire.extend_from_slice(&segment_payload_len.to_be_bytes());
    wire.extend_from_slice(&segment.payload);
    Ok(wire)
}

fn encode_sync_ack(
    session_id: LinkSessionId,
    acknowledgement: SyncAcknowledgement,
) -> Result<Vec<u8>, CaptureLinkError> {
    let mut wire = data_prefix(session_id, 3, SYNC_ACK_BYTES)?;
    wire.extend_from_slice(&acknowledgement.transfer_id);
    wire.extend_from_slice(&acknowledgement.persisted_through.to_be_bytes());
    wire.extend_from_slice(&acknowledgement.full_sha256);
    Ok(wire)
}

fn encode_sync_range(
    session_id: LinkSessionId,
    range: MissingRange,
) -> Result<Vec<u8>, CaptureLinkError> {
    let mut wire = data_prefix(session_id, 4, SYNC_RANGE_BYTES)?;
    wire.extend_from_slice(&range.transfer_id);
    wire.extend_from_slice(&range.start.to_be_bytes());
    wire.extend_from_slice(&range.end.to_be_bytes());
    Ok(wire)
}

fn decode_data(
    wire: &[u8],
    session_id: LinkSessionId,
    config: CaptureLinkConfig,
) -> Result<DataEvent, CaptureLinkError> {
    if wire.len() < DATA_HEADER_BYTES || wire[..4] != DATA_MAGIC || wire[4] != WIRE_VERSION {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    if wire[6..22] != *session_id.as_bytes() {
        return Err(CaptureLinkError::SessionBindingMismatch);
    }
    let payload_len =
        usize::try_from(read_u32(wire, 22)?).map_err(|_| CaptureLinkError::InvalidEnvelope)?;
    if payload_len != wire.len() - DATA_HEADER_BYTES {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    let payload = &wire[DATA_HEADER_BYTES..];
    match wire[5] {
        1 => decode_preview(wire, session_id, config).map(DataEvent::Preview),
        2 => decode_sync_segment(payload, config).map(DataEvent::Segment),
        3 => decode_sync_ack(payload).map(DataEvent::Acknowledgement),
        4 => decode_sync_range(payload).map(DataEvent::MissingRange),
        _ => Err(CaptureLinkError::InvalidEnvelope),
    }
}

fn decode_preview(
    wire: &[u8],
    session_id: LinkSessionId,
    config: CaptureLinkConfig,
) -> Result<PreviewFrame, CaptureLinkError> {
    if wire.len() < DATA_HEADER_BYTES + PREVIEW_HEADER_BYTES
        || wire.len() > DATA_HEADER_BYTES + PREVIEW_HEADER_BYTES + config.max_preview_bytes
        || wire[..4] != DATA_MAGIC
        || wire[4] != WIRE_VERSION
        || wire[5] != 1
        || wire[6..22] != *session_id.as_bytes()
    {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    let payload_len =
        usize::try_from(read_u32(wire, 22)?).map_err(|_| CaptureLinkError::InvalidEnvelope)?;
    if payload_len != wire.len() - DATA_HEADER_BYTES {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    let payload = &wire[DATA_HEADER_BYTES..];
    Ok(PreviewFrame {
        generation: read_u32(payload, 0)?,
        sequence: read_u64(payload, 4)?,
        capture_timestamp_ns: read_u64(payload, 12)?,
        payload: payload[PREVIEW_HEADER_BYTES..].to_vec(),
    })
}

fn decode_sync_segment(
    payload: &[u8],
    config: CaptureLinkConfig,
) -> Result<SyncSegment, CaptureLinkError> {
    if payload.len() < SYNC_SEGMENT_HEADER_BYTES
        || payload.len() > SYNC_SEGMENT_HEADER_BYTES + config.max_sync_segment_bytes
    {
        return Err(CaptureLinkError::PayloadLimit);
    }
    let payload_len =
        usize::try_from(read_u32(payload, 64)?).map_err(|_| CaptureLinkError::InvalidEnvelope)?;
    if payload_len != payload.len() - SYNC_SEGMENT_HEADER_BYTES {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    let segment = SyncSegment {
        transfer_id: array_at(payload, 0)?,
        offset: read_u64(payload, 16)?,
        total_bytes: read_u64(payload, 24)?,
        payload_sha256: array_at(payload, 32)?,
        payload: payload[SYNC_SEGMENT_HEADER_BYTES..].to_vec(),
    };
    segment.validate()?;
    Ok(segment)
}

fn decode_sync_ack(payload: &[u8]) -> Result<SyncAcknowledgement, CaptureLinkError> {
    if payload.len() != SYNC_ACK_BYTES {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    Ok(SyncAcknowledgement {
        transfer_id: array_at(payload, 0)?,
        persisted_through: read_u64(payload, 16)?,
        full_sha256: array_at(payload, 24)?,
    })
}

fn decode_sync_range(payload: &[u8]) -> Result<MissingRange, CaptureLinkError> {
    if payload.len() != SYNC_RANGE_BYTES {
        return Err(CaptureLinkError::InvalidEnvelope);
    }
    MissingRange {
        transfer_id: array_at(payload, 0)?,
        start: read_u64(payload, 16)?,
        end: read_u64(payload, 24)?,
    }
    .validate()
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CaptureLinkError> {
    Ok(u32::from_be_bytes(array_at(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, CaptureLinkError> {
    Ok(u64::from_be_bytes(array_at(bytes, offset)?))
}

fn array_at<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], CaptureLinkError> {
    let end = offset
        .checked_add(N)
        .ok_or(CaptureLinkError::InvalidEnvelope)?;
    bytes
        .get(offset..end)
        .ok_or(CaptureLinkError::InvalidEnvelope)?
        .try_into()
        .map_err(|_| CaptureLinkError::InvalidEnvelope)
}
