use mutsuki_link_core::{
    Connection, ConnectionMetadata, ConnectionQuality, EndpointAddress, EndpointId,
    IdentityEvidence, IdentityStatus, MemoryConnection, MemoryTransportConfig, PeerId,
    ProtocolVersion, RealtimeDatagram, RealtimeFlowId, RealtimePriority, ReceivedRealtimeDatagram,
    SecurityExpectation, SecurityLevel, SecurityPolicy, SendOutcome, SessionContinuity,
    SessionId as LinkSessionId, SessionInfo, SessionKeyBinding, TransportError, TransportErrorKind,
    TransportKind, TransportSecurityEvidence, authenticate_session, memory_transport_pair,
};
use mutsuki_link_pairing::{KeyState, LinkPermission, TrustRecord};
use nana_capture_link::{
    CONTROL_PROTOCOL, CaptureAuthorization, CaptureLink, CaptureLinkConfig, CaptureLinkError,
    CaptureRole, ControlEvent, DataEvent, MissingRange, PREVIEW_NAMESPACE, PreviewFrame,
    PreviewTransportMode, SYNC_NAMESPACE, SyncAcknowledgement, SyncSegment,
    authorize_capture_session,
};
use std::collections::BTreeSet;
use std::time::Instant;

struct RealtimeMemoryConnection(MemoryConnection);

impl Connection for RealtimeMemoryConnection {
    fn metadata(&self) -> &ConnectionMetadata {
        self.0.metadata()
    }

    fn try_send(&mut self, message: &[u8]) -> Result<(), TransportError> {
        self.0.try_send(message)
    }

    fn try_receive(&mut self) -> Result<Option<Vec<u8>>, TransportError> {
        self.0.try_receive()
    }

    fn try_send_control(&mut self, message: &[u8]) -> Result<(), TransportError> {
        self.0.try_send_control(message)
    }

    fn try_receive_control(&mut self) -> Result<Option<Vec<u8>>, TransportError> {
        self.0.try_receive_control()
    }

    fn max_datagram_payload(&self) -> Option<usize> {
        Some(2 * 1024 * 1024 - 14)
    }

    fn try_send_latest(
        &mut self,
        datagram: RealtimeDatagram<'_>,
    ) -> Result<SendOutcome, TransportError> {
        if datagram.deadline <= Instant::now() {
            return Ok(SendOutcome::DroppedExpired);
        }
        let mut wire = Vec::with_capacity(14 + datagram.payload.len());
        wire.extend_from_slice(&datagram.flow.0.to_be_bytes());
        wire.extend_from_slice(&datagram.generation.to_be_bytes());
        wire.extend_from_slice(&datagram.sequence.to_be_bytes());
        wire.extend_from_slice(datagram.payload);
        self.0.try_send_datagram(&wire)?;
        Ok(SendOutcome::Enqueued)
    }

    fn try_receive_realtime(&mut self) -> Result<Option<ReceivedRealtimeDatagram>, TransportError> {
        let Some(wire) = self.0.try_receive_datagram()? else {
            return Ok(None);
        };
        if wire.len() < 14 {
            return Err(TransportError::new(
                TransportErrorKind::Other,
                "test realtime envelope is invalid",
            ));
        }
        Ok(Some(ReceivedRealtimeDatagram {
            flow: RealtimeFlowId(u16::from_be_bytes([wire[0], wire[1]])),
            generation: u32::from_be_bytes(wire[2..6].try_into().unwrap()),
            sequence: u64::from_be_bytes(wire[6..14].try_into().unwrap()),
            priority: RealtimePriority::Disposable,
            received_at: Instant::now(),
            payload: wire[14..].to_vec(),
        }))
    }

    fn close_write(&mut self) -> Result<(), TransportError> {
        self.0.close_write()
    }

    fn close_read(&mut self) -> Result<(), TransportError> {
        self.0.close_read()
    }

    fn abort(&mut self) {
        self.0.abort();
    }
}

fn endpoint(address: &str) -> EndpointAddress {
    EndpointAddress {
        scheme: "quic".to_owned(),
        address: address.to_owned(),
    }
}

fn authorization(
    role: CaptureRole,
    peer: u8,
    link_session: u8,
    key_state: KeyState,
    namespaces: &[&str],
) -> Result<CaptureAuthorization, CaptureLinkError> {
    authorization_with_datagram(role, peer, link_session, key_state, namespaces, true)
}

fn authorization_with_datagram(
    role: CaptureRole,
    peer: u8,
    link_session: u8,
    key_state: KeyState,
    namespaces: &[&str],
    datagram_allowed: bool,
) -> Result<CaptureAuthorization, CaptureLinkError> {
    let peer_id = PeerId::from_bytes([peer; 32]);
    let mut permissions = namespaces
        .iter()
        .map(|namespace| LinkPermission::OpenNamespace((*namespace).to_owned()))
        .chain(std::iter::once(LinkPermission::Connect))
        .collect::<BTreeSet<_>>();
    if datagram_allowed {
        permissions.insert(LinkPermission::Datagram);
    }
    let record = TrustRecord {
        peer_id,
        public_key: vec![peer; 32],
        alias: "capture peer".to_owned(),
        first_paired_at_unix_ms: 1,
        permissions,
        key_state,
        last_pairing_challenge_hash: [7; 32],
        previous_key_fingerprints: Vec::new(),
    };
    let fingerprint = record.public_key_fingerprint();
    let local_endpoint = endpoint("local");
    let remote_endpoint = endpoint("remote");
    let session = SessionInfo {
        session_id: LinkSessionId::from_bytes([link_session; 16]),
        peer_id,
        protocols: Vec::new(),
        continuity: SessionContinuity::default(),
        quality: ConnectionQuality::default(),
        close_reason: None,
    };
    let evidence = TransportSecurityEvidence {
        transport: TransportKind::Quic,
        security_level: SecurityLevel::AuthenticatedEncrypted,
        mutually_authenticated: true,
        local_peer_credential_verified: false,
        development_plaintext: false,
        identity: IdentityEvidence {
            peer_id,
            public_key_fingerprint: fingerprint,
            key_epoch: 4,
            status: IdentityStatus::Active {
                valid_until_unix_ms: 2_000,
            },
        },
        session_key: Some(SessionKeyBinding {
            key_id: [6; 32],
            forward_secure: true,
            handshake_transcript_hash: [5; 32],
            local_endpoint: local_endpoint.clone(),
            remote_endpoint: remote_endpoint.clone(),
            link_version: ProtocolVersion::new(1, 0),
        }),
    };
    let expectation = SecurityExpectation {
        peer_id,
        public_key_fingerprint: fingerprint,
        minimum_key_epoch: 4,
        handshake_transcript_hash: [5; 32],
        local_endpoint,
        remote_endpoint,
        link_version: ProtocolVersion::new(1, 0),
        now_unix_ms: 1_000,
    };
    let authenticated =
        authenticate_session(&session, &evidence, &expectation, SecurityPolicy::default())
            .expect("fixture security evidence is valid");
    authorize_capture_session(authenticated, role, &record, 11)
}

fn all_namespaces() -> [&'static str; 3] {
    [CONTROL_PROTOCOL, PREVIEW_NAMESPACE, SYNC_NAMESPACE]
}

fn link_pair(
    datagram_capacity: usize,
    queue_capacity: usize,
    link_session: u8,
) -> (CaptureLink<MemoryConnection>, CaptureLink<MemoryConnection>) {
    let (device_connection, studio_connection) = memory_transport_pair(
        EndpointId::from_bytes([1; 16]),
        EndpointId::from_bytes([2; 16]),
        MemoryTransportConfig {
            queue_capacity,
            max_message_bytes: 2 * 1024 * 1024,
            datagram_capacity,
        },
    );
    let device_authorization = authorization(
        CaptureRole::Device,
        2,
        link_session,
        KeyState::Active,
        &all_namespaces(),
    )
    .unwrap();
    let studio_authorization = authorization(
        CaptureRole::Studio,
        1,
        link_session,
        KeyState::Active,
        &all_namespaces(),
    )
    .unwrap();
    (
        CaptureLink::new(
            device_connection,
            device_authorization,
            CaptureLinkConfig::default(),
        )
        .unwrap(),
        CaptureLink::new(
            studio_connection,
            studio_authorization,
            CaptureLinkConfig::default(),
        )
        .unwrap(),
    )
}

fn exchange_hello<C: Connection>(device: &mut CaptureLink<C>, studio: &mut CaptureLink<C>) {
    device.try_send_hello().unwrap();
    studio.try_send_hello().unwrap();
    assert_eq!(
        device.poll_control().unwrap(),
        Some(ControlEvent::PeerReady)
    );
    assert_eq!(
        studio.poll_control().unwrap(),
        Some(ControlEvent::PeerReady)
    );
    assert!(device.remote_ready());
    assert!(studio.remote_ready());
}

#[test]
fn trust_scope_is_fail_closed_for_missing_or_revoked_permissions() {
    let missing_sync = authorization(
        CaptureRole::Device,
        2,
        9,
        KeyState::Active,
        &[CONTROL_PROTOCOL, PREVIEW_NAMESPACE],
    );
    assert_eq!(missing_sync.unwrap_err(), CaptureLinkError::Unauthorized);

    let revoked = authorization(
        CaptureRole::Device,
        2,
        9,
        KeyState::Revoked {
            revoked_at_unix_ms: 1_500,
        },
        &all_namespaces(),
    );
    assert_eq!(revoked.unwrap_err(), CaptureLinkError::PeerRevoked);

    let rotated = authorization(
        CaptureRole::Device,
        2,
        9,
        KeyState::Rotated {
            rotated_at_unix_ms: 1_500,
            new_peer_id: PeerId::from_bytes([3; 32]),
        },
        &all_namespaces(),
    );
    assert_eq!(rotated.unwrap_err(), CaptureLinkError::PeerRotated);
}

#[test]
fn realtime_transport_falls_back_without_explicit_datagram_permission() {
    let (device_connection, studio_connection) = memory_transport_pair(
        EndpointId::from_bytes([1; 16]),
        EndpointId::from_bytes([2; 16]),
        MemoryTransportConfig {
            datagram_capacity: 2,
            ..MemoryTransportConfig::default()
        },
    );
    let device_authorization = authorization_with_datagram(
        CaptureRole::Device,
        2,
        9,
        KeyState::Active,
        &all_namespaces(),
        false,
    )
    .unwrap();
    let studio_authorization = authorization(
        CaptureRole::Studio,
        1,
        9,
        KeyState::Active,
        &all_namespaces(),
    )
    .unwrap();
    let mut device = CaptureLink::new(
        RealtimeMemoryConnection(device_connection),
        device_authorization,
        CaptureLinkConfig::default(),
    )
    .unwrap();
    let mut studio = CaptureLink::new(
        RealtimeMemoryConnection(studio_connection),
        studio_authorization,
        CaptureLinkConfig::default(),
    )
    .unwrap();
    assert_eq!(
        device.preview_mode(),
        PreviewTransportMode::ReliableLatestOnly
    );
    assert_eq!(studio.preview_mode(), PreviewTransportMode::Datagram);
    exchange_hello(&mut device, &mut studio);
    assert_eq!(
        device.preview_mode(),
        PreviewTransportMode::ReliableLatestOnly
    );
    assert_eq!(
        studio.preview_mode(),
        PreviewTransportMode::ReliableLatestOnly
    );
}

#[test]
fn reliable_fallback_keeps_control_independent_and_only_delivers_latest_preview() {
    let (mut device, mut studio) = link_pair(0, 1, 9);
    assert_eq!(
        device.preview_mode(),
        PreviewTransportMode::ReliableLatestOnly
    );
    exchange_hello(&mut device, &mut studio);

    let segment = SyncSegment::new([3; 16], 0, 4, b"data".to_vec()).unwrap();
    device.try_send_sync_segment(&segment).unwrap();
    device
        .try_send_preview(PreviewFrame {
            generation: 1,
            sequence: 1,
            capture_timestamp_ns: 10,
            payload: vec![1; 16],
        })
        .unwrap();
    device
        .try_send_preview(PreviewFrame {
            generation: 1,
            sequence: 2,
            capture_timestamp_ns: 20,
            payload: vec![2; 16],
        })
        .unwrap();
    device.try_send_control(b"pause-capture").unwrap();

    assert_eq!(
        studio.poll_control().unwrap(),
        Some(ControlEvent::Application(b"pause-capture".to_vec()))
    );
    assert_eq!(
        studio.poll_data().unwrap(),
        Some(DataEvent::Segment(segment))
    );
    device.poll_outbound().unwrap();
    assert_eq!(
        studio.poll_data().unwrap(),
        Some(DataEvent::Preview(PreviewFrame {
            generation: 1,
            sequence: 2,
            capture_timestamp_ns: 20,
            payload: vec![2; 16],
        }))
    );
    assert_eq!(device.telemetry().preview_replacements, 1);

    let acknowledgement = SyncAcknowledgement {
        transfer_id: [3; 16],
        persisted_through: 4,
        full_sha256: [8; 32],
    };
    studio.try_send_acknowledgement(acknowledgement).unwrap();
    assert_eq!(
        device.poll_data().unwrap(),
        Some(DataEvent::Acknowledgement(acknowledgement))
    );
    let range = MissingRange {
        transfer_id: [3; 16],
        start: 4,
        end: 8,
    };
    studio.try_send_missing_range(range).unwrap();
    assert_eq!(
        device.poll_data().unwrap(),
        Some(DataEvent::MissingRange(range))
    );
}

#[test]
fn datagram_preview_is_session_bound_and_digest_is_checked_before_send() {
    let (device_connection, studio_connection) = memory_transport_pair(
        EndpointId::from_bytes([1; 16]),
        EndpointId::from_bytes([2; 16]),
        MemoryTransportConfig {
            queue_capacity: 4,
            max_message_bytes: 2 * 1024 * 1024,
            datagram_capacity: 2,
        },
    );
    let mut device = CaptureLink::new(
        RealtimeMemoryConnection(device_connection),
        authorization(
            CaptureRole::Device,
            2,
            9,
            KeyState::Active,
            &all_namespaces(),
        )
        .unwrap(),
        CaptureLinkConfig::default(),
    )
    .unwrap();
    let mut studio = CaptureLink::new(
        RealtimeMemoryConnection(studio_connection),
        authorization(
            CaptureRole::Studio,
            1,
            9,
            KeyState::Active,
            &all_namespaces(),
        )
        .unwrap(),
        CaptureLinkConfig::default(),
    )
    .unwrap();
    assert_eq!(device.preview_mode(), PreviewTransportMode::Datagram);
    exchange_hello(&mut device, &mut studio);

    let preview = PreviewFrame {
        generation: 4,
        sequence: 12,
        capture_timestamp_ns: 99,
        payload: vec![6; 128],
    };
    let durable = SyncSegment::new([6; 16], 0, 4, b"sync".to_vec()).unwrap();
    device.try_send_sync_segment(&durable).unwrap();
    device.try_send_preview(preview.clone()).unwrap();
    assert_eq!(
        studio.poll_data().unwrap(),
        Some(DataEvent::Segment(durable))
    );
    assert_eq!(
        studio.poll_data().unwrap(),
        Some(DataEvent::Preview(preview))
    );

    let mut corrupted = SyncSegment::new([4; 16], 0, 4, b"data".to_vec()).unwrap();
    corrupted.payload[0] ^= 0xff;
    assert_eq!(
        device.try_send_sync_segment(&corrupted).unwrap_err(),
        CaptureLinkError::DigestMismatch
    );
}

#[test]
fn hello_rejects_a_different_link_session() {
    let (left_connection, right_connection) = memory_transport_pair(
        EndpointId::from_bytes([1; 16]),
        EndpointId::from_bytes([2; 16]),
        MemoryTransportConfig::default(),
    );
    let left_authorization = authorization(
        CaptureRole::Device,
        2,
        7,
        KeyState::Active,
        &all_namespaces(),
    )
    .unwrap();
    let right_authorization = authorization(
        CaptureRole::Studio,
        1,
        8,
        KeyState::Active,
        &all_namespaces(),
    )
    .unwrap();
    let mut left = CaptureLink::new(
        left_connection,
        left_authorization,
        CaptureLinkConfig::default(),
    )
    .unwrap();
    let mut right = CaptureLink::new(
        right_connection,
        right_authorization,
        CaptureLinkConfig::default(),
    )
    .unwrap();
    left.try_send_hello().unwrap();
    assert_eq!(
        right.poll_control().unwrap_err(),
        CaptureLinkError::SessionBindingMismatch
    );
}

#[test]
fn reconnect_requires_a_new_authenticated_session_and_resets_runtime_state() {
    let (mut device, mut studio) = link_pair(0, 2, 9);
    exchange_hello(&mut device, &mut studio);

    let (same_connection, _) = memory_transport_pair(
        EndpointId::from_bytes([3; 16]),
        EndpointId::from_bytes([4; 16]),
        MemoryTransportConfig::default(),
    );
    let same_session = authorization(
        CaptureRole::Device,
        2,
        9,
        KeyState::Active,
        &all_namespaces(),
    )
    .unwrap();
    assert_eq!(
        device.reconnect(same_connection, same_session).unwrap_err(),
        CaptureLinkError::SessionBindingMismatch
    );

    let (new_connection, _) = memory_transport_pair(
        EndpointId::from_bytes([5; 16]),
        EndpointId::from_bytes([6; 16]),
        MemoryTransportConfig::default(),
    );
    let new_session = authorization(
        CaptureRole::Device,
        2,
        10,
        KeyState::Active,
        &all_namespaces(),
    )
    .unwrap();
    device.reconnect(new_connection, new_session).unwrap();
    assert!(!device.remote_ready());
    assert_eq!(
        device.telemetry(),
        nana_capture_link::CaptureLinkTelemetry::default()
    );
}

#[test]
#[ignore = "manual synthetic performance smoke"]
fn capture_link_performance_smoke() {
    const PREVIEW_ITERATIONS: u64 = 200_000;
    const SYNC_ITERATIONS: u64 = 1_024;
    const SYNC_PAYLOAD_BYTES: usize = 64 * 1024;

    let (mut device, mut studio) = link_pair(0, 1, 9);
    exchange_hello(&mut device, &mut studio);
    let blocker = SyncSegment::new([1; 16], 0, 1, vec![1]).unwrap();
    device.try_send_sync_segment(&blocker).unwrap();

    let preview_started = Instant::now();
    for sequence in 1..=PREVIEW_ITERATIONS {
        device
            .try_send_preview(PreviewFrame {
                generation: 1,
                sequence,
                capture_timestamp_ns: sequence,
                payload: vec![7; 256],
            })
            .unwrap();
    }
    let preview_elapsed = preview_started.elapsed();
    let preview_telemetry = device.telemetry();
    assert!(preview_telemetry.preview_pending);
    assert_eq!(
        preview_telemetry.preview_replacements,
        PREVIEW_ITERATIONS - 1
    );
    assert_eq!(preview_telemetry.pending_sync, 0);

    assert_eq!(
        studio.poll_data().unwrap(),
        Some(DataEvent::Segment(blocker))
    );
    device.poll_outbound().unwrap();
    let delivered = studio.poll_data().unwrap().unwrap();
    assert!(matches!(
        delivered,
        DataEvent::Preview(PreviewFrame {
            sequence: PREVIEW_ITERATIONS,
            ..
        })
    ));

    let (mut sender, mut receiver) = link_pair(0, 1, 10);
    exchange_hello(&mut sender, &mut receiver);
    let sync_payload = vec![9; SYNC_PAYLOAD_BYTES];
    let sync_started = Instant::now();
    for sequence in 0..SYNC_ITERATIONS {
        let mut transfer_id = [0; 16];
        transfer_id[..8].copy_from_slice(&sequence.to_be_bytes());
        let segment = SyncSegment::new(
            transfer_id,
            0,
            u64::try_from(SYNC_PAYLOAD_BYTES).unwrap(),
            sync_payload.clone(),
        )
        .unwrap();
        sender.try_send_sync_segment(&segment).unwrap();
        assert_eq!(
            receiver.poll_data().unwrap(),
            Some(DataEvent::Segment(segment))
        );
    }
    let sync_elapsed = sync_started.elapsed();

    let preview_ns_per_submit = preview_elapsed.as_secs_f64() * 1_000_000_000.0 / 200_000.0;
    let sync_mib_per_second = 64.0 / sync_elapsed.as_secs_f64();
    println!(
        "preview_ns_per_submit={preview_ns_per_submit:.3} preview_replacements={} sync_mib_per_second={sync_mib_per_second:.3} sync_messages={} preview_elapsed_us={} sync_elapsed_us={}",
        preview_telemetry.preview_replacements,
        sender.telemetry().sent_sync_messages,
        preview_elapsed.as_micros(),
        sync_elapsed.as_micros()
    );
}
