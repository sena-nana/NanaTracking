import base64
import hashlib
import http.client
import json
import threading
from io import BytesIO
from pathlib import Path

import pytest

from nana_tracking.data.capture import CaptureChunk, CaptureSessionManifest
from nana_tracking.data.studio import (
    CaptureQualitySample,
    CaptureStudio,
    NormalizedPreviewPoint,
    PreviewMetadata,
    StudioSessionDefinition,
)
from nana_tracking.data.studio_server import make_capture_studio_server


def test_studio_control_quality_preview_chunk_and_restart_closure(tmp_path: Path) -> None:
    studio = CaptureStudio.create(tmp_path, definition())
    started = studio.issue_control(
        "start",
        take_id="take-1",
        action_script_id="basic-v1",
    )
    studio.acknowledge_command(
        revision=started.revision,
        command_id=started.command_id,
        device_id="iphone-1",
        applied_at_ns=started.issued_at_ns + 1,
    )
    first_quality = studio.publish_quality(quality(sequence=0, timestamp_ns=1_000_000_000))
    assert first_quality.acceptable
    gap_quality = studio.publish_quality(quality(sequence=3, timestamp_ns=1_200_000_000))
    assert set(gap_quality.flags) == {"sequence_gap", "timestamp_gap"}

    preview = b"\xff\xd8preview\xff\xd9"
    studio.publish_preview(
        PreviewMetadata(
            session_id="session-1",
            take_id="take-1",
            sequence=3,
            capture_timestamp_ns=1_200_000_000,
            byte_length=len(preview),
        ),
        preview,
    )
    assert studio.preview_bytes() == preview
    with pytest.raises(ValueError, match="sequence regressed"):
        studio.publish_preview(
            PreviewMetadata(
                session_id="session-1",
                take_id="take-1",
                sequence=2,
                capture_timestamp_ns=1_100_000_000,
                byte_length=len(preview),
            ),
            preview,
        )

    payload = b"bounded-capture-chunk"
    descriptor = chunk(payload)
    acknowledgement = studio.receive_chunk(descriptor, BytesIO(payload))
    assert acknowledgement.chunk_id == descriptor.chunk_id

    studio.issue_control("stop", take_id="take-1")
    studio.issue_control(
        "retake",
        take_id="take-2",
        action_script_id="basic-v1",
        retake_of="take-1",
    )
    studio.issue_control("stop", take_id="take-2")
    studio.issue_control("end")
    manifest = studio.finalize_receiver_session()
    manifest.verify_files(tmp_path / "receiver")

    with (tmp_path / ".studio-state" / "commands.jsonl").open("ab") as stream:
        stream.write(b'{"command_id":"torn')
    with (tmp_path / ".studio-state" / "command-acks.jsonl").open("ab") as stream:
        stream.write(b'{"command_id":"torn')
    reopened = CaptureStudio(tmp_path)
    state = reopened.state()
    assert state.status == "complete"
    assert state.acknowledged_revision == 1
    assert state.received_chunk_count == 1
    assert {take.take_id: take.status for take in state.takes} == {
        "take-1": "replaced",
        "take-2": "stopped",
    }
    assert CaptureSessionManifest.load(tmp_path / "receiver" / "session.json") == manifest


def test_authenticated_studio_http_runs_control_preview_and_streaming_chunk(
    tmp_path: Path,
) -> None:
    CaptureStudio.create(tmp_path, definition())
    server = make_capture_studio_server(
        tmp_path,
        host="127.0.0.1",
        port=0,
        token="secret-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = request(server.server_port, "GET", "/api/state")
        assert status == 401

        control_body = json.dumps(
            {
                "action": "start",
                "take_id": "take-http",
                "action_script_id": "basic-v1",
            }
        ).encode()
        status, response = request(
            server.server_port,
            "POST",
            "/api/control",
            body=control_body,
            headers=authorized_json_headers(control_body),
        )
        assert status == 201
        command = json.loads(response)

        status, response = request(
            server.server_port,
            "GET",
            "/api/commands?after=0",
            headers=authorization(),
        )
        assert status == 200
        assert json.loads(response)[0]["command_id"] == command["command_id"]

        preview = b"\xff\xd8http-preview\xff\xd9"
        preview_metadata = PreviewMetadata(
            session_id="session-1",
            take_id="take-http",
            sequence=1,
            capture_timestamp_ns=5,
            byte_length=len(preview),
        )
        status, _ = request(
            server.server_port,
            "PUT",
            "/api/preview",
            body=preview,
            headers=authorization()
            | {
                "Content-Length": str(len(preview)),
                "X-Nana-Preview": encode_header(preview_metadata.model_dump(mode="json")),
            },
        )
        assert status == 200
        status, returned_preview = request(
            server.server_port,
            "GET",
            "/api/preview",
            headers=authorization(),
        )
        assert status == 200
        assert returned_preview == preview

        payload = b"streamed-without-base64"
        descriptor = chunk(payload)
        status, response = request(
            server.server_port,
            "PUT",
            f"/api/chunks/{descriptor.chunk_id}",
            body=payload,
            headers=authorization()
            | {
                "Content-Length": str(len(payload)),
                "X-Nana-Chunk": encode_header(descriptor.model_dump(mode="json")),
            },
        )
        assert status == 200
        assert json.loads(response)["sha256"] == descriptor.sha256
        status, response = request(
            server.server_port,
            "GET",
            "/api/receiver-index",
            headers=authorization(),
        )
        assert status == 200
        assert json.loads(response)[0]["chunk_id"] == descriptor.chunk_id
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def definition() -> StudioSessionDefinition:
    return StudioSessionDefinition(
        session_id="session-1",
        subject_id="subject-1",
        device_id="iphone-1",
        device_model="iPhone17,1",
        os_version="iOS 20.0",
        ntp_mapping_revision="mapping-v1",
        consent_record_id="consent-1",
        license_record_ids=["nana-synthetic-smoke"],
        created_at_ns=1,
    )


def quality(*, sequence: int, timestamp_ns: int) -> CaptureQualitySample:
    return CaptureQualitySample(
        session_id="session-1",
        take_id="take-1",
        sequence=sequence,
        capture_timestamp_ns=timestamp_ns,
        luminance=0.5,
        clipped_fraction=0.0,
        occluded_fraction=0.0,
        tracking_state="normal",
        face_mesh=[NormalizedPreviewPoint(x=0.25, y=0.75)],
        parameters={"jaw.open": 0.5},
    )


def chunk(payload: bytes) -> CaptureChunk:
    return CaptureChunk(
        chunk_id="arkit-0",
        take_id="take-1",
        kind="arkit",
        relative_path="chunks/take-1/arkit/00000000000000000000-00000000000000000003-arkit-0.bin",
        sequence_start=0,
        sequence_end=3,
        capture_timestamp_start_ns=1,
        capture_timestamp_end_ns=4,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def authorization() -> dict[str, str]:
    return {"Authorization": "Bearer secret-token"}


def authorized_json_headers(body: bytes) -> dict[str, str]:
    return authorization() | {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }


def encode_header(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode()


def request(
    port: int,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()
