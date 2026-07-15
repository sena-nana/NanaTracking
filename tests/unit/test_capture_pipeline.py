import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from nana_tracking.cli import app
from nana_tracking.data.capture import (
    ArkitMapping,
    ArkitMappingRule,
    CaptureChunk,
    CaptureSessionManifest,
    ChunkAcknowledgement,
    FrozenCaptureDataset,
    LatestPreview,
    LocalChunkStore,
    PreviewFrame,
    RawArkitFrame,
    build_training_manifest_from_frozen,
    convert_arkit_frames,
    freeze_capture_dataset,
    reconcile_chunks,
    run_capture_pipeline_smoke,
    write_capture_records,
)
from nana_tracking.data.schema import CameraCalibration, CaptureConditions, RgbFrame

runner = CliRunner()


def test_local_first_chunks_survive_disconnect_and_reconcile_after_restart(
    tmp_path: Path,
) -> None:
    ios = LocalChunkStore(tmp_path / "ios")
    first = ios.write_chunk(
        chunk_id="take-1-arkit-0",
        take_id="take-1",
        kind="arkit",
        sequence_start=0,
        sequence_end=3,
        capture_timestamp_start_ns=100,
        capture_timestamp_end_ns=130,
        payload=b"first-four-frames",
    )
    second = ios.write_chunk(
        chunk_id="take-1-arkit-1",
        take_id="take-1",
        kind="arkit",
        sequence_start=4,
        sequence_end=7,
        capture_timestamp_start_ns=140,
        capture_timestamp_end_ns=170,
        payload=b"next-four-frames",
    )
    manifest = ios.finalize(
        session_id="session-1",
        subject_id="subject-1",
        device_id="iphone-1",
        device_model="iPhone17,1",
        os_version="iOS 20.0",
        ntp_mapping_revision="arkit-to-ntp/1.0.0-smoke",
        consent_record_id="consent-1",
        license_record_ids=["nana-synthetic-smoke"],
    )
    manifest.verify_files(tmp_path / "ios")

    studio = LocalChunkStore(tmp_path / "studio")
    persisted = studio.receive_chunk(first, b"first-four-frames")
    first_ack = ChunkAcknowledgement(chunk_id=persisted.chunk_id, sha256=persisted.sha256)
    ios.acknowledge(first_ack)

    reopened = LocalChunkStore(tmp_path / "ios")
    assert [chunk.chunk_id for chunk in reopened.pending_chunks()] == [second.chunk_id]
    partial = reconcile_chunks(manifest.chunks, [first_ack])
    assert not partial.complete
    assert partial.missing_ranges[0].sequence_start == 4
    assert partial.missing_ranges[0].sequence_end == 7

    with pytest.raises(ValueError, match="does not match descriptor"):
        studio.receive_chunk(second, b"corrupt")
    persisted = studio.receive_chunk(second, b"next-four-frames")
    second_ack = ChunkAcknowledgement(chunk_id=persisted.chunk_id, sha256=persisted.sha256)
    reopened.acknowledge(second_ack)
    assert reopened.pending_chunks() == []
    assert reconcile_chunks(manifest.chunks, [first_ack, second_ack]).complete


def test_local_journals_recover_only_a_crash_torn_tail(tmp_path: Path) -> None:
    root = tmp_path / "ios"
    store = LocalChunkStore(root)
    chunk = store.write_chunk(
        chunk_id="chunk-0",
        take_id="take-1",
        kind="arkit",
        sequence_start=0,
        sequence_end=0,
        capture_timestamp_start_ns=1,
        capture_timestamp_end_ns=1,
        payload=b"payload",
    )
    chunks_journal = root / ".capture-state" / "chunks.jsonl"
    with chunks_journal.open("ab") as stream:
        stream.write(b'{"chunk_id":"torn')
    reopened = LocalChunkStore(root)
    assert reopened.pending_chunks() == [chunk]
    assert chunks_journal.read_bytes().endswith(b"\n")
    assert b'"torn' not in chunks_journal.read_bytes()

    acknowledgement = ChunkAcknowledgement(chunk_id=chunk.chunk_id, sha256=chunk.sha256)
    reopened.acknowledge(acknowledgement)
    acknowledgement_journal = root / ".capture-state" / "acknowledged.jsonl"
    acknowledgement_journal.write_bytes(acknowledgement_journal.read_bytes().removesuffix(b"\n"))
    assert LocalChunkStore(root).acknowledgements() == [acknowledgement]
    assert acknowledgement_journal.read_bytes().endswith(b"\n")

    with chunks_journal.open("ab") as stream:
        stream.write(b"not-json\n")
    with pytest.raises(ValueError, match="invalid journal entry"):
        LocalChunkStore(root)


def test_capture_studio_cli_emits_ack_only_after_durable_receive(tmp_path: Path) -> None:
    sender_root = tmp_path / "sender"
    sender = LocalChunkStore(sender_root)
    payload = b"durable-payload"
    chunk = sender.write_chunk(
        chunk_id="chunk-0",
        take_id="take-1",
        kind="arkit",
        sequence_start=0,
        sequence_end=0,
        capture_timestamp_start_ns=1,
        capture_timestamp_end_ns=1,
        payload=payload,
    )
    descriptor = tmp_path / "chunk.json"
    descriptor.write_text(chunk.model_dump_json(), encoding="utf-8")
    payload_path = tmp_path / "chunk.bin"
    payload_path.write_bytes(payload)
    receiver_root = tmp_path / "receiver"

    received = runner.invoke(
        app,
        ["data", "capture-receive", str(receiver_root), str(descriptor), str(payload_path)],
    )
    assert received.exit_code == 0
    assert ChunkAcknowledgement.model_validate_json(received.stdout) == ChunkAcknowledgement(
        chunk_id=chunk.chunk_id,
        sha256=chunk.sha256,
    )
    index = runner.invoke(app, ["data", "capture-receiver-index", str(receiver_root)])
    assert index.exit_code == 0
    assert chunk.chunk_id in index.stdout
    pending = runner.invoke(app, ["data", "capture-pending", str(sender_root)])
    assert pending.exit_code == 0
    assert chunk.chunk_id in pending.stdout


def test_capture_manifest_rejects_overlap_truncation_and_preview_as_training_chunk(
    tmp_path: Path,
) -> None:
    store = LocalChunkStore(tmp_path)
    chunk = store.write_chunk(
        chunk_id="rgb-0",
        take_id="take-1",
        kind="rgb",
        sequence_start=0,
        sequence_end=0,
        capture_timestamp_start_ns=1,
        capture_timestamp_end_ns=1,
        payload=b"rgb",
    )
    manifest = store.finalize(
        session_id="session-1",
        subject_id="subject-1",
        device_id="device-1",
        device_model="model-1",
        os_version="os-1",
        ntp_mapping_revision="mapping-1",
        consent_record_id="consent-1",
        license_record_ids=["nana-synthetic-smoke"],
    )
    duplicate = chunk.model_copy(
        update={
            "chunk_id": "rgb-1",
            "relative_path": (
                "chunks/take-1/rgb/00000000000000000000-00000000000000000000-rgb-1.bin"
            ),
        }
    )
    with pytest.raises(ValidationError, match="overlapping capture chunks"):
        CaptureSessionManifest.create(
            session_id="session-1",
            subject_id="subject-1",
            device_id="device-1",
            device_model="model-1",
            os_version="os-1",
            ntp_mapping_revision="mapping-1",
            consent_record_id="consent-1",
            license_record_ids=["nana-synthetic-smoke"],
            chunks=[chunk, duplicate],
        )
    with pytest.raises(ValidationError):
        CaptureChunk.model_validate(
            {
                **chunk.model_dump(mode="json"),
                "chunk_id": "preview-0",
                "kind": "preview",
            }
        )

    (tmp_path / chunk.relative_path).write_bytes(b"rg")
    with pytest.raises(ValueError, match="length mismatch"):
        manifest.verify_files(tmp_path)


def test_preview_is_latest_only_and_has_no_durable_chunk_surface() -> None:
    preview = LatestPreview()
    preview.publish(PreviewFrame(sequence=1, capture_timestamp_ns=10, jpeg=b"one"))
    preview.publish(PreviewFrame(sequence=2, capture_timestamp_ns=20, jpeg=b"two"))
    assert preview.dropped == 1
    assert preview.take().sequence == 2  # type: ignore[union-attr]
    assert preview.take() is None


def test_raw_arkit_can_be_rederived_under_a_new_mapping_without_mutation(
    tmp_path: Path,
) -> None:
    frame = raw_frame(
        record_id="record-1",
        subject_id="subject-1",
        session_id="session-1",
        device_id="device-1",
        sequence=1,
        timestamp_ns=100,
    )
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(frame.model_dump_json() + "\n", encoding="utf-8")
    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    original = mapping(offset=0.0, revision="mapping-v1")
    revised = mapping(offset=0.1, revision="mapping-v2")
    first = convert_arkit_frames(RawArkitFrame.load_jsonl(raw_path), original)
    second = convert_arkit_frames(RawArkitFrame.load_jsonl(raw_path), revised)
    first_value = first[0].teachers[0].labels["jaw.open"]
    second_value = second[0].teachers[0].labels["jaw.open"]
    assert first_value.value == pytest.approx(0.5)
    assert second_value.value == pytest.approx(0.6)
    assert first_value.method == "mapping-v1:jawOpen"
    assert second_value.method == "mapping-v2:jawOpen"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before


def test_raw_arkit_v11_preserves_asynchronous_depth_timing_and_unknown_confidence() -> None:
    frame = raw_frame(
        record_id="record-depth-v11",
        subject_id="subject-1",
        session_id="session-1",
        device_id="device-1",
        sequence=1,
        timestamp_ns=100,
    )
    payload = frame.model_dump(mode="json") | {
        "schema_version": "nana-raw-arkit-frame/1.1.0",
        "depth_confidence": 0.0,
        "depth_capture_timestamp_ns": 95,
        "depth_width": 4,
        "depth_height": 3,
        "depth_pixel_format": "float32-le-meters",
        "depth_confidence_source": "unavailable",
        "depth_accuracy": "relative",
        "depth_quality": "high",
        "depth_filtered": True,
    }
    v11 = RawArkitFrame.model_validate(payload)

    record = convert_arkit_frames([v11], mapping())[0]

    assert record.depth[0].capture_timestamp_ns == 95
    assert record.depth[0].state == "observed"
    assert record.depth[0].confidence == 0.0
    limited = RawArkitFrame.model_validate(payload | {"tracking_state": "limited"})
    limited_depth = convert_arkit_frames([limited], mapping())[0].depth[0]
    assert limited_depth.capture_timestamp_ns == 95
    assert limited_depth.state == "unavailable"
    with pytest.raises(ValidationError, match="complete timing"):
        RawArkitFrame.model_validate(payload | {"depth_capture_timestamp_ns": None})
    with pytest.raises(ValidationError, match="confidence must be zero"):
        RawArkitFrame.model_validate(payload | {"depth_confidence": 0.5})


def test_frozen_capture_dataset_reuses_license_and_identity_device_split_gates(
    tmp_path: Path,
) -> None:
    identities = [
        ("train-subject", "train-session", "dev-main"),
        ("validation-subject", "validation-session", "dev-main"),
        ("test-subject", "test-session", "dev-heldout"),
    ]
    manifests: list[Path] = []
    frames: list[RawArkitFrame] = []
    for index, (subject, session, device) in enumerate(identities):
        root = tmp_path / session
        store = LocalChunkStore(root)
        rgb = store.write_chunk(
            chunk_id=f"{session}-rgb",
            take_id="take-1",
            kind="rgb",
            sequence_start=0,
            sequence_end=0,
            capture_timestamp_start_ns=100,
            capture_timestamp_end_ns=100,
            payload=f"rgb-{session}".encode(),
        )
        depth = store.write_chunk(
            chunk_id=f"{session}-depth",
            take_id="take-1",
            kind="depth",
            sequence_start=0,
            sequence_end=0,
            capture_timestamp_start_ns=100,
            capture_timestamp_end_ns=100,
            payload=f"depth-{session}".encode(),
        )
        store.write_chunk(
            chunk_id=f"{session}-arkit",
            take_id="take-1",
            kind="arkit",
            sequence_start=0,
            sequence_end=0,
            capture_timestamp_start_ns=100,
            capture_timestamp_end_ns=100,
            payload=f"raw-{session}".encode(),
        )
        store.finalize(
            session_id=session,
            subject_id=subject,
            device_id=device,
            device_model="fixture-device",
            os_version="fixture-os",
            ntp_mapping_revision="mapping-v1",
            consent_record_id=f"consent-{subject}",
            license_record_ids=["nana-synthetic-smoke"],
        )
        manifests.append(root / "session.json")
        frame = raw_frame(
            record_id=f"record-{index}",
            subject_id=subject,
            session_id=session,
            device_id=device,
            sequence=0,
            timestamp_ns=100,
        )
        frames.append(
            frame.model_copy(
                update={
                    "rgb": frame.rgb.model_copy(update={"uri": rgb.relative_path}),
                    "depth_uri": depth.relative_path,
                }
            )
        )

    records_path = tmp_path / "records.jsonl"
    mapping_path = tmp_path / "mapping.json"
    mapping().save(mapping_path)
    write_capture_records(convert_arkit_frames(frames, mapping()), records_path)
    output = tmp_path / "dataset" / "frozen.json"
    frozen = freeze_capture_dataset(
        data_revision="capture-smoke-v1",
        session_manifests=manifests,
        capture_records=records_path,
        arkit_mappings=[mapping_path],
        license_registry=Path("configs/data/license-registry.json"),
        license_record_ids=["nana-synthetic-smoke"],
        held_out_test_devices={"dev-heldout"},
        validation_identities=1,
        seed=17,
        smoke_only=True,
        output=output,
    )
    assert frozen.frozen
    assert frozen.split_plan.splits["test"].devices == ["dev-heldout"]
    FrozenCaptureDataset.load(output).verify(output)

    with pytest.raises(ValueError, match="smoke"):
        freeze_capture_dataset(
            data_revision="invalid-production",
            session_manifests=manifests,
            capture_records=records_path,
            arkit_mappings=[mapping_path],
            license_registry=Path("configs/data/license-registry.json"),
            license_record_ids=["nana-synthetic-smoke"],
            held_out_test_devices={"dev-heldout"},
            validation_identities=1,
            seed=17,
            smoke_only=False,
            output=tmp_path / "production.json",
        )


def test_capture_pipeline_smoke_closes_local_record_sync_derive_and_freeze(
    tmp_path: Path,
) -> None:
    report = run_capture_pipeline_smoke(
        tmp_path / "capture-smoke",
        mapping_path=Path("configs/data/arkit-to-ntp-v1-smoke.json"),
        license_registry=Path("configs/data/license-registry.json"),
        label_catalog_path=Path("configs/data/ntp-v1-label-catalog.json"),
    )

    assert report["smoke_only"] is True
    assert report["session_count"] == 3
    assert report["chunk_acknowledgement_count"] == 12
    assert report["derived_record_count"] == 3
    frozen_path = Path(str(report["dataset"]))
    FrozenCaptureDataset.load(frozen_path).verify(frozen_path)
    training_manifest = build_training_manifest_from_frozen(
        frozen_path,
        label_catalog_path=Path("configs/data/ntp-v1-label-catalog.json"),
        output=tmp_path / "training-manifest.json",
    )
    training_manifest.verify_files(tmp_path / "training-manifest.json")


def mapping(*, offset: float = 0.0, revision: str = "mapping-v1") -> ArkitMapping:
    return ArkitMapping(
        mapping_revision=revision,
        teacher_source_id="synthetic-truedepth",
        license_record_id="nana-synthetic-smoke",
        rules=[
            ArkitMappingRule(
                target_signal="jaw.open",
                source_blendshape="jawOpen",
                offset=offset,
                minimum=0.0,
                maximum=1.0,
                confidence=0.9,
            )
        ],
        smoke_only=True,
    )


def raw_frame(
    *,
    record_id: str,
    subject_id: str,
    session_id: str,
    device_id: str,
    sequence: int,
    timestamp_ns: int,
) -> RawArkitFrame:
    identity = [1.0, 0.0, 0.0, 0.0] * 4
    return RawArkitFrame(
        record_id=record_id,
        subject_id=subject_id,
        session_id=session_id,
        take_id="take-1",
        device_id=device_id,
        action_script_id="basic-smoke",
        consent_record_id=f"consent-{subject_id}",
        capture_timestamp_ns=timestamp_ns,
        sequence=sequence,
        rgb=RgbFrame(
            uri=f"rgb/{record_id}.heic",
            width=640,
            height=480,
            exposure_duration_ns=10_000_000,
            iso=100.0,
            frame_duration_ns=16_666_667,
        ),
        camera=CameraCalibration(
            intrinsics=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            distortion_model="none",
            distortion_coefficients=[],
        ),
        blendshapes={"jawOpen": 0.5},
        head_transform_column_major=identity,
        left_eye_transform_column_major=identity,
        right_eye_transform_column_major=identity,
        depth_uri=f"depth/{record_id}.bin",
        depth_confidence=0.8,
        tracking_state="normal",
        conditions=CaptureConditions(lighting="normal"),
    )
