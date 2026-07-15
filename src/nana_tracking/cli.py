"""Command-line entrypoint for reproducible project workflows."""

import json
import time
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import typer
from rich.console import Console

from nana_tracking.config import ExperimentConfig, load_config
from nana_tracking.contracts import ModelPackageMetadata
from nana_tracking.data.capture import (
    ArkitMapping,
    CaptureChunk,
    CaptureSessionManifest,
    ChunkAcknowledgement,
    FrozenCaptureDataset,
    LocalChunkStore,
    RawArkitFrame,
    build_training_manifest_from_frozen,
    convert_arkit_frames,
    freeze_capture_dataset,
    reconcile_chunks,
    run_capture_pipeline_smoke,
    write_capture_records,
)
from nana_tracking.data.executors import benchmark_backends
from nana_tracking.data.labeling import LabelCatalog, materialize_dataset, write_materialized_labels
from nana_tracking.data.manifest import DatasetManifest
from nana_tracking.data.schema import CaptureRecord
from nana_tracking.data.strategy import (
    ActorSplitManifest,
    CaptureSplitPlan,
    ExpressionCacheManifest,
    ExpressionCacheRecord,
    LicenseRegistry,
    PipelineStage,
    load_clip_index,
    split_captures,
    split_clips_by_actor,
)
from nana_tracking.data.studio import (
    CaptureStudio,
    ControlAction,
    StudioSessionDefinition,
)
from nana_tracking.data.studio_server import make_capture_studio_server
from nana_tracking.doctor import doctor_report
from nana_tracking.evaluation import (
    ExpressionAblationConfig,
    FailureSample,
    benchmark_face_basic_package,
    benchmark_face_spatial_package,
    benchmark_full_set_package,
    benchmark_rgb_roi_preprocessor,
    benchmark_temporal_refiner,
    fit_confidence_calibration,
    render_failure_report,
    run_expression_ablation_smoke,
    validate_face_basic_acceptance,
)
from nana_tracking.evaluation import (
    evaluate as evaluate_model,
)
from nana_tracking.evaluation.capture import benchmark_capture_store
from nana_tracking.evaluation.standard import BenchmarkReport, EvaluationStandard
from nana_tracking.export import create_model_package, verify_model_package
from nana_tracking.personalization import (
    fit_level_a_calibration,
    train_level_b_adapter,
    verify_level_b_adapter,
)
from nana_tracking.training import train as train_model

app = typer.Typer(no_args_is_help=True, help="NanaTracking training and ONNX tooling.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset manifest commands.")
evaluation_app = typer.Typer(no_args_is_help=True, help="Evaluation standard commands.")
studio_app = typer.Typer(no_args_is_help=True, help="Capture Studio operator commands.")
app.add_typer(data_app, name="data")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(studio_app, name="studio")
console = Console()


def _print_json(payload: object) -> None:
    console.print_json(json.dumps(payload, default=str))


@studio_app.command("create")
def studio_create_command(
    root: Annotated[Path, typer.Argument(file_okay=False)],
    session_id: Annotated[str, typer.Option()],
    subject_id: Annotated[str, typer.Option()],
    device_id: Annotated[str, typer.Option()],
    device_model: Annotated[str, typer.Option()],
    os_version: Annotated[str, typer.Option()],
    ntp_mapping_revision: Annotated[str, typer.Option()],
    consent_record_id: Annotated[str, typer.Option()],
    license_records: Annotated[str, typer.Option(help="Comma-separated license record IDs")],
) -> None:
    """Create one durable Capture Studio session."""

    studio = CaptureStudio.create(
        root,
        StudioSessionDefinition(
            session_id=session_id,
            subject_id=subject_id,
            device_id=device_id,
            device_model=device_model,
            os_version=os_version,
            ntp_mapping_revision=ntp_mapping_revision,
            consent_record_id=consent_record_id,
            license_record_ids=sorted(
                {record.strip() for record in license_records.split(",") if record.strip()}
            ),
            created_at_ns=time.time_ns(),
        ),
    )
    _print_json(studio.state().model_dump(mode="json"))


@studio_app.command("state")
def studio_state_command(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Show the durable session, take, control, quality, and receiver state."""

    _print_json(CaptureStudio(root).state().model_dump(mode="json"))


@studio_app.command("control")
def studio_control_command(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    action: Annotated[ControlAction, typer.Argument()],
    take_id: Annotated[str | None, typer.Option()] = None,
    action_script_id: Annotated[str | None, typer.Option()] = None,
    retake_of: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Issue a validated start, pause, stop, retake, or end command."""

    command = CaptureStudio(root).issue_control(
        action,
        take_id=take_id,
        action_script_id=action_script_id,
        retake_of=retake_of,
    )
    _print_json(command.model_dump(mode="json"))


@studio_app.command("finalize")
def studio_finalize_command(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Finalize the verified receiver chunks into a capture session manifest."""

    manifest = CaptureStudio(root).finalize_receiver_session()
    _print_json(
        {
            "session_id": manifest.session_id,
            "chunk_count": len(manifest.chunks),
            "manifest_sha256": manifest.manifest_sha256,
            "manifest": root / "receiver" / "session.json",
        }
    )


@studio_app.command("serve")
def studio_serve_command(
    root: Annotated[Path, typer.Argument(file_okay=False)],
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8765,
    token_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    tls_cert: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    tls_key: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Serve the operator UI and authenticated iOS control/synchronization API."""

    token = token_file.read_text(encoding="utf-8").strip() if token_file is not None else None
    if token_file is not None and not token:
        raise typer.BadParameter("token file cannot be empty")
    server = make_capture_studio_server(
        root,
        host=host,
        port=port,
        token=token,
        tls_cert=tls_cert,
        tls_key=tls_key,
    )
    scheme = "https" if tls_cert is not None else "http"
    console.print(f"Capture Studio listening on {scheme}://{host}:{server.server_port}")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@app.command()
def doctor() -> None:
    """Report Python 3.14, GIL/JIT, accelerator, and ORT provider state."""

    _print_json(doctor_report())


@data_app.command("validate")
def validate_data(manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Validate the complete dataset contract and automatic quality gates."""

    result = materialize_dataset(manifest)
    _print_json(result.quality.model_dump(mode="json"))
    if result.quality.error_count:
        raise typer.Exit(code=1)


@data_app.command("materialize-labels")
def materialize_labels_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Deterministically materialize available or explicitly unavailable NTP labels."""

    result = materialize_dataset(manifest)
    if result.quality.error_count:
        _print_json({"output": None, "quality": result.quality.model_dump(mode="json")})
        raise typer.Exit(code=1)
    write_materialized_labels(result, output)
    _print_json({"output": output, "quality": result.quality.model_dump(mode="json")})


@data_app.command("schema")
def data_schema_command(
    kind: Annotated[
        Literal[
            "manifest",
            "capture-record",
            "label-catalog",
            "license-registry",
            "actor-splits",
            "capture-splits",
            "expression-cache",
            "expression-record",
            "capture-session",
            "raw-arkit-frame",
            "arkit-mapping",
            "frozen-capture-dataset",
        ],
        typer.Argument(),
    ] = "manifest",
) -> None:
    """Print the authoritative JSON schema for a versioned data contract."""

    models = {
        "manifest": DatasetManifest,
        "capture-record": CaptureRecord,
        "label-catalog": LabelCatalog,
        "license-registry": LicenseRegistry,
        "actor-splits": ActorSplitManifest,
        "capture-splits": CaptureSplitPlan,
        "expression-cache": ExpressionCacheManifest,
        "expression-record": ExpressionCacheRecord,
        "capture-session": CaptureSessionManifest,
        "raw-arkit-frame": RawArkitFrame,
        "arkit-mapping": ArkitMapping,
        "frozen-capture-dataset": FrozenCaptureDataset,
    }
    _print_json(models[kind].model_json_schema())


@data_app.command("validate-licenses")
def validate_licenses_command(
    registry: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    stage: Annotated[PipelineStage, typer.Option()],
    records: Annotated[str, typer.Option(help="Comma-separated license record IDs")],
    production: Annotated[bool, typer.Option()] = True,
) -> None:
    """Fail closed unless every requested source is admitted for the pipeline stage."""

    loaded = LicenseRegistry.load(registry)
    loaded.verify_local_license_texts(registry)
    admitted = loaded.admit(
        (record.strip() for record in records.split(",") if record.strip()),
        stage=stage,
        production=production,
    )
    _print_json(
        {
            "registry_revision": loaded.revision,
            "stage": stage,
            "production": production,
            "admitted_records": [record.record_id for record in admitted],
        }
    )


@data_app.command("split-actors")
def split_actors_command(
    index: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    seed: Annotated[int, typer.Option(min=0)] = 17,
    validation_actors: Annotated[int, typer.Option(min=1)] = 1,
    test_actors: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Build deterministic actor-isolated CREMA-D train/validation/test splits."""

    manifest = split_clips_by_actor(
        load_clip_index(index),
        seed=seed,
        validation_actors=validation_actors,
        test_actors=test_actors,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_json({"output": output, "splits": manifest.splits})


@data_app.command("split-captures")
def split_captures_command(
    records: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    held_out_test_devices: Annotated[str, typer.Option()],
    seed: Annotated[int, typer.Option(min=0)] = 17,
    validation_identities: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Split F captures by identity with explicit device-held-out test identities."""

    plan = split_captures(
        CaptureRecord.load_jsonl(records),
        seed=seed,
        held_out_test_devices={
            device.strip() for device in held_out_test_devices.split(",") if device.strip()
        },
        validation_identities=validation_identities,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_json({"output": output, "splits": plan.splits})


@data_app.command("capture-verify")
def capture_verify_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Verify a finalized local capture session and every declared chunk digest."""

    loaded = CaptureSessionManifest.load(manifest)
    loaded.verify_files(manifest.parent)
    _print_json(
        {
            "session_id": loaded.session_id,
            "manifest_sha256": loaded.manifest_sha256,
            "chunk_count": len(loaded.chunks),
            "mapping_revision": loaded.ntp_mapping_revision,
        }
    )


@data_app.command("capture-receive")
def capture_receive_command(
    receiver_root: Annotated[Path, typer.Argument(file_okay=False)],
    descriptor: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    payload: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Durably receive and verify one chunk before emitting its acknowledgement."""

    chunk = CaptureChunk.model_validate_json(descriptor.read_text(encoding="utf-8"))
    persisted = LocalChunkStore(receiver_root).receive_chunk(chunk, payload.read_bytes())
    acknowledgement = ChunkAcknowledgement(
        chunk_id=persisted.chunk_id,
        sha256=persisted.sha256,
    )
    _print_json(acknowledgement.model_dump(mode="json"))


@data_app.command("capture-receiver-index")
def capture_receiver_index_command(
    receiver_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Emit the durable receiver inventory used for manifest reconciliation."""

    acknowledgements = [
        ChunkAcknowledgement(chunk_id=chunk.chunk_id, sha256=chunk.sha256)
        for chunk in LocalChunkStore(receiver_root).chunks()
    ]
    _print_json([item.model_dump(mode="json") for item in acknowledgements])


@data_app.command("capture-pending")
def capture_pending_command(
    sender_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """List local chunks that remain durable and unacknowledged after reconnect."""

    pending = LocalChunkStore(sender_root).pending_chunks()
    _print_json([chunk.model_dump(mode="json") for chunk in pending])


@data_app.command("capture-reconcile")
def capture_reconcile_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    remote_index: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Compare durable receiver acknowledgements with a finalized capture session."""

    loaded = CaptureSessionManifest.load(manifest)
    payload = json.loads(remote_index.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise typer.BadParameter("remote index must be a JSON array")
    acknowledgements = [ChunkAcknowledgement.model_validate(item) for item in payload]
    result = reconcile_chunks(loaded.chunks, acknowledgements)
    _print_json(result.model_dump(mode="json") | {"complete": result.complete})
    if not result.complete:
        raise typer.Exit(code=1)


@data_app.command("capture-convert-arkit")
def capture_convert_arkit_command(
    raw_frames: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    mapping: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Regenerate versioned NTP teacher records from immutable raw ARKit frames."""

    loaded_mapping = ArkitMapping.load(mapping)
    records = convert_arkit_frames(RawArkitFrame.load_jsonl(raw_frames), loaded_mapping)
    write_capture_records(records, output)
    _print_json(
        {
            "output": output,
            "record_count": len(records),
            "mapping_revision": loaded_mapping.mapping_revision,
            "smoke_only": loaded_mapping.smoke_only,
        }
    )


@data_app.command("capture-freeze")
def capture_freeze_command(
    session_manifests: Annotated[
        str, typer.Option(help="Comma-separated finalized session.json paths")
    ],
    capture_records: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    arkit_mappings: Annotated[
        str, typer.Option(help="Comma-separated versioned ARKit mapping paths")
    ],
    license_registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    license_records: Annotated[str, typer.Option(help="Comma-separated admitted record IDs")],
    held_out_test_devices: Annotated[str, typer.Option()],
    data_revision: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    validation_identities: Annotated[int, typer.Option(min=1)] = 1,
    seed: Annotated[int, typer.Option(min=0)] = 17,
    smoke_only: Annotated[bool, typer.Option()] = False,
) -> None:
    """Freeze verified sessions and derived records behind license and group-split gates."""

    manifests = [Path(item.strip()) for item in session_manifests.split(",") if item.strip()]
    if not manifests or any(not path.is_file() for path in manifests):
        raise typer.BadParameter("every session manifest path must exist")
    frozen = freeze_capture_dataset(
        data_revision=data_revision,
        session_manifests=manifests,
        capture_records=capture_records,
        arkit_mappings=[Path(item.strip()) for item in arkit_mappings.split(",") if item.strip()],
        license_registry=license_registry,
        license_record_ids=[
            record.strip() for record in license_records.split(",") if record.strip()
        ],
        held_out_test_devices={
            device.strip() for device in held_out_test_devices.split(",") if device.strip()
        },
        validation_identities=validation_identities,
        seed=seed,
        smoke_only=smoke_only,
        output=output,
    )
    _print_json(
        {
            "output": output,
            "dataset_sha256": frozen.dataset_sha256,
            "splits": frozen.split_plan.splits,
            "smoke_only": frozen.smoke_only,
        }
    )


@data_app.command("capture-verify-frozen")
def capture_verify_frozen_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Re-verify a frozen capture dataset before any training entrypoint consumes it."""

    frozen = FrozenCaptureDataset.load(manifest)
    frozen.verify(manifest)
    _print_json(
        {
            "data_revision": frozen.data_revision,
            "dataset_sha256": frozen.dataset_sha256,
            "frozen": frozen.frozen,
            "smoke_only": frozen.smoke_only,
        }
    )


@data_app.command("capture-build-training-manifest")
def capture_build_training_manifest_command(
    frozen: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    label_catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Expose a verified frozen capture revision through the training DatasetManifest contract."""

    manifest = build_training_manifest_from_frozen(
        frozen,
        label_catalog_path=label_catalog,
        output=output,
    )
    _print_json(
        {
            "output": output,
            "data_revision": manifest.data_revision,
            "digest": manifest.digest,
            "smoke_only": manifest.smoke_only,
        }
    )


@data_app.command("capture-smoke")
def capture_smoke_command(
    work_dir: Annotated[Path, typer.Option("--work-dir", file_okay=False)] = Path(
        "runs/capture-smoke"
    ),
    mapping: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/data/arkit-to-ntp-v1-smoke.json"
    ),
    license_registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/data/license-registry.json"
    ),
    label_catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/data/ntp-v1-label-catalog.json"
    ),
) -> None:
    """Run the deterministic synthetic local capture-to-frozen-dataset closure."""

    _print_json(
        run_capture_pipeline_smoke(
            work_dir,
            mapping_path=mapping,
            license_registry=license_registry,
            label_catalog_path=label_catalog,
        )
    )


@evaluation_app.command("validate-standard")
def validate_evaluation_standard(
    standard: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate shared metrics, fixed sequences, and the benchmark report template."""

    validated = EvaluationStandard.load(standard)
    sequences, report = validated.validate_assets(standard)
    _print_json(
        {
            "standard_revision": validated.standard_revision,
            "profiles": [suite.profile for suite in validated.suites],
            "metric_count": len(validated.metrics),
            "fixed_sequence_count": len(sequences.sequences),
            "report_schema_version": report.schema_version,
        }
    )


@evaluation_app.command("benchmark-capture-store")
def benchmark_capture_store_command(
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    chunks: Annotated[int, typer.Option(min=8)] = 256,
    payload_bytes: Annotated[int, typer.Option(min=1024)] = 64 * 1024,
) -> None:
    """Measure durable local recording, verified streaming receive, ACK, and restart indexing."""

    _print_json(
        benchmark_capture_store(
            output,
            chunk_count=chunks,
            payload_bytes=payload_bytes,
        )
    )


@evaluation_app.command("report-schema")
def evaluation_report_schema() -> None:
    """Print the machine-readable benchmark report schema."""

    _print_json(BenchmarkReport.model_json_schema())


@evaluation_app.command("render-failures")
def render_failures_command(
    samples: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Render a local report from versioned failure-sample JSONL."""

    loaded = FailureSample.load_jsonl(samples)
    render_failure_report(loaded, output)
    _print_json({"output": output, "sample_count": len(loaded)})


@evaluation_app.command("validate-face-basic-acceptance")
def validate_face_basic_acceptance_command(
    bundle: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate all Issue #7 production evidence as one digest-pinned bundle."""

    result = validate_face_basic_acceptance(bundle)
    _print_json(result.model_dump(mode="json"))
    if not result.passed:
        raise typer.Exit(code=1)


@app.command("train")
def train_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    resume: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Train a registered model and save a reproducible checkpoint."""

    result = train_model(load_config(config), resume=resume)
    _print_json(
        {
            "run_dir": result.run_dir,
            "checkpoint": result.checkpoint,
            "final_step": result.final_step,
            "final_loss": result.final_loss,
        }
    )


@app.command("evaluate")
def evaluate_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Evaluate a checkpoint and write per-head metrics."""

    _print_json(evaluate_model(load_config(config), checkpoint, output_path=output))


@app.command("export")
def export_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Export and parity-check a complete ONNX model package."""

    _print_json(create_model_package(load_config(config), checkpoint, output))


@app.command("verify-export")
def verify_export_command(
    package: Annotated[Path, typer.Option("--package", exists=True, file_okay=False)],
) -> None:
    """Verify package contents, digest, and ORT fixed-vector parity."""

    _print_json(verify_model_package(package))


@app.command("calibrate-level-a")
def calibrate_level_a_command(
    capture: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    package: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    user_slot: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Fit a resettable Level A profile from neutral/range/confidence NPZ arrays."""

    verify_model_package(package)
    contract = ModelPackageMetadata.model_validate_json(
        (package / "runtime-metadata.json").read_text(encoding="utf-8")
    )
    with np.load(capture) as values:
        profile = fit_level_a_calibration(
            values["neutral"],
            values["range"],
            values["confidence"],
            user_slot=user_slot,
            model_family=contract.model_family,
            model_version=contract.model_version,
            feature_revision=contract.feature_revision,
            signal_registry_revision=contract.signal_registry_revision,
            normalization_revision=contract.normalization_revision,
            calibration_revision=contract.calibration_revision,
        )
    profile.save(output)
    _print_json({"output": output, "user_slot": user_slot, "signal_count": len(profile.signals)})


@app.command("fit-confidence-calibration")
def fit_confidence_calibration_command(
    evidence: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    package: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Fit held-out per-signal monotonic confidence calibration."""

    verify_model_package(package)
    metadata = ModelPackageMetadata.model_validate_json(
        (package / "runtime-metadata.json").read_text(encoding="utf-8")
    )
    with np.load(evidence) as values:
        calibration = fit_confidence_calibration(
            values["predicted"],
            values["correct"],
            signal_ids=metadata.supported_signals,
            model_family=metadata.model_family,
            model_version=metadata.model_version,
            signal_registry_revision=metadata.signal_registry_revision,
        )
    calibration.save(output)
    _print_json({"output": output, "signal_count": len(calibration.curves)})


@app.command("train-level-b-adapter")
def train_level_b_adapter_command(
    capture: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    package: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    user_slot: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    steps: Annotated[int, typer.Option(min=1)] = 200,
) -> None:
    """Train and verify a bounded offline residual adapter without touching the base encoder."""

    verify_model_package(package)
    metadata = ModelPackageMetadata.model_validate_json(
        (package / "runtime-metadata.json").read_text(encoding="utf-8")
    )
    with np.load(capture) as values:
        adapter = train_level_b_adapter(
            values["base_values"],
            values["target_values"],
            values["confidence"],
            output,
            user_slot=user_slot,
            base_model_family=metadata.model_family,
            base_model_version=metadata.model_version,
            base_model_digest=metadata.model_digest,
            feature_revision=metadata.feature_revision,
            signal_registry_revision=metadata.signal_registry_revision,
            normalization_revision=metadata.normalization_revision,
            calibration_revision=metadata.calibration_revision,
            signal_ids=metadata.supported_signals,
            steps=steps,
            smoke_only=metadata.smoke_only,
        )
    parity = verify_level_b_adapter(
        output,
        user_slot=user_slot,
        base_model_family=metadata.model_family,
        base_model_version=metadata.model_version,
        base_model_digest=metadata.model_digest,
        feature_revision=metadata.feature_revision,
    )
    _print_json({"output": output, "metadata": adapter, "parity": parity})


@app.command("benchmark-face-basic")
def benchmark_face_basic_command(
    package: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    providers: Annotated[str, typer.Option()] = "CPUExecutionProvider",
    warmup: Annotated[int, typer.Option(min=1)] = 20,
    iterations: Annotated[int, typer.Option(min=1)] = 200,
    tensorrt_fp16: Annotated[bool, typer.Option()] = False,
) -> None:
    """Benchmark a FaceBasic package on explicitly selected target providers."""

    report = benchmark_face_basic_package(
        package,
        output,
        providers=[provider.strip() for provider in providers.split(",") if provider.strip()],
        warmup=warmup,
        iterations=iterations,
        tensorrt_fp16=tensorrt_fp16,
    )
    _print_json(report)


@app.command("benchmark-roi-preprocess")
def benchmark_roi_preprocess_command(
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    input_width: Annotated[int, typer.Option(min=1)] = 1280,
    input_height: Annotated[int, typer.Option(min=1)] = 720,
    roi_side: Annotated[int, typer.Option(min=1)] = 640,
    output_sizes: Annotated[str, typer.Option()] = "64,96,128",
    roi_positions: Annotated[int, typer.Option(min=1)] = 32,
    frames_per_roi: Annotated[int, typer.Option(min=1)] = 5,
    warmup: Annotated[int, typer.Option(min=1)] = 100,
    iterations: Annotated[int, typer.Option(min=1)] = 2_000,
) -> None:
    """Benchmark bounded moving-ROI preprocessing independently of model inference."""

    sizes = tuple(int(size.strip()) for size in output_sizes.split(",") if size.strip())
    _print_json(
        benchmark_rgb_roi_preprocessor(
            output,
            input_width=input_width,
            input_height=input_height,
            roi_side=roi_side,
            output_sizes=sizes,
            roi_positions=roi_positions,
            frames_per_roi=frames_per_roi,
            warmup=warmup,
            iterations=iterations,
        )
    )


@app.command("benchmark-face-spatial")
def benchmark_face_spatial_command(
    package: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    providers: Annotated[str, typer.Option()] = "CPUExecutionProvider",
    warmup: Annotated[int, typer.Option(min=1)] = 20,
    iterations: Annotated[int, typer.Option(min=1)] = 200,
    tensorrt_fp16: Annotated[bool, typer.Option()] = False,
) -> None:
    """Benchmark a FaceSpatial package on explicitly selected target providers."""

    report = benchmark_face_spatial_package(
        package,
        output,
        providers=[provider.strip() for provider in providers.split(",") if provider.strip()],
        warmup=warmup,
        iterations=iterations,
        tensorrt_fp16=tensorrt_fp16,
    )
    _print_json(report)


@app.command("benchmark-full-set")
def benchmark_full_set_command(
    package: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    providers: Annotated[str, typer.Option()] = "CPUExecutionProvider",
    warmup: Annotated[int, typer.Option(min=1)] = 20,
    iterations: Annotated[int, typer.Option(min=1)] = 200,
    tensorrt_fp16: Annotated[bool, typer.Option()] = False,
) -> None:
    """Benchmark the low-cadence FullSet upper-body package."""

    report = benchmark_full_set_package(
        package,
        output,
        providers=[provider.strip() for provider in providers.split(",") if provider.strip()],
        warmup=warmup,
        iterations=iterations,
        tensorrt_fp16=tensorrt_fp16,
    )
    _print_json(report)


@app.command("benchmark-temporal")
def benchmark_temporal_command(
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    frames: Annotated[int, typer.Option(min=100)] = 2_000,
    seed: Annotated[int, typer.Option(min=0)] = 31,
) -> None:
    """Benchmark causal refinement jitter, peak retention, and overhead."""

    _print_json(benchmark_temporal_refiner(output, frames=frames, seed=seed))


@app.command("benchmark-expression-ablation")
def benchmark_expression_ablation_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Run the complete synthetic smoke-only frozen-F to expression ablation suite."""

    _print_json(run_expression_ablation_smoke(ExpressionAblationConfig.load(config), output))


@app.command("smoke")
def smoke_command(
    work_dir: Annotated[Path, typer.Option("--work-dir", file_okay=False)] = Path("runs/smoke"),
) -> None:
    """Run the synthetic train/resume/evaluate/export/verify pipeline."""

    config = ExperimentConfig.model_validate(
        {
            "data": {"samples": 8, "batch_size": 4},
            "model": {"name": "smoke"},
            "training": {"seed": 7, "max_steps": 1, "device": "cpu"},
            "evaluation": {"atol": 1e-5, "rtol": 1e-4},
            "export": {"model_family": "nana-smoke", "model_version": "0.0.0-smoke"},
            "reproducibility": {
                "output_dir": work_dir / "runs",
                "data_revision": "synthetic-v1",
                "ntp_schema_revision": "smoke-ntp-v0",
                "signal_registry_revision": "smoke-signals-v0",
                "feature_revision": "smoke-features-v1",
            },
        }
    )
    initial = train_model(config)
    resumed_config = config.model_copy(
        update={"training": config.training.model_copy(update={"max_steps": 2})}
    )
    resumed = train_model(resumed_config, resume=initial.checkpoint)
    evaluation = evaluate_model(resumed_config, resumed.checkpoint)
    package_dir = work_dir / "model-package"
    parity = create_model_package(resumed_config, resumed.checkpoint, package_dir)
    verification = verify_model_package(package_dir)
    _print_json(
        {
            "checkpoint": resumed.checkpoint,
            "final_step": resumed.final_step,
            "evaluation": evaluation,
            "export_parity": parity,
            "package_verification": verification,
            "package": package_dir,
            "warning": "Synthetic smoke evidence is not face-tracking acceptance evidence.",
        }
    )


@app.command("benchmark-python")
def benchmark_python_command(
    output: Annotated[Path, typer.Option("--output")] = Path(
        "artifacts/benchmarks/python-executors.json"
    ),
    items: Annotated[int, typer.Option(min=1)] = 32,
    rounds: Annotated[int, typer.Option(min=1)] = 5_000,
    workers: Annotated[int, typer.Option(min=1)] = 2,
    buffersize: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    """Compare bounded Python 3.14 executor throughput without setting a pass threshold."""

    report = benchmark_backends(
        items=items,
        rounds=rounds,
        workers=workers,
        buffersize=buffersize,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json({"output": output, "results": report})


def main() -> None:
    app()


if __name__ == "__main__":
    main()
