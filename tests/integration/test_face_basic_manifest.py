import hashlib
import json
from pathlib import Path

import torch
from torchvision.io import write_png

from nana_tracking.config import load_config
from nana_tracking.data.labeling import LabelCatalog
from nana_tracking.data.loaders import create_loader
from nana_tracking.data.manifest import DatasetManifest, dataset_digest
from nana_tracking.data.schema import (
    CaptureConditions,
    CaptureRecord,
    LabelObservation,
    RgbFrame,
    TeacherFrame,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_manifest(tmp_path: Path) -> Path:
    catalog_path = Path("configs/data/ntp-v1-label-catalog.json").resolve()
    catalog = LabelCatalog.load(catalog_path)
    examples = CaptureRecord.load_jsonl(Path("examples/records/synthetic-capture-v1.jsonl"))
    records: list[CaptureRecord] = []
    pose = {
        name: LabelObservation(
            value=value,
            confidence=0.95,
            state="observed",
            evidence="geometry",
            method="head-camera-pose/1.0.0",
        )
        for name, value in {
            "head.pose.position.x": 0.0,
            "head.pose.position.y": 0.0,
            "head.pose.position.z": 0.1,
            "head.pose.orientation.x": 0.0,
            "head.pose.orientation.y": 0.0,
            "head.pose.orientation.z": 0.0,
            "head.pose.orientation.w": 1.0,
        }.items()
    }
    basic = {
        signal.stable_name: LabelObservation(
            value=0.0,
            confidence=0.9,
            state="observed",
            evidence="derived",
            method="ntp-face-orthogonal/1.0.0",
        )
        for signal in catalog.signals[:36]
    }
    for index, example in enumerate(examples):
        image_path = tmp_path / f"frame-{index}.png"
        image = torch.full((3, 64, 64), index * 32, dtype=torch.uint8)
        write_png(image, str(image_path))
        rgb = RgbFrame(
            uri=str(image_path),
            width=64,
            height=64,
            exposure_duration_ns=1_000_000,
            iso=100.0,
            frame_duration_ns=16_666_667,
        )
        teacher = TeacherFrame(
            source_id="synthetic-truedepth",
            capture_timestamp_ns=example.capture_timestamp_ns,
            labels={**basic, **pose},
        )
        records.append(
            example.model_copy(
                update={
                    "rgb": rgb,
                    "teachers": [teacher],
                    "depth": [],
                    "conditions": CaptureConditions(lighting="normal"),
                }
            )
        )
    record_path = tmp_path / "records.jsonl"
    record_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )

    payload = json.loads(Path("examples/manifests/synthetic-v1.json").read_text(encoding="utf-8"))
    payload["data_revision"] = "synthetic-face-basic-loader-v1"
    payload["digest"] = "0" * 64
    payload["label_catalog"] = {"path": str(catalog_path), "sha256": _sha256(catalog_path)}
    payload["record_files"] = [
        {"path": str(record_path), "sha256": _sha256(record_path), "record_count": 3}
    ]
    manifest = DatasetManifest.model_validate(payload)
    manifest = manifest.model_copy(update={"digest": dataset_digest(manifest)})
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def test_manifest_loader_preserves_complete_basic_pose_and_identity_split(
    tmp_path: Path,
) -> None:
    manifest = _build_manifest(tmp_path)
    config = load_config(Path("configs/face-basic-smoke.yaml"))
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"dataset": "manifest", "manifest": manifest, "batch_size": 1}
            )
        }
    )
    batch = next(iter(create_loader(config, split="train", shuffle=False)))
    assert batch.images.shape == (1, 3, 64, 64)
    assert batch.targets["rig"].shape == (1, 36)
    assert batch.targets["pose"].shape == (1, 7)
    assert batch.targets["visibility"].item() == 0
    assert batch.targets["identity"].item() == 0
    assert torch.all(batch.label_confidence["rig"] == 0.9)
    assert torch.all(batch.label_confidence["pose"] == 0.95)
