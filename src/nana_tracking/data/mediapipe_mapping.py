"""Optional MediaPipe proposal helper; outputs can never become frame supervision."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torchvision.io import ImageReadMode, decode_image, write_png
from torchvision.utils import draw_keypoints

from nana_tracking.data.multiface import (
    AnchorVote,
    SEMANTIC_ANCHORS,
    SemanticAnchorMapping,
)

MEDIAPIPE_SEMANTIC_INDICES = (
    133,
    33,
    362,
    263,
    107,
    70,
    336,
    300,
    1,
    98,
    327,
    61,
    291,
    13,
    14,
    152,
)


class MediaPipeReviewSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    identity_id: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    image: Path
    projected_mesh: Path


def _load_review_index(path: Path) -> list[MediaPipeReviewSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("MediaPipe review index must be a non-empty JSON array")
    return [MediaPipeReviewSample.model_validate(item) for item in payload]


def generate_mediapipe_anchor_votes(
    *,
    review_index: Path,
    model_asset: Path,
    expected_model_sha256: str,
    output: Path,
) -> dict[str, object]:
    """Run pinned MediaPipe only to propose nearest Multiface vertices."""

    actual_model_sha256 = hashlib.sha256(model_asset.read_bytes()).hexdigest()
    if actual_model_sha256 != expected_model_sha256:
        raise ValueError("MediaPipe Face Landmarker model digest mismatch")
    try:
        mp: Any = importlib.import_module("mediapipe")
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe mapping helper is optional; run with --extra anchor-mapping"
        ) from error
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_asset))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    samples = _load_review_index(review_index)
    votes: list[AnchorVote] = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for sample in samples:
            image = mp.Image.create_from_file(str(sample.image))
            result: Any = landmarker.detect(image)
            if len(result.face_landmarks) != 1:
                raise ValueError(f"MediaPipe did not find exactly one face: {sample.sample_id}")
            mesh = cast(
                NDArray[np.float64],
                np.asarray(
                    np.load(sample.projected_mesh, allow_pickle=False),
                    dtype=np.float64,
                ),
            )
            if mesh.shape != (7_306, 2) or not np.isfinite(mesh).all():
                raise ValueError(
                    f"projected Multiface mesh must have shape (7306, 2): {sample.sample_id}"
                )
            landmarks = result.face_landmarks[0]
            for semantic_name, mediapipe_index in zip(
                SEMANTIC_ANCHORS,
                MEDIAPIPE_SEMANTIC_INDICES,
                strict=True,
            ):
                point = np.asarray(
                    [
                        cast(float, landmarks[mediapipe_index].x),
                        cast(float, landmarks[mediapipe_index].y),
                    ],
                    dtype=np.float64,
                )
                distances = np.linalg.norm(mesh - point, axis=1)
                vertex_id = int(np.argmin(distances))
                votes.append(
                    AnchorVote(
                        semantic_name=semantic_name,
                        mediapipe_index=mediapipe_index,
                        multiface_vertex_id=vertex_id,
                        distance=float(distances[vertex_id]),
                        identity_id=sample.identity_id,
                        expression=sample.expression,
                        camera_id=sample.camera_id,
                        sample_id=sample.sample_id,
                    )
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(vote.model_dump_json() + "\n" for vote in votes),
        encoding="utf-8",
    )
    return {
        "schema_version": "nana-mediapipe-anchor-votes/1.0.0",
        "output": str(output),
        "sample_count": len(samples),
        "vote_count": len(votes),
        "mediapipe_model_sha256": actual_model_sha256,
        "training_labels_created": False,
    }


def render_anchor_review_overlays(
    *,
    review_index: Path,
    mapping_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Render reviewed Multiface vertex locations; overlays are evidence, not labels."""

    mapping = SemanticAnchorMapping.load(mapping_path)
    if len(mapping.correspondences) != len(SEMANTIC_ANCHORS):
        raise ValueError("overlay rendering requires all 16 candidate correspondences")
    samples = _load_review_index(review_index)
    output_directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    outputs: list[str] = []
    vertex_ids = [item.multiface_vertex_id for item in mapping.correspondences]
    for sample in samples:
        image = decode_image(str(sample.image), mode=ImageReadMode.RGB)
        mesh = np.asarray(
            np.load(sample.projected_mesh, allow_pickle=False),
            dtype=np.float32,
        )
        if mesh.shape != (7_306, 2):
            raise ValueError(f"projected mesh has invalid shape: {sample.sample_id}")
        points = torch.from_numpy(mesh[vertex_ids].copy())
        points[:, 0] *= image.shape[2]
        points[:, 1] *= image.shape[1]
        overlay = draw_keypoints(
            image,
            points.unsqueeze(0),
            colors="red",
            radius=3,
        )
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", sample.sample_id)
        output = output_directory / f"{safe_name}.png"
        write_png(overlay, str(output))
        digest.update(output.name.encode())
        digest.update(output.read_bytes())
        outputs.append(str(output))
    return {
        "schema_version": "nana-anchor-overlay-review/1.0.0",
        "mapping_revision": mapping.mapping_revision,
        "sample_count": len(outputs),
        "overlay_sha256": digest.hexdigest(),
        "outputs": outputs,
        "training_labels_created": False,
    }
