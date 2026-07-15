"""Framework-neutral capture records for versioned multi-sensor datasets."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class RgbFrame(DataModel):
    uri: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    exposure_duration_ns: int = Field(gt=0)
    iso: FiniteFloat = Field(gt=0)
    frame_duration_ns: int = Field(gt=0)


class CameraCalibration(DataModel):
    intrinsics: tuple[
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
    ]
    distortion_model: Literal["none", "brown-conrady", "fisheye"]
    distortion_coefficients: list[FiniteFloat]


class CaptureConditions(DataModel):
    lighting: Literal["normal", "dim", "backlit", "mixed"]
    occlusions: set[
        Literal[
            "glasses",
            "headphones",
            "hair",
            "hand",
            "clothing",
            "self_occlusion",
            "out_of_frame",
        ]
    ] = Field(default_factory=set)


class LabelObservation(DataModel):
    value: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    state: Literal["observed", "fused", "predicted", "unavailable"]
    evidence: Literal["teacher_label", "geometry", "derived"]
    method: str = Field(min_length=1)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.state == "unavailable":
            if self.value is not None or self.confidence != 0.0 or not self.unavailable_reason:
                raise ValueError(
                    "unavailable observations require no value, zero confidence, and a reason"
                )
        elif self.value is None or self.unavailable_reason is not None:
            raise ValueError("usable observations require a value and no unavailable reason")
        return self


class TeacherFrame(DataModel):
    source_id: str = Field(min_length=1)
    capture_timestamp_ns: int = Field(ge=0)
    labels: dict[str, LabelObservation]


class DepthObservation(DataModel):
    source_id: str = Field(min_length=1)
    capture_timestamp_ns: int = Field(ge=0)
    state: Literal["observed", "fused", "predicted", "unavailable"]
    confidence: float = Field(ge=0.0, le=1.0)
    depth_uri: str | None = None
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.state == "unavailable":
            if self.depth_uri is not None or self.confidence != 0.0 or not self.unavailable_reason:
                raise ValueError("unavailable depth requires no URI, zero confidence, and a reason")
        elif self.depth_uri is None or self.unavailable_reason is not None:
            raise ValueError("depth observations require a URI and no unavailable reason")
        return self


class CaptureRecord(DataModel):
    schema_version: Literal["ntp-capture/1.0.0"] = "ntp-capture/1.0.0"
    record_id: str = Field(min_length=1)
    identity_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    environment_id: str | None = None
    action_script_id: str | None = None
    consent_record_id: str | None = None
    human_review_status: Literal["pending", "approved", "rejected"] | None = None
    capture_timestamp_ns: int = Field(ge=0)
    sequence: int = Field(ge=0)
    rgb: RgbFrame
    camera: CameraCalibration
    teachers: list[TeacherFrame] = Field(min_length=1)
    depth: list[DepthObservation] = Field(default_factory=list)
    conditions: CaptureConditions

    @model_validator(mode="after")
    def validate_teacher_uniqueness(self) -> Self:
        source_ids = [teacher.source_id for teacher in self.teachers]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("a capture record may contain at most one frame per teacher source")
        return self

    @classmethod
    def load_jsonl(cls, path: Path) -> list[Self]:
        records: list[Self] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    records.append(cls.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(f"invalid capture record at {path}:{line_number}") from error
        if not records:
            raise ValueError(f"capture record file is empty: {path}")
        return records
