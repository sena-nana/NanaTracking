"""Durable append-only JSONL journals with crash-tail recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel


def load_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    """Load a journal, repairing only its final non-newline-terminated record."""

    if not path.is_file():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        line_start = payload.rfind(b"\n") + 1
        tail = payload[line_start:]
        try:
            model.model_validate_json(tail)
        except ValueError:
            _truncate_and_sync(path, line_start)
            payload = payload[:line_start]
        else:
            _append_and_sync(path, b"\n")
            payload += b"\n"

    values: list[T] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(model.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid journal entry at {path}:{line_number}") from error
    return values


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    _append_and_sync(path, encoded)
    if created:
        _sync_directory(path.parent)


def _append_and_sync(path: Path, payload: bytes) -> None:
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _truncate_and_sync(path: Path, size: int) -> None:
    with path.open("r+b") as stream:
        stream.truncate(size)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
