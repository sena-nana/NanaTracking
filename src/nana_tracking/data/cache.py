"""Internal Zstandard cache helpers; model packages do not use this format."""

import json
from compression import zstd
from pathlib import Path
from typing import Any


def write_json_zstd(payload: Any, path: Path, *, level: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(zstd.compress(raw, level=level))


def read_json_zstd(path: Path) -> Any:
    return json.loads(zstd.decompress(path.read_bytes()))
