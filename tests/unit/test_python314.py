import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nana_tracking.data.cache import read_json_zstd, write_json_zstd
from nana_tracking.data.executors import ExecutorBackend, map_values


@pytest.mark.parametrize("backend", ["inline", "multiprocessing", "interpreter"])
def test_executor_backends_match(backend: ExecutorBackend) -> None:
    expected = map_values(range(4), backend="inline", rounds=100)
    assert map_values(range(4), backend=backend, rounds=100) == expected


def test_executor_worker_error_propagates() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        map_values([0, -1], backend="interpreter", rounds=10)


def test_executor_map_buffers_input() -> None:
    produced: list[int] = []

    def values():
        for value in range(100):
            produced.append(value)
            yield value

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.map(json.dumps, values(), buffersize=1)
        next(result)
        assert len(produced) < 100


def test_zstd_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cache.json.zst"
    payload = {"revision": "synthetic-v1", "values": [1, 2, 3]}
    write_json_zstd(payload, path)
    assert read_json_zstd(path) == payload
