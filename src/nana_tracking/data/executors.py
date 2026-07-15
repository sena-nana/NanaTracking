"""Python 3.14 bounded executor abstraction for pure-Python preprocessing."""

import json
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import InterpreterPoolExecutor, ProcessPoolExecutor
from functools import partial
from typing import Literal

ExecutorBackend = Literal["inline", "multiprocessing", "interpreter"]


def cpu_transform(value: int, *, rounds: int = 2_000) -> int:
    """Pickle-safe deterministic CPU work used for executor validation."""

    if value < 0:
        raise ValueError("values must be non-negative")
    state = value + 0x9E3779B9
    for _ in range(rounds):
        state = ((state ^ (state >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return state


def map_values(
    values: Iterable[int],
    *,
    backend: ExecutorBackend,
    workers: int = 2,
    buffersize: int = 2,
    rounds: int = 2_000,
) -> list[int]:
    transform = partial(cpu_transform, rounds=rounds)
    if backend == "inline":
        return [transform(value) for value in values]
    if backend == "interpreter":
        return _map_interpreter_broker(
            list(values),
            workers=workers,
            buffersize=buffersize,
            rounds=rounds,
        )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(transform, values, buffersize=buffersize))


def run_interpreter_pool(
    values: list[int],
    *,
    workers: int,
    buffersize: int,
    rounds: int,
) -> list[int]:
    """Run only inside the isolated broker process, never beside PyTorch autograd."""

    transform = partial(cpu_transform, rounds=rounds)
    with InterpreterPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(transform, values, buffersize=buffersize))


def _map_interpreter_broker(
    values: list[int],
    *,
    workers: int,
    buffersize: int,
    rounds: int,
) -> list[int]:
    request = json.dumps(
        {
            "values": values,
            "workers": workers,
            "buffersize": buffersize,
            "rounds": rounds,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-m", "nana_tracking.data.interpreter_worker"],
        input=request,
        check=True,
        capture_output=True,
        text=True,
    )
    response = json.loads(completed.stdout)
    if not response["ok"]:
        if response["error_type"] == "ValueError":
            raise ValueError(response["message"])
        raise RuntimeError(
            f"interpreter broker failed: {response['error_type']}: {response['message']}"
        )
    return [int(value) for value in response["values"]]


def benchmark_backends(
    *,
    items: int = 32,
    rounds: int = 5_000,
    workers: int = 2,
    buffersize: int = 2,
) -> dict[str, dict[str, float | int]]:
    # Keep these imports out of module initialization: ``_tracemalloc`` is not
    # subinterpreter-safe, while workers only need ``cpu_transform``.
    import time
    import tracemalloc

    values = list(range(items))
    baseline: list[int] | None = None
    report: dict[str, dict[str, float | int]] = {}
    for backend in ("inline", "multiprocessing", "interpreter"):
        tracemalloc.start()
        started = time.perf_counter()
        result = map_values(
            values,
            backend=backend,
            workers=workers,
            buffersize=buffersize,
            rounds=rounds,
        )
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if baseline is None:
            baseline = result
        elif result != baseline:
            raise RuntimeError(f"{backend} produced results that differ from inline")
        report[backend] = {
            "items": items,
            "elapsed_seconds": elapsed,
            "items_per_second": items / elapsed,
            "peak_traced_bytes": peak,
        }
    return report
