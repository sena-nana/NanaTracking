"""Local capture-store performance smoke evidence."""

from __future__ import annotations

import json
import os
import platform
import statistics
import tempfile
import time
from io import BytesIO
from pathlib import Path

from nana_tracking.data.capture import CaptureChunk, ChunkAcknowledgement, LocalChunkStore


def benchmark_capture_store(
    output: Path,
    *,
    chunk_count: int = 256,
    payload_bytes: int = 64 * 1024,
) -> dict[str, object]:
    if chunk_count < 8 or payload_bytes < 1024:
        raise ValueError("capture benchmark requires at least 8 chunks of at least 1024 bytes")
    payload = bytes(range(256)) * (payload_bytes // 256) + bytes(range(payload_bytes % 256))
    write_latencies_ns: list[int] = []
    receive_latencies_ns: list[int] = []
    acknowledge_latencies_ns: list[int] = []
    with tempfile.TemporaryDirectory(prefix="nana-capture-benchmark-") as temporary:
        root = Path(temporary)
        sender = LocalChunkStore(root / "sender")
        chunks: list[CaptureChunk] = []
        wall_start = time.perf_counter_ns()
        for sequence in range(chunk_count):
            started = time.perf_counter_ns()
            chunks.append(
                sender.write_chunk(
                    chunk_id=f"chunk-{sequence:06d}",
                    take_id="take-benchmark",
                    kind="arkit",
                    sequence_start=sequence,
                    sequence_end=sequence,
                    capture_timestamp_start_ns=sequence * 16_666_667,
                    capture_timestamp_end_ns=sequence * 16_666_667,
                    payload=payload,
                )
            )
            write_latencies_ns.append(time.perf_counter_ns() - started)
        write_wall_ns = time.perf_counter_ns() - wall_start

        torn_tail = b'{"chunk_id":"crash-torn-tail"'
        chunks_journal = root / "sender" / ".capture-state" / "chunks.jsonl"
        with chunks_journal.open("ab") as stream:
            stream.write(torn_tail)
            stream.flush()
            os.fsync(stream.fileno())
        started = time.perf_counter_ns()
        reopened = LocalChunkStore(root / "sender")
        restart_index_ns = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        pending = reopened.pending_chunks()
        pending_scan_ns = time.perf_counter_ns() - started
        if pending != chunks:
            raise ValueError("capture benchmark restart changed the pending chunk inventory")
        if not chunks_journal.read_bytes().endswith(b"\n"):
            raise ValueError("capture benchmark did not repair the crash-torn journal tail")

        receiver = LocalChunkStore(root / "receiver")
        wall_start = time.perf_counter_ns()
        for chunk in chunks:
            started = time.perf_counter_ns()
            persisted = receiver.receive_chunk_stream(chunk, BytesIO(payload))
            receive_latencies_ns.append(time.perf_counter_ns() - started)
            acknowledgement = ChunkAcknowledgement(
                chunk_id=persisted.chunk_id,
                sha256=persisted.sha256,
            )
            started = time.perf_counter_ns()
            reopened.acknowledge(acknowledgement)
            acknowledge_latencies_ns.append(time.perf_counter_ns() - started)
        sync_wall_ns = time.perf_counter_ns() - wall_start
        if reopened.pending_chunks():
            raise ValueError("capture benchmark did not close every acknowledgement")

    total_bytes = chunk_count * payload_bytes
    report: dict[str, object] = {
        "schema": "nana-capture-store-benchmark/1.1.0",
        "smoke_only": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "chunk_count": chunk_count,
        "payload_bytes": payload_bytes,
        "total_payload_bytes": total_bytes,
        "local_write": _latency_report(write_latencies_ns, write_wall_ns, total_bytes),
        "verified_receive": _latency_report(
            receive_latencies_ns,
            sync_wall_ns,
            total_bytes,
        ),
        "acknowledgement": _latency_report(
            acknowledge_latencies_ns,
            sum(acknowledge_latencies_ns),
            0,
        ),
        "restart_index_ms": restart_index_ns / 1_000_000,
        "restart_recovered_torn_tail_bytes": len(torn_tail),
        "pending_scan_ms": pending_scan_ns / 1_000_000,
        "design": {
            "payload_upload": "bounded binary stream",
            "journal": "append-only fsync with startup-only crash-tail recovery",
            "lookup": "startup index plus constant-time ID/path checks",
            "preview": "single latest slot outside durable chunks",
        },
        "warning": (
            "Synthetic filesystem smoke does not prove iOS flash, Windows disk, LAN, or production "
            "capture throughput."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _latency_report(latencies_ns: list[int], wall_ns: int, total_bytes: int) -> dict[str, float]:
    ordered = sorted(latencies_ns)
    return {
        "mean_ms": statistics.fmean(ordered) / 1_000_000,
        "p50_ms": _percentile(ordered, 0.50) / 1_000_000,
        "p95_ms": _percentile(ordered, 0.95) / 1_000_000,
        "p99_ms": _percentile(ordered, 0.99) / 1_000_000,
        "throughput_mib_s": (
            0.0 if total_bytes == 0 else (total_bytes / (1024 * 1024)) / (wall_ns / 1_000_000_000)
        ),
    }


def _percentile(ordered: list[int], fraction: float) -> int:
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]
