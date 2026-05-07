from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
SIZES = (256, 512, 1024)
WARMUP = 5
ITERS = 20
FORGE_RESULT_RE = re.compile(
    r"FORGE_RESULT:\s+size=(\d+)x\1\s+latency_ms=([0-9.]+)\s+gflops=([0-9.]+)"
)
CPU_RE = re.compile(r"^\s*CPU:\s*(.+)$", re.MULTILINE)


def benchmark_numpy() -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}

    for size in SIZES:
        a = (np.arange(size * size, dtype=np.float32) % 17) * 0.1
        b = (np.arange(size * size, dtype=np.float32) % 11) * 0.2
        a = a.reshape((size, size))
        b = b.reshape((size, size))

        for _ in range(WARMUP):
            _ = np.matmul(a, b)

        latencies = []
        for _ in range(ITERS):
            start = time.perf_counter()
            _ = np.matmul(a, b)
            latencies.append(time.perf_counter() - start)

        mean_s = sum(latencies) / len(latencies)
        gflops = (2.0 * size**3) / (mean_s * 1e9)
        results[size] = {
            "mean_ms": mean_s * 1000.0,
            "gflops": gflops,
        }

    return results


def benchmark_forge() -> tuple[str, dict[int, dict[str, float]]]:
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"

    command = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "-p",
        "forge-driver",
        "--",
        "bench",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        message = completed.stdout.strip()
        if completed.stderr.strip():
            message = f"{message}\n{completed.stderr.strip()}".strip()
        raise SystemExit(message or "Forge benchmark failed")

    cpu_match = CPU_RE.search(completed.stdout)
    cpu = cpu_match.group(1).strip() if cpu_match else "unknown"

    results: dict[int, dict[str, float]] = {}
    for size_str, latency_str, gflops_str in FORGE_RESULT_RE.findall(completed.stdout):
        size = int(size_str)
        results[size] = {
            "mean_ms": float(latency_str),
            "gflops": float(gflops_str),
        }

    missing = [size for size in SIZES if size not in results]
    if missing:
        raise SystemExit(f"Forge benchmark output missing sizes: {missing}")

    return cpu, results


def format_table(cpu: str, forge: dict[int, dict[str, float]], numpy: dict[int, dict[str, float]]) -> str:
    lines = [
        "Forge vs NumPy SGEMM Benchmark",
        f"CPU: {cpu}",
        "",
        "  Size          Forge Latency   Forge GFLOPS   NumPy Latency   NumPy GFLOPS   NumPy/Forge",
        "  ───────────────────────────────────────────────────────────────────────────────────────",
    ]

    for size in SIZES:
        forge_row = forge[size]
        numpy_row = numpy[size]
        ratio = numpy_row["gflops"] / forge_row["gflops"] if forge_row["gflops"] else 0.0
        lines.append(
            f"  {size}×{size:<8} {forge_row['mean_ms']:>11.2f} ms   {forge_row['gflops']:>10.1f}   "
            f"{numpy_row['mean_ms']:>11.2f} ms   {numpy_row['gflops']:>10.1f}   {ratio:>11.2f}x"
        )

    lines.extend(
        [
            "  ───────────────────────────────────────────────────────────────────────────────────────",
            f"  Warmup: {WARMUP}  Timed: {ITERS}  Native CPU flags enabled via RUSTFLAGS=\"-C target-cpu=native\"",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    forge_cpu, forge_results = benchmark_forge()
    numpy_results = benchmark_numpy()

    output = format_table(forge_cpu, forge_results, numpy_results)
    print(output)

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "forge_numpy_compare.txt").write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
