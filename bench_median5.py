"""
Forge vs NumPy SGEMM — Median-of-5 Benchmark
==============================================
Runs Forge (cargo run --release bench) and NumPy matmul 5 independent times
each and reports the **median** GFLOPS / latency for every matrix size.

Usage:
    python bench_median5.py
"""
from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import io
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
SIZES = (256, 512, 1024)
WARMUP = 5
ITERS = 20
RUNS = 5  # number of full benchmark passes for the median

FORGE_RESULT_RE = re.compile(
    r"FORGE_RESULT:\s+size=(\d+)x\1\s+latency_ms=([0-9.]+)\s+gflops=([0-9.]+)"
)
CPU_RE = re.compile(r"^\s*CPU:\s*(.+)$", re.MULTILINE)

BORDER = "=" * 90
THIN   = "-" * 90


# ── NumPy ────────────────────────────────────────────────────────────────────

def numpy_single_run() -> dict[int, dict[str, float]]:
    """One complete NumPy benchmark pass over all sizes."""
    results: dict[int, dict[str, float]] = {}

    for size in SIZES:
        a = (np.arange(size * size, dtype=np.float32) % 17) * 0.1
        b = (np.arange(size * size, dtype=np.float32) % 11) * 0.2
        a = a.reshape((size, size))
        b = b.reshape((size, size))

        # warmup
        for _ in range(WARMUP):
            _ = np.matmul(a, b)

        latencies: list[float] = []
        for _ in range(ITERS):
            start = time.perf_counter()
            _ = np.matmul(a, b)
            latencies.append(time.perf_counter() - start)

        mean_s = sum(latencies) / len(latencies)
        gflops = (2.0 * size ** 3) / (mean_s * 1e9)
        results[size] = {"mean_ms": mean_s * 1000.0, "gflops": gflops}

    return results


def benchmark_numpy_median() -> dict[int, dict[str, float]]:
    """Run NumPy RUNS times and return median metrics per size."""
    all_runs: dict[int, list[dict[str, float]]] = {s: [] for s in SIZES}

    for i in range(RUNS):
        print(f"  NumPy  run {i + 1}/{RUNS} …", end=" ", flush=True)
        single = numpy_single_run()
        for size in SIZES:
            all_runs[size].append(single[size])
        gf = ", ".join(f"{single[s]['gflops']:.1f}" for s in SIZES)
        print(f"GFLOPS [{', '.join(f'{s}' for s in SIZES)}] = [{gf}]")

    median_results: dict[int, dict[str, float]] = {}
    for size in SIZES:
        gflops_list = [r["gflops"] for r in all_runs[size]]
        latency_list = [r["mean_ms"] for r in all_runs[size]]
        median_results[size] = {
            "median_ms": statistics.median(latency_list),
            "median_gflops": statistics.median(gflops_list),
            "all_gflops": gflops_list,
        }
    return median_results


# ── Forge ────────────────────────────────────────────────────────────────────

def forge_single_run() -> tuple[str, dict[int, dict[str, float]]]:
    """One complete Forge benchmark pass (invokes cargo)."""
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"

    command = [
        "cargo", "run", "--release", "--quiet",
        "-p", "forge-driver", "--", "bench",
    ]
    completed = subprocess.run(
        command, cwd=ROOT_DIR, env=env,
        capture_output=True, text=True,
    )

    if completed.returncode != 0:
        msg = (completed.stdout.strip() + "\n" + completed.stderr.strip()).strip()
        raise SystemExit(f"Forge benchmark failed:\n{msg}")

    cpu_match = CPU_RE.search(completed.stdout)
    cpu = cpu_match.group(1).strip() if cpu_match else "unknown"

    results: dict[int, dict[str, float]] = {}
    for size_str, latency_str, gflops_str in FORGE_RESULT_RE.findall(completed.stdout):
        size = int(size_str)
        results[size] = {
            "mean_ms": float(latency_str),
            "gflops": float(gflops_str),
        }

    missing = [s for s in SIZES if s not in results]
    if missing:
        raise SystemExit(f"Forge output missing sizes: {missing}\n{completed.stdout}")

    return cpu, results


def benchmark_forge_median() -> tuple[str, dict[int, dict[str, float]]]:
    """Run Forge RUNS times and return median metrics per size."""
    all_runs: dict[int, list[dict[str, float]]] = {s: [] for s in SIZES}
    cpu = "unknown"

    for i in range(RUNS):
        print(f"  Forge  run {i + 1}/{RUNS} …", end=" ", flush=True)
        cpu, single = forge_single_run()
        for size in SIZES:
            all_runs[size].append(single[size])
        gf = ", ".join(f"{single[s]['gflops']:.1f}" for s in SIZES)
        print(f"GFLOPS [{', '.join(f'{s}' for s in SIZES)}] = [{gf}]")

    median_results: dict[int, dict[str, float]] = {}
    for size in SIZES:
        gflops_list = [r["gflops"] for r in all_runs[size]]
        latency_list = [r["mean_ms"] for r in all_runs[size]]
        median_results[size] = {
            "median_ms": statistics.median(latency_list),
            "median_gflops": statistics.median(gflops_list),
            "all_gflops": gflops_list,
        }
    return cpu, median_results


# ── Report ───────────────────────────────────────────────────────────────────

def format_report(
    cpu: str,
    forge: dict[int, dict[str, float]],
    numpy_res: dict[int, dict[str, float]],
) -> str:
    lines = [
        BORDER,
        "  Forge vs NumPy SGEMM — Median of 5 Runs",
        f"  CPU: {cpu}",
        BORDER,
        "",
        f"  {'Size':<12} {'Forge ms':>10} {'Forge GF/s':>12} {'NumPy ms':>10} {'NumPy GF/s':>12} {'Forge/NumPy':>13}",
        "  " + THIN[2:],
    ]

    for size in SIZES:
        f = forge[size]
        n = numpy_res[size]
        ratio = f["median_gflops"] / n["median_gflops"] if n["median_gflops"] else 0.0
        tag = f"{ratio:.2f}x"
        if ratio >= 1.0:
            tag = f"{tag} ✓"
        lines.append(
            f"  {size}×{size:<8} {f['median_ms']:>10.2f} {f['median_gflops']:>12.1f}"
            f" {n['median_ms']:>10.2f} {n['median_gflops']:>12.1f} {tag:>13}"
        )

    lines.append("  " + THIN[2:])
    lines.append("")

    # per-run detail
    lines.append("  Per-run GFLOPS detail (all 5 runs):")
    for size in SIZES:
        f_all = ", ".join(f"{g:.1f}" for g in forge[size]["all_gflops"])
        n_all = ", ".join(f"{g:.1f}" for g in numpy_res[size]["all_gflops"])
        lines.append(f"    {size}×{size}  Forge: [{f_all}]")
        lines.append(f"    {' ' * len(f'{size}×{size}')}  NumPy: [{n_all}]")

    lines.extend(["", BORDER])
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Force UTF-8 output on Windows to avoid cp1252 issues
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(BORDER)
    print("  Building Forge (release) …")
    print(BORDER)

    # Pre-build so compilation time doesn't affect run 1
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    subprocess.run(
        ["cargo", "build", "--release", "-p", "forge-driver"],
        cwd=ROOT_DIR, env=env,
        capture_output=True, text=True,
    )

    print()
    print("  Running Forge benchmark (5 passes) …")
    cpu, forge_results = benchmark_forge_median()

    print()
    print("  Running NumPy benchmark (5 passes) …")
    numpy_results = benchmark_numpy_median()

    print()
    report = format_report(cpu, forge_results, numpy_results)
    print(report)

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "median5_comparison.txt"
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
