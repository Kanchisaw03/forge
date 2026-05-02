#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime

try:
    import torch
except Exception:
    print("torch is not installed. Install with: python3 -m pip install torch", file=sys.stderr)
    sys.exit(1)


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    idx = int((p * n + 99) / 100) - 1
    idx = max(0, min(idx, n - 1))
    return values[idx]


def stats_for_size(n):
    a = (torch.arange(n * n, dtype=torch.float32) % 17) * 0.1
    b = (torch.arange(n * n, dtype=torch.float32) % 11) * 0.2
    a = a.view(n, n)
    b = b.view(n, n)

    with torch.no_grad():
        for _ in range(5):
            _ = torch.mm(a, b)

        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            _ = torch.mm(a, b)
            end = time.perf_counter()
            latencies.append(end - start)

    mean_s = sum(latencies) / len(latencies)
    flops = 2.0 * (n ** 3)
    gflops = (flops / mean_s) / 1e9
    mean_ms = mean_s * 1000.0
    p50_ms = percentile([v * 1000.0 for v in latencies], 50)
    p95_ms = percentile([v * 1000.0 for v in latencies], 95)
    p99_ms = percentile([v * 1000.0 for v in latencies], 99)
    return {
        "size": n,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "gflops": gflops,
    }


def format_row(size, tool, mean_ms, p50_ms, p95_ms, p99_ms, gflops):
    return f"{size:<8} {tool:<11} {mean_ms:>9.3f} {p50_ms:>9.3f} {p95_ms:>9.3f} {p99_ms:>9.3f} {gflops:>9.3f}"


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"MKL available: {torch.backends.mkl.is_available()}")

    rows = [stats_for_size(n) for n in (256, 512, 1024)]

    header = "Size     Tool         Mean(ms)   P50(ms)    P95(ms)    P99(ms)    GFLOPS"
    table_lines = [header]
    for row in rows:
        size_label = f"{row['size']}x{row['size']}"
        table_lines.append(
            format_row(
                size_label,
                "PyTorch",
                row["mean_ms"],
                row["p50_ms"],
                row["p95_ms"],
                row["p99_ms"],
                row["gflops"],
            )
        )

    cpu = cpu_model()
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    results_path = os.path.join(results_dir, "torch_results.txt")
    with open(results_path, "w", encoding="utf-8") as handle:
        handle.write(f"timestamp={timestamp}\n")
        handle.write(f"cpu_model={cpu}\n")
        handle.write("tool=PyTorch\n")
        handle.write(f"torch_version={torch.__version__}\n")
        handle.write(f"threads={torch.get_num_threads()}\n")
        handle.write(f"mkl_available={torch.backends.mkl.is_available()}\n")
        for row in rows:
            handle.write(
                "size={size} mean_ms={mean_ms:.3f} p50_ms={p50_ms:.3f} p95_ms={p95_ms:.3f} "
                "p99_ms={p99_ms:.3f} gflops={gflops:.3f}\n".format(**row)
            )
        handle.write("\n")
        handle.write("\n".join(table_lines))
        handle.write("\n")

    print("\n".join(table_lines))


if __name__ == "__main__":
    main()
