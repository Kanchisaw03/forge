#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS_DIR="$ROOT_DIR/results"
mkdir -p "$RESULTS_DIR"

"$ROOT_DIR/scripts/bench_forge.sh"
python3 "$ROOT_DIR/scripts/bench_numpy.py"
python3 "$ROOT_DIR/scripts/bench_torch.py"

ROOT_DIR="$ROOT_DIR" python3 - <<'PY'
import datetime
import os
import subprocess

root_dir = os.environ.get("ROOT_DIR", os.getcwd())
results_dir = os.path.join(root_dir, "results")


def load_results(path):
    data = {}
    cpu = "unknown"
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("cpu_model="):
                cpu = line.split("=", 1)[1]
            if line.startswith("size="):
                parts = line.split()
                size = int(parts[0].split("=", 1)[1])
                row = {}
                for part in parts[1:]:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        row[key] = value
                data[size] = row
    return cpu, data


def repo_url():
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=root_dir,
            text=True,
        ).strip()
    except Exception:
        return "github.com/unknown/unknown"
    if url.startswith("git@github.com:"):
        url = url[len("git@github.com:"):]
    if url.startswith("https://github.com/"):
        url = url[len("https://github.com/"):]
    if url.endswith(".git"):
        url = url[:-4]
    return f"github.com/{url}"


def fmt_latency(value):
    return f"{float(value):.2f}"


def fmt_gflops(value):
    return f"{float(value):.1f}"


cpu_forge, forge = load_results(os.path.join(results_dir, "forge_results.txt"))
_, numpy_res = load_results(os.path.join(results_dir, "numpy_results.txt"))
_, torch_res = load_results(os.path.join(results_dir, "torch_results.txt"))

cpu = cpu_forge or "unknown"
date = datetime.datetime.utcnow().strftime("%Y-%m-%d")

sizes = [256, 512]

rows = []
for size in sizes:
    rows.append((
        f"{size}\u00d7{size}",
        "Forge",
        fmt_latency(forge[size]["mean_ms"]),
        fmt_gflops(forge[size]["gflops"]),
    ))
    rows.append((
        f"{size}\u00d7{size}",
        "NumPy (MKL)",
        fmt_latency(numpy_res[size]["mean_ms"]),
        fmt_gflops(numpy_res[size]["gflops"]),
    ))
    rows.append((
        f"{size}\u00d7{size}",
        "PyTorch CPU",
        fmt_latency(torch_res[size]["mean_ms"]),
        fmt_gflops(torch_res[size]["gflops"]),
    ))

def make_row(size, tool, latency, gflops, gpu="No"):
    return f"\u2551  {size:<8}  {tool:<14}  {latency:>12}  {gflops:>7}  {gpu:^10} \u2551"

header = make_row("Size", "Tool", "Latency (ms)", "GFLOPS", "GPU Needed")
width = len(header)

line_top = "\u2554" + "\u2550" * (width - 2) + "\u2557"
line_mid = "\u2560" + "\u2550" * (width - 2) + "\u2563"
line_bot = "\u255a" + "\u2550" * (width - 2) + "\u255d"

title_text = f"SGEMM Benchmark Comparison \u2014 {cpu} \u2014 {date}"
max_title = width - 4
if len(title_text) > max_title:
    title_text = title_text[: max_title - 3] + "..."
title = f"\u2551  {title_text:<{width - 4}} \u2551"

lines = [
    line_top,
    title,
    line_mid,
    header,
    line_mid,
]

for idx, (size, tool, latency, gflops) in enumerate(rows):
    lines.append(make_row(size, tool, latency, gflops))
    if idx == 2:
        lines.append(line_mid)

lines.append(line_mid)
lines.append(f"\u2551  {'Test: 5 warmup + 20 timed iterations per measurement':<{width - 4}} \u2551")
lines.append(f"\u2551  {'Build: RUSTFLAGS=\"-C target-cpu=native\" cargo build':<{width - 4}} \u2551")
lines.append(line_bot)

comparison_path = os.path.join(results_dir, "comparison.txt")
with open(comparison_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines))
    handle.write("\n")

forge_256 = fmt_gflops(forge[256]["gflops"])
numpy_256 = fmt_gflops(numpy_res[256]["gflops"])
torch_256 = fmt_gflops(torch_res[256]["gflops"])
repo = repo_url()

snippet_path = os.path.join(results_dir, "social_snippet.txt")
with open(snippet_path, "w", encoding="utf-8") as handle:
    handle.write(f"Forge: {forge_256} GFLOPS | NumPy: {numpy_256} GFLOPS | PyTorch: {torch_256} GFLOPS\n")
    handle.write(f"256\u00d7256 SGEMM, no GPU, {cpu}, AVX2+FMA\n")
    handle.write(f"{repo}\n")

print("Wrote results/comparison.txt and results/social_snippet.txt")
PY
