#!/usr/bin/env python3
import datetime
import os
import subprocess
import sys


def repo_url(root_dir: str) -> str:
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


def months_since_first_commit(root_dir: str) -> int:
    try:
        out = subprocess.check_output(
            ["git", "log", "--reverse", "--format=%cI"],
            cwd=root_dir,
            text=True,
        ).splitlines()
    except Exception:
        return 1
    if not out:
        return 1
    first = out[0].strip().replace("Z", "+00:00")
    start = datetime.datetime.fromisoformat(first)
    now = datetime.datetime.now(tz=start.tzinfo)
    months = (now.year - start.year) * 12 + (now.month - start.month)
    return max(1, months)


def parse_comparison(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    cpu_model = "unknown"
    date = "unknown"
    for line in lines:
        if "SGEMM Benchmark Comparison" in line:
            clean = line.replace("\u2551", "").strip()
            parts = clean.split("\u2014")
            if len(parts) >= 3:
                cpu_model = parts[1].strip()
                date = parts[2].strip()
            break

    gflops = {}
    for line in lines:
        if "256" not in line:
            continue
        clean = line.replace("\u2551", "").replace("\u00d7", "x").strip()
        if not clean.startswith("256x256"):
            continue
        parts = clean.split()
        if not parts:
            continue
        size = parts[0]
        if size != "256x256":
            continue
        idx = None
        for i in range(1, len(parts)):
            try:
                float(parts[i])
                idx = i
                break
            except ValueError:
                continue
        if idx is None or idx + 1 >= len(parts):
            continue
        tool = " ".join(parts[1:idx])
        latency = parts[idx]
        gflops_val = parts[idx + 1]
        gflops[tool] = gflops_val

    return cpu_model, date, gflops


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(root_dir, "results")
    comparison_path = os.path.join(results_dir, "comparison.txt")
    if not os.path.exists(comparison_path):
        print("comparison.txt not found. Run scripts/bench_all.sh first.", file=sys.stderr)
        sys.exit(1)

    cpu, date, gflops = parse_comparison(comparison_path)
    forge = gflops.get("Forge", "0.0")
    numpy = gflops.get("NumPy (MKL)", "0.0")
    torch = gflops.get("PyTorch CPU", "0.0")
    months = months_since_first_commit(root_dir)
    repo = repo_url(root_dir)

    linkedin = (
        f"I spent {months} months building a compiler that doesn't need a GPU.\n\n"
        "Everyone said: just use CUDA. just use PyTorch.\n\n"
        "Here's what happened when I benchmarked it:\n\n"
        f"Forge (CPU, no GPU):  {forge} GFLOPS\n"
        f"NumPy MKL (CPU):      {numpy} GFLOPS\n"
        f"PyTorch CPU:          {torch} GFLOPS\n\n"
        f"256x256 SGEMM. {cpu}. No GPU anywhere.\n\n"
        "7 crates. 8 optimizer passes. Written in Rust.\n"
        "Output is human-readable Rust source you can inspect.\n\n"
        "GitHub in the comments.\n\n"
        "What's the most expensive infrastructure assumption\n"
        "your team has never questioned?"
    )

    twitter = (
        "1/6 Built a CPU-only kernel compiler in Rust (Forge).\n"
        f"256x256 SGEMM: Forge {forge} GFLOPS, NumPy {numpy} GFLOPS, PyTorch {torch} GFLOPS.\n\n"
        "2/6 No GPU, no CUDA runtime, just AVX2+FMA.\n"
        f"CPU: {cpu}.\n\n"
        "3/6 Forge compiles kernels to human-readable Rust source with AVX2 intrinsics.\n"
        "SSA IR, 8-pass optimizer, and rustc as a validation gate.\n\n"
        "4/6 Bench method: 5 warmup + 20 timed iterations per measurement.\n"
        "Same matrix sizes across tools.\n\n"
        "5/6 If you're CPU-bound or GPU-scarce, this is the path I wanted.\n"
        "Fast, debuggable, deployable on commodity servers.\n\n"
        f"6/6 Repo: {repo}"
    )

    hn = (
        f"Title: Forge: AVX2 CPU kernel compiler, {forge} GFLOPS on 256x256 SGEMM\n"
        "Body:\n"
        "Forge is a Rust compiler that lowers kernels to typed SSA, runs an 8-pass optimizer,\n"
        "and emits AVX2 intrinsics as Rust source validated by rustc.\n\n"
        f"Bench: 256x256 SGEMM on {cpu}.\n"
        f"Forge {forge} GFLOPS | NumPy {numpy} GFLOPS | PyTorch {torch} GFLOPS.\n\n"
        f"Repo: {repo}"
    )

    reddit_rust = (
        "Title: Forge: a Rust AVX2 kernel compiler (CPU-only, no GPU)\n\n"
        f"Bench numbers (256x256 SGEMM on {cpu}):\n"
        f"Forge {forge} GFLOPS | NumPy {numpy} GFLOPS | PyTorch {torch} GFLOPS\n\n"
        "Forge compiles kernels through SSA IR and an 8-pass optimizer, then emits Rust + AVX2.\n"
        "Output is readable, and rustc is the validation gate.\n\n"
        f"Repo: {repo}"
    )

    reddit_ml = (
        "Title: CPU-only kernel compiler vs NumPy and PyTorch (256x256 SGEMM)\n\n"
        f"I benchmarked Forge (a Rust AVX2 kernel compiler) on {cpu}.\n"
        f"256x256 SGEMM: Forge {forge} GFLOPS | NumPy {numpy} GFLOPS | PyTorch {torch} GFLOPS.\n\n"
        "Forge compiles kernels to typed SSA, runs 8 optimizer passes, and emits AVX2 Rust.\n"
        "No GPU or CUDA runtime required.\n\n"
        f"Repo: {repo}"
    )

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "linkedin_post.txt"), "w", encoding="utf-8") as handle:
        handle.write(linkedin + "\n")
    with open(os.path.join(results_dir, "twitter_thread.txt"), "w", encoding="utf-8") as handle:
        handle.write(twitter + "\n")
    with open(os.path.join(results_dir, "hn_post.txt"), "w", encoding="utf-8") as handle:
        handle.write(hn + "\n")
    with open(os.path.join(results_dir, "reddit_rust.txt"), "w", encoding="utf-8") as handle:
        handle.write(reddit_rust + "\n")
    with open(os.path.join(results_dir, "reddit_ml.txt"), "w", encoding="utf-8") as handle:
        handle.write(reddit_ml + "\n")

    print("Generated social assets in results/")


if __name__ == "__main__":
    main()
