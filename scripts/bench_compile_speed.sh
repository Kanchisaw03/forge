#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS_DIR="$ROOT_DIR/results"
mkdir -p "$RESULTS_DIR"

( cd "$ROOT_DIR" && RUSTFLAGS="-C target-cpu=native" cargo build --release -p forge-driver )

BIN="$ROOT_DIR/target/release/forge-driver"
if [ ! -x "$BIN" ]; then
  echo "ERROR: forge-driver binary not found at $BIN" >&2
  exit 1
fi

KERNEL_SRC="$ROOT_DIR/examples/matmul.forge"
if [ ! -f "$KERNEL_SRC" ]; then
  echo "ERROR: kernel source not found at $KERNEL_SRC" >&2
  exit 1
fi

time_run() {
  local out
  if command -v /usr/bin/time >/dev/null 2>&1; then
    out=$(/usr/bin/time -p "$@" 2>&1 >/dev/null)
  else
    out=$( { time -p "$@" >/dev/null; } 2>&1 )
  fi
  echo "$out" | awk '/^real/ {print $2; exit}'
}

runs=()
for _ in 1 2 3 4 5; do
  t=$(time_run "$BIN" compile "$KERNEL_SRC")
  runs+=("$t")
done

mean_s=$(printf '%s\n' "${runs[@]}" | awk '{sum+=$1} END {printf "%.6f", sum/NR}')
min_s=$(printf '%s\n' "${runs[@]}" | sort -n | head -n 1)
mean_ms=$(awk -v s="$mean_s" 'BEGIN {printf "%.2f", s * 1000.0}')
min_ms=$(awk -v s="$min_s" 'BEGIN {printf "%.2f", s * 1000.0}')

CPU_MODEL=$(awk -F: '/model name/ {gsub(/^ /, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u "+%Y-%m-%dT%H:%M:%SZ")

RESULTS_FILE="$RESULTS_DIR/compile_speed.txt"
{
  echo "timestamp=$TIMESTAMP"
  echo "cpu_model=$CPU_MODEL"
  echo "forge_compile_mean_ms=$mean_ms"
  echo "forge_compile_min_ms=$min_ms"
  echo "runs_seconds=$(IFS=,; echo "${runs[*]}")"
  echo ""
  echo "Reference compile times (ms):"
  echo "- Triton JIT first run: 4000-12000 (source: triton-lang.org docs)"
  echo "- torch.compile(): 2000-8000 (source: pytorch.org docs)"
} > "$RESULTS_FILE"

echo "Wrote $RESULTS_FILE"
