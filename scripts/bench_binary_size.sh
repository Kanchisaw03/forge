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

stat_bytes() {
  if command -v stat >/dev/null 2>&1; then
    stat -c%s "$1"
  else
    wc -c < "$1" | tr -d ' '
  fi
}

format_kb() {
  awk -v b="$1" 'BEGIN {printf "%.2f", b / 1024.0}'
}

format_mb() {
  awk -v b="$1" 'BEGIN {printf "%.2f", b / (1024.0 * 1024.0)}'
}

time_run() {
  local out
  if command -v /usr/bin/time >/dev/null 2>&1; then
    out=$(/usr/bin/time -p "$@" 2>&1 >/dev/null)
  else
    out=$( { time -p "$@" >/dev/null; } 2>&1 )
  fi
  echo "$out" | awk '/^real/ {print $2; exit}'
}

min_time=999999
for _ in 1 2 3; do
  t=$(time_run "$BIN" --help)
  if awk -v a="$t" -v b="$min_time" 'BEGIN {exit !(a < b)}'; then
    min_time="$t"
  fi
done

BIN_BYTES=$(stat_bytes "$BIN")
BIN_KB=$(format_kb "$BIN_BYTES")
BIN_MB=$(format_mb "$BIN_BYTES")
MIN_MS=$(awk -v s="$min_time" 'BEGIN {printf "%.2f", s * 1000.0}')

KERNEL_SRC="$ROOT_DIR/examples/matmul.forge"
OUT_PATH=$("$BIN" compile "$KERNEL_SRC")
if [ ! -f "$OUT_PATH" ]; then
  echo "ERROR: compiled kernel output not found at $OUT_PATH" >&2
  exit 1
fi

KERNEL_BYTES=$(stat_bytes "$OUT_PATH")
KERNEL_KB=$(format_kb "$KERNEL_BYTES")
KERNEL_MB=$(format_mb "$KERNEL_BYTES")

CPU_MODEL=$(awk -F: '/model name/ {gsub(/^ /, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u "+%Y-%m-%dT%H:%M:%SZ")

RESULTS_FILE="$RESULTS_DIR/binary_size.txt"
{
  echo "timestamp=$TIMESTAMP"
  echo "cpu_model=$CPU_MODEL"
  echo "forge_driver_bytes=$BIN_BYTES"
  echo "forge_driver_kb=$BIN_KB"
  echo "forge_driver_mb=$BIN_MB"
  echo "forge_driver_cold_start_min_ms=$MIN_MS"
  echo "kernel_output_path=$OUT_PATH"
  echo "kernel_output_bytes=$KERNEL_BYTES"
  echo "kernel_output_kb=$KERNEL_KB"
  echo "kernel_output_mb=$KERNEL_MB"
  echo ""
  echo "Reference sizes (MB):"
  echo "- CUDA runtime: 2400"
  echo "- PyTorch CPU: 750"
  echo "- llama.cpp: 12"
  echo "- Forge output: ${KERNEL_MB}"
} > "$RESULTS_FILE"

echo "Wrote $RESULTS_FILE"
