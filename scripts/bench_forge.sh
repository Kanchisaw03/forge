#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS_DIR="$ROOT_DIR/results"
mkdir -p "$RESULTS_DIR"

if ! grep -q avx2 /proc/cpuinfo; then
  echo "ERROR: AVX2 not detected in /proc/cpuinfo." >&2
  exit 1
fi

check_ac_power() {
  local supply
  for supply in /sys/class/power_supply/*; do
    if [ -f "$supply/type" ] && grep -qi "mains" "$supply/type"; then
      if [ -f "$supply/online" ] && [ "$(cat "$supply/online")" != "1" ]; then
        echo "WARN: system is not on AC power; results may vary." >&2
      fi
      return
    fi
  done
  echo "WARN: unable to determine AC power state." >&2
}

check_ac_power

if command -v cpupower >/dev/null 2>&1; then
  if ! cpupower frequency-set -g performance >/dev/null 2>&1; then
    echo "WARN: cpupower present but unable to set performance governor." >&2
  fi
else
  echo "WARN: cpupower not installed; skipping governor tuning." >&2
fi

echo "Building Forge (release)..."
( cd "$ROOT_DIR" && cargo build --release -p forge-driver )

BIN="$ROOT_DIR/target/release/forge-driver"
if [ ! -x "$BIN" ]; then
  echo "ERROR: forge-driver binary not found at $BIN" >&2
  exit 1
fi

mean_of() {
  printf '%s\n' "$@" | awk '{sum+=$1} END {if (NR==0) {print "0.000"} else {printf "%.3f", sum/NR}}'
}

percentile_of() {
  local p="$1"
  shift
  local sorted
  mapfile -t sorted < <(printf '%s\n' "$@" | sort -n)
  local n=${#sorted[@]}
  if [ "$n" -eq 0 ]; then
    printf "0.000"
    return
  fi
  local idx=$(( (p * n + 99) / 100 - 1 ))
  if [ "$idx" -lt 0 ]; then
    idx=0
  fi
  if [ "$idx" -ge "$n" ]; then
    idx=$((n - 1))
  fi
  printf "%.3f" "${sorted[$idx]}"
}

parse_field() {
  local key="$1"
  local line="$2"
  echo "$line" | awk -v key="$key" -F'[ =]' '{for (i=1; i<=NF; i++) if ($i==key) {print $(i+1); exit}}'
}

declare -A mean_ms p50_ms p95_ms p99_ms mean_gflops runs_ms runs_gflops

run_size() {
  local size="$1"
  local ms_list=()
  local gflops_list=()

  for _ in 1 2 3; do
    local out
    out=$("$BIN" bench "$ROOT_DIR/README.md" --size "$size" --warmup 5 --iters 20 | grep "size=")
    local ms
    local gf
    ms=$(parse_field "latency_ms" "$out")
    gf=$(parse_field "gflops" "$out")
    ms_list+=("$ms")
    gflops_list+=("$gf")
  done

  local ms_csv
  local gf_csv
  ms_csv=$(IFS=,; echo "${ms_list[*]}")
  gf_csv=$(IFS=,; echo "${gflops_list[*]}")

  runs_ms[$size]="$ms_csv"
  runs_gflops[$size]="$gf_csv"
  mean_ms[$size]=$(mean_of "${ms_list[@]}")
  p50_ms[$size]=$(percentile_of 50 "${ms_list[@]}")
  p95_ms[$size]=$(percentile_of 95 "${ms_list[@]}")
  p99_ms[$size]=$(percentile_of 99 "${ms_list[@]}")
  mean_gflops[$size]=$(mean_of "${gflops_list[@]}")
}

for size in 256 512 1024; do
  echo "Running Forge SGEMM ${size}x${size}..."
  run_size "$size"
done

CPU_MODEL=$(awk -F: '/model name/ {gsub(/^ /, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u "+%Y-%m-%dT%H:%M:%SZ")

RESULTS_FILE="$RESULTS_DIR/forge_results.txt"
{
  echo "timestamp=$TIMESTAMP"
  echo "cpu_model=$CPU_MODEL"
  echo "tool=Forge"
  for size in 256 512 1024; do
    echo "size=$size runs_ms=${runs_ms[$size]} runs_gflops=${runs_gflops[$size]} mean_ms=${mean_ms[$size]} p50_ms=${p50_ms[$size]} p95_ms=${p95_ms[$size]} p99_ms=${p99_ms[$size]} gflops=${mean_gflops[$size]}"
  done
  echo ""
  printf "%s\n" "Size     Tool         Mean(ms)   P50(ms)    P95(ms)    P99(ms)    GFLOPS"
  for size in 256 512 1024; do
    printf "%s\n" "$(printf "%-8s %-11s %9s %9s %9s %9s %9s" "${size}x${size}" "Forge" "${mean_ms[$size]}" "${p50_ms[$size]}" "${p95_ms[$size]}" "${p99_ms[$size]}" "${mean_gflops[$size]}")"
  done
} > "$RESULTS_FILE"

echo "Forge SGEMM summary:"
printf "%s\n" "Size     Tool         Mean(ms)   P50(ms)    P95(ms)    P99(ms)    GFLOPS"
for size in 256 512 1024; do
  printf "%s\n" "$(printf "%-8s %-11s %9s %9s %9s %9s %9s" "${size}x${size}" "Forge" "${mean_ms[$size]}" "${p50_ms[$size]}" "${p95_ms[$size]}" "${p99_ms[$size]}" "${mean_gflops[$size]}")"
done
echo "Wrote $RESULTS_FILE"
