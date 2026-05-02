#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS_DIR="$ROOT_DIR/results"

cat <<'EOF'
Forge Benchmark Suite
- Runs Forge, NumPy, and PyTorch SGEMM benchmarks
- Captures binary size and compile speed metrics
- Generates social media assets
Estimated runtime: 10-20 minutes on a typical laptop
EOF

echo "Checking prerequisites..."
missing=0

if ! command -v cargo >/dev/null 2>&1; then
  echo "Missing cargo. Install via https://rustup.rs" >&2
  missing=1
fi
if ! command -v rustc >/dev/null 2>&1; then
  echo "Missing rustc. Install via https://rustup.rs" >&2
  missing=1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Missing python3. Install Python 3.8+ and ensure python3 is on PATH." >&2
  missing=1
else
  if python3 - <<'PY'
import sys
if sys.version_info < (3, 8):
    raise SystemExit(1)
PY
  then
    :
  else
    echo "Python 3.8+ is required." >&2
    missing=1
  fi
fi

if ! grep -q avx2 /proc/cpuinfo; then
  echo "Missing AVX2 CPU support (grep avx2 /proc/cpuinfo)." >&2
  missing=1
fi

if command -v python3 >/dev/null 2>&1; then
  if python3 - <<'PY'
try:
    import numpy  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    :
  else
    echo "NumPy missing. Install with: python3 -m pip install numpy" >&2
    missing=1
  fi

  if python3 - <<'PY'
try:
    import torch  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    :
  else
    echo "PyTorch missing. Install with: python3 -m pip install torch" >&2
    missing=1
  fi
fi

if [ "$missing" -ne 0 ]; then
  echo "Prerequisite check failed. Fix the issues above and re-run." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

"$ROOT_DIR/scripts/bench_all.sh"
"$ROOT_DIR/scripts/bench_binary_size.sh"
"$ROOT_DIR/scripts/bench_compile_speed.sh"
python3 "$ROOT_DIR/scripts/generate_social_assets.py"

echo "All benchmarks complete. Your results are in results/"
echo "Your social media posts are ready in results/"
echo "Run: cat results/linkedin_post.txt"
