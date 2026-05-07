#!/usr/bin/env bash
set -euo pipefail

ALGO=${1:-sac}
STEPS=${2:-500000}
ROAD=${3:-iso_8608_class_c}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== Training ${ALGO} for ${STEPS} timesteps on ${ROAD} road ==="
python training/train.py \
  --algo    "${ALGO}" \
  --timesteps "${STEPS}" \
  --road    "${ROAD}" \
  --eval-road speed_bump
