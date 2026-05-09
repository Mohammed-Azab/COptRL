#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Installing quarter_car_core ==="
pip install -e "${REPO_ROOT}/quarter_car_ws/src/quarter_car_core"

echo "=== Installing Python training requirements ==="
pip install -r "${REPO_ROOT}/requirements.txt"

echo ""
echo "=== Done ==="
echo ""