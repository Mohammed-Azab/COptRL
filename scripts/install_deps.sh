#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Installing quarter_car_core (pure Python, editable) ==="
pip install -e "${REPO_ROOT}/quarter_car_ws/src/quarter_car_core"

echo "=== Installing Python training requirements ==="
pip install -r "${REPO_ROOT}/requirements.txt"

echo ""
echo "=== Done ==="
echo ""
echo "For ROS2 packages (requires ROS2 Humble sourced):"
echo "  cd ${REPO_ROOT}/quarter_car_ws && colcon build"
echo ""
echo "Verify zero-ROS import:"
echo "  python -c \"from quarter_car_core import quarter_car_env; print('OK')\""
