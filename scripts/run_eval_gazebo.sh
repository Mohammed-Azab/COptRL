#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?"Usage: $0 <model_path> [algo] [road_profile]"}
ALGO=${2:-sac}
ROAD=${3:-speed_bump}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "${REPO_ROOT}/quarter_car_ws/install/setup.bash"

echo "=== Launching Gazebo evaluation ==="
echo "    model: ${MODEL_PATH}"
echo "    algo:  ${ALGO}"
echo "    road:  ${ROAD}"
echo ""
echo "NOTE: No RViz2. Gazebo is the only 3D view. rqt_plot shows signals."
echo ""

ros2 launch quarter_car_sim eval_gazebo.launch.py \
  model_path:="${MODEL_PATH}" \
  algo:="${ALGO}" \
  road_profile:="${ROAD}"
