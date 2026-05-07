#!/usr/bin/env bash
# Build the ROS 2 workspace and launch the passive simulation for visualisation.
# Run this from anywhere — it locates the project root automatically.
# Works whether or not a Python venv is currently active.
#
# Usage:
#   bash scripts/build_and_launch.sh                          # passive sim (default)
#   bash scripts/build_and_launch.sh --launch training        # passive sim (explicit)
#   bash scripts/build_and_launch.sh --launch eval_gazebo \
#       --model models/sac_best.zip --algo sac                # full Gazebo eval
#   bash scripts/build_and_launch.sh --build-only             # just build, don't launch

set -euo pipefail

# ── locate repo root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS="${REPO_ROOT}/quarter_car_ws"

# ── parse args ────────────────────────────────────────────────────────────────
LAUNCH_TARGET="training"
MODEL_PATH=""
ALGO="sac"
BUILD_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --launch)      LAUNCH_TARGET="$2"; shift 2 ;;
        --model)       MODEL_PATH="$2";    shift 2 ;;
        --algo)        ALGO="$2";          shift 2 ;;
        --build-only)  BUILD_ONLY=1;       shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── strip venv from PATH so ros2 / colcon use the system Python ───────────────
# (an active .venv shadows importlib.metadata and breaks ros2cli)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo ">>> Temporarily removing venv from PATH for ROS 2 commands"
    export PATH="${PATH//${VIRTUAL_ENV}\/bin:/}"
    unset VIRTUAL_ENV
    unset PYTHONHOME 2>/dev/null || true
fi

# ── source ROS 2 Humble ───────────────────────────────────────────────────────
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ERROR: ROS 2 Humble not found at /opt/ros/humble/setup.bash"
    exit 1
fi
source /opt/ros/humble/setup.bash
echo ">>> Sourced ROS 2 Humble"

# ── install quarter_car_core into the system / user Python ───────────────────
# This lets ROS 2 nodes import it without any sys.path tricks.
echo ">>> Installing quarter_car_core into user Python"
pip3 install --user -q -e "${WS}/src/quarter_car_core"

# ── build the ROS 2 workspace ─────────────────────────────────────────────────
echo ">>> Building workspace (colcon --symlink-install)"
cd "${WS}"
colcon build --symlink-install \
    --packages-select quarter_car_sim quarter_car_controllers \
    --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20

# ── source the install overlay ───────────────────────────────────────────────
source "${WS}/install/setup.bash"
echo ">>> Workspace sourced"

[[ $BUILD_ONLY -eq 1 ]] && { echo ">>> Build complete (--build-only)."; exit 0; }

# ── launch ────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

case "${LAUNCH_TARGET}" in
    training)
        echo ""
        echo ">>> Launching passive simulation (50 Hz, speed_bump, Gazebo disabled)"
        echo "    rqt_plot will show /car/acceleration + /car/comfort_score + /car/reward"
        echo ""
        ros2 launch quarter_car_sim training.launch.py
        ;;

    eval_gazebo)
        if [[ -z "${MODEL_PATH}" ]]; then
            echo "ERROR: --model <path_to_model.zip> is required for eval_gazebo"
            exit 1
        fi
        echo ""
        echo ">>> Launching Gazebo evaluation"
        echo "    model : ${MODEL_PATH}"
        echo "    algo  : ${ALGO}"
        echo ""
        ros2 launch quarter_car_sim eval_gazebo.launch.py \
            model_path:="${MODEL_PATH}" \
            algo:="${ALGO}"
        ;;

    compare)
        if [[ -z "${MODEL_PATH}" ]]; then
            echo "ERROR: --model <path_to_model.zip> is required for compare"
            exit 1
        fi
        echo ""
        echo ">>> Launching controller comparison (RL vs passive vs LQR stub vs MPC stub)"
        echo ""
        ros2 launch quarter_car_sim compare.launch.py \
            model_path:="${MODEL_PATH}" \
            algo:="${ALGO}"
        ;;

    *)
        echo "ERROR: Unknown launch target '${LAUNCH_TARGET}'"
        echo "       Valid targets: training | eval_gazebo | compare"
        exit 1
        ;;
esac
