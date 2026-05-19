default:
    @just --list

# Install Python deps and the gym_env package
install:
    pip install -r requirements.txt
    pip install -e gym_env/

# (Re)install only the gym_env package
build-gym-env:
    pip install -e src/gym_env/

# Run the test suite (isolates from broken ROS pytest plugins)
test:
    PYTHONPATH=src/gym_env python -m pytest tests/ -v -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint
