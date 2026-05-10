default:
    @just --list

# Install Python deps and the gym_env package
install:
    pip install -r requirements.txt
    pip install -e gym_env/

# (Re)install only the gym_env package
build-gym-env:
    pip install -e gym_env/

# Train an RL agent
# Usage: just train algo=sac timesteps=500000 road=iso_8608_class_c seed=66
train algo="sac" timesteps="500000" road="iso_8608_class_c" eval-road="speed_bump" seed="66":
    python training/train.py \
        --algo {{algo}} \
        --timesteps {{timesteps}} \
        --road {{road}} \
        --eval-road {{eval-road}} \
        --seed {{seed}}

# Evaluate a trained model
# Usage: just eval algo=sac road=speed_bump [model_path=path/to/model.zip]
eval algo="sac" road="speed_bump" model_path="" episodes="3":
    python training/evaluate.py \
        --algo {{algo}} \
        --road {{road}} \
        --episodes {{episodes}} \
        {{ if model_path != "" { "--model_path " + model_path } else { "" } }}
