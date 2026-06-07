set dotenv-load
set shell := ["bash", "-c"]

venv := ".venv/bin/python3"
pip  := ".venv/bin/pip"

default:
    @just --list

# setup
install:
    {{pip}} install -r requirements.txt
    {{pip}} install -e src/gym_env

torch-cpu:
    {{pip}} install torch --index-url https://download.pytorch.org/whl/cpu

build-gym-env:
    {{pip}} install -e src/gym_env

install-dashboard:
    {{pip}} install optuna-dashboard

# train PPO — use --c for curriculum, e.g. just train speed_bump --c --n-envs 4
train road="speed_bump" *args="":
    PYTHONPATH=src {{venv}} src/train/train.py --algo PPO --road {{road}} \
        $(echo "{{args}}" | sed 's/--c\b/--curriculum/g')

# dummy constant-speed baseline
dummy *args="":
    PYTHONPATH=src {{venv}} src/train/dummy_agent/train.py {{args}}

# evaluate a trained model — e.g. just eval models/.../PPO_final.zip --save-plots
eval model *args="":
    PYTHONPATH=src {{venv}} src/eval/eval.py --algo PPO --model_path {{model}} {{args}}

# compare agent vs baselines
compare model *args="":
    PYTHONPATH=src {{venv}} src/eval/compare.py --algo PPO --model-path {{model}} {{args}}

# Optuna PPO search — e.g. just tune --trials 50 --no-curriculum
tune *args="":
    PYTHONPATH=src:src/tune:src/train {{venv}} src/tune/tune.py {{args}}

# tuning with SQLite storage for live dashboard
tune-db study="ppo_study" *args="":
    PYTHONPATH=src:src/tune:src/train {{venv}} src/tune/tune.py \
        --storage sqlite:///tune.db --study-name {{study}} {{args}}

# open Optuna dashboard (requires: just install-dashboard)
dashboard db="tune.db":
    {{venv}} -m optuna_dashboard sqlite:///{{db}}

# TensorBoard for all runs
tb:
    {{venv}} -m tensorboard.main --logdir logs/tensorboard

# TensorBoard for one run
tb-run run:
    {{venv}} -m tensorboard.main --logdir logs/tensorboard/{{run}}

# tests
test *args="":
    PYTHONPATH=src .venv/bin/pytest tests/ -v {{args}}
