set dotenv-load
set shell := ["bash", "-c"]

venv := ".venv/bin/python3"
pip  := ".venv/bin/pip"

default:
    @just --list

# ── setup ─────────────────────────────────────────────────────────────────────

# install all requirements + the gym_env package
install:
    {{pip}} install -r requirements.txt
    {{pip}} install -e src/gym_env

# install CPU-only torch — saves ~1.7 GB vs the default CUDA build
torch-cpu:
    {{pip}} install torch --index-url https://download.pytorch.org/whl/cpu

# reinstall only the gym_env package after source changes
build-gym-env:
    {{pip}} install -e src/gym_env

# install optuna-dashboard for live tuning monitoring
install-dashboard:
    {{pip}} install optuna-dashboard

# ── training ─────────────────────────────────────────────────────────────────

# train PPO  — extra args: --timesteps 1000000 --n-envs 4 --seed 42 --run-name x
train road="speed_bump" *args="":
    PYTHONPATH=src {{venv}} src/train/train.py --algo PPO --road {{road}} {{args}}

# train PPO with 3-level curriculum
train-c road="speed_bump" *args="":
    PYTHONPATH=src {{venv}} src/train/train.py --algo PPO --road {{road}} --curriculum {{args}}

# run the dummy constant-speed baseline  — extra args: --episodes 5 --road speed_bump
dummy *args="":
    PYTHONPATH=src {{venv}} src/train/dummy_agent/train.py {{args}}

# ── evaluation ────────────────────────────────────────────────────────────────

# evaluate a trained model  — extra args: --n-episodes 10 --save-plots --road speed_bump
eval model *args="":
    PYTHONPATH=src {{venv}} src/eval/eval.py --algo PPO --model_path {{model}} {{args}}

# compare agent vs passive/random baselines  — extra args: --save-plots --road speed_bump
compare model *args="":
    PYTHONPATH=src {{venv}} src/eval/compare.py --algo PPO --model-path {{model}} {{args}}

# ── hyperparameter tuning ─────────────────────────────────────────────────────

# run Optuna PPO search  — extra args: --trials 50 --timesteps 100000 --no-curriculum
tune *args="":
    PYTHONPATH=src:src/tune:src/train {{venv}} src/tune/tune.py {{args}}

# run tuning with persistent SQLite storage (enables live dashboard monitoring)
tune-db study="ppo_study" *args="":
    PYTHONPATH=src:src/tune:src/train {{venv}} src/tune/tune.py \
        --storage sqlite:///tune.db --study-name {{study}} {{args}}

# open Optuna dashboard — run alongside tune-db  (requires: just install-dashboard)
dashboard db="tune.db":
    {{venv}} -m optuna_dashboard sqlite:///{{db}}

# ── monitoring ────────────────────────────────────────────────────────────────

# launch TensorBoard for all training runs
tb:
    {{venv}} -m tensorboard.main --logdir logs/tensorboard

# launch TensorBoard for one specific run tag
tb-run run:
    {{venv}} -m tensorboard.main --logdir logs/tensorboard/{{run}}

# ── tests ─────────────────────────────────────────────────────────────────────

# run the full pytest suite
test *args="":
    PYTHONPATH=src .venv/bin/pytest tests/ -v {{args}}
