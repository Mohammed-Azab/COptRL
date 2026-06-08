set dotenv-load
set shell := ["bash", "-c"]

venv := ".venv/bin/python3"
pip  := ".venv/bin/pip3"

default:
    @just --list

# setup
install:
    {{pip}} install -r requirements.txt
    {{pip}} install -e src/gym_env

install-torch-cpu:
    {{pip}} install torch --index-url https://download.pytorch.org/whl/cpu

build-gym-env:
    {{pip}} install -e src/gym_env

install-dashboard:
    {{pip}} install optuna-dashboard

# train PPO — road defaults to speed_bump
# usage:  just train                           (speed_bump, no curriculum)
#         just train speed_bump --c            (explicit road + curriculum shorthand)
#         just train --curriculum              (flags only → road defaults to speed_bump)
#         just train speed_bump --c --n-envs 4
train road="speed_bump" *args="":
    #!/usr/bin/env bash
    r="{{road}}"
    extra="{{args}}"
    if [[ "$r" == --* ]]; then extra="$r $extra"; r="speed_bump"; fi
    extra=$(echo "$extra" | sed 's/--c\b/--curriculum/g')
    PYTHONPATH=src {{venv}} src/train/train.py --algo PPO --road "$r" $extra

# Run dummy agent
dummy *args="":
    PYTHONPATH=src {{venv}} src/train/dummy_agent/train.py {{args}}

# Evaluate a trained model -> just eval models/.../PPO_final.zip --save-plots
eval model *args="":
    PYTHONPATH=src {{venv}} src/eval/eval.py --algo PPO --model_path {{model}} {{args}}

# compare agent vs baselines
compare model *args="":
    PYTHONPATH=src {{venv}} src/eval/compare.py --algo PPO --model-path {{model}} {{args}}

# Optuna PPO search —>just tune --trials 50 --no-curriculum
# Tuning with SQLite storage for live dashboard
tune study="myPPO_study" *args="":
    PYTHONPATH=src:src/tune:src/train {{venv}} src/tune/tune.py \
        --storage sqlite:///tune.db --study-name {{study}} {{args}}

# open Optuna dashboard — run just install-dashboard first if missing
dashboard db="tune.db":
    @test -f .venv/bin/optuna-dashboard || (echo "not installed — run: just install-dashboard" && exit 1)
    .venv/bin/optuna-dashboard sqlite:///{{db}}

# TensorBoard for all runs
tb:
    {{venv}} -m tensorboard.main --logdir logs/tensorboard

# TensorBoard for one run
tb-run run:
    {{venv}} -m tensorboard.main --logdir logs/tensorboard/{{run}}

# tests
test *args="":
    PYTHONPATH=src .venv/bin/pytest tests/ -v {{args}}
