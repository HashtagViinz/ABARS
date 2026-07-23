#!/bin/bash

#SBATCH --job-name=ABARS_test_1
#SBATCH --partition=department_only
#SBATCH --output=logs/logs.out
#SBATCH --error=logs/err.err
#SBATCH --gres=gpu:1

uv run main.py dataset_prepare