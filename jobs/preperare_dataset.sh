#!/bin/bash
#SBATCH --job-name=PREP_DATASET_ABARS
#SBATCH --partition=department_only
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

uv run main.py dataset_prepare