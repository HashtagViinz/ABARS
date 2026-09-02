#!/bin/bash
#SBATCH --job-name=TUNE_V26N
#SBATCH --partition=department_only
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --gres=gpu:quadro_rtx_6000:2
#SBATCH --cpus-per-task=8

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
uv run main.py tune --model V26_small
