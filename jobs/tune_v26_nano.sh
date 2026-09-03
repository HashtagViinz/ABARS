#!/bin/bash
#SBATCH --job-name=TUNE_V26N
#SBATCH --partition=department_only
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --gres=gpu:quadro_rtx_6000:2
#SBATCH --cpus-per-task=8

export MLFLOW_TRACKING_URI="sqlite:////home/vsalvatore/ABARS/mlflow.db" 
export NCCL_P2P_DISABLE=1         # Disabilita il Peer-to-Peer diretto se la risorsa condivisa dà problemi
export NCCL_IB_DISABLE=1          # Disabilita InfiniBand forzando i socket standard TCP
export OMP_NUM_THREADS=1          # Evita conflitti di CPU nei worker
export PYTHONUNBUFFERED=1         # Stampa i log in tempo reale senza buffering
uv run main.py tune --model v26_nano
