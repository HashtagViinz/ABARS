#!/bin/bash
#SBATCH --job-name=ABARS_CLAHE
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

# Lancia il training del modello Small (che usa gli iperparametri Tuned in automatico)
# e applica il filtro CLAHE on-the-fly tramite Albumentations.
# --name: Nome con cui il run apparira' nella dashboard di MLflow
# --filter: Il filtro da applicare (clahe, sharpen, white_balance, all)

uv run main.py train --model small --epochs 80 --patience 10 --name tuned_clahe --filter clahe
