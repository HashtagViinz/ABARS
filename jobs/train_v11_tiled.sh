#!/bin/bash
#SBATCH --job-name=ABARS_V11_TILED
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

# === ISTRUZIONI PER YOLO11 + TILING ===
# Questo job lancia la nuovissima architettura YOLO11 (Small) 
# combinata con la strategia di Offline Tiling 2x2.
# --model v11_small: Seleziona YOLO11
# --name v11_tuned_tiled: Nome univoco per distinguerlo nella dashboard MLflow
# --tile: Applica lo slicing offline per migliorare la detection degli oggetti piccoli

uv run main.py train --model v11_small --epochs 200 --patience 20 --name v11_tuned_tiled --tile
