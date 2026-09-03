#!/bin/bash
#SBATCH --job-name=abars_train_v26_nano_tile_clahe
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

# === ISTRUZIONI PER I FILTRI ===
# Puoi passare uno o piu' filtri separati da spazio al parametro --filter.
# Lo script si occupera' in automatico di generare la copia del dataset 
# (se non esiste gia') prima di avviare l'addestramento.
#
# Esempi:
# --filter clahe
# --filter grayscale
# --filter clahe grayscale

uv run main.py train --model v26_nano --epochs 200 --patience 20 --name v26n_tuned_tiled_clahe --tile --filter clahe
