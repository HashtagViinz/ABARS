#!/bin/bash
#SBATCH --job-name=ABARS_TILED
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

# === ISTRUZIONI PER IL TILING E FILTRI ===
# Puoi passare il parametro --tile per avviare il partizionamento 
# del dataset (Slicing 2x2) prima dell'addestramento.
# Se passi anche --filter, il codice prima applicherà i filtri 
# sulle immagini ad alta risoluzione e poi taglierà il risultato.
#
# Esempi:
# --tile                        -> Solo Tiling (RGB)
# --tile --filter clahe         -> Applica CLAHE e poi Tiling
# --tile --filter clahe grayscale -> Applica entrambi i filtri e poi Tiling
#
# Il sistema genererà le cartelle invisibilmente e passerà lo YAML a YOLO.

uv run main.py train --model v8_small --epochs 200 --patience 20 --name tuned_tiled --tile
