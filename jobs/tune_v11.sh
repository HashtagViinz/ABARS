#!/bin/bash
#SBATCH --job-name=ABARS_TUNE_V11
#SBATCH --partition=department_only
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --gres=gpu:quadro_rtx_6000:2

# --- FIX PER IL DEADLOCK MULTI-GPU SU SLURM ---
export NCCL_P2P_DISABLE=1         # Disabilita il Peer-to-Peer diretto se la risorsa condivisa dà problemi
export NCCL_IB_DISABLE=1          # Disabilita InfiniBand forzando i socket standard TCP
export OMP_NUM_THREADS=1          # Evita conflitti di CPU nei worker
export PYTHONUNBUFFERED=1         # Stampa i log in tempo reale senza buffering

echo "=== JOB START ==="
echo "Nodo allocato: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=== nvidia-smi ===="
nvidia-smi
echo "================="

uv run main.py tune --model v11_small
