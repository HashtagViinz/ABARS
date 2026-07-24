#!/bin/bash
#SBATCH --job-name=TEST_HW
#SBATCH --partition=department_only

LOG_DIR="logs"
LOG_FILE="test_hardware"

log_file="${LOG_DIR}/${LOG_FILE}.log"
log_err_file="${LOG_DIR}/${LOG_FILE}.err"

#SBATCH --output=${log_file}
#SBATCH --error=${log_err_file}

#SBATCH --gres=gpu:quadro_rtx_6000:2

# --- FIX PER IL DEADLOCK MULTI-GPU SU SLURM ---
export NCCL_P2P_DISABLE=1         # Disabilita il Peer-to-Peer diretto se la risorsa condivisa dà problemi
export NCCL_IB_DISABLE=1          # Disabilita InfiniBand forzando i socket standard TCP
export OMP_NUM_THREADS=1          # Evita conflitti di CPU nei worker
export PYTHONUNBUFFERED=1         # Stampa i log in tempo reale senza buffering

# --- DIAGNOSTICA INIZIALE (scritta nei log) ---
echo "=== JOB START ==="
echo "Nodo allocato: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "================="

uv run main.py hw_test