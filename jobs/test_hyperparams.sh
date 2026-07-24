#!/bin/bash


LOG_DIR="logs"
LOG_FILE="test_hyperparams"

log_file="${LOG_DIR}/${LOG_FILE}.log"
log_err_file="${LOG_DIR}/${LOG_FILE}.err"

# Clean old Log File
rm -rf ${log_file} ${log_err_file}


#SBATCH --job-name=TEST_HyperParams
#SBATCH --partition=department_only
#SBATCH --output=${log_file}
#SBATCH --error=${log_err_file}

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

uv run main.py tune --model nano