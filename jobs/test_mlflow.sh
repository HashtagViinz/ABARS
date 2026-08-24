#!/bin/bash
#SBATCH --job-name=test_mlflow
#SBATCH --partition=department_only
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-task=4

# Configura l'URI per il cluster in modo esplicito
export MLFLOW_TRACKING_URI="sqlite:////home/vsalvatore/ABARS/mlflow.db" 

# --- FIX PER IL DEADLOCK MULTI-GPU SU SLURM ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- DIAGNOSTICA INIZIALE ---
echo "=== MLFLOW TEST START ==="
echo "Nodo allocato: $HOSTNAME"
echo "========================="

# Esegui il test
uv run main.py mlflow_test
