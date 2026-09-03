#!/bin/bash
# Questo script lancia l'analisi statistica del dataset in tempo reale.
# Non richiede sbatch perché non usa la GPU.

echo "Avvio analisi del dataset Tiled..."
export PYTHONUNBUFFERED=1

# Assumiamo che il file yaml si chiami uavod10_tiled.yaml
# Se il tuo file si chiama diversamente (es. dataset.yaml), modificalo qui.
uv run main.py analyze_dataset --dataset dataset_tiled/uavod10_tiled.yaml
