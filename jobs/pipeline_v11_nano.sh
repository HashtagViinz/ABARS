#!/bin/bash
echo "Sottomissione Tuning V11 Nano..."
TUNE_JOB_ID=$(sbatch --parsable jobs/tune_v11_nano.sh)
echo "Tuning sottomesso con ID: $TUNE_JOB_ID"

echo "Sottomissione Training vincolato alla fine del Tuning..."
TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$TUNE_JOB_ID jobs/train_v11_nano_tiled.sh)
echo "Training sottomesso con ID: $TRAIN_JOB_ID (In attesa di $TUNE_JOB_ID...)"
