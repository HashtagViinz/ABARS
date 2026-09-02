#!/bin/bash
echo "Sottomissione Tuning V26 Small..."
TUNE_JOB_ID=$(sbatch --parsable jobs/tune_v26_small.sh)
echo "Tuning sottomesso con ID: $TUNE_JOB_ID"

echo "Sottomissione Training vincolato alla fine del Tuning..."
TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$TUNE_JOB_ID jobs/train_v26_small_tiled.sh)
echo "Training sottomesso con ID: $TRAIN_JOB_ID (In attesa di $TUNE_JOB_ID...)"
