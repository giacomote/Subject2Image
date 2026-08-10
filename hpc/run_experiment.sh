#!/bin/bash

# ============================================================
# SLURM directives
# ============================================================

#SBATCH --job-name=personalization_pipeline
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Preliminary operations
# ============================================================

export PYTHONNOUSERSITE=1

mkdir -p logs

module purge
module load cuda/11.8.0
module load python/3.13.13

cd ~/NeuralVisionaries

source ~/NeuralVisionaries/.venv/bin/activate

# ============================================================
# Launch Python script
# ============================================================

echo "=================================================="
echo "Job name        : $SLURM_JOB_NAME"
echo "Job ID          : $SLURM_JOB_ID"
echo "Running on node : $(hostname)"
echo "Run started at  : $(date)"
echo -e "==================================================\n"

python personalization_pipeline.py

echo -e "\n=================================================="
echo "Job completed at: $(date)"
echo "=================================================="