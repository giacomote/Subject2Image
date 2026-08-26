#!/bin/bash

# ============================================================
# SLURM directives
# ============================================================

#SBATCH --job-name=s2i-infer
#SBATCH --account=cvcs2026
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Checking submit directory
# ============================================================

if [[ "$SLURM_SUBMIT_DIR" != *"Subject2Image"* ]]; then
    echo "[ERROR] Please submit the job from the root of the repository" >&2
    exit 1
fi

# ============================================================
# Mode validation
# ============================================================

MODE=$1

if [ "$MODE" == "baseline" ]; then
    TARGET_FILE="pipeline_baseline/infer.py"
elif [ "$MODE" == "modified" ]; then
    TARGET_FILE="pipeline_modified/infer.py"
else
    echo "[ERROR] Missing mode specification" >&2
    echo "Usage: sbatch $0 [baseline|modified]" >&2
    exit 1
fi

# ============================================================
# Preliminary operations
# ============================================================

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

export HF_HOME="/work/cvcs2026/neural_visionaries/.hf_cache"
export HF_TOKEN_PATH="~/.cache/huggingface/token"

mkdir -p $HF_HOME
mkdir -p logs

module purge
module load cuda/12.6.3
module load python/3.11.15

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# ============================================================
# Launch Python script
# ============================================================

GPU_RAW=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n1)
GPU_INFO=$(echo "$GPU_RAW" | awk -F',' '{sub(/^NVIDIA */, "", $1); printf "%s (%.0f GB)", $1, $2/1024}')

echo "=================================================="
echo "Job name        : $SLURM_JOB_NAME"
echo "Job ID          : $SLURM_JOB_ID"
echo "Running on node : $(hostname)"
echo "Assigned GPU    : $GPU_INFO"
echo "Run started at  : $(date '+%a %b %d %H:%M:%S %Z %Y')"
echo -e "==================================================\n"

python -u $TARGET_FILE

echo -e "\n=================================================="
echo "Job completed at: $(date '+%a %b %d %H:%M:%S %Z %Y')"
echo "=================================================="