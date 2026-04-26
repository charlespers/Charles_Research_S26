#!/bin/bash
#SBATCH --job-name=iris_standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/standard_%j.out
#SBATCH --error=logs/standard_%j.out

set -e
mkdir -p logs

module purge
module load anaconda3/2024.2

# Properly initialize conda for non-interactive shells
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lk_gpu

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "JAX devices: $(python -c 'import jax; print(jax.devices())')"

python pipeline_iris/run_pipeline.py --standard-grid --stretch-scenarios --gpu --sweep-only
