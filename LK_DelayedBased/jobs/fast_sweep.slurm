#!/bin/bash
#SBATCH --job-name=iris_fast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fast_%j.out
#SBATCH --error=logs/fast_%j.out

set -e
mkdir -p logs

module purge
module load anaconda3/2024.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lk_gpu

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "JAX devices: $(python -c 'import jax; print(jax.devices())')"

# 1080 configs x 3 scenarios = 3240 GPU runs, single JIT compilation
# Expected runtime: ~30 min on A100
python pipeline_iris/run_pipeline.py --fast-grid --stretch-scenarios --gpu --sweep-only
echo "SUCCESS"
