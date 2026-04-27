#!/bin/bash
#SBATCH --job-name=iris_big
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/big_%j.out
#SBATCH --error=logs/big_%j.out

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

# 2304 configs x 7 scenarios = 16128 GPU runs, 2 JIT compilations (noise on/off)
# Expected runtime: ~5-6 h on V100; 8 h wall time gives comfortable buffer
python pipeline_iris/run_pipeline.py --rich-grid --many-scenarios --gpu --sweep-only
echo "SUCCESS"
