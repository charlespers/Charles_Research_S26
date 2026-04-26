#!/bin/bash
#SBATCH --job-name=iris_quick
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/quick_%j.out
#SBATCH --error=logs/quick_%j.out

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

python pipeline_iris/run_pipeline.py --quick --stretch-scenarios --gpu --sweep-only
echo "SUCCESS"
