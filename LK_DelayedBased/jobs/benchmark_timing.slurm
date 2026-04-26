#!/bin/bash
#SBATCH --job-name=benchmark_timing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/timing_%j.out
#SBATCH --error=logs/timing_%j.out

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

python benchmark_timing.py --output-dir outputs/timing
