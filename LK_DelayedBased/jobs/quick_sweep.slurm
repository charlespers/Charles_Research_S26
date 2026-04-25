#!/bin/bash
#SBATCH --job-name=iris_quick
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/quick_%j.out
#SBATCH --error=logs/quick_%j.out

mkdir -p logs
module purge
module load anaconda3/2024.2
conda activate lk_gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $SLURM_SUBMIT_DIR

python pipeline_iris/run_pipeline.py --quick --stretch-scenarios --gpu --sweep-only
