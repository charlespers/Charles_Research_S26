#!/bin/bash
#SBATCH --job-name=iris_standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/standard_%j.out
#SBATCH --error=logs/standard_%j.out

mkdir -p logs
module purge
module load anaconda3/2024.2
conda activate lk_gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $SLURM_SUBMIT_DIR

python pipeline_iris/run_pipeline.py --standard-grid --stretch-scenarios --gpu --sweep-only
