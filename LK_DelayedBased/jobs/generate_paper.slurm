#!/bin/bash
#SBATCH --job-name=paper_figures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/paper_%j.out
#SBATCH --error=logs/paper_%j.out
#
# Generates all 9 paper figures. Requires:
#   outputs/timing/timing_results.csv  (from benchmark_timing.slurm)
#   pipeline_iris/outputs/sweeps/fab_long/**/sweep_long.csv  (from standard_sweep.slurm)
#   outputs/analysis/mlp_classification/  (from mlp step below)
#
# Run this AFTER benchmark_timing.slurm and standard_sweep.slurm have both finished.

mkdir -p logs
module purge
module load anaconda3/2024.2
conda activate lk_gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $SLURM_SUBMIT_DIR

# Identify most recent sweep CSV
SWEEP_CSV=$(ls -t pipeline_iris/outputs/sweeps/fab_long/*/sweep_long.csv 2>/dev/null | head -1)
if [ -z "$SWEEP_CSV" ]; then
  echo "ERROR: No sweep_long.csv found. Run standard_sweep.slurm first." >&2
  exit 1
fi
echo "Using sweep: $SWEEP_CSV"

# Step 1: Train MLP harness
python mlp_classification_harness.py \
    --sweep-csv "$SWEEP_CSV" \
    --output-dir outputs/analysis/mlp_classification

# Step 2: Generate all 9 figures
python generate_figures.py \
    --timing-csv outputs/timing/timing_results.csv \
    --sweep-csv "$SWEEP_CSV" \
    --mlp-dir outputs/analysis/mlp_classification \
    --output-dir paper/figures

echo "All figures written to paper/figures/"
