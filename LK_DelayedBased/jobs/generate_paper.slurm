#!/bin/bash
#SBATCH --job-name=paper_figures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/paper_%j.out
#SBATCH --error=logs/paper_%j.out

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

SWEEP_CSV=$(ls -t pipeline_iris/outputs/sweeps/fab_long/*/sweep_long.csv 2>/dev/null | head -1)
if [ -z "$SWEEP_CSV" ]; then
  echo "ERROR: No sweep_long.csv found. Run standard_sweep.slurm first." >&2
  exit 1
fi
echo "Using sweep: $SWEEP_CSV"

python mlp_classification_harness.py --sweep-csv "$SWEEP_CSV" --output-dir outputs/analysis/mlp_classification

python generate_figures.py \
    --timing-csv outputs/timing/timing_results.csv \
    --sweep-csv "$SWEEP_CSV" \
    --mlp-dir outputs/analysis/mlp_classification \
    --output-dir paper/figures

echo "All figures written to paper/figures/"
