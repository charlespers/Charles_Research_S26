# Princeton Adroit GPU Cluster — Setup & Workflow

## One-time environment setup

Run these once on the **login node** (no GPU needed):

```bash
module load anaconda3/2024.2
conda create -n lk_gpu python=3.11 -y
conda activate lk_gpu
pip install -r requirements.txt
pip install "jax[cuda12]"
```

`CUDA_ERROR_NO_DEVICE` during install is normal — login node has no GPU.

## Get the code on Adroit

```bash
mkdir ~/lk_gpu_proj
cd ~/lk_gpu_proj
git clone <your-repo-url> LK_DelayedBased
cd LK_DelayedBased
mkdir -p logs
```

To update after local changes: `git pull`

## Session startup (every session)

```bash
module load anaconda3/2024.2
conda activate lk_gpu
cd ~/lk_gpu_proj/LK_DelayedBased
```

---

## Complete paper generation workflow

### Step 0 — Interactive sanity check (optional, ~2 min)

```bash
srun --partition=gpu --gres=gpu:1 --mem=8G --time=00:10:00 --pty bash
module load anaconda3/2024.2 && conda activate lk_gpu
cd ~/lk_gpu_proj/LK_DelayedBased
python pipeline_iris/run_pipeline.py --quick --gpu
```

### Step 1 — GPU benchmark timing (Fig 1 data)

```bash
sbatch jobs/benchmark_timing.slurm
# Produces: outputs/timing/timing_results.csv  (~15 min)
```

### Step 2 — Standard fab sweep (Fig 2 and meta-learning data)

```bash
sbatch jobs/standard_sweep.slurm
# Produces: pipeline_iris/outputs/sweeps/fab_long/*/sweep_long.csv  (~3-8 hr)
```

Wait for both Step 1 and Step 2 to finish before Step 3.

### Step 3 — Train MLP + generate all 9 paper figures

```bash
sbatch jobs/generate_paper.slurm
# Produces: paper/figures/fig{1..9}_*.{pdf,png}  (~30 min on GPU)
```

### Step 4 — Compile paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

You need `neurips_2024.sty` from neurips.cc. Download and place in `paper/`.

---

## Individual job files

| File | Purpose | Est. time |
|------|---------|-----------|
| `jobs/quick_sweep.slurm` | 3-scenario quick sweep (dev/test) | ~5 min |
| `jobs/standard_sweep.slurm` | Full 7k-config sweep (paper data) | ~3–8 hr |
| `jobs/benchmark_timing.slurm` | CPU vs GPU timing sweep | ~15 min |
| `jobs/generate_paper.slurm` | MLP harness + all 9 figures | ~30 min |

---

## Submit, monitor, cancel

```bash
sbatch jobs/<name>.slurm
squeue -u $USER
tail -f logs/<name>_<JOBID>.out
scancel <JOBID>
```

---

## Key output locations

```
outputs/timing/timing_results.csv          # GPU speedup data (Fig 1)
outputs/timing/timing_summary.json
pipeline_iris/outputs/sweeps/fab_long/     # Sweep results (Figs 2, 3, 9)
outputs/analysis/mlp_classification/      # MLP harness outputs (Figs 4, 5, 6)
paper/figures/                             # All 9 paper figures
```

---

## Running scripts manually (interactive compute node)

```bash
srun --partition=gpu --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash
module load anaconda3/2024.2 && conda activate lk_gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd ~/lk_gpu_proj/LK_DelayedBased

# GPU benchmark
python benchmark_timing.py

# Train MLP + figures (after sweep)
SWEEP=$(ls -t pipeline_iris/outputs/sweeps/fab_long/*/sweep_long.csv | head -1)
python mlp_classification_harness.py --sweep-csv "$SWEEP"
python generate_figures.py --sweep-csv "$SWEEP"
```
