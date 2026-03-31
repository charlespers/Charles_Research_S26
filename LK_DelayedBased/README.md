# Model 7 Iris Reservoir

Four coupled lasers used as a physical reservoir computer for Iris classification.

## Files

```
model7_reservoir.py       CPU simulator
model7_reservoir_gpu.py   GPU simulator (JAX)
benchmark_iris.py         Iris benchmark function
fab_sweep.py              Parameter sweep
fab_design.py             Latin-Hypercube sampling
fab_meta_model.py         Recommender model training
pipeline_common.py        CLI logic
output_io.py              File/directory helpers
pipeline_iris/
  run_pipeline.py         ← main entry point
compare_cpu_gpu.py        CPU vs GPU timing check
run_gpu_test.slurm        SLURM job for Adroit
```

## Setup

```bash
pip install -r requirements.txt
pip install "jax[cuda12]"   # GPU only
```

## Run

```bash
# Quick (4 configs, ~2 min CPU / seconds GPU)
python pipeline_iris/run_pipeline.py --quick --gpu

# Full sweep
python pipeline_iris/run_pipeline.py --standard-grid --gpu
```

Outputs go to `pipeline_iris/outputs/`.

## All flags

| Flag | What it does |
|------|-------------|
| `--quick` | 4-config grid, fastest |
| `--standard-grid` | ~7 000-config grid |
| `--stretch-scenarios` | 3 train/test splits |
| `--lhc` | Latin-Hypercube sampling |
| `--n-lhc N` | LHC sample count (default 24) |
| `--n-replicates N` | Replicates per config |
| `--seed N` | Random seed |
| `--sweep-only` | Skip recommender training |
| `--gpu` | Use JAX GPU backend |
