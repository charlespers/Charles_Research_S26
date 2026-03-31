The setup was slightly challenging so I'll replicate it here: (Refer to AROIT.md for more details)

From the terminal:

```
ssh <netid>@adroit.princeton.edu
*log in
```

Then install the requirements

```bash
pip install -r requirements.txt
pip install "jax[cuda12]"   # GPU only
```

To run the scripts, test with first, and sweep with second

```bash
# Quick (4 configs, ~2 min CPU / seconds GPU)
python pipeline_iris/run_pipeline.py --quick --gpu

# Full sweep
python pipeline_iris/run_pipeline.py --standard-grid --gpu
```

Outputs go to `pipeline_iris/outputs/`.

All flags:

| Flag                  | What it does                  |
| --------------------- | ----------------------------- |
| `--quick`             | 4-config grid, fastest        |
| `--standard-grid`     | ~7 000-config grid            |
| `--stretch-scenarios` | 3 train/test splits           |
| `--lhc`               | Latin-Hypercube sampling      |
| `--n-lhc N`           | LHC sample count (default 24) |
| `--n-replicates N`    | Replicates per config         |
| `--seed N`            | Random seed                   |
| `--sweep-only`        | Skip recommender training     |
| `--gpu`               | Use JAX GPU backend           |
