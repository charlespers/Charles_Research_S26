# Iris Pipeline

```bash
# From LK_DelayedBased/

# Fastest run (4 configs, no recommender)
python pipeline_iris/run_pipeline.py --quick --sweep-only

# Quick run + GPU
python pipeline_iris/run_pipeline.py --quick --gpu

# Full sweep + recommender
python pipeline_iris/run_pipeline.py --standard-grid --gpu
```

Outputs go to `pipeline_iris/outputs/`.

> Recommender training needs `--stretch-scenarios` (3 scenarios minimum for LOOCV).
