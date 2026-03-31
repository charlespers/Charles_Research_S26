# Testing GPU vs CPU

## Quick check (interactive GPU node)

```bash
srun --partition=gpu --gres=gpu:1 --mem=16G --time=00:30:00 --pty bash
module load anaconda3/2024.2
conda activate lk_gpu
cd /LK_DelayedBased

python compare_cpu_gpu.py
```

Prints CPU time, GPU time (first call + cached), speedup, and accuracy diff.
With `noise_on=False` accuracy diff should be **0.0000**.

## Full test suite (batch job)

```bash
sbatch run_gpu_test.slurm
squeue -u $USER
tail -f logs/gpu_test_<JOBID>.out
```

Expected runtime: 20–40 min.

## Tests the following

| Test                          | Pass criterion        |
| ----------------------------- | --------------------- |
| Single-step ODE (CPU vs GPU)  | Relative error < 1e-9 |
| CPU baseline (150 samples)    | Runs, prints accuracy |
| GPU first call (JIT compile)  | Runs cleanly          |
| GPU second call (cached)      | Time << CPU           |
| CPU vs GPU accuracy (3 seeds) | Difference < 0.05     |
| Quick vs standard grid        | Best acc within 10 pp |
