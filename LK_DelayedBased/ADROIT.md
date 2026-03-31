Once on the log in node you need to run the following one time - after you can skip this on subsequent sections

```bash
module load anaconda3/2024.2
conda create -n lk_gpu python=3.11 -y
conda activate lk_gpu
pip install -r requirements.txt
pip install "jax[cuda12]"
```

The `CUDA_ERROR_NO_DEVICE` warning during install is normal — the login node has no GPU.

Make sure to copy the code over to adroit something like

```bash
mkdir <file_name>
cd /<file_name>
git clone <your-repo-url>
cd <file_name/LK_DelayedBased>
mkdir -p logs
```

To update: `git pull`

Thene very session you must run

```bash
module load anaconda3/2024.2
conda activate lk_gpu
cd <file_name>/LK_DelayedBased
```

For intereactive compute

```bash
srun --partition=gpu --gres=gpu:1 --mem=16G --time=00:30:00 --pty bash
# prompt changes to a compute node, then:
module load anaconda3/2024.2
conda activate lk_gpu
cd /LK_DelayedBased

python pipeline_iris/run_pipeline.py --quick --gpu
```

For batch jobs

**Quick sweep** (`jobs/quick_sweep.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=iris_quick
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/quick_%j.out

mkdir -p logs
module purge
module load anaconda3/2024.2
conda activate lk_gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $SLURM_SUBMIT_DIR

python pipeline_iris/run_pipeline.py --quick --gpu --sweep-only
```

**Standard sweep** (`jobs/standard_sweep.slurm`) — same as above but:

```bash
#SBATCH --mem=32G
#SBATCH --time=08:00:00
...
python pipeline_iris/run_pipeline.py --standard-grid --gpu
```

**Submit / monitor:**

```bash
sbatch jobs/quick_sweep.slurm
squeue -u $USER
tail -f logs/quick_<JOBID>.out
scancel <JOBID>
```
