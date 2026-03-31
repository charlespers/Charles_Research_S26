#!/bin/bash
#SBATCH --job-name=lk_gpu_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err
#SBATCH --mail-type=END,FAIL

# ── Environment ────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "=========================================="

mkdir -p logs

module purge
module load anaconda3/2024.2
conda activate lk_gpu   # edit this name if you used a different env name

# Make sure JAX can see the GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd $SLURM_SUBMIT_DIR

echo ""
echo "--- Python / JAX info ---"
python -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
x = jnp.ones((4,4))
print('Test matmul shape:', (x @ x).shape)
"

# ── Test 1: Single-step + feature-level correctness ───────────────────────────
echo ""
echo "=========================================="
echo "TEST 1: Single-step ODE and feature correctness"
echo "=========================================="
python model7_reservoir_gpu.py --validate
VALIDATE_EXIT=$?
echo "Exit code: $VALIDATE_EXIT"

# ── Test 2: CPU Iris baseline (timed) ─────────────────────────────────────────
echo ""
echo "=========================================="
echo "TEST 2: CPU Iris benchmark (baseline timing)"
echo "=========================================="
python - <<'PYEOF'
import time
from benchmark_iris import run_iris_model7_benchmark

t0 = time.perf_counter()
res = run_iris_model7_benchmark(
    motif="auxiliary",
    dt_ns=5e-4,
    store_every=400,
    noise_on=True,
    base_seed=30,
    t_span_ns=(0.0, 20.0),
    washout_ns=10.0,
    spacing_m=50e-6,
    lambda0_m=850e-9,
    n_air=1.0,
    I_th_A=17.35e-3,
    default_k=15.0,
    i_min=0.5,
    i_max=1.5,
    use_gpu=False,
)
t1 = time.perf_counter()
print(f"CPU accuracy : {res['accuracy_test']:.4f}  ({res['accuracy_test']*100:.2f}%)")
print(f"CPU wall time: {t1-t0:.2f}s")
PYEOF
CPU_EXIT=$?

# ── Test 3: GPU Iris benchmark — first call (includes JIT compile) ─────────────
echo ""
echo "=========================================="
echo "TEST 3: GPU Iris benchmark — first call (JIT warmup included)"
echo "=========================================="
python - <<'PYEOF'
import time
from benchmark_iris import run_iris_model7_benchmark

t0 = time.perf_counter()
res = run_iris_model7_benchmark(
    motif="auxiliary",
    dt_ns=5e-4,
    store_every=400,
    noise_on=True,
    base_seed=30,
    t_span_ns=(0.0, 20.0),
    washout_ns=10.0,
    spacing_m=50e-6,
    lambda0_m=850e-9,
    n_air=1.0,
    I_th_A=17.35e-3,
    default_k=15.0,
    i_min=0.5,
    i_max=1.5,
    use_gpu=True,
)
t1 = time.perf_counter()
print(f"GPU accuracy (1st call): {res['accuracy_test']:.4f}  ({res['accuracy_test']*100:.2f}%)")
print(f"GPU wall time (1st):     {t1-t0:.2f}s  (includes XLA compile)")
PYEOF
GPU1_EXIT=$?

# ── Test 4: GPU Iris benchmark — second call (JIT cache hit) ──────────────────
echo ""
echo "=========================================="
echo "TEST 4: GPU Iris benchmark — second call (cached JIT)"
echo "=========================================="
python - <<'PYEOF'
import time
from benchmark_iris import run_iris_model7_benchmark

# warm up
run_iris_model7_benchmark(
    motif="auxiliary", dt_ns=5e-4, store_every=400, noise_on=True, base_seed=30,
    t_span_ns=(0.0, 20.0), washout_ns=10.0, spacing_m=50e-6,
    lambda0_m=850e-9, n_air=1.0, I_th_A=17.35e-3, default_k=15.0,
    i_min=0.5, i_max=1.5, use_gpu=True,
)

# timed second call
t0 = time.perf_counter()
res = run_iris_model7_benchmark(
    motif="auxiliary", dt_ns=5e-4, store_every=400, noise_on=True, base_seed=42,
    t_span_ns=(0.0, 20.0), washout_ns=10.0, spacing_m=50e-6,
    lambda0_m=850e-9, n_air=1.0, I_th_A=17.35e-3, default_k=15.0,
    i_min=0.5, i_max=1.5, use_gpu=True,
)
t1 = time.perf_counter()
print(f"GPU accuracy (2nd call): {res['accuracy_test']:.4f}  ({res['accuracy_test']*100:.2f}%)")
print(f"GPU wall time (2nd):     {t1-t0:.2f}s  (JIT cache hit)")
PYEOF
GPU2_EXIT=$?

# ── Test 5: Full accuracy validation (3 seeds, CPU vs GPU) ────────────────────
echo ""
echo "=========================================="
echo "TEST 5: Full accuracy validation — CPU vs GPU (3 seeds)"
echo "=========================================="
python model7_reservoir_gpu.py --validate-full
FULL_EXIT=$?

# ── Test 6: Quick grid vs standard grid ───────────────────────────────────────
echo ""
echo "=========================================="
echo "TEST 6: Quick grid vs standard grid accuracy"
echo "=========================================="
python model7_reservoir_gpu.py --validate-grids
GRID_EXIT=$?

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Test 1 (single-step + features):  $([ $VALIDATE_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 2 (CPU baseline):            $([ $CPU_EXIT -eq 0 ]      && echo PASS || echo FAIL)"
echo "Test 3 (GPU first call):          $([ $GPU1_EXIT -eq 0 ]     && echo PASS || echo FAIL)"
echo "Test 4 (GPU cached):              $([ $GPU2_EXIT -eq 0 ]     && echo PASS || echo FAIL)"
echo "Test 5 (CPU vs GPU accuracy):     $([ $FULL_EXIT -eq 0 ]     && echo PASS || echo FAIL)"
echo "Test 6 (quick vs standard grid):  $([ $GRID_EXIT -eq 0 ]     && echo PASS || echo FAIL)"

OVERALL=$(( VALIDATE_EXIT + CPU_EXIT + GPU1_EXIT + GPU2_EXIT + FULL_EXIT + GRID_EXIT ))
echo ""
echo "Finished: $(date)"
if [ $OVERALL -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "SOME TESTS FAILED (see above)"
fi
exit $OVERALL
