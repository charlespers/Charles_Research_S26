# Explaining This Project From First Principles

A complete explanation of what this project does, starting from absolute basics
and building up to the full research system.

---

## Part 1: The Simplest Idea — Using Physics to Compute

### What is a computer?

At its core, a computer takes an input, does something complicated to it, and
produces an output. The "something complicated" is usually billions of transistors
switching on and off inside a chip.

But what if you could use a *physical system* — a spinning fluid, a vibrating
membrane, or a flickering laser — to do the complicated part? The physical system
would scramble your input into a rich, high-dimensional signal, and you'd only
need a simple rule at the end to read out the answer.

This idea is called **physical reservoir computing**.

### The core insight

Imagine you drop a pebble into a pond. The ripples spread out and bounce off
the edges in complicated patterns. If you then drop a second pebble somewhere
else, the ripple patterns mix together. Now imagine taking snapshots of the pond
at many points and asking: "can I predict tomorrow's weather from these
snapshots?" Surprisingly, yes — *if* the pond is rich and complicated enough,
its state encodes useful information about what drove it.

A **reservoir** is exactly that pond. It is:
1. A fixed physical system (not trained — just existing)
2. Driven by your input data
3. Producing a high-dimensional, nonlinear response
4. Read out by a simple linear model at the end

The only thing you train is the linear readout. The reservoir does the heavy
lifting for free, just by existing.

---

## Part 2: Why Lasers?

### What is a semiconductor laser?

A semiconductor laser is a tiny chip (smaller than a grain of sand) that
produces coherent light when you pump electricity into it. Think of it like an
LED that has been pushed into a state where all the light pulses in sync.

Inside the laser:
- **Electrons** get pumped in by the current. They sit in excited energy states.
- When an excited electron drops to a lower state, it emits a **photon** (a
  particle of light).
- The photons bounce back and forth between mirrors, stimulating more emission.
- The result: a coherent beam of light that oscillates at a precise frequency.

We track two things inside each laser:
- **The optical field** — the actual light oscillation, described by two numbers:
  `x` (real part) and `y` (imaginary part). The intensity is `S = x² + y²`.
- **The carrier number N** — how many excited electrons are waiting to emit.

### Why are lasers interesting as reservoirs?

Semiconductor lasers are:
- **Fast** — they respond on nanosecond timescales (billionths of a second)
- **Nonlinear** — the relationship between input and output is complex, not
  just proportional
- **Sensitive to coupling** — when lasers talk to each other via light, they
  produce rich, chaotic dynamics
- **Physically realizable** — they can be fabricated on chips and operated at
  room temperature

The combination of speed, nonlinearity, and coupling makes laser networks a
natural reservoir for machine learning.

---

## Part 3: The Lang-Kobayashi Model

### What is a mathematical model?

When physicists want to simulate a laser, they write down equations that
describe how the laser's state changes over time. These are called
**rate equations** — they tell you the *rate of change* of each variable.

### The Lang-Kobayashi equations (simplified)

For a single laser with optical feedback, the Lang-Kobayashi (LK) model says:

```
dx/dt = (gain - loss) * x + light injected from other lasers
dy/dt = (gain - loss) * y + light injected from other lasers
dN/dt = pump current - electron decay - electrons consumed making photons
```

Where:
- `x, y` describe the optical field (like the amplitude of a wave)
- `N` is the carrier (excited electron) count
- `gain` depends on how many electrons are available: more electrons → more
  amplification
- `loss` is how quickly the field decays
- The `light injected` term is what makes a *network* of lasers interesting —
  each laser can receive light from other lasers, delayed by the travel time

### The time delay

Light takes time to travel between lasers. If laser 1 is 50 micrometers from
laser 2, the light takes about 0.17 picoseconds to travel between them. This
**time delay** (τ) means each laser is driven by the *past state* of its
neighbors, not their current state.

This is crucial: time-delayed coupling creates **memory** in the system. The
current state of the laser depends on what happened τ nanoseconds ago. Memory
is what makes reservoir computers good at processing sequential data.

### Four lasers in a line

This project uses **four lasers arranged in a straight line**, coupled to their
neighbors. Each laser injects some of its light into adjacent lasers:

```
[Laser 1] ←→ [Laser 2] ←→ [Laser 3] ←→ [Laser 4]
```

The coupling pattern (which lasers talk to which, and how strongly) is called
the **coupling motif** or **coupling topology**. Different topologies produce
very different dynamics.

---

## Part 4: Using the Reservoir for Machine Learning

### The Iris dataset

The Iris dataset is a classic machine learning benchmark. It contains 150 flower
measurements: 4 numbers per flower (sepal length, sepal width, petal length,
petal width), and the task is to predict which of 3 species the flower belongs
to.

The input is 4 numbers → the output is one of 3 classes.

### Driving the lasers with data

To use the laser reservoir for machine learning, we do the following:

1. **Normalize** each flower's 4 measurements to lie between 0 and 1.
2. **Map** these 4 numbers to 4 pump currents — one per laser. Higher feature
   value → more current pumped into that laser → more excited electrons → more
   light output.
3. **Run** the simulation: integrate the LK equations for 20 nanoseconds.
   The first 10 ns is "washout" — we discard it to let the initial conditions
   fade. The last 10 ns is where the laser states carry information about the
   input.
4. **Extract features**: for each of the 4 lasers, compute 5 statistics from
   the photon intensity time series:
   - mean, min, max, standard deviation, and mean carrier number
   - That's 4 lasers × 5 statistics = **20 numbers** per flower
5. **Train a linear classifier** (Ridge regression) on these 20 numbers to
   predict the flower species.

The *reservoir* (the laser array) transforms 4 input numbers into 20 rich
features. The linear classifier is then trained in milliseconds. The reservoir
itself is never trained — only the final linear layer is.

---

## Part 5: The Fabrication Parameter Problem

### Why optimization is hard

The laser array has many physical parameters that affect its behavior:

| Parameter | What it controls | Example values |
|-----------|-----------------|----------------|
| `motif` | Which lasers couple to which | auxiliary, chain, relay |
| `default_k` | How strongly lasers are coupled | 10–20 |
| `i_min`, `i_max` | Range of pump currents | 0.45–0.55, 1.35–1.55 |
| `spacing_m` | Physical distance between lasers | 45–55 μm |
| `noise_on` | Include spontaneous emission | True / False |
| `I_th_A` | Threshold current | 16.5–18.2 mA |
| `lambda0_m` | Light wavelength | 840–860 nm |
| `dt_ns` | Simulation timestep | 0.4–0.5 ps |

If you take just a few values for each parameter and combine them, you get
over **7,000 configurations** to test. This is the *fabrication parameter
sweep*.

### Why this is slow on CPU

For each configuration, you need to:
1. Simulate the 4-laser system for 150 Iris samples
2. Each sample requires integrating 40,000 Euler steps (20 ns / 0.0005 ns)
3. Each step touches 12 variables (x, y, N for 4 lasers)

On a CPU, running all 150 samples sequentially takes **15–45 seconds per
configuration**. For 7,000 configurations, that's **30–90 CPU-hours**. That
is not a reasonable research workflow.

---

## Part 6: GPU Acceleration with JAX

### What is a GPU?

A GPU (Graphics Processing Unit) was originally designed to render video game
graphics — computing the color of millions of pixels simultaneously. It turns
out the same hardware is perfect for any computation where you need to do the
*same operation on thousands of independent inputs at once*.

Our Iris simulation is exactly that: 150 independent flower samples, each
requiring the same integration, with no dependencies between them.

### What is JAX?

JAX is a Python library from Google that lets you write NumPy-like code and
then:
1. **JIT-compile** it to run efficiently on CPUs or GPUs
2. **Vectorize** it with `vmap` — automatically parallelize over a batch
3. **Differentiate** it (we don't use this, but it's why JAX was built)

### How we use JAX

**`jax.lax.scan`** compiles the 40,000-step integration loop into a single
fused GPU kernel. Instead of Python calling a function 40,000 times (slow), the
GPU executes a tight loop in hardware (fast).

**`jax.vmap`** vectorizes the single-sample simulation over all 150 Iris
samples at once. The GPU executes all samples in parallel.

**`jax.jit`** compiles everything the first time it runs. Subsequent calls reuse
the compiled kernel — this is why the "cached" GPU call is so much faster than
the "warmup" call.

### The speedup

On a GPU, the cached kernel call achieves **50–100× speedup** over the CPU
baseline:
- CPU: ~30 seconds for 112 training samples
- GPU (warmup): ~30 seconds (includes compilation)
- GPU (cached): ~0.3 seconds

This transforms a 90-hour CPU sweep into a **3–4 hour GPU sweep**. That is
the central practical contribution of this project.

### How the ring buffer works

Time-delayed coupling requires remembering past states. When laser 2 is at
step `n`, it needs the state of laser 1 at step `n - τ/Δt`. We store this
in a **ring buffer** — a circular array of fixed depth:

```
ring[0] = state at time 0
ring[1] = state at time Δt
ring[2] = state at time 2Δt
...
ring[depth-1] = state at time (depth-1)·Δt

After ring fills: ring[n % depth] = state at time n·Δt
To look back by d steps: ring[(n - d + depth) % depth]
```

This is a classic CS trick, and JAX can handle it with `jax.lax.scan` because
the ring depth is fixed at compile time.

---

## Part 7: The Meta-Learning MLP

### What is meta-learning?

Meta-learning is "learning to learn." In this context: instead of sweeping
7,000 configurations every time we want to find good fab parameters, we train
a model to *predict* which configuration will work best for a new task —
without running the sweep.

### The two models

**MLP Regressor** (continuous prediction):
- Input: 7 numbers describing the *workload* (how many training samples, how
  long the simulation window, what kind of task)
- Output: 9 numbers describing the *optimal fab configuration* (best `k`, best
  `i_min`, best spacing, etc.)
- Trained on: the best configuration found for each historical sweep scenario

**MLP Motif Classifier** (discrete prediction):
- Input: same 7 workload numbers
- Output: which coupling topology (motif) will work best — one of {auxiliary,
  chain, relay, competitive, mixed}
- This is harder, but choosing the right topology is the most impactful decision

### Leave-One-Out Cross-Validation (LOOCV)

Since we only have a handful of scenarios (each sweep produces one "optimal
configuration" per scenario), we use LOOCV: train on all scenarios except one,
test on the held-out one, repeat for each scenario. This measures how well the
MLP generalizes to entirely new workload conditions.

### Why this matters

Once the MLP is trained, a new user can describe their task in 7 numbers and
instantly receive a recommended fab configuration — no sweep required. This
is the path from "research simulator" to "practical engineering tool."

---

## Part 8: The Figures and Their Story

Each figure in the paper tells one part of the story:

| Figure | What it shows | Why it matters |
|--------|--------------|----------------|
| 1 — Speedup bar chart | CPU vs GPU time at different batch sizes | The GPU speedup makes the sweep feasible |
| 2 — Accuracy heatmap | Iris accuracy across motif and coupling strength | There is a rich optimization landscape worth sweeping |
| 3 — Baseline comparison | Reservoir vs Ridge / MLP / RF / GBT | The reservoir is competitive despite using only a linear readout |
| 4 — Regressor R² | How well the MLP predicts each fab parameter | Some parameters are predictable from workload descriptors |
| 5 — Confusion matrix | Which motifs the MLP predicts correctly | The classifier generalizes across workload shapes |
| 6 — Feature importance | Which workload features drive fab recommendations | Training set size matters most |
| 7 — Laser trajectories | The physical dynamics under 3 motifs | Different topologies produce qualitatively different physics |
| 8 — Ablation | Accuracy and speed vs time resolution and decimation | The default parameters are justified empirically |
| 9 — t-SNE | 20D reservoir features projected to 2D | The 3 Iris classes are linearly separable in reservoir space |

---

## Part 9: How the Code Is Organized

```
model7_reservoir.py        — CPU simulator (NumPy, pure Python loops)
model7_reservoir_gpu.py    — GPU simulator (JAX, lax.scan + vmap)
benchmark_iris.py          — Run one Iris classification experiment
compare_cpu_gpu.py         — Quick CPU vs GPU timing printout

fab_sweep.py               — Run many (scenario, config) pairs, save CSV
fab_design.py              — Latin Hypercube sampling for the sweep
fab_meta_model.py          — Train MLP recommender on sweep results
pipeline_iris/run_pipeline.py — All-in-one entry point

benchmark_timing.py        — [NEW] Structured timing sweep → CSV for Figure 1
mlp_classification_harness.py — [NEW] LOOCV training harness (classification only)
generate_figures.py        — [NEW] Generate all 9 paper figures
paper/main.tex             — [NEW] Full NeurIPS paper
```

### Running the full pipeline

```bash
# 1. Generate timing data (on a GPU node)
python benchmark_timing.py

# 2. Run the full fab sweep (on a GPU node, takes ~3.5 hours)
python pipeline_iris/run_pipeline.py --standard-grid --stretch-scenarios --gpu

# 3. Train the MLP harness on sweep results
python mlp_classification_harness.py \
    --sweep-csv pipeline_iris/outputs/sweeps/fab_long/<run_id>/sweep_long.csv

# 4. Generate all 9 figures
python generate_figures.py

# 5. Compile the paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex
```

---

## Part 10: The Big Picture

Here is the complete chain of ideas:

```
Problem:  Photonic reservoirs are powerful but have too many fab parameters to optimize by hand.
          Systematic sweeps are too slow on CPU.

Solution: GPU-accelerate the simulator → 50–100× speedup → systematic sweep becomes feasible.

Result 1: We can now map the full accuracy landscape (Fig 2) and identify the best configs.

Result 2: We train an MLP on the sweep data → instant fab recommendations for new workloads.

Result 3: The optimized reservoir matches classical ML baselines on Iris (Fig 3).

Result 4: t-SNE (Fig 9) and trajectory plots (Fig 7) explain *why* it works — the physics
          creates rich, separable feature spaces.

Takeaway: GPU simulation is the key enabling technology for systematic photonic ML research.
```

This is not just a cool demo. It is a methodological contribution: **you can
now treat photonic fab parameters as a hyperparameter optimization problem**,
with the GPU-accelerated simulator as the objective function. The same
framework extends naturally to other photonic systems, other tasks, and
active-learning loops that adaptively query the most informative configurations.

---

*Generated as part of the GPU-accelerated photonic reservoir computing research project.*
*See `paper/main.tex` for the full academic write-up.*
