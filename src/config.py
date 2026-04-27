"""Experiment configuration: seeds, sizes, capacities, and file paths."""

import numpy as np
import torch


SEED = 0

# Sample sizes
N = 400
N_EVAL = 10_000

# Problem dimensions
TREATMENT = 10
DIMENSION_DATA = 30

T = TREATMENT
D = DIMENSION_DATA

# Inverse temperature for the softmax policy and the inner objective.
# Lower τ → softmax sharpens toward argmax (G's policy class converges to
# the deterministic LP solution).
TAU = 0.1

# Data-generating process knobs
SIGMA_Y = 0.05
PROPENSITY_STRENGTH = 0.7
OUTCOME_STRENGTH = 2.0
TREATMENT_EFFECT_STRENGTH = 6.0  # nonlinear basis E[X(1-X)]=1/6 → 3× compensation vs old linear DGP
CLIP_PROPENSITY = 0.02

# Distinct seeds for train and eval splits
TRAIN_SEED = SEED
EVAL_SEED = 10_000

# Per-treatment capacity vector. Control unconstrained (B[0]=1); 10% cap on
# each of the 9 non-control arms. With T=10, S2 splits N samples 10 ways
# (≈ N/10 per-arm samples for outcome regression) → at small N, S2's
# per-arm m̂_t is essentially noise. G's MLP shares its trunk across all
# treatments and benefits from the full N samples for the shared features.
B = np.array([1.0] + [0.1] * 9, dtype=float)

assert len(B) == T, f"B must have length T={T}, got len(B)={len(B)}"
assert B.sum() >= 1.0, "Feasibility requires sum(B) >= 1."

# Snapshot paths
TRAIN_DATA_PATH = "train_data.npz"
EVAL_DATA_PATH = "eval_data.npz"
CONFIG_PATH = "experiment_config.npz"
RESULTS_CSV_PATH = "all_method_results.csv"


def setup_global_state():
    """Set torch default dtype and seed numpy/torch."""
    torch.set_default_dtype(torch.float64)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
