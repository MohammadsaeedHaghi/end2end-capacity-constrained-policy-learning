"""Experiment configuration: seeds, sizes, capacities, and file paths."""

import numpy as np
import torch


SEED = 0

# Sample sizes
N = 400
N_EVAL = 10_000

# Problem dimensions
TREATMENT = 4
DIMENSION_DATA = 35

T = TREATMENT
D = DIMENSION_DATA

# Inverse temperature for the softmax policy and the inner objective
TAU = 0.5

# Data-generating process knobs
SIGMA_Y = 0.5
PROPENSITY_STRENGTH = 0.7
OUTCOME_STRENGTH = 1.0
CLIP_PROPENSITY = 0.02

# Distinct seeds for train and eval splits
TRAIN_SEED = SEED
EVAL_SEED = 10_000

# Per-treatment capacity vector (b_0 = 1 leaves arm 0 effectively unconstrained)
B = np.array([1.00, 0.10, 0.10, 0.08], dtype=float)

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
