"""Synthetic observational-data generator and (de)serialization helpers."""

import numpy as np


def generate_data(
    N,
    seed=0,
    d=30,
    T=15,
    sigma_y=0.5,
    propensity_strength=0.7,
    outcome_strength=1.0,
    treatment_effect_strength=1.0,
    clip_propensity=0.02,
):
    """
    Generate synthetic observational data with d features and T treatments.

    Returns dict with keys:
        X      : covariates, shape (N, d)
        T      : observed treatment, shape (N,)
        Y      : observed outcome, shape (N,)
        e_T    : observed propensity P(T_i | X_i), shape (N,)
        Y_pot  : potential outcomes, shape (N, T)
        E      : full propensity matrix, shape (N, T)
        Beta   : outcome coefficients, shape (T, d)
        Alpha  : propensity coefficients, shape (T, d)
    """
    rng = np.random.default_rng(seed)

    # Non-negative covariates, X ~ Uniform[0, 1]^d.
    X = rng.uniform(0.0, 1.0, size=(N, d))

    # Hidden basis for treatment effects: phi(X) = X(1-X), elementwise.
    # Under X ~ U[0,1], cov(phi_j(X), X_j) = 0, so OLS / Lasso of Y on X
    # project the phi-component to zero slope -> linear methods can only
    # estimate constant per-treatment offsets and cannot personalize.
    # Tree, KNN, and MLP-based methods recover the nonlinear shape.
    phi_X = X * (1.0 - X)

    # Shared baseline direction in X-space (random, unit L2). Linear methods
    # CAN fit this; the misspecification is isolated to the treatment-effect
    # contrast, which is the part that drives policy decisions.
    beta_base = rng.normal(size=d)
    beta_base = beta_base / np.linalg.norm(beta_base)

    # Treatment-effect vectors β_dev_t in phi-space: deterministic, sparse,
    # non-negative, nested. Each treatment t > 0 activates two more entries
    # than t-1, with magnitudes [1.0, 0.6] per pair. Combined with phi(X) ≥ 0
    # (since X ∈ [0, 1]), this gives strict elementwise dominance on
    # conditional means:
    #     E[Y^t | X] >= E[Y^{t-1} | X]   for all t >= 1.
    features_per_step = 2
    unit_magnitudes = np.array([1.0, 0.6])
    beta_dev = np.zeros((T, d))
    for t in range(1, T):
        beta_dev[t] = beta_dev[t - 1]
        start = (t - 1) * features_per_step
        end = min(start + features_per_step, d)
        if start < d:
            beta_dev[t, start:end] = unit_magnitudes[: end - start]

    # Y^t(x) = OS · β_base · x + TE · β_dev_t · phi(x) + ε
    # Y^0(x) = OS · β_base · x + ε                       (control)
    linear_baseline = outcome_strength * (X @ beta_base)             # (N,)
    nonlinear_effect = treatment_effect_strength * (phi_X @ beta_dev.T)  # (N, T)
    noise = sigma_y * rng.normal(size=(N, T))
    Y_pot = linear_baseline[:, None] + nonlinear_effect + noise

    # Diagnostic Beta: per-treatment coefficient on the *visible* X if linear
    # methods were oracle-correct. Since the treatment effect lives in phi(X),
    # this is just β_base (shared across t) for every treatment — i.e. linear
    # methods see no per-treatment X-direction, only constant offsets.
    Beta = (outcome_strength * beta_base[None, :]).repeat(T, axis=0)

    Alpha = rng.normal(size=(T, d))
    Alpha = Alpha / np.linalg.norm(Alpha, axis=1, keepdims=True)
    Alpha[0] = 0.0

    # Center X around its mean (0.5) for the propensity score so that no
    # treatment is structurally over-/under-represented when X >= 0.
    S = propensity_strength * ((X - 0.5) @ Alpha.T)
    S_shift = S - S.max(axis=1, keepdims=True)
    E = np.exp(S_shift)
    E = E / E.sum(axis=1, keepdims=True)

    if clip_propensity is not None:
        E = np.clip(E, clip_propensity, 1.0)
        E = E / E.sum(axis=1, keepdims=True)

    T_obs = np.array([rng.choice(T, p=E[i]) for i in range(N)])
    Y_obs = Y_pot[np.arange(N), T_obs]
    e_obs = E[np.arange(N), T_obs]

    return {
        "X": X,
        "T": T_obs,
        "Y": Y_obs,
        "e_T": e_obs,
        "Y_pot": Y_pot,
        "E": E,
        "Beta": Beta,
        "Alpha": Alpha,
    }


def load_npz_as_dict(path):
    file = np.load(path, allow_pickle=True)
    return {k: file[k] for k in file.files}


def load_experiment(train_path, eval_path, config_path):
    return (
        load_npz_as_dict(train_path),
        load_npz_as_dict(eval_path),
        load_npz_as_dict(config_path),
    )
