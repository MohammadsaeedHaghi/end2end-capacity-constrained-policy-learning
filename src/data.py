"""Synthetic observational-data generator and (de)serialization helpers."""

import numpy as np
from scipy.stats import norm as _norm


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

    # Correlated covariates via Gaussian copula:
    # - Z ~ N(0, Σ_AR1) with Σ_AR1[i,j] = rho^|i-j| (rho = 0.7)
    # - X[i] = Φ(Z[i])  → uniform marginals on [0, 1] but correlated jointly.
    # Marginal-symmetry of the φ basis below still gives cov(φ_j(X), X_k) = 0
    # for all j, k under this copula, so linear methods stay linear-blind to
    # treatment effects. KNN's L2 distance, by contrast, is misled by the
    # 24 correlated noise dimensions on top of the few signal dims.
    rho = 0.7
    idx = np.arange(d)
    Sigma_AR1 = rho ** np.abs(idx[:, None] - idx[None, :])
    L_chol = np.linalg.cholesky(Sigma_AR1 + 1e-9 * np.eye(d))
    Z = rng.normal(size=(N, d)) @ L_chol.T
    X = _norm.cdf(Z)

    # Three nonlinear bands stacked into phi(X), each in [0, ·]:
    #     phi_a = X (1 - X)        ∈ [0, 0.25]   (parabola, mean 1/6)
    #     phi_b = sin(π X)          ∈ [0, 1]      (sinusoid, mean 2/π)
    #     phi_c = |X - 0.5|         ∈ [0, 0.5]    (V-shape,  mean 1/4)
    # All three are even functions of (Z = Φ⁻¹(X)) and so have cov 0 with
    # any X_k under the Gaussian copula → linear-blind to treatment effects.
    phi_a = X * (1.0 - X)
    phi_b = np.sin(np.pi * X)
    phi_c = np.abs(X - 0.5)
    phi_X = np.concatenate([phi_a, phi_b, phi_c], axis=1)   # (N, 3d)
    phi_dim = 3 * d

    # Shared baseline direction in raw X-space (random unit norm). Linear
    # methods CAN fit this; misspecification is isolated to the treatment-
    # effect contrast that drives policy decisions.
    beta_base = rng.normal(size=d)
    beta_base = beta_base / np.linalg.norm(beta_base)

    # Treatment-effect vectors β_dev_t in phi-space, deterministic, sparse,
    # non-negative, nested. Each treatment t > 0 activates a new X-coordinate
    # and routes it through all three phi bands simultaneously, with
    # magnitudes [1.0, 0.5, 0.7] for (phi_a, phi_b, phi_c) respectively.
    # Combined with phi(X) ≥ 0, this gives strict elementwise dominance on
    # conditional means:
    #     E[Y^t | X] >= E[Y^{t-1} | X]   for all t >= 1.
    mag_a, mag_b, mag_c = 1.0, 0.5, 0.7
    beta_dev = np.zeros((T, phi_dim))
    for t in range(1, T):
        beta_dev[t] = beta_dev[t - 1]
        k = t - 1                  # X-coordinate index activated by step t
        if k < d:
            beta_dev[t, k] = mag_a              # phi_a band: index k
            beta_dev[t, d + k] = mag_b          # phi_b band: index d+k
            beta_dev[t, 2 * d + k] = mag_c      # phi_c band: index 2d+k

    # Y^t(x) = OS · β_base · x + TE · β_dev_t · phi(x) + ε
    # Y^0(x) = OS · β_base · x + ε                       (control)
    linear_baseline = outcome_strength * (X @ beta_base)                 # (N,)
    nonlinear_effect = treatment_effect_strength * (phi_X @ beta_dev.T)  # (N, T)
    noise = sigma_y * rng.normal(size=(N, T))
    Y_pot = linear_baseline[:, None] + nonlinear_effect + noise

    # Diagnostic Beta: per-treatment coefficient on the *visible* X if linear
    # methods were oracle-correct. Since the treatment effect lives in phi(X),
    # this is just β_base (shared across t) for every treatment.
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
