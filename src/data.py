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

    X = rng.normal(size=(N, d))

    beta_base = rng.normal(size=d)
    beta_base = beta_base / np.linalg.norm(beta_base)

    beta_dev = rng.normal(size=(T, d))
    beta_dev = beta_dev / np.linalg.norm(beta_dev, axis=1, keepdims=True)

    Beta = outcome_strength * (beta_base[None, :] + 0.5 * beta_dev)
    Beta[0] = 0.5 * beta_base  # baseline / control

    noise = sigma_y * rng.normal(size=(N, T))
    Y_pot = X @ Beta.T + noise

    Alpha = rng.normal(size=(T, d))
    Alpha = Alpha / np.linalg.norm(Alpha, axis=1, keepdims=True)
    Alpha[0] = 0.0

    S = propensity_strength * (X @ Alpha.T)
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
