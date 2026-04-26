"""Evaluation utilities for soft and one-hot policies."""

import numpy as np
import torch

from .policy import softmax_policy, oracle_value_soft, oracle_value_onehot


def evaluate_GF_model(model, mu_train, eval_data, b, tau, T, tag):
    """
    Apply the trained policy (theta, mu_train) to eval data.
    mu_train is frozen — it is part of the policy, not re-solved on eval.
    """
    if not torch.is_tensor(mu_train):
        mu_train = torch.tensor(mu_train)

    with torch.no_grad():
        X_e = torch.tensor(eval_data["X"])
        M_e = model(X_e)
        pi_e = softmax_policy(M_e, mu_train, tau)

    pi_np = pi_e.numpy()
    V_orc = oracle_value_soft(pi_np, eval_data["Y_pot"])
    alloc = pi_np.mean(axis=0)

    return {
        "tag": tag,
        "V_IPW": np.nan,
        "V_oracle": V_orc,
        "alloc": alloc,
        "cap_viol_sup": float(np.maximum(alloc - b, 0.0).max()),
        "cap_ok": bool(np.all(alloc <= b + 1e-3)),
        "method": tag,
        "mu": mu_train.detach().cpu().numpy(),
        "lp_status": "NA",
        "lp_time": np.nan,
        "total_time": np.nan,
    }


def evaluate_greedy_no_cap_from_model(model, eval_data, b, T, tag="greedy_no_cap_from_G"):
    with torch.no_grad():
        M_eval = model(torch.tensor(eval_data["X"])).numpy()

    pi_greedy = np.zeros_like(M_eval)
    pi_greedy[np.arange(len(M_eval)), M_eval.argmax(axis=1)] = 1.0

    V_greedy = oracle_value_soft(pi_greedy, eval_data["Y_pot"])
    alloc = pi_greedy.mean(axis=0)

    return {
        "tag": tag,
        "V_IPW": np.nan,
        "V_oracle": V_greedy,
        "alloc": alloc,
        "cap_viol_sup": float(np.maximum(alloc - b, 0.0).max()),
        "cap_ok": bool(np.all(alloc <= b + 1e-3)),
        "method": tag,
        "mu": np.full(T, np.nan),
        "lp_status": "NA",
        "lp_time": np.nan,
        "total_time": np.nan,
    }


def evaluate_policy(pi_onehot, assignments, eval_data, b, tag):
    """Evaluate a one-hot policy on evaluation data (used by S2 pipeline)."""
    T_obs = eval_data["T"]
    Y_obs = eval_data["Y"]
    e_T_obs = eval_data["e_T"]
    Y_pot = eval_data["Y_pot"]

    match = (assignments == T_obs).astype(float)

    V_ipw = float((match * Y_obs / e_T_obs).mean())
    V_oracle = oracle_value_onehot(pi_onehot, Y_pot)

    alloc = pi_onehot.mean(axis=0)
    cap_viol = np.maximum(alloc - np.asarray(b), 0.0)
    cap_viol_sup = float(cap_viol.max())
    cap_ok = bool(np.all(alloc <= np.asarray(b) + 1e-3))

    print(
        f"[eval] {tag:12s}  "
        f"V_IPW={V_ipw: .4f}  "
        f"V_oracle={V_oracle: .4f}  "
        f"capviol={cap_viol_sup:.3e}  "
        f"cap_ok={cap_ok}"
    )

    return {
        "tag": tag,
        "V_IPW": V_ipw,
        "V_oracle": V_oracle,
        "alloc": alloc,
        "cap_viol_sup": cap_viol_sup,
        "cap_ok": cap_ok,
    }
