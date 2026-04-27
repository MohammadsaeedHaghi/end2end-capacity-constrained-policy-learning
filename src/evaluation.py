"""Evaluation utilities for soft and one-hot policies."""

import numpy as np
import torch

from .policy import (
    softmax_policy,
    oracle_value_soft,
    oracle_value_onehot,
    ipw_value_np,
    dr_value_np,
)


def evaluate_GF_model(model, mu_train, train_data, eval_data, m_hat_eval,
                      b, tau, T, tag):
    """
    Apply the trained policy (theta, mu_train) to both splits.
    mu_train is frozen — it is part of the policy, not re-solved on eval.
    """
    if not torch.is_tensor(mu_train):
        mu_train = torch.tensor(mu_train)

    with torch.no_grad():
        X_e = torch.tensor(eval_data["X"])
        pi_e = softmax_policy(model(X_e), mu_train, tau).numpy()

        X_t = torch.tensor(train_data["X"])
        pi_t = softmax_policy(model(X_t), mu_train, tau).numpy()

    V_ipw_train = ipw_value_np(
        pi_t, train_data["T"], train_data["Y"], train_data["e_T"]
    )
    V_ipw_eval = ipw_value_np(
        pi_e, eval_data["T"], eval_data["Y"], eval_data["e_T"]
    )
    V_dr_eval = dr_value_np(
        pi_e, eval_data["T"], eval_data["Y"], eval_data["e_T"], m_hat_eval
    )
    V_orc = oracle_value_soft(pi_e, eval_data["Y_pot"])
    alloc = pi_e.mean(axis=0)

    return {
        "tag": tag,
        "V_IPW_train": V_ipw_train,
        "V_IPW_eval": V_ipw_eval,
        "V_DR_eval": V_dr_eval,
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


def evaluate_greedy_no_cap_from_model(model, train_data, eval_data, m_hat_eval,
                                      b, T, tag="greedy_no_cap_from_G"):
    """Greedy argmax over trained scores, ignoring capacity."""
    with torch.no_grad():
        M_eval = model(torch.tensor(eval_data["X"])).numpy()
        M_train = model(torch.tensor(train_data["X"])).numpy()

    pi_e = np.zeros_like(M_eval)
    pi_e[np.arange(len(M_eval)), M_eval.argmax(axis=1)] = 1.0

    pi_t = np.zeros_like(M_train)
    pi_t[np.arange(len(M_train)), M_train.argmax(axis=1)] = 1.0

    V_ipw_train = ipw_value_np(
        pi_t, train_data["T"], train_data["Y"], train_data["e_T"]
    )
    V_ipw_eval = ipw_value_np(
        pi_e, eval_data["T"], eval_data["Y"], eval_data["e_T"]
    )
    V_dr_eval = dr_value_np(
        pi_e, eval_data["T"], eval_data["Y"], eval_data["e_T"], m_hat_eval
    )
    V_orc = oracle_value_soft(pi_e, eval_data["Y_pot"])
    alloc = pi_e.mean(axis=0)

    return {
        "tag": tag,
        "V_IPW_train": V_ipw_train,
        "V_IPW_eval": V_ipw_eval,
        "V_DR_eval": V_dr_eval,
        "V_oracle": V_orc,
        "alloc": alloc,
        "cap_viol_sup": float(np.maximum(alloc - b, 0.0).max()),
        "cap_ok": bool(np.all(alloc <= b + 1e-3)),
        "method": tag,
        "mu": np.full(T, np.nan),
        "lp_status": "NA",
        "lp_time": np.nan,
        "total_time": np.nan,
    }


def evaluate_policy(pi_onehot_eval, assignments_eval,
                    pi_onehot_train, assignments_train,
                    train_data, eval_data, m_hat_eval, b, tag):
    """Evaluate a one-hot policy on both splits (used by S2 pipeline)."""
    V_ipw_train = ipw_value_np(
        pi_onehot_train, train_data["T"], train_data["Y"], train_data["e_T"]
    )
    V_ipw_eval = ipw_value_np(
        pi_onehot_eval, eval_data["T"], eval_data["Y"], eval_data["e_T"]
    )
    V_dr_eval = dr_value_np(
        pi_onehot_eval, eval_data["T"], eval_data["Y"], eval_data["e_T"], m_hat_eval
    )
    V_oracle = oracle_value_onehot(pi_onehot_eval, eval_data["Y_pot"])

    alloc = pi_onehot_eval.mean(axis=0)
    cap_viol = np.maximum(alloc - np.asarray(b), 0.0)
    cap_viol_sup = float(cap_viol.max())
    cap_ok = bool(np.all(alloc <= np.asarray(b) + 1e-3))

    print(
        f"[eval] {tag:12s}  "
        f"V_IPW_train={V_ipw_train: .4f}  "
        f"V_IPW_eval={V_ipw_eval: .4f}  "
        f"V_DR_eval={V_dr_eval: .4f}  "
        f"V_oracle={V_oracle: .4f}  "
        f"capviol={cap_viol_sup:.3e}  "
        f"cap_ok={cap_ok}"
    )

    return {
        "tag": tag,
        "V_IPW_train": V_ipw_train,
        "V_IPW_eval": V_ipw_eval,
        "V_DR_eval": V_dr_eval,
        "V_oracle": V_oracle,
        "alloc": alloc,
        "cap_viol_sup": cap_viol_sup,
        "cap_ok": cap_ok,
    }
