"""Random and oracle-greedy-no-cap baselines."""

import numpy as np

from .policy import (
    oracle_value_soft,
    oracle_value_onehot,
    ipw_value_np,
    dr_value_np,
)


def evaluate_random_policy(train_data, eval_data, m_hat_eval, b, T):
    pi_e = np.full((eval_data["X"].shape[0], T), 1.0 / T)
    pi_t = np.full((train_data["X"].shape[0], T), 1.0 / T)

    V_ipw_train = ipw_value_np(
        pi_t, train_data["T"], train_data["Y"], train_data["e_T"]
    )
    V_ipw_eval = ipw_value_np(
        pi_e, eval_data["T"], eval_data["Y"], eval_data["e_T"]
    )
    V_dr_eval = dr_value_np(
        pi_e, eval_data["T"], eval_data["Y"], eval_data["e_T"], m_hat_eval
    )
    V_oracle = oracle_value_soft(pi_e, eval_data["Y_pot"])
    alloc = pi_e.mean(axis=0)

    print(
        f"[baseline] random uniform  "
        f"V_IPW_train={V_ipw_train: .4f}  "
        f"V_IPW_eval={V_ipw_eval: .4f}  "
        f"V_DR_eval={V_dr_eval: .4f}  "
        f"V_oracle={V_oracle: .4f}"
    )

    return {
        "tag": "random",
        "V_IPW_train": V_ipw_train,
        "V_IPW_eval": V_ipw_eval,
        "V_DR_eval": V_dr_eval,
        "V_oracle": V_oracle,
        "alloc": alloc,
        "cap_viol_sup": float(np.maximum(alloc - b, 0.0).max()),
        "cap_ok": bool(np.all(alloc <= b + 1e-3)),
        "method": "random",
        "mu": np.full(T, np.nan),
        "lp_status": "NA",
        "lp_time": np.nan,
        "total_time": np.nan,
    }


def evaluate_oracle_greedy_no_cap(train_data, eval_data, m_hat_eval, b, T):
    Y_pot_e = eval_data["Y_pot"]
    Y_pot_t = train_data["Y_pot"]

    a_e = Y_pot_e.argmax(axis=1)
    pi_e = np.zeros_like(Y_pot_e)
    pi_e[np.arange(len(Y_pot_e)), a_e] = 1.0

    a_t = Y_pot_t.argmax(axis=1)
    pi_t = np.zeros_like(Y_pot_t)
    pi_t[np.arange(len(Y_pot_t)), a_t] = 1.0

    V_ipw_train = ipw_value_np(
        pi_t, train_data["T"], train_data["Y"], train_data["e_T"]
    )
    V_ipw_eval = ipw_value_np(
        pi_e, eval_data["T"], eval_data["Y"], eval_data["e_T"]
    )
    V_dr_eval = dr_value_np(
        pi_e, eval_data["T"], eval_data["Y"], eval_data["e_T"], m_hat_eval
    )
    V_oracle = oracle_value_onehot(pi_e, Y_pot_e)

    alloc = pi_e.mean(axis=0)
    cap_viol_sup = float(np.maximum(alloc - b, 0.0).max())

    print(
        f"[baseline] oracle greedy-no-cap  "
        f"V_IPW_train={V_ipw_train: .4f}  "
        f"V_IPW_eval={V_ipw_eval: .4f}  "
        f"V_DR_eval={V_dr_eval: .4f}  "
        f"V_oracle={V_oracle: .4f}  "
        f"capviol={cap_viol_sup:.3e}"
    )

    return {
        "tag": "oracle_greedy_no_cap",
        "V_IPW_train": V_ipw_train,
        "V_IPW_eval": V_ipw_eval,
        "V_DR_eval": V_dr_eval,
        "V_oracle": V_oracle,
        "alloc": alloc,
        "cap_viol_sup": cap_viol_sup,
        "cap_ok": bool(np.all(alloc <= b + 1e-3)),
        "method": "oracle_greedy_no_cap",
        "mu": np.full(T, np.nan),
        "lp_status": "NA",
        "lp_time": np.nan,
        "total_time": np.nan,
    }
