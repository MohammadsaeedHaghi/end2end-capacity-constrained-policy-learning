"""Random and oracle-greedy-no-cap baselines."""

import numpy as np

from .policy import oracle_value_soft, oracle_value_onehot


def evaluate_random_policy(eval_data, b, T):
    N_eval_local = eval_data["X"].shape[0]
    pi_rand = np.full((N_eval_local, T), 1.0 / T)

    V_oracle = oracle_value_soft(pi_rand, eval_data["Y_pot"])
    print(f"[baseline] random uniform  V_oracle={V_oracle: .4f}")

    alloc = pi_rand.mean(axis=0)
    return {
        "tag": "random",
        "V_IPW": np.nan,
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


def evaluate_oracle_greedy_no_cap(eval_data, b, T):
    Y_pot = eval_data["Y_pot"]
    N_eval_local = Y_pot.shape[0]

    assignments = Y_pot.argmax(axis=1)
    pi_onehot = np.zeros_like(Y_pot)
    pi_onehot[np.arange(N_eval_local), assignments] = 1.0

    V_oracle = oracle_value_onehot(pi_onehot, Y_pot)

    alloc = pi_onehot.mean(axis=0)
    cap_viol_sup = float(np.maximum(alloc - b, 0.0).max())

    print(
        f"[baseline] oracle greedy-no-cap  "
        f"V_oracle={V_oracle: .4f}  "
        f"capviol={cap_viol_sup:.3e}"
    )

    return {
        "tag": "oracle_greedy_no_cap",
        "V_IPW": np.nan,
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
