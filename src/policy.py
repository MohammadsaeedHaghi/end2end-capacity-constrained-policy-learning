"""Softmax policy + IPW and oracle value functionals."""

import torch


def softmax_policy(M, mu, tau):
    """pi_{t,i}(x) = softmax_t((m_{t,i} - mu_t) / tau)."""
    return torch.softmax((M - mu.unsqueeze(0)) / tau, dim=1)


def ipw_value(pi, T_obs, Y, e_T):
    """V_IPW(theta) = (1/N) sum_i pi_{T_i,i} * Y_i / e_{T_i}(X_i)."""
    pi_t = pi.gather(1, T_obs.unsqueeze(1)).squeeze(1)
    return (pi_t * Y / e_T).mean()


def oracle_value_soft(pi_np, Y_pot_np):
    """Oracle value of a soft policy given counterfactual outcomes."""
    return float((pi_np * Y_pot_np).sum(axis=1).mean())


def oracle_value_onehot(pi_onehot, Y_pot):
    """Oracle value of a one-hot policy given counterfactual outcomes."""
    return float((pi_onehot * Y_pot).sum(axis=1).mean())
