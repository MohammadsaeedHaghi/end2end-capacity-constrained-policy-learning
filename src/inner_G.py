"""
Inner layer for the convex Lagrangian dual G(mu).

    G(mu) = (tau / N) * sum_i log sum_t exp((m_{t,i} - mu_t) / tau) + b . mu

Solved as a CVXPYLayer so gradients flow back through diffcp.
"""

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def build_mu_layer_G(N, T, tau, b):
    M_param = cp.Parameter((N, T))
    mu_var = cp.Variable(T, nonneg=True)

    ones_col = np.ones((N, 1))
    U = M_param - ones_col @ cp.reshape(mu_var, (1, T))

    obj = (tau / N) * cp.sum(cp.log_sum_exp(U / tau, axis=1)) + b @ mu_var
    prob = cp.Problem(cp.Minimize(obj))

    assert prob.is_dpp(), "G is not DPP — cvxpylayers will refuse"

    return CvxpyLayer(prob, parameters=[M_param], variables=[mu_var])


# Module-level cache of the CVXPYLayer parametrized for the training (N, T).
MU_LAYER_G = None


def initialize_G_layer(N, T, tau, b):
    """Build and cache the CVXPYLayer for the given training (N, T, tau, b)."""
    global MU_LAYER_G
    MU_LAYER_G = build_mu_layer_G(N, T, tau, b)
    print("[G] CVXPYLayer initialized.")


def mu_of_M_G(M):
    """M: (N, T) torch -> mu*: (T,) torch. Gradients flow via diffcp."""
    if MU_LAYER_G is None:
        raise RuntimeError("Call initialize_G_layer(N, T, tau, b) first.")

    mu_star, = MU_LAYER_G(
        M,
        solver_args={"solve_method": "SCS", "eps": 1e-9},
    )
    return mu_star
