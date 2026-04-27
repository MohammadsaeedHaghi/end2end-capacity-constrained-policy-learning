"""
Inner layer for the convex Lagrangian dual G(mu).

    G(mu) = (tau / N) * sum_i log sum_t exp((m_{t,i} - mu_t) / tau) + b . mu

Solved as a CVXPYLayer for the training-time gradient path, and via scipy
L-BFGS-B for forward-only evaluation at arbitrary N (used by μ-on-eval
variants where the cached CVXPYLayer is size-locked to training N).
"""

import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from scipy.optimize import minimize


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
        solver_args={"solve_method": "SCS", "eps": 1e-6, "max_iters": 5000},
    )
    return mu_star


def _G_torch(mu, M, b, tau):
    U = (M - mu.unsqueeze(0)) / tau
    return tau * torch.logsumexp(U, dim=1).mean() + (mu * b).sum()


def solve_G_scipy(M_t, b, tau, mu_init=None, tol=1e-10, maxiter=500):
    """Forward-only G solve at arbitrary N. No gradients. Used to fit μ on
    eval data (different N than training, so the cached CVXPYLayer can't be
    reused)."""
    T_ = M_t.shape[1]

    mu0 = np.zeros(T_) if mu_init is None else mu_init.detach().cpu().numpy()
    mu0 = np.maximum(mu0, 0.0)

    M_det = M_t.detach()
    b_t = b if torch.is_tensor(b) else torch.tensor(b)
    b_det = b_t.detach()

    def fg(mu_np):
        with torch.enable_grad():
            mu = torch.tensor(mu_np, dtype=M_det.dtype, requires_grad=True)
            Gv = _G_torch(mu, M_det, b_det, float(tau))
            g, = torch.autograd.grad(Gv, mu)
        return float(Gv.item()), g.detach().numpy().astype(np.float64)

    res = minimize(
        fg, mu0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * T_,
        options={"ftol": tol, "gtol": tol, "maxiter": maxiter},
    )
    return torch.tensor(np.maximum(res.x, 0.0), dtype=M_t.dtype)
