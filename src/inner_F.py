"""
Inner layer for the literal non-convex F(mu).

    F(mu) = (1/N) sum_i sum_t sigma_{t,i}(mu) * (m_{t,i} - mu_t) + b . mu

Forward solves argmin_{mu >= 0} F via scipy L-BFGS-B with warm starts.
Backward applies the implicit-function theorem to the KKT system, with the
active set (mu_t = 0) handled by augmenting with multipliers.
"""

import numpy as np
import torch
from scipy.optimize import minimize


def _F_torch(mu, M, b, tau):
    U = (M - mu.unsqueeze(0)) / tau
    sigma = torch.softmax(U, dim=1)
    V = M - mu.unsqueeze(0)
    return (sigma * V).sum(dim=1).mean() + (mu * b).sum()


def _solve_F_inner(M_t, b_t, tau, mu_init=None, tol=1e-10, maxiter=200):
    """argmin_{mu >= 0} F(mu; M) via scipy L-BFGS-B."""
    T_ = M_t.shape[1]

    mu0 = np.zeros(T_) if mu_init is None else mu_init.detach().cpu().numpy()
    mu0 = np.maximum(mu0, 0.0)

    M_det = M_t.detach()
    b_det = b_t.detach()

    def fg(mu_np):
        with torch.enable_grad():
            mu = torch.tensor(mu_np, dtype=M_det.dtype, requires_grad=True)
            Fv = _F_torch(mu, M_det, b_det, tau)
            g, = torch.autograd.grad(Fv, mu)
        return float(Fv.item()), g.detach().numpy().astype(np.float64)

    res = minimize(
        fg,
        mu0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * T_,
        options={"ftol": tol, "gtol": tol, "maxiter": maxiter},
    )

    mu_star = torch.tensor(np.maximum(res.x, 0.0), dtype=M_t.dtype)
    info = {
        "fun": res.fun,
        "nit": res.nit,
        "gnorm": float(np.max(np.abs(res.jac))),
        "msg": res.message,
    }
    return mu_star, info


# Module-level state: warm start across consecutive forward calls and the most
# recent solver info for diagnostics.
_F_STATE = {"mu_warm": None, "last_info": None}


def reset_F_state():
    _F_STATE["mu_warm"] = None
    _F_STATE["last_info"] = None


class ImplicitMuF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, M, b, tau):
        mu_init = _F_STATE["mu_warm"]
        mu_star, info = _solve_F_inner(M, b, float(tau), mu_init=mu_init)

        _F_STATE["mu_warm"] = mu_star.detach().clone()
        _F_STATE["last_info"] = info

        ctx.save_for_backward(M.detach(), mu_star.detach(), b.detach())
        ctx.tau = float(tau)
        return mu_star

    @staticmethod
    def backward(ctx, grad_mu):
        M, mu_star, b = ctx.saved_tensors
        tau = ctx.tau

        T_ = mu_star.numel()
        eps = 1e-7

        active = mu_star < eps
        A_idx = active.nonzero(as_tuple=True)[0]
        n_A = int(active.sum())
        n_sys = T_ + n_A

        M_d = M.clone().detach().requires_grad_(True)
        mu_d = mu_star.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            Fv = _F_torch(mu_d, M_d, b, tau)
            g_mu, = torch.autograd.grad(Fv, mu_d, create_graph=True)

            H = torch.zeros(T_, T_, dtype=M.dtype)
            for k in range(T_):
                row, = torch.autograd.grad(g_mu[k], mu_d, retain_graph=True)
                H[k] = row

            J = torch.zeros(n_sys, n_sys, dtype=M.dtype)
            J[:T_, :T_] = H + 1e-6 * torch.eye(T_, dtype=M.dtype)

            for j, a in enumerate(A_idx):
                a = a.item()
                J[a, T_ + j] = -1.0
                J[T_ + j, a] = 1.0

            dFp_dM = torch.zeros(n_sys, M_d.numel(), dtype=M.dtype)
            for k in range(T_):
                gk, = torch.autograd.grad(g_mu[k], M_d, retain_graph=True)
                dFp_dM[k] = gk.reshape(-1)

            du_dM = torch.linalg.solve(J, -dFp_dM)
            grad_M = (du_dM[:T_].t() @ grad_mu).reshape(M.shape)

        return grad_M, None, None


def mu_of_M_F(M, b, tau):
    """M: (N, T) -> mu*: (T,). Gradients via IFT on F."""
    return ImplicitMuF.apply(
        M,
        b if torch.is_tensor(b) else torch.tensor(b),
        float(tau),
    )
