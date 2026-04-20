# end2end-capacity-constrained-policy-learning

Bilevel IPW policy learning with capacity constraints, implemented two ways
side-by-side (CVXPYLayer vs PyTorch implicit differentiation).

## Problem

Given observational data $(X_i, T_i, Y_i, e_i)$ over 3 treatments, learn a
softmax policy $\pi_{\theta,t}(x)$ that maximizes the IPW estimator

$$\hat V_{\text{IPW}}(\theta) = \frac{1}{N}\sum_{i=1}^N \pi_{\theta,T_i}(X_i)\,\frac{Y_i}{\hat e_{T_i}(X_i)}$$

where the policy

$$\pi_{\theta,t}(x) = \frac{\exp((m_{t,\theta}(x)-\mu_{\theta,t})/\tau)}{\sum_s \exp((m_{s,\theta}(x)-\mu_{\theta,s})/\tau)}$$

uses shadow prices $\mu_\theta$ that themselves solve an inner optimization
enforcing per-treatment capacity budgets $b_t$.

## Inner objectives

We implement both:

- **G (convex dual, recommended)**: the Lagrangian dual of the
  entropy-regularized capacity-constrained primal,
  $\min_{\mu\ge 0}\; \tfrac{\tau}{N}\sum_i \log\sum_t \exp((m_{t,i}-\mu_t)/\tau) + b^\top\mu$.
  Convex, DPP-compliant, solved inside a **CVXPYLayer**; gradients back to
  $\theta$ via diffcp's implicit differentiation through the KKT system.

- **F (user's literal form)**:
  $\min_{\mu\ge 0}\; \tfrac{1}{N}\sum_i \sum_t \sigma_{t,i}(\mu)(m_{t,i}-\mu_t) + b^\top\mu$.
  Non-convex in $\mu$ (the softmax-weighted average has indefinite curvature),
  so it is not CVXPY-expressible. Solved with scipy L-BFGS-B inside a custom
  `torch.autograd.Function` whose backward uses the implicit-function theorem
  on $\nabla_\mu F(\mu^\star, M)=0$, restricted to the inactive set.

For the identity that relates them,
$\sum_t \sigma_t(u) u_t = \tau\log\sum_t e^{u_t/\tau} - \tau H(\sigma(u))$,
so $F(\mu) = G(\mu) - (\tau/N)\sum_i H(\sigma_i(\mu))$: the two dual objectives
differ by the policy entropy. They yield different $\mu^\star$ in general, but
on this DGP they converge to effectively the same learned policy.

## Data-generating process

- $X_1, X_2 \sim \mathcal N(0,1)$ i.i.d.
- $Y^t = \tfrac{1}{2}X_1 + X_2 + (2\mathbf 1[t=1]-1)\tfrac{1}{4}X_1 + (2\mathbf 1[t=2]-1)\tfrac{1}{4}X_2$ for $t\in\{0,1,2\}$
- Softmax propensities from scores $s_0=0$, $s_1 = \tfrac{1}{2}X_1-\tfrac{1}{2}X_2$,
  $s_2 = -\tfrac{1}{2}X_1+\tfrac{1}{2}X_2$; observed treatment $T\sim\text{Categorical}(e(x))$.
- Only the observed tuple $(X_1, X_2, T, Y, e_T)$ is used during training.

## Run

```bash
# Create the project-local conda env (once)
conda create -y -p ./.conda-env python=3.11
./.conda-env/bin/pip install "numpy<2" torch cvxpy cvxpylayers diffcp scipy matplotlib jupyter

# Train both pipelines end-to-end
./.conda-env/bin/python ipw_policy.py
```

## What you should see

With $N=500$, $T=3$, $\tau=1$, $b=(1.0, 0.4, 0.4)$, 200 Adam steps:

| Path                                | Final $\hat V_{\text{IPW}}$ | Eval $V_{\text{oracle}}$ | Alloc              | Wall  |
|-------------------------------------|-----------------------------|--------------------------|--------------------|-------|
| Random policy                       | —                           | −0.009                   | (1/3, 1/3, 1/3)    | —     |
| Greedy from trained scores (no cap) | —                           |  0.275                   | (0.19, 0.41, 0.40) | —     |
| G via CVXPYLayer                    |  0.333                      |  0.272                   | (0.20, 0.40, 0.40) | ~80s  |
| F via implicit diff                 |  0.334                      |  0.272                   | (0.20, 0.40, 0.40) | ~1.3s |

Where $V_{\text{oracle}}(\pi) = (1/N)\sum_i \sum_t \pi_{t,i}\,Y^t_i$ is
computed using the counterfactual outcomes from the DGP — available only
because the data is synthetic, and used strictly for evaluation.

## File layout

- `ipw_policy.py` — blocked, notebook-pasteable end-to-end script.
- `.conda-env/` — project-local conda env (gitignored).
