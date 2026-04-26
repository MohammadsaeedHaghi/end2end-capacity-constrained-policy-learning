"""Run all methods on the snapshotted data and emit the comparison table."""

from src import config
from src.data import load_experiment
from src.inner_G import initialize_G_layer
from src.train import train_GF
from src.evaluation import evaluate_GF_model, evaluate_greedy_no_cap_from_model
from src.baselines import evaluate_random_policy, evaluate_oracle_greedy_no_cap
from src.s2_dual import run_dual_method
from src.comparison import make_comparison_table


def main():
    config.setup_global_state()

    train_data, eval_data, cfg = load_experiment(
        config.TRAIN_DATA_PATH,
        config.EVAL_DATA_PATH,
        config.CONFIG_PATH,
    )

    N = int(cfg["N"])
    N_EVAL = int(cfg["N_EVAL"])
    T = int(cfg["T"])
    D = int(cfg["D"])
    TAU = float(cfg["TAU"])
    B = cfg["B"]

    print("[main] Loaded data.")
    print("[main] train shapes:",
          {k: v.shape for k, v in train_data.items() if hasattr(v, "shape")})
    print("[main] eval shapes:",
          {k: v.shape for k, v in eval_data.items() if hasattr(v, "shape")})
    print("[main] B:", B, " sum(B):", B.sum())

    initialize_G_layer(N=N, T=T, tau=TAU, b=B)

    results = []

    print("\n========= BASELINES =========")
    results.append(evaluate_random_policy(eval_data, b=B, T=T))
    results.append(evaluate_oracle_greedy_no_cap(eval_data, b=B, T=T))

    print("\n========= TRAIN G (CVXPYLayer, convex dual) =========")
    model_G, mu_G, hist_G = train_GF(
        kind="G",
        train_data=train_data,
        D=D, T=T, tau=TAU, b=B,
        steps=200, lr=5e-3, log_every=20, seed=1,
    )
    results.append(evaluate_GF_model(
        model=model_G, mu_train=mu_G, eval_data=eval_data,
        b=B, tau=TAU, T=T, tag="G",
    ))
    results.append(evaluate_greedy_no_cap_from_model(
        model=model_G, eval_data=eval_data, b=B, T=T,
    ))

    print("\n========= TRAIN F (implicit diff, non-convex literal) =========")
    model_F, mu_F, hist_F = train_GF(
        kind="F",
        train_data=train_data,
        D=D, T=T, tau=TAU, b=B,
        steps=200, lr=5e-3, log_every=20, seed=1,
    )
    results.append(evaluate_GF_model(
        model=model_F, mu_train=mu_F, eval_data=eval_data,
        b=B, tau=TAU, T=T, tag="F",
    ))

    print("\n========= S2 SAMPLE-BASED DUAL-PRICE METHODS =========")
    for method in ["linear", "lasso", "tree", "knn", "dr"]:
        print(f"\n--- Running S2 method: {method} ---")
        results.append(run_dual_method(
            method_name=method,
            train_data=train_data,
            eval_data=eval_data,
            T=T, b=B,
            verbose_lp=False,
        ))

    df = make_comparison_table(results, b=B, N=N, N_EVAL=N_EVAL, T=T, D=D)
    df.to_csv(config.RESULTS_CSV_PATH, index=False)
    print(f"\n[main] Results saved to {config.RESULTS_CSV_PATH}.")


if __name__ == "__main__":
    main()
