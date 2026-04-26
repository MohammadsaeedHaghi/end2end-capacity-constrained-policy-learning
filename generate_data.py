"""Generate synthetic train/eval splits and snapshot the experiment config."""

import numpy as np

from src import config
from src.data import generate_data


def main():
    config.setup_global_state()

    train_data = generate_data(
        N=config.N,
        seed=config.TRAIN_SEED,
        d=config.D,
        T=config.T,
        sigma_y=config.SIGMA_Y,
        propensity_strength=config.PROPENSITY_STRENGTH,
        outcome_strength=config.OUTCOME_STRENGTH,
        clip_propensity=config.CLIP_PROPENSITY,
    )

    eval_data = generate_data(
        N=config.N_EVAL,
        seed=config.EVAL_SEED,
        d=config.D,
        T=config.T,
        sigma_y=config.SIGMA_Y,
        propensity_strength=config.PROPENSITY_STRENGTH,
        outcome_strength=config.OUTCOME_STRENGTH,
        clip_propensity=config.CLIP_PROPENSITY,
    )

    np.savez(config.TRAIN_DATA_PATH, **train_data)
    np.savez(config.EVAL_DATA_PATH, **eval_data)
    np.savez(
        config.CONFIG_PATH,
        N=config.N,
        N_EVAL=config.N_EVAL,
        T=config.T,
        D=config.D,
        TAU=config.TAU,
        B=config.B,
        SIGMA_Y=config.SIGMA_Y,
        PROPENSITY_STRENGTH=config.PROPENSITY_STRENGTH,
        OUTCOME_STRENGTH=config.OUTCOME_STRENGTH,
        CLIP_PROPENSITY=config.CLIP_PROPENSITY,
        TRAIN_SEED=config.TRAIN_SEED,
        EVAL_SEED=config.EVAL_SEED,
    )

    print(
        f"Saved {config.TRAIN_DATA_PATH}, "
        f"{config.EVAL_DATA_PATH}, {config.CONFIG_PATH}."
    )
    print("Train shapes:",
          {k: v.shape for k, v in train_data.items() if hasattr(v, "shape")})
    print("Eval shapes:",
          {k: v.shape for k, v in eval_data.items() if hasattr(v, "shape")})
    print("Empirical train treatment proportions:",
          np.bincount(train_data["T"], minlength=config.T) / config.N)
    print("B:", config.B, " sum(B):", config.B.sum())


if __name__ == "__main__":
    main()
