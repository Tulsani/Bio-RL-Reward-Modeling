"""Build the training-only factual outcome example library."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from factual_examples import (
    build_factual_outcome_library,
    generate_cross_fitted_behavior_context,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads(
        (PROJECT_ROOT / "config" / "factual_examples_config.json").read_text(
            encoding="utf-8"
        )
    )
    trajectories = pd.read_csv(PROJECT_ROOT / "trajectories_train.csv")
    behavior_context = generate_cross_fitted_behavior_context(
        trajectories,
        n_splits=config["cross_validation_folds"],
        random_seed=config["random_seed"],
    )
    library, diagnostics = build_factual_outcome_library(
        trajectories,
        behavior_context,
        lower_reward_quantile=config["lower_reward_quantile"],
        upper_reward_quantile=config["upper_reward_quantile"],
        minimum_logged_propensity=config["minimum_logged_propensity"],
        max_examples_per_cell=config["max_examples_per_cell"],
        minimum_examples_per_cell=config["minimum_examples_per_cell"],
    )
    diagnostics["config"] = config

    library_path = PROJECT_ROOT / config["library_path"]
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library.to_csv(library_path, index=False)
    diagnostics_path = PROJECT_ROOT / config["diagnostics_path"]
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
