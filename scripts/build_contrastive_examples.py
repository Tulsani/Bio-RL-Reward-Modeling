"""Generate the training-only supported contrastive-example library."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from contrastive_examples import (
    build_contrastive_examples,
    generate_cross_fitted_estimates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads(
        (PROJECT_ROOT / "config" / "contrastive_examples_config.json").read_text(
            encoding="utf-8"
        )
    )
    value_config = json.loads(
        (PROJECT_ROOT / "config" / "value_model_config.json").read_text(
            encoding="utf-8"
        )
    )
    trajectories = pd.read_csv(PROJECT_ROOT / "trajectories_train.csv")
    estimates = generate_cross_fitted_estimates(
        trajectories,
        n_splits=config["cross_validation_folds"],
        random_seed=config["random_seed"],
        ridge_config=value_config["models"]["ridge"],
        nonlinear_config=value_config["models"]["hist_gradient_boosting"],
    )
    examples, diagnostics = build_contrastive_examples(
        trajectories,
        estimates,
        minimum_propensity=config["minimum_propensity"],
        minimum_advantage=config["minimum_advantage"],
        maximum_value_disagreement=config["maximum_value_disagreement"],
        max_examples_per_action=config["max_examples_per_action"],
        require_preferred_matches_logged=config[
            "require_preferred_matches_logged"
        ],
        minimum_logged_reward=config["minimum_logged_reward"],
        minimum_examples_per_action=config["minimum_examples_per_action"],
    )
    diagnostics["config"] = config

    examples_path = PROJECT_ROOT / config["examples_path"]
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(examples_path, index=False)

    diagnostics_path = PROJECT_ROOT / config["diagnostics_path"]
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
