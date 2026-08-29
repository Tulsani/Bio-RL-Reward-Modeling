"""Diagnose maintain-versus-intervention bias with training-only estimates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from contrastive_examples import (
    diagnose_action_reward_bias,
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
    diagnostics = diagnose_action_reward_bias(trajectories, estimates)
    diagnostics["config"] = {
        "cross_validation_folds": config["cross_validation_folds"],
        "random_seed": config["random_seed"],
    }

    output_path = PROJECT_ROOT / config["bias_diagnostics_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "data_scope": diagnostics["data_scope"],
        "global_model_rankings": diagnostics["global_model_rankings"],
        "model_agreement": diagnostics["model_agreement"],
        "logged_maintain_states": diagnostics["logged_maintain_states"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
