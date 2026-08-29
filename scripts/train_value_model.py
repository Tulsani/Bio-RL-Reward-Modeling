"""Fit and diagnose the training-only action-conditioned reward model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from value_model import ActionRewardModel, compare_action_reward_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "value_model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trajectories = pd.read_csv(PROJECT_ROOT / "trajectories_train.csv")

    comparison = compare_action_reward_models(
        trajectories,
        model_configs=config["models"],
        n_splits=config["cross_validation_folds"],
        random_seed=config["random_seed"],
        baseline_name=config["baseline_model"],
    )
    selected_name = config["selected_model"]
    diagnostics = comparison["models"][selected_name]
    diagnostics["model_comparison"] = {
        "baseline": comparison["baseline"],
        "deltas_vs_baseline": comparison["deltas_vs_baseline"],
        "delta_definition": comparison["delta_definition"],
        "overall": {
            name: result["overall"]
            for name, result in comparison["models"].items()
        },
        "per_action": {
            name: result["per_action"]
            for name, result in comparison["models"].items()
        },
        "map_below_65": {
            name: result["map_below_65_slice"]["true"]
            for name, result in comparison["models"].items()
        },
    }
    diagnostics["config"] = config

    model = ActionRewardModel(
        random_seed=config["random_seed"],
        **config["models"][selected_name],
    ).fit(trajectories)

    artifact_path = PROJECT_ROOT / config["artifact_path"]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    metrics_path = PROJECT_ROOT / config["metrics_path"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(diagnostics["overall"], indent=2, sort_keys=True))
    print(
        json.dumps(
            diagnostics["model_comparison"], indent=2, sort_keys=True
        )
    )
    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path.relative_to(PROJECT_ROOT)),
                "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
