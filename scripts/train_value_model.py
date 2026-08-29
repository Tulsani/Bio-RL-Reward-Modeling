"""Fit and diagnose the training-only action-conditioned reward model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from value_model import ActionRewardModel, cross_validate_action_reward_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "value_model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trajectories = pd.read_csv(PROJECT_ROOT / "trajectories_train.csv")

    diagnostics = cross_validate_action_reward_model(
        trajectories,
        n_splits=config["cross_validation_folds"],
        alpha=config["alpha"],
        random_seed=config["random_seed"],
    )
    diagnostics["config"] = config

    model = ActionRewardModel(alpha=config["alpha"]).fit(trajectories)

    artifact_path = PROJECT_ROOT / config["artifact_path"]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    metrics_path = PROJECT_ROOT / config["metrics_path"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(diagnostics["overall"], indent=2, sort_keys=True))
    print(json.dumps(diagnostics["per_action"], indent=2, sort_keys=True))
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
