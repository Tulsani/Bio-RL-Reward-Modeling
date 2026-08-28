"""Run a small, diverse, cached smoke test of the hosted zero-shot policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_client import LLMClient
from mistral_backend import MistralBackend
from policy import ZeroShotLLMPolicy
from prompts import ACTION_NAMES, OBSERVATION_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args()


def load_model_config() -> dict:
    path = PROJECT_ROOT / "config" / "model_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def select_diverse_rows(
    trajectories: pd.DataFrame, limit: int
) -> list[tuple[str, pd.Series]]:
    """Select deterministic cases using pre-action state fields only."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    candidates: list[tuple[str, int]] = []

    def add(label: str, indices) -> None:
        candidates.extend((label, int(index)) for index in indices)

    add("lowest_map", trajectories.nsmallest(2, "map_mm_hg").index)
    add(
        "map_near_65",
        (trajectories["map_mm_hg"] - 65.0).abs().nsmallest(2).index,
    )
    add(
        "highest_lactate",
        trajectories.dropna(subset=["lactate_mmol_l"])
        .nlargest(2, "lactate_mmol_l")
        .index,
    )
    add(
        "missing_lactate",
        trajectories[trajectories["lactate_mmol_l"].isna()].index[:2],
    )
    add(
        "vasopressor_active",
        trajectories[
            trajectories["vasopressor_active_pre_action"] == 1
        ].index[:2],
    )
    add(
        "recent_iv_fluids",
        trajectories[trajectories["iv_fluids_previous_6h_ml"] > 0]
        .sort_values("iv_fluids_previous_6h_ml", ascending=False)
        .index[:2],
    )

    selected: list[tuple[str, pd.Series]] = []
    seen: set[int] = set()
    for label, index in candidates:
        if index in seen:
            continue
        seen.add(index)
        selected.append((label, trajectories.loc[index]))
        if len(selected) == limit:
            return selected

    for index, row in trajectories.iterrows():
        if index not in seen:
            selected.append(("deterministic_fill", row))
        if len(selected) == limit:
            break
    return selected


def main() -> None:
    args = parse_args()
    config = load_model_config()
    trajectories = pd.read_csv(PROJECT_ROOT / "trajectories_train.csv")
    selected = select_diverse_rows(trajectories, args.limit)
    observations = [
        {column: row[column] for column in OBSERVATION_COLUMNS}
        for _, row in selected
    ]

    backend = MistralBackend(
        random_seed=config["random_seed"],
        max_tokens=config["max_tokens"],
        dotenv_path=PROJECT_ROOT / ".env",
    )
    client = LLMClient(
        model_id=config["model_id"],
        prompt_version=config["prompt_version"],
        cache_dir=PROJECT_ROOT / config["cache_dir"],
        backend=backend,
        temperature=config["temperature"],
    )
    policy = ZeroShotLLMPolicy(client)
    probabilities = policy.predict_proba(observations)

    cases = []
    for (label, _), row_probabilities, result in zip(
        selected, probabilities, policy.call_results, strict=True
    ):
        cases.append(
            {
                "selection_reason": label,
                "chosen_action": ACTION_NAMES[int(np.argmax(row_probabilities))],
                "probabilities": {
                    action: round(float(probability), 6)
                    for action, probability in zip(
                        ACTION_NAMES, row_probabilities, strict=True
                    )
                },
                "rationale": result.decision.rationale,
                "cache_hit": result.cache_hit,
                "fallback_used": result.fallback_used,
                "failure_reason": result.failure_reason,
            }
        )

    report = {
        "model_id": policy.model_id,
        "prompt_version": policy.prompt_version,
        "num_cases": len(cases),
        "diagnostics": policy.diagnostics(),
        "cases": cases,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
