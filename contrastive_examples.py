"""Build supported contrastive examples from training-only cross-fitted models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from behavior_model import BehaviorPolicyModel
from prompts import ACTION_NAMES, OBSERVATION_COLUMNS
from reward import compute_reward
from value_model import (
    LOGGED_ACTION_COLUMN,
    PATIENT_ID_COLUMN,
    ActionRewardModel,
    action_support_mask,
    patient_group_value_folds,
)

RIDGE_MODEL = "ridge"
NONLINEAR_MODEL = "hist_gradient_boosting"
MODEL_NAMES = (RIDGE_MODEL, NONLINEAR_MODEL)


@dataclass(frozen=True)
class CrossFittedEstimates:
    """Out-of-fold nuisance estimates aligned with the source DataFrame rows."""

    behavior_probabilities: np.ndarray
    action_values: dict[str, np.ndarray]
    fold_assignments: np.ndarray

    def validate(self, n_rows: int) -> None:
        expected_shape = (n_rows, len(ACTION_NAMES))
        if self.behavior_probabilities.shape != expected_shape:
            raise ValueError("Cross-fitted behavior probabilities have invalid shape.")
        if set(self.action_values) != set(MODEL_NAMES):
            raise ValueError(f"action_values must contain models {MODEL_NAMES}.")
        if any(values.shape != expected_shape for values in self.action_values.values()):
            raise ValueError("Cross-fitted action values have invalid shape.")
        arrays = [self.behavior_probabilities, *self.action_values.values()]
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Cross-fitted estimates must be finite.")
        if not np.allclose(self.behavior_probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Behavior probabilities must sum to one per row.")
        if self.fold_assignments.shape != (n_rows,) or (self.fold_assignments < 1).any():
            raise ValueError("Every row must have a positive fold assignment.")


def generate_cross_fitted_estimates(
    trajectories: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_seed: int = 42,
    ridge_config: dict[str, Any] | None = None,
    nonlinear_config: dict[str, Any] | None = None,
) -> CrossFittedEstimates:
    """Fit all nuisance models without predicting on a model's training patients."""
    _validate_training_trajectories(trajectories)
    ridge_parameters = {"model_type": RIDGE_MODEL, "alpha": 1.0}
    nonlinear_parameters = {
        "model_type": NONLINEAR_MODEL,
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    }
    if ridge_config is not None:
        ridge_parameters.update(ridge_config)
    if nonlinear_config is not None:
        nonlinear_parameters.update(nonlinear_config)

    n_rows = len(trajectories)
    behavior = np.full((n_rows, len(ACTION_NAMES)), np.nan, dtype=float)
    values = {
        name: np.full((n_rows, len(ACTION_NAMES)), np.nan, dtype=float)
        for name in MODEL_NAMES
    }
    fold_assignments = np.zeros(n_rows, dtype=int)

    for fold, (train_index, validation_index) in enumerate(
        patient_group_value_folds(
            trajectories, n_splits=n_splits, random_seed=random_seed
        ),
        start=1,
    ):
        train_fold = trajectories.iloc[train_index]
        validation_fold = trajectories.iloc[validation_index]
        behavior_model = BehaviorPolicyModel(random_seed=random_seed).fit(train_fold)
        behavior[validation_index] = behavior_model.predict_proba(validation_fold)

        model_parameters = {
            RIDGE_MODEL: ridge_parameters,
            NONLINEAR_MODEL: nonlinear_parameters,
        }
        for name, parameters in model_parameters.items():
            reward_model = ActionRewardModel(
                random_seed=random_seed, **parameters
            ).fit(train_fold)
            values[name][validation_index] = reward_model.predict_values(
                validation_fold
            )
        fold_assignments[validation_index] = fold

    estimates = CrossFittedEstimates(
        behavior_probabilities=behavior,
        action_values=values,
        fold_assignments=fold_assignments,
    )
    estimates.validate(n_rows)
    return estimates


def build_contrastive_examples(
    trajectories: pd.DataFrame,
    estimates: CrossFittedEstimates,
    *,
    minimum_propensity: float = 0.02,
    minimum_advantage: float = 0.10,
    maximum_value_disagreement: float = 0.50,
    max_examples_per_action: int = 100,
    require_preferred_matches_logged: bool = True,
    minimum_logged_reward: float | None = 0.0,
    minimum_examples_per_action: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select model-agreed preferred actions and supported near negatives.

    Both reward models must rank the same supported action first. The strongest
    remaining supported action by average modeled reward becomes the near
    negative, and both model-specific advantages must clear the margin.
    """
    _validate_training_trajectories(trajectories)
    estimates.validate(len(trajectories))
    if minimum_advantage < 0:
        raise ValueError("minimum_advantage must be non-negative.")
    if maximum_value_disagreement < 0:
        raise ValueError("maximum_value_disagreement must be non-negative.")
    if max_examples_per_action <= 0:
        raise ValueError("max_examples_per_action must be positive.")
    if minimum_logged_reward is not None and not np.isfinite(minimum_logged_reward):
        raise ValueError("minimum_logged_reward must be finite or None.")
    if minimum_examples_per_action <= 0:
        raise ValueError("minimum_examples_per_action must be positive.")

    support = action_support_mask(
        estimates.behavior_probabilities,
        n_rows=len(trajectories),
        minimum_propensity=minimum_propensity,
    )
    ridge_values = estimates.action_values[RIDGE_MODEL]
    nonlinear_values = estimates.action_values[NONLINEAR_MODEL]
    average_values = (ridge_values + nonlinear_values) / 2.0
    logged_rewards = _logged_rewards(trajectories)
    candidates = []
    stage_counts = {
        "source_rows": len(trajectories),
        "at_least_two_supported_actions": 0,
        "models_agree_on_preferred_action": 0,
        "agreed_preferred_action_is_supported": 0,
        "value_disagreement_within_limit": 0,
        "both_advantages_clear_margin": 0,
        "preferred_matches_logged_action": 0,
        "logged_reward_clears_threshold": 0,
    }

    for row_position in range(len(trajectories)):
        supported_indices = np.flatnonzero(support[row_position])
        if len(supported_indices) < 2:
            continue
        stage_counts["at_least_two_supported_actions"] += 1

        ridge_preferred = int(np.argmax(ridge_values[row_position]))
        nonlinear_preferred = int(np.argmax(nonlinear_values[row_position]))
        if ridge_preferred != nonlinear_preferred:
            continue
        stage_counts["models_agree_on_preferred_action"] += 1
        if not support[row_position, ridge_preferred]:
            continue
        stage_counts["agreed_preferred_action_is_supported"] += 1

        alternative_indices = supported_indices[
            supported_indices != ridge_preferred
        ]
        near_negative = alternative_indices[
            np.argmax(average_values[row_position, alternative_indices])
        ]
        compared_indices = np.array([ridge_preferred, near_negative])
        value_disagreement = float(
            np.max(
                np.abs(
                    ridge_values[row_position, compared_indices]
                    - nonlinear_values[row_position, compared_indices]
                )
            )
        )
        if value_disagreement > maximum_value_disagreement:
            continue
        stage_counts["value_disagreement_within_limit"] += 1

        ridge_advantage = float(
            ridge_values[row_position, ridge_preferred]
            - ridge_values[row_position, near_negative]
        )
        nonlinear_advantage = float(
            nonlinear_values[row_position, nonlinear_preferred]
            - nonlinear_values[row_position, near_negative]
        )
        if min(ridge_advantage, nonlinear_advantage) < minimum_advantage:
            continue
        stage_counts["both_advantages_clear_margin"] += 1

        source = trajectories.iloc[row_position]
        preferred_matches_logged = bool(
            ACTION_NAMES[ridge_preferred] == source[LOGGED_ACTION_COLUMN]
        )
        if require_preferred_matches_logged and not preferred_matches_logged:
            continue
        stage_counts["preferred_matches_logged_action"] += 1
        if (
            minimum_logged_reward is not None
            and logged_rewards[row_position] < minimum_logged_reward
        ):
            continue
        stage_counts["logged_reward_clears_threshold"] += 1

        candidate = {
            "source_patient_id": source[PATIENT_ID_COLUMN],
            "source_time_step": int(source["time_step"]),
            "cross_fit_fold": int(estimates.fold_assignments[row_position]),
            **{column: source[column] for column in OBSERVATION_COLUMNS},
            "preferred_action": ACTION_NAMES[ridge_preferred],
            "near_negative_action": ACTION_NAMES[near_negative],
            "logged_action": source[LOGGED_ACTION_COLUMN],
            "preferred_matches_logged": preferred_matches_logged,
            "preferred_propensity": float(
                estimates.behavior_probabilities[row_position, ridge_preferred]
            ),
            "near_negative_propensity": float(
                estimates.behavior_probabilities[row_position, near_negative]
            ),
            "ridge_preferred_value": float(
                ridge_values[row_position, ridge_preferred]
            ),
            "ridge_near_negative_value": float(
                ridge_values[row_position, near_negative]
            ),
            "nonlinear_preferred_value": float(
                nonlinear_values[row_position, ridge_preferred]
            ),
            "nonlinear_near_negative_value": float(
                nonlinear_values[row_position, near_negative]
            ),
            "ridge_advantage": ridge_advantage,
            "nonlinear_advantage": nonlinear_advantage,
            "robust_advantage": min(ridge_advantage, nonlinear_advantage),
            "value_disagreement": value_disagreement,
        }
        candidates.append(candidate)

    candidate_frame = pd.DataFrame(candidates)
    selected = _balance_candidates(candidate_frame, max_examples_per_action)
    diagnostics = _selection_diagnostics(
        trajectories,
        selected,
        stage_counts=stage_counts,
        candidates_before_balance=candidate_frame,
        minimum_propensity=minimum_propensity,
        minimum_advantage=minimum_advantage,
        maximum_value_disagreement=maximum_value_disagreement,
        max_examples_per_action=max_examples_per_action,
        require_preferred_matches_logged=require_preferred_matches_logged,
        minimum_logged_reward=minimum_logged_reward,
        minimum_examples_per_action=minimum_examples_per_action,
    )
    return selected, diagnostics


def _balance_candidates(
    candidates: pd.DataFrame, max_examples_per_action: int
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    action_order = {action: index for index, action in enumerate(ACTION_NAMES)}
    ranked = candidates.assign(
        _action_order=candidates["preferred_action"].map(action_order)
    ).sort_values(
        ["_action_order", "robust_advantage", "preferred_propensity", "source_patient_id", "source_time_step"],
        ascending=[True, False, False, True, True],
        kind="stable",
    )
    selected = ranked.groupby("preferred_action", sort=False).head(
        max_examples_per_action
    )
    return selected.drop(columns="_action_order").reset_index(drop=True)


def _selection_diagnostics(
    trajectories: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    stage_counts: dict[str, int],
    candidates_before_balance: pd.DataFrame,
    minimum_propensity: float,
    minimum_advantage: float,
    maximum_value_disagreement: float,
    max_examples_per_action: int,
    require_preferred_matches_logged: bool,
    minimum_logged_reward: float | None,
    minimum_examples_per_action: int,
) -> dict[str, Any]:
    selected_by_action = {
        action: int((selected.get("preferred_action") == action).sum())
        if not selected.empty
        else 0
        for action in ACTION_NAMES
    }
    underrepresented_actions = [
        action
        for action, count in selected_by_action.items()
        if count < minimum_examples_per_action
    ]
    diagnostics: dict[str, Any] = {
        "thresholds": {
            "minimum_propensity": minimum_propensity,
            "minimum_advantage": minimum_advantage,
            "maximum_value_disagreement": maximum_value_disagreement,
            "max_examples_per_action": max_examples_per_action,
            "require_preferred_matches_logged": require_preferred_matches_logged,
            "minimum_logged_reward": minimum_logged_reward,
            "minimum_examples_per_action": minimum_examples_per_action,
        },
        "selection_stages": {
            **stage_counts,
            "candidates_before_balance": len(candidates_before_balance),
            "selected_examples": len(selected),
        },
        "candidates_before_balance_by_preferred_action": {
            action: int(
                (
                    candidates_before_balance.get("preferred_action") == action
                ).sum()
            )
            if not candidates_before_balance.empty
            else 0
            for action in ACTION_NAMES
        },
        "selected_by_preferred_action": selected_by_action,
        "policy_improvement_readiness": {
            "approved": not underrepresented_actions,
            "minimum_examples_per_action": minimum_examples_per_action,
            "underrepresented_actions": underrepresented_actions,
            "reason": (
                None
                if not underrepresented_actions
                else "Candidate library lacks action-balanced model support."
            ),
        },
    }
    if selected.empty:
        diagnostics.update(
            {
                "preferred_matches_logged_rate": None,
                "map_below_65": {"rows": 0, "rate": 0.0},
                "robust_advantage": None,
                "propensity": None,
            }
        )
        return diagnostics

    low_map = selected["map_mm_hg"] < 65.0
    diagnostics.update(
        {
            "preferred_matches_logged_rate": float(
                selected["preferred_matches_logged"].mean()
            ),
            "map_below_65": {
                "rows": int(low_map.sum()),
                "rate": float(low_map.mean()),
                "source_rate": float((trajectories["map_mm_hg"] < 65.0).mean()),
            },
            "robust_advantage": _quantiles(selected["robust_advantage"]),
            "preferred_propensity": _quantiles(selected["preferred_propensity"]),
            "near_negative_propensity": _quantiles(
                selected["near_negative_propensity"]
            ),
            "value_disagreement": _quantiles(selected["value_disagreement"]),
        }
    )
    return diagnostics


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def _logged_rewards(trajectories: pd.DataFrame) -> np.ndarray:
    if "reward" in trajectories.columns:
        rewards = pd.to_numeric(trajectories["reward"], errors="coerce").to_numpy()
    else:
        try:
            rewards = trajectories.apply(compute_reward, axis=1).to_numpy(dtype=float)
        except KeyError as error:
            raise ValueError(
                f"Cannot select examples because outcome column {error.args[0]!r} is missing."
            ) from error
    if not np.isfinite(rewards).all():
        raise ValueError("Logged rewards used for selection must be finite.")
    return rewards


def _validate_training_trajectories(trajectories: pd.DataFrame) -> None:
    if not isinstance(trajectories, pd.DataFrame) or trajectories.empty:
        raise ValueError("Contrastive source trajectories must be non-empty.")
    required = set(OBSERVATION_COLUMNS) | {
        PATIENT_ID_COLUMN,
        "time_step",
        "split",
        LOGGED_ACTION_COLUMN,
    }
    missing = sorted(required - set(trajectories.columns))
    if missing:
        raise ValueError(f"Missing contrastive source columns: {missing}")
    if set(trajectories["split"].dropna().unique()) != {"train"}:
        raise ValueError("Contrastive examples may use only the training split.")
    if trajectories["split"].isna().any():
        raise ValueError("Split values cannot be missing.")
