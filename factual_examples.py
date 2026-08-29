"""Training-only factual outcome examples and state-matched retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from behavior_model import BehaviorPolicyModel
from prompts import ACTION_NAMES, OBSERVATION_COLUMNS
from reward import compute_reward
from value_model import (
    LOGGED_ACTION_COLUMN,
    PATIENT_ID_COLUMN,
    patient_group_value_folds,
)

FAVORABLE = "favorable"
UNFAVORABLE = "unfavorable"
OUTCOME_LABELS = (FAVORABLE, UNFAVORABLE)
CATEGORICAL_FEATURES = ("sex", "previous_action")
NUMERIC_FEATURES = tuple(
    column for column in OBSERVATION_COLUMNS if column not in CATEGORICAL_FEATURES
)


@dataclass(frozen=True)
class CrossFittedBehaviorContext:
    """Behavior probabilities and fold IDs aligned to source rows."""

    probabilities: np.ndarray
    fold_assignments: np.ndarray

    def validate(self, n_rows: int) -> None:
        expected_shape = (n_rows, len(ACTION_NAMES))
        if self.probabilities.shape != expected_shape:
            raise ValueError(f"Behavior probabilities must have shape {expected_shape}.")
        if not np.isfinite(self.probabilities).all() or (
            self.probabilities < 0.0
        ).any():
            raise ValueError("Behavior probabilities must be finite and non-negative.")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Behavior probabilities must sum to one per row.")
        if self.fold_assignments.shape != (n_rows,) or (
            self.fold_assignments < 1
        ).any():
            raise ValueError("Every row must have a positive fold assignment.")


def generate_cross_fitted_behavior_context(
    trajectories: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_seed: int = 42,
) -> CrossFittedBehaviorContext:
    """Generate patient-disjoint behavior probabilities without reward models."""
    _validate_training_trajectories(trajectories)
    probabilities = np.full(
        (len(trajectories), len(ACTION_NAMES)), np.nan, dtype=float
    )
    fold_assignments = np.zeros(len(trajectories), dtype=int)
    for fold, (train_index, validation_index) in enumerate(
        patient_group_value_folds(
            trajectories, n_splits=n_splits, random_seed=random_seed
        ),
        start=1,
    ):
        model = BehaviorPolicyModel(random_seed=random_seed).fit(
            trajectories.iloc[train_index]
        )
        probabilities[validation_index] = model.predict_proba(
            trajectories.iloc[validation_index]
        )
        fold_assignments[validation_index] = fold

    context = CrossFittedBehaviorContext(probabilities, fold_assignments)
    context.validate(len(trajectories))
    return context


def build_factual_outcome_library(
    trajectories: pd.DataFrame,
    behavior_context: CrossFittedBehaviorContext,
    *,
    lower_reward_quantile: float = 0.25,
    upper_reward_quantile: float = 0.75,
    minimum_logged_propensity: float = 0.02,
    max_examples_per_cell: int = 100,
    minimum_examples_per_cell: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build supported favorable/unfavorable factual examples per action.

    Outcome labels are relative to the reward distribution within each logged
    action. They do not assert that one action is counterfactually optimal.
    """
    _validate_training_trajectories(trajectories)
    behavior_context.validate(len(trajectories))
    if not 0.0 <= lower_reward_quantile < upper_reward_quantile <= 1.0:
        raise ValueError("Reward quantiles must satisfy 0 <= lower < upper <= 1.")
    if not 0.0 <= minimum_logged_propensity <= 1.0:
        raise ValueError("minimum_logged_propensity must be between zero and one.")
    if max_examples_per_cell <= 0 or minimum_examples_per_cell <= 0:
        raise ValueError("Example-count thresholds must be positive.")

    rewards = _logged_rewards(trajectories)
    actions = trajectories[LOGGED_ACTION_COLUMN].to_numpy()
    action_indices = np.array(
        [ACTION_NAMES.index(action) for action in actions], dtype=int
    )
    logged_propensities = behavior_context.probabilities[
        np.arange(len(trajectories)), action_indices
    ]
    thresholds = {}
    candidate_cells: dict[tuple[str, str], pd.DataFrame] = {}

    for action in ACTION_NAMES:
        action_mask = actions == action
        action_rewards = rewards[action_mask]
        lower = float(np.quantile(action_rewards, lower_reward_quantile))
        upper = float(np.quantile(action_rewards, upper_reward_quantile))
        thresholds[action] = {
            "lower": lower,
            "upper": upper,
            "logged_rows": int(action_mask.sum()),
        }
        supported = logged_propensities >= minimum_logged_propensity
        for outcome_label, outcome_mask in (
            (FAVORABLE, rewards >= upper),
            (UNFAVORABLE, rewards <= lower),
        ):
            selected = np.flatnonzero(action_mask & supported & outcome_mask)
            cell = _candidate_frame(
                trajectories,
                behavior_context,
                logged_propensities,
                rewards,
                selected,
                action,
                outcome_label,
            )
            ascending_reward = outcome_label == UNFAVORABLE
            cell = cell.sort_values(
                ["_selection_reward", "logged_action_propensity", "source_patient_id", "source_time_step"],
                ascending=[ascending_reward, False, True, True],
                kind="stable",
            )
            cell = cell.drop_duplicates("source_patient_id").head(
                max_examples_per_cell
            )
            candidate_cells[(action, outcome_label)] = cell

    library = pd.concat(candidate_cells.values(), ignore_index=True)
    library = library.drop(columns="_selection_reward")
    diagnostics = _library_diagnostics(
        trajectories,
        library,
        thresholds=thresholds,
        candidate_cells=candidate_cells,
        lower_reward_quantile=lower_reward_quantile,
        upper_reward_quantile=upper_reward_quantile,
        minimum_logged_propensity=minimum_logged_propensity,
        max_examples_per_cell=max_examples_per_cell,
        minimum_examples_per_cell=minimum_examples_per_cell,
    )
    return library, diagnostics


class FactualExampleRetriever:
    """Retrieve state-nearest factual examples from every action/outcome cell."""

    def __init__(self) -> None:
        self.preprocessor = _retrieval_preprocessor()
        self.library: pd.DataFrame | None = None
        self._features: np.ndarray | None = None

    def fit(self, library: pd.DataFrame) -> FactualExampleRetriever:
        _validate_library(library)
        self.library = library.reset_index(drop=True).copy()
        transformed = self.preprocessor.fit_transform(
            self.library.loc[:, OBSERVATION_COLUMNS]
        )
        self._features = np.asarray(transformed, dtype=float)
        return self

    def retrieve(
        self,
        observation: pd.DataFrame | pd.Series | dict[str, Any],
        *,
        examples_per_cell: int = 1,
        exclude_patient_id: str | None = None,
    ) -> pd.DataFrame:
        """Return nearest examples, equally represented across all six cells."""
        if self.library is None or self._features is None:
            raise RuntimeError("Fit FactualExampleRetriever before retrieval.")
        if examples_per_cell <= 0:
            raise ValueError("examples_per_cell must be positive.")
        query = _one_row_frame(observation)
        query_features = np.asarray(
            self.preprocessor.transform(query.loc[:, OBSERVATION_COLUMNS]),
            dtype=float,
        )[0]
        distances = np.linalg.norm(self._features - query_features, axis=1)
        retrieved = []
        for action in ACTION_NAMES:
            for outcome_label in OUTCOME_LABELS:
                mask = (
                    (self.library["action"] == action)
                    & (self.library["outcome_label"] == outcome_label)
                ).to_numpy()
                if exclude_patient_id is not None:
                    mask = mask & (
                        self.library["source_patient_id"] != exclude_patient_id
                    ).to_numpy()
                indices = np.flatnonzero(mask)
                if len(indices) < examples_per_cell:
                    raise ValueError(
                        f"Not enough examples for {action}/{outcome_label} after exclusions."
                    )
                nearest = indices[np.argsort(distances[indices], kind="stable")][
                    :examples_per_cell
                ]
                cell = self.library.iloc[nearest].copy()
                cell["retrieval_distance"] = distances[nearest]
                retrieved.append(cell)
        return pd.concat(retrieved, ignore_index=True)


def prompt_safe_factual_records(examples: pd.DataFrame) -> list[dict[str, Any]]:
    """Strip rewards, propensities, distances, and provenance before prompting."""
    _validate_library(examples)
    records = []
    for _, row in examples.iterrows():
        state = {
            column: None if pd.isna(row[column]) else row[column]
            for column in OBSERVATION_COLUMNS
        }
        records.append(
            {
                "state": state,
                "logged_action": row["action"],
                "observed_outcome_label": row["outcome_label"],
            }
        )
    return records


def _candidate_frame(
    trajectories: pd.DataFrame,
    behavior_context: CrossFittedBehaviorContext,
    logged_propensities: np.ndarray,
    rewards: np.ndarray,
    selected: np.ndarray,
    action: str,
    outcome_label: str,
) -> pd.DataFrame:
    source = trajectories.iloc[selected]
    frame = pd.DataFrame(
        {
            "source_patient_id": source[PATIENT_ID_COLUMN].to_numpy(),
            "source_time_step": source["time_step"].to_numpy(dtype=int),
            "cross_fit_fold": behavior_context.fold_assignments[selected],
        }
    )
    for column in OBSERVATION_COLUMNS:
        frame[column] = source[column].to_numpy()
    frame["action"] = action
    frame["outcome_label"] = outcome_label
    frame["logged_action_propensity"] = logged_propensities[selected]
    frame["_selection_reward"] = rewards[selected]
    return frame


def _library_diagnostics(
    trajectories: pd.DataFrame,
    library: pd.DataFrame,
    *,
    thresholds: dict[str, Any],
    candidate_cells: dict[tuple[str, str], pd.DataFrame],
    lower_reward_quantile: float,
    upper_reward_quantile: float,
    minimum_logged_propensity: float,
    max_examples_per_cell: int,
    minimum_examples_per_cell: int,
) -> dict[str, Any]:
    selected_counts = {
        action: {
            label: int(
                ((library["action"] == action) & (library["outcome_label"] == label)).sum()
            )
            for label in OUTCOME_LABELS
        }
        for action in ACTION_NAMES
    }
    insufficient_cells = [
        f"{action}/{label}"
        for action in ACTION_NAMES
        for label in OUTCOME_LABELS
        if selected_counts[action][label] < minimum_examples_per_cell
    ]
    return {
        "data_scope": "training_only_cross_fitted_behavior_support",
        "thresholds": {
            "lower_reward_quantile": lower_reward_quantile,
            "upper_reward_quantile": upper_reward_quantile,
            "minimum_logged_propensity": minimum_logged_propensity,
            "max_examples_per_cell": max_examples_per_cell,
            "minimum_examples_per_cell": minimum_examples_per_cell,
            "per_action_reward_thresholds": thresholds,
        },
        "candidates_after_support_and_patient_deduplication": {
            action: {
                label: len(candidate_cells[(action, label)])
                for label in OUTCOME_LABELS
            }
            for action in ACTION_NAMES
        },
        "selected_counts": selected_counts,
        "selected_rows": len(library),
        "unique_source_patients": int(library["source_patient_id"].nunique()),
        "logged_action_propensity": _quantiles(
            library["logged_action_propensity"].to_numpy()
        ),
        "map_below_65": {
            "selected_rate": float((library["map_mm_hg"] < 65.0).mean()),
            "source_rate": float((trajectories["map_mm_hg"] < 65.0).mean()),
        },
        "policy_improvement_readiness": {
            "approved": not insufficient_cells,
            "insufficient_cells": insufficient_cells,
            "reason": (
                None
                if not insufficient_cells
                else "One or more factual action/outcome cells lack examples."
            ),
        },
        "interpretation": (
            "Outcome labels are relative within logged action and are factual associations, "
            "not counterfactual optimal-action labels."
        ),
    }


def _retrieval_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )


def _one_row_frame(
    observation: pd.DataFrame | pd.Series | dict[str, Any],
) -> pd.DataFrame:
    if isinstance(observation, pd.DataFrame):
        frame = observation.copy()
    elif isinstance(observation, pd.Series):
        frame = observation.to_frame().T
    elif isinstance(observation, dict):
        frame = pd.DataFrame([observation])
    else:
        raise TypeError("Observation must be a DataFrame, Series, or dictionary.")
    if len(frame) != 1:
        raise ValueError("Retrieval requires exactly one observation.")
    missing = sorted(set(OBSERVATION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing observation columns: {missing}")
    return frame


def _validate_library(library: pd.DataFrame) -> None:
    required = set(OBSERVATION_COLUMNS) | {
        "source_patient_id",
        "action",
        "outcome_label",
    }
    if not isinstance(library, pd.DataFrame) or library.empty:
        raise ValueError("Factual example library must be a non-empty DataFrame.")
    missing = sorted(required - set(library.columns))
    if missing:
        raise ValueError(f"Missing factual example columns: {missing}")
    if not set(library["action"]).issubset(ACTION_NAMES):
        raise ValueError("Factual example library contains unknown actions.")
    if not set(library["outcome_label"]).issubset(OUTCOME_LABELS):
        raise ValueError("Factual example library contains unknown outcome labels.")


def _validate_training_trajectories(trajectories: pd.DataFrame) -> None:
    if not isinstance(trajectories, pd.DataFrame) or trajectories.empty:
        raise ValueError("Factual source trajectories must be non-empty.")
    required = set(OBSERVATION_COLUMNS) | {
        PATIENT_ID_COLUMN,
        "time_step",
        "split",
        LOGGED_ACTION_COLUMN,
    }
    missing = sorted(required - set(trajectories.columns))
    if missing:
        raise ValueError(f"Missing factual source columns: {missing}")
    if trajectories["split"].isna().any() or set(trajectories["split"].unique()) != {
        "train"
    }:
        raise ValueError("Factual examples may use only the training split.")
    unknown_actions = set(trajectories[LOGGED_ACTION_COLUMN].unique()) - set(
        ACTION_NAMES
    )
    if unknown_actions:
        raise ValueError(f"Unknown logged actions: {sorted(unknown_actions)}")


def _logged_rewards(trajectories: pd.DataFrame) -> np.ndarray:
    if "reward" in trajectories.columns:
        rewards = pd.to_numeric(trajectories["reward"], errors="coerce").to_numpy()
    else:
        try:
            rewards = trajectories.apply(compute_reward, axis=1).to_numpy(dtype=float)
        except KeyError as error:
            raise ValueError(
                f"Cannot label outcomes because column {error.args[0]!r} is missing."
            ) from error
    if not np.isfinite(rewards).all():
        raise ValueError("Factual example rewards must be finite.")
    return rewards


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }
