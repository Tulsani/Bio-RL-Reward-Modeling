"""Training-only behavior-policy probability model and diagnostics."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prompts import ACTION_NAMES, OBSERVATION_COLUMNS

LOGGED_ACTION_COLUMN = "observed_clinician_action"
PATIENT_ID_COLUMN = "patient_id"
CATEGORICAL_FEATURES = ("sex", "previous_action")
NUMERIC_FEATURES = tuple(
    column
    for column in OBSERVATION_COLUMNS
    if column not in CATEGORICAL_FEATURES
)


@dataclass(frozen=True)
class BehaviorFoldMetrics:
    fold: int
    train_patients: int
    validation_patients: int
    validation_rows: int
    log_loss: float
    multiclass_brier: float
    accuracy: float
    macro_f1: float
    top_label_ece: float


class BehaviorPolicyModel:
    """Estimate the logged behavior propensity ``b(a | s)``.

    The model intentionally does not use class weights: behavior-policy
    probabilities should approximate the logged action distribution rather than
    rebalance rare actions as a predictive-classification objective would.
    """

    def __init__(
        self,
        *,
        regularization_c: float = 1.0,
        max_iter: int = 2_000,
        random_seed: int = 42,
    ) -> None:
        if regularization_c <= 0:
            raise ValueError("regularization_c must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        self.regularization_c = regularization_c
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.pipeline = self._build_pipeline()
        self._is_fitted = False

    def fit(self, trajectories: pd.DataFrame) -> BehaviorPolicyModel:
        """Fit on training trajectories using pre-action fields only."""
        _validate_frame(trajectories, require_patient_id=False)
        _require_training_split(trajectories)
        _require_all_actions(trajectories[LOGGED_ACTION_COLUMN])

        features = trajectories.loc[:, OBSERVATION_COLUMNS]
        actions = trajectories[LOGGED_ACTION_COLUMN]
        self.pipeline.fit(features, actions)
        self._is_fitted = True
        return self

    def predict_proba(self, observations: pd.DataFrame) -> np.ndarray:
        """Return ``b(a | s)`` in canonical action order."""
        if not self._is_fitted:
            raise RuntimeError("Fit BehaviorPolicyModel before prediction.")
        missing = sorted(set(OBSERVATION_COLUMNS) - set(observations.columns))
        if missing:
            raise ValueError(f"Missing observation columns: {missing}")

        raw_probabilities = self.pipeline.predict_proba(
            observations.loc[:, OBSERVATION_COLUMNS]
        )
        classifier = self.pipeline.named_steps["classifier"]
        class_to_index = {
            action: index for index, action in enumerate(classifier.classes_)
        }
        probabilities = raw_probabilities[
            :, [class_to_index[action] for action in ACTION_NAMES]
        ]
        _validate_probabilities(probabilities)
        return probabilities

    def _build_pipeline(self) -> Pipeline:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
                (
                    "categorical",
                    categorical_pipeline,
                    list(CATEGORICAL_FEATURES),
                ),
            ]
        )
        classifier = LogisticRegression(
            C=self.regularization_c,
            class_weight=None,
            max_iter=self.max_iter,
            random_state=self.random_seed,
            solver="lbfgs",
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )


def patient_group_folds(
    trajectories: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield stratified folds with disjoint patients."""
    _validate_frame(trajectories, require_patient_id=True)
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")
    if trajectories[PATIENT_ID_COLUMN].nunique() < n_splits:
        raise ValueError("Number of patients must be at least n_splits.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )
    yield from splitter.split(
        trajectories,
        y=trajectories[LOGGED_ACTION_COLUMN],
        groups=trajectories[PATIENT_ID_COLUMN],
    )


def cross_validate_behavior_model(
    trajectories: pd.DataFrame,
    *,
    n_splits: int = 5,
    regularization_c: float = 1.0,
    max_iter: int = 2_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Return patient-grouped out-of-fold propensity diagnostics."""
    _validate_frame(trajectories, require_patient_id=True)
    _require_training_split(trajectories)
    _require_all_actions(trajectories[LOGGED_ACTION_COLUMN])

    out_of_fold = np.full(
        (len(trajectories), len(ACTION_NAMES)), np.nan, dtype=float
    )
    fold_metrics = []

    for fold, (train_index, validation_index) in enumerate(
        patient_group_folds(
            trajectories, n_splits=n_splits, random_seed=random_seed
        ),
        start=1,
    ):
        train_fold = trajectories.iloc[train_index]
        validation_fold = trajectories.iloc[validation_index]
        model = BehaviorPolicyModel(
            regularization_c=regularization_c,
            max_iter=max_iter,
            random_seed=random_seed,
        ).fit(train_fold)
        probabilities = model.predict_proba(validation_fold)
        out_of_fold[validation_index] = probabilities

        fold_metrics.append(
            _fold_metrics(
                fold,
                train_fold,
                validation_fold,
                probabilities,
            )
        )

    if not np.isfinite(out_of_fold).all():
        raise RuntimeError("Every row must receive an out-of-fold prediction.")

    actions = trajectories[LOGGED_ACTION_COLUMN].to_numpy()
    action_indices = np.array(
        [ACTION_NAMES.index(action) for action in actions], dtype=int
    )
    logged_propensity = out_of_fold[np.arange(len(actions)), action_indices]
    predicted_actions = np.asarray(ACTION_NAMES)[out_of_fold.argmax(axis=1)]
    per_action = {}
    for index, action in enumerate(ACTION_NAMES):
        target = actions == action
        action_logged_propensity = logged_propensity[target]
        per_action[action] = {
            "binary_brier": float(
                np.mean((out_of_fold[:, index] - target.astype(float)) ** 2)
            ),
            "one_vs_rest_ece": _binary_ece(
                target, out_of_fold[:, index]
            ),
            "logged_count": int(target.sum()),
            "logged_propensity_mean": float(action_logged_propensity.mean()),
            "logged_propensity_p05": float(
                np.quantile(action_logged_propensity, 0.05)
            ),
            "logged_propensity_median": float(
                np.median(action_logged_propensity)
            ),
        }

    return {
        "n_splits": n_splits,
        "rows": len(trajectories),
        "patients": trajectories[PATIENT_ID_COLUMN].nunique(),
        "actions": list(ACTION_NAMES),
        "class_weight": None,
        "overall": {
            "log_loss": _multiclass_log_loss(actions, out_of_fold),
            "multiclass_brier": _multiclass_brier(
                actions, out_of_fold
            ),
            "accuracy": float(accuracy_score(actions, predicted_actions)),
            "macro_f1": float(
                f1_score(
                    actions,
                    predicted_actions,
                    labels=list(ACTION_NAMES),
                    average="macro",
                    zero_division=0,
                )
            ),
            "top_label_ece": _top_label_ece(actions, out_of_fold),
        },
        "observed_action_rates": {
            action: float(np.mean(actions == action)) for action in ACTION_NAMES
        },
        "mean_predicted_probabilities": {
            action: float(out_of_fold[:, index].mean())
            for index, action in enumerate(ACTION_NAMES)
        },
        "argmax_action_rates": {
            action: float(np.mean(predicted_actions == action))
            for action in ACTION_NAMES
        },
        "per_action": per_action,
        "map_below_65_slice": _binary_state_slice(
            trajectories["map_mm_hg"].to_numpy() < 65.0,
            actions,
            out_of_fold,
        ),
        "logged_action_propensity": {
            "min": float(logged_propensity.min()),
            "p01": float(np.quantile(logged_propensity, 0.01)),
            "p05": float(np.quantile(logged_propensity, 0.05)),
            "median": float(np.median(logged_propensity)),
            "mean": float(logged_propensity.mean()),
        },
        "folds": [asdict(metrics) for metrics in fold_metrics],
    }


def _fold_metrics(
    fold: int,
    train_fold: pd.DataFrame,
    validation_fold: pd.DataFrame,
    probabilities: np.ndarray,
) -> BehaviorFoldMetrics:
    actions = validation_fold[LOGGED_ACTION_COLUMN].to_numpy()
    predicted = np.asarray(ACTION_NAMES)[probabilities.argmax(axis=1)]
    return BehaviorFoldMetrics(
        fold=fold,
        train_patients=train_fold[PATIENT_ID_COLUMN].nunique(),
        validation_patients=validation_fold[PATIENT_ID_COLUMN].nunique(),
        validation_rows=len(validation_fold),
        log_loss=_multiclass_log_loss(actions, probabilities),
        multiclass_brier=_multiclass_brier(actions, probabilities),
        accuracy=float(accuracy_score(actions, predicted)),
        macro_f1=float(
            f1_score(
                actions,
                predicted,
                labels=list(ACTION_NAMES),
                average="macro",
                zero_division=0,
            )
        ),
        top_label_ece=_top_label_ece(actions, probabilities),
    )


def _multiclass_brier(
    actions: np.ndarray, probabilities: np.ndarray
) -> float:
    indices = np.array(
        [ACTION_NAMES.index(action) for action in actions], dtype=int
    )
    targets = np.eye(len(ACTION_NAMES))[indices]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def _multiclass_log_loss(
    actions: np.ndarray, probabilities: np.ndarray
) -> float:
    indices = np.array(
        [ACTION_NAMES.index(action) for action in actions], dtype=int
    )
    logged_probabilities = probabilities[np.arange(len(actions)), indices]
    epsilon = np.finfo(float).eps
    return float(-np.log(np.clip(logged_probabilities, epsilon, 1.0)).mean())


def _top_label_ece(
    actions: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(ACTION_NAMES)[probabilities.argmax(axis=1)]
    correct = predicted == actions
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in pairwise(edges):
        if upper == 1.0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        if in_bin.any():
            weight = float(in_bin.mean())
            ece += weight * abs(
                float(correct[in_bin].mean())
                - float(confidence[in_bin].mean())
            )
    return float(ece)


def _binary_ece(
    target: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in pairwise(edges):
        if upper == 1.0:
            in_bin = (probability >= lower) & (probability <= upper)
        else:
            in_bin = (probability >= lower) & (probability < upper)
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(target[in_bin].mean())
                - float(probability[in_bin].mean())
            )
    return float(ece)


def _binary_state_slice(
    mask: np.ndarray, actions: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    result = {}
    for label, selected in (("false", ~mask), ("true", mask)):
        result[label] = {
            "rows": int(selected.sum()),
            "observed_action_rates": {
                action: float(np.mean(actions[selected] == action))
                for action in ACTION_NAMES
            },
            "mean_predicted_probabilities": {
                action: float(probabilities[selected, index].mean())
                for index, action in enumerate(ACTION_NAMES)
            },
        }
    return result


def _validate_frame(
    trajectories: pd.DataFrame, *, require_patient_id: bool
) -> None:
    if not isinstance(trajectories, pd.DataFrame) or trajectories.empty:
        raise ValueError("Behavior-model trajectories must be a non-empty DataFrame.")
    required = set(OBSERVATION_COLUMNS) | {LOGGED_ACTION_COLUMN}
    if require_patient_id:
        required.add(PATIENT_ID_COLUMN)
    missing = sorted(required - set(trajectories.columns))
    if missing:
        raise ValueError(f"Missing required behavior-model columns: {missing}")
    if trajectories[LOGGED_ACTION_COLUMN].isna().any():
        raise ValueError("Logged actions cannot be missing.")
    if require_patient_id and trajectories[PATIENT_ID_COLUMN].isna().any():
        raise ValueError("Patient IDs cannot be missing.")
    unknown_actions = set(
        trajectories[LOGGED_ACTION_COLUMN].dropna().unique()
    ) - set(ACTION_NAMES)
    if unknown_actions:
        raise ValueError(f"Unknown logged actions: {sorted(unknown_actions)}")


def _require_training_split(trajectories: pd.DataFrame) -> None:
    if "split" not in trajectories.columns:
        raise ValueError("A split column is required to enforce training-only fit.")
    if trajectories["split"].isna().any():
        raise ValueError("Split values cannot be missing.")
    splits = set(trajectories["split"].dropna().unique())
    if splits != {"train"}:
        raise ValueError(
            f"Behavior model may fit only the training split, got {sorted(splits)}."
        )


def _require_all_actions(actions: pd.Series) -> None:
    missing = sorted(set(ACTION_NAMES) - set(actions.dropna().unique()))
    if missing:
        raise ValueError(f"Training data is missing actions: {missing}")


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(ACTION_NAMES):
        raise ValueError("Behavior probabilities must have shape [N, 3].")
    if not np.isfinite(probabilities).all():
        raise ValueError("Behavior probabilities must be finite.")
    if (probabilities < 0).any():
        raise ValueError("Behavior probabilities must be non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Behavior probability rows must sum to one.")
