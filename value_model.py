"""Training-only action-conditioned one-step reward model.

This module estimates ``m(s, a) = E[r_t | s_t=s, a_t=a]``.  It is a direct
reward model, not a fitted-Q evaluation implementation and therefore does not
include discounted future rewards.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prompts import ACTION_NAMES, OBSERVATION_COLUMNS
from reward import compute_reward

LOGGED_ACTION_COLUMN = "observed_clinician_action"
PATIENT_ID_COLUMN = "patient_id"
REWARD_COLUMN = "reward"
CATEGORICAL_FEATURES = ("sex", "previous_action")
NUMERIC_FEATURES = tuple(
    column for column in OBSERVATION_COLUMNS if column not in CATEGORICAL_FEATURES
)
ModelType = Literal["ridge", "hist_gradient_boosting"]
MODEL_TYPES = ("ridge", "hist_gradient_boosting")


@dataclass(frozen=True)
class ValueFoldMetrics:
    fold: int
    train_patients: int
    validation_patients: int
    validation_rows: int
    rmse: float
    mae: float
    r2: float


class ActionRewardModel:
    """Estimate one-step reward for each action with a separate ridge model.

    Separate outcome models let state coefficients differ by action. This is a
    simple T-learner baseline; predictions for actions other than the logged
    action are modeled counterfactual estimates, never observed outcomes.
    """

    def __init__(
        self,
        *,
        model_type: ModelType = "ridge",
        alpha: float = 1.0,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter <= 0 or max_leaf_nodes < 2 or min_samples_leaf <= 0:
            raise ValueError("Tree size and iteration parameters must be positive.")
        if l2_regularization < 0:
            raise ValueError("l2_regularization must be non-negative.")
        self.model_type = model_type
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_seed = random_seed
        self.models: dict[str, Pipeline] = {}
        self._is_fitted = False

    def fit(self, trajectories: pd.DataFrame) -> ActionRewardModel:
        """Fit one model per logged action using training trajectories only."""
        _validate_frame(trajectories, require_patient_id=False)
        _require_training_split(trajectories)
        _require_all_actions(trajectories[LOGGED_ACTION_COLUMN])
        rewards = _reward_targets(trajectories)

        self.models = {}
        for action in ACTION_NAMES:
            selected = trajectories[LOGGED_ACTION_COLUMN] == action
            model = _build_pipeline(
                model_type=self.model_type,
                alpha=self.alpha,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                random_seed=self.random_seed,
            )
            model.fit(
                trajectories.loc[selected, OBSERVATION_COLUMNS],
                rewards[selected],
            )
            self.models[action] = model

        self._is_fitted = True
        return self

    def predict_values(self, observations: pd.DataFrame) -> np.ndarray:
        """Return modeled one-step rewards in canonical action order."""
        if not self._is_fitted:
            raise RuntimeError("Fit ActionRewardModel before prediction.")
        _validate_observations(observations)
        features = observations.loc[:, OBSERVATION_COLUMNS]
        values = np.column_stack(
            [self.models[action].predict(features) for action in ACTION_NAMES]
        )
        if values.shape != (len(observations), len(ACTION_NAMES)):
            raise RuntimeError("Action-value predictions have an invalid shape.")
        if not np.isfinite(values).all():
            raise RuntimeError("Action-value predictions must be finite.")
        return values

    def predict_logged_values(self, trajectories: pd.DataFrame) -> np.ndarray:
        """Select each row's modeled reward for its logged action."""
        _validate_actions(trajectories)
        values = self.predict_values(trajectories)
        indices = np.array(
            [ACTION_NAMES.index(action) for action in trajectories[LOGGED_ACTION_COLUMN]],
            dtype=int,
        )
        return values[np.arange(len(trajectories)), indices]

    def predict_supported_values(
        self,
        observations: pd.DataFrame,
        behavior_probabilities: np.ndarray,
        *,
        minimum_propensity: float = 0.02,
    ) -> np.ndarray:
        """Mask modeled values lacking minimum behavior-policy support with NaN."""
        support = action_support_mask(
            behavior_probabilities,
            n_rows=len(observations),
            minimum_propensity=minimum_propensity,
        )
        return np.where(support, self.predict_values(observations), np.nan)


def action_support_mask(
    behavior_probabilities: np.ndarray,
    *,
    n_rows: int | None = None,
    minimum_propensity: float = 0.02,
) -> np.ndarray:
    """Return actions whose estimated behavior propensity clears a threshold."""
    probabilities = np.asarray(behavior_probabilities, dtype=float)
    expected_rows = len(probabilities) if n_rows is None else n_rows
    expected_shape = (expected_rows, len(ACTION_NAMES))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"behavior_probabilities must have shape {expected_shape}."
        )
    if not 0.0 <= minimum_propensity <= 1.0:
        raise ValueError("minimum_propensity must be between zero and one.")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Behavior probabilities must be finite and non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Behavior probabilities must sum to one per row.")
    return probabilities >= minimum_propensity


def patient_group_value_folds(
    trajectories: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield action-stratified folds with disjoint patients."""
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


def cross_validate_action_reward_model(
    trajectories: pd.DataFrame,
    *,
    n_splits: int = 5,
    model_type: ModelType = "ridge",
    alpha: float = 1.0,
    learning_rate: float = 0.05,
    max_iter: int = 200,
    max_leaf_nodes: int = 15,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Return patient-grouped factual out-of-fold reward diagnostics."""
    _validate_frame(trajectories, require_patient_id=True)
    _require_training_split(trajectories)
    _require_all_actions(trajectories[LOGGED_ACTION_COLUMN])
    rewards = _reward_targets(trajectories)
    out_of_fold = np.full(len(trajectories), np.nan, dtype=float)
    fold_metrics = []

    for fold, (train_index, validation_index) in enumerate(
        patient_group_value_folds(
            trajectories, n_splits=n_splits, random_seed=random_seed
        ),
        start=1,
    ):
        train_fold = trajectories.iloc[train_index]
        validation_fold = trajectories.iloc[validation_index]
        model = ActionRewardModel(
            model_type=model_type,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_seed=random_seed,
        ).fit(train_fold)
        predictions = model.predict_logged_values(validation_fold)
        out_of_fold[validation_index] = predictions
        fold_metrics.append(
            _fold_metrics(
                fold,
                train_fold,
                validation_fold,
                rewards[validation_index],
                predictions,
            )
        )

    if not np.isfinite(out_of_fold).all():
        raise RuntimeError("Every row must receive an out-of-fold prediction.")

    actions = trajectories[LOGGED_ACTION_COLUMN].to_numpy()
    return {
        "estimand": "one_step_expected_reward",
        "model_type": f"per_action_{model_type}_t_learner",
        "n_splits": n_splits,
        "rows": len(trajectories),
        "patients": trajectories[PATIENT_ID_COLUMN].nunique(),
        "actions": list(ACTION_NAMES),
        "overall": _regression_metrics(rewards, out_of_fold),
        "per_action": {
            action: {
                "count": int((actions == action).sum()),
                **_regression_metrics(
                    rewards[actions == action], out_of_fold[actions == action]
                ),
            }
            for action in ACTION_NAMES
        },
        "map_below_65_slice": _state_slice_metrics(
            trajectories["map_mm_hg"].to_numpy() < 65.0,
            rewards,
            out_of_fold,
        ),
        "folds": [asdict(metrics) for metrics in fold_metrics],
        "counterfactual_warning": (
            "Only logged-action predictions are validated. Other action values "
            "are modeled counterfactual estimates and require support checks."
        ),
    }


def compare_action_reward_models(
    trajectories: pd.DataFrame,
    *,
    model_configs: dict[str, dict[str, Any]],
    n_splits: int = 5,
    random_seed: int = 42,
    baseline_name: str = "ridge",
) -> dict[str, Any]:
    """Compare model specifications under identical patient-grouped folds."""
    if baseline_name not in model_configs:
        raise ValueError("baseline_name must identify an entry in model_configs.")
    if len(model_configs) < 2:
        raise ValueError("At least two model configurations are required.")

    results = {
        name: cross_validate_action_reward_model(
            trajectories,
            n_splits=n_splits,
            random_seed=random_seed,
            **parameters,
        )
        for name, parameters in model_configs.items()
    }
    baseline = results[baseline_name]
    deltas = {}
    for name, result in results.items():
        if name == baseline_name:
            continue
        deltas[name] = {
            "overall": _metric_deltas(result["overall"], baseline["overall"]),
            "per_action": {
                action: _metric_deltas(
                    result["per_action"][action],
                    baseline["per_action"][action],
                )
                for action in ACTION_NAMES
            },
            "map_below_65": _metric_deltas(
                result["map_below_65_slice"]["true"],
                baseline["map_below_65_slice"]["true"],
            ),
        }
    return {
        "baseline": baseline_name,
        "models": results,
        "deltas_vs_baseline": deltas,
        "delta_definition": "candidate minus baseline; lower RMSE/MAE and higher R2 are better",
    }


def _build_pipeline(
    *,
    model_type: ModelType,
    alpha: float,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    random_seed: int,
) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if model_type == "ridge":
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=model_type == "ridge",
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ]
    )
    if model_type == "ridge":
        regressor = Ridge(alpha=alpha)
    else:
        regressor = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=random_seed,
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def _metric_deltas(
    candidate: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    return {
        metric: float(candidate[metric] - baseline[metric])
        for metric in ("rmse", "mae", "r2")
    }


def _fold_metrics(
    fold: int,
    train_fold: pd.DataFrame,
    validation_fold: pd.DataFrame,
    rewards: np.ndarray,
    predictions: np.ndarray,
) -> ValueFoldMetrics:
    metrics = _regression_metrics(rewards, predictions)
    return ValueFoldMetrics(
        fold=fold,
        train_patients=train_fold[PATIENT_ID_COLUMN].nunique(),
        validation_patients=validation_fold[PATIENT_ID_COLUMN].nunique(),
        validation_rows=len(validation_fold),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
    )


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
    }


def _state_slice_metrics(
    mask: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    return {
        label: {
            "rows": int(selected.sum()),
            **_regression_metrics(actual[selected], predicted[selected]),
        }
        for label, selected in (("false", ~mask), ("true", mask))
    }


def _reward_targets(trajectories: pd.DataFrame) -> np.ndarray:
    if REWARD_COLUMN in trajectories.columns:
        rewards = pd.to_numeric(trajectories[REWARD_COLUMN], errors="coerce").to_numpy()
    else:
        try:
            rewards = trajectories.apply(compute_reward, axis=1).to_numpy(dtype=float)
        except KeyError as error:
            raise ValueError(
                f"Cannot compute rewards because outcome column {error.args[0]!r} is missing."
            ) from error
    if not np.isfinite(rewards).all():
        raise ValueError("Reward targets must be finite.")
    return rewards.astype(float, copy=False)


def _validate_observations(observations: pd.DataFrame) -> None:
    if not isinstance(observations, pd.DataFrame) or observations.empty:
        raise ValueError("Observations must be a non-empty DataFrame.")
    missing = sorted(set(OBSERVATION_COLUMNS) - set(observations.columns))
    if missing:
        raise ValueError(f"Missing observation columns: {missing}")


def _validate_actions(trajectories: pd.DataFrame) -> None:
    if LOGGED_ACTION_COLUMN not in trajectories.columns:
        raise ValueError(f"Missing logged action column: {LOGGED_ACTION_COLUMN}")
    if trajectories[LOGGED_ACTION_COLUMN].isna().any():
        raise ValueError("Logged actions cannot be missing.")
    unknown = set(trajectories[LOGGED_ACTION_COLUMN].unique()) - set(ACTION_NAMES)
    if unknown:
        raise ValueError(f"Unknown logged actions: {sorted(unknown)}")


def _validate_frame(
    trajectories: pd.DataFrame, *, require_patient_id: bool
) -> None:
    _validate_observations(trajectories)
    _validate_actions(trajectories)
    if require_patient_id:
        if PATIENT_ID_COLUMN not in trajectories.columns:
            raise ValueError(f"Missing patient ID column: {PATIENT_ID_COLUMN}")
        if trajectories[PATIENT_ID_COLUMN].isna().any():
            raise ValueError("Patient IDs cannot be missing.")
    _reward_targets(trajectories)


def _require_training_split(trajectories: pd.DataFrame) -> None:
    if "split" not in trajectories.columns:
        raise ValueError("A split column is required to enforce training-only fit.")
    if trajectories["split"].isna().any():
        raise ValueError("Split values cannot be missing.")
    splits = set(trajectories["split"].unique())
    if splits != {"train"}:
        raise ValueError("Value models may fit only the training split.")


def _require_all_actions(actions: pd.Series) -> None:
    missing = sorted(set(ACTION_NAMES) - set(actions.unique()))
    if missing:
        raise ValueError(f"Training data must contain every action; missing: {missing}")
