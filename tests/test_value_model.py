import numpy as np
import pandas as pd
import pytest

from prompts import ACTION_NAMES, OBSERVATION_COLUMNS
from value_model import (
    ActionRewardModel,
    action_support_mask,
    cross_validate_action_reward_model,
    patient_group_value_folds,
)


def make_training_data(num_patients=15):
    rows = []
    for patient_index in range(num_patients):
        for action_index, action in enumerate(ACTION_NAMES):
            state_signal = float(patient_index + action_index)
            row = {column: state_signal for column in OBSERVATION_COLUMNS}
            row.update(
                {
                    "patient_id": f"patient_{patient_index:02d}",
                    "split": "train",
                    "sex": "F" if patient_index % 2 else "M",
                    "previous_action": ACTION_NAMES[
                        (action_index - 1) % len(ACTION_NAMES)
                    ],
                    "map_mm_hg": 60.0 + patient_index,
                    "observed_clinician_action": action,
                    "reward": 0.1 * patient_index + action_index,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_action_reward_model_returns_canonical_value_matrix():
    trajectories = make_training_data()
    model = ActionRewardModel().fit(trajectories)

    values = model.predict_values(trajectories.iloc[:4])

    assert values.shape == (4, len(ACTION_NAMES))
    assert np.isfinite(values).all()
    assert tuple(model.models) == ACTION_NAMES


def test_logged_value_predictions_select_the_logged_action_column():
    trajectories = make_training_data()
    model = ActionRewardModel().fit(trajectories)
    values = model.predict_values(trajectories)
    indices = [ACTION_NAMES.index(action) for action in trajectories["observed_clinician_action"]]

    logged = model.predict_logged_values(trajectories)

    assert logged == pytest.approx(values[np.arange(len(values)), indices])


def test_value_model_handles_missing_and_unknown_categories():
    trajectories = make_training_data()
    trajectories.loc[0, "lactate_mmol_l"] = np.nan
    model = ActionRewardModel().fit(trajectories)
    observation = trajectories.iloc[[0]].copy()
    observation.loc[:, "sex"] = "UNKNOWN"
    observation.loc[:, "previous_action"] = "UNKNOWN"

    assert np.isfinite(model.predict_values(observation)).all()


def test_value_fit_rejects_non_training_data():
    trajectories = make_training_data()
    trajectories.loc[0, "split"] = "validation"

    with pytest.raises(ValueError, match="only the training split"):
        ActionRewardModel().fit(trajectories)


def test_patient_group_value_folds_never_split_a_patient():
    trajectories = make_training_data()

    folds = list(patient_group_value_folds(trajectories, n_splits=3))

    assert len(folds) == 3
    for train_index, validation_index in folds:
        train_patients = set(trajectories.iloc[train_index]["patient_id"])
        validation_patients = set(trajectories.iloc[validation_index]["patient_id"])
        assert train_patients.isdisjoint(validation_patients)


def test_cross_validation_reports_factual_per_action_metrics():
    diagnostics = cross_validate_action_reward_model(
        make_training_data(), n_splits=3
    )

    assert diagnostics["estimand"] == "one_step_expected_reward"
    assert diagnostics["patients"] == 15
    assert set(diagnostics["per_action"]) == set(ACTION_NAMES)
    assert len(diagnostics["folds"]) == 3
    assert np.isfinite(list(diagnostics["overall"].values())).all()
    assert diagnostics["map_below_65_slice"]["true"]["rows"] > 0
    assert "counterfactual" in diagnostics["counterfactual_warning"]


def test_support_mask_hides_weakly_supported_action_values():
    trajectories = make_training_data()
    model = ActionRewardModel().fit(trajectories)
    observations = trajectories.iloc[:2]
    behavior_probabilities = np.array(
        [[0.97, 0.02, 0.01], [0.80, 0.10, 0.10]], dtype=float
    )

    supported = model.predict_supported_values(
        observations, behavior_probabilities, minimum_propensity=0.02
    )

    assert np.isfinite(supported[0, :2]).all()
    assert np.isnan(supported[0, 2])
    assert np.isfinite(supported[1]).all()


def test_support_mask_rejects_invalid_probability_vectors():
    with pytest.raises(ValueError, match="sum to one"):
        action_support_mask(np.array([[0.5, 0.3, 0.3]]))


def test_post_action_columns_are_not_model_features():
    trajectories = make_training_data()
    trajectories["next_6h_map_delta"] = np.arange(len(trajectories))
    model = ActionRewardModel().fit(trajectories)

    for action_model in model.models.values():
        transformed_columns = action_model.named_steps["preprocessor"].feature_names_in_
        assert set(transformed_columns) == set(OBSERVATION_COLUMNS)
        assert "next_6h_map_delta" not in transformed_columns


def test_predict_before_fit_fails_clearly():
    with pytest.raises(RuntimeError, match="Fit ActionRewardModel"):
        ActionRewardModel().predict_values(make_training_data())
