import numpy as np
import pandas as pd
import pytest

from behavior_model import (
    BehaviorPolicyModel,
    cross_validate_behavior_model,
    patient_group_folds,
)
from prompts import ACTION_NAMES, OBSERVATION_COLUMNS


def make_training_data(num_patients=12):
    rows = []
    for patient_index in range(num_patients):
        for action_index, action in enumerate(ACTION_NAMES):
            row = {column: float(patient_index) for column in OBSERVATION_COLUMNS}
            row.update(
                {
                    "patient_id": f"patient_{patient_index:02d}",
                    "split": "train",
                    "sex": "F" if patient_index % 2 else "M",
                    "previous_action": ACTION_NAMES[
                        (action_index - 1) % len(ACTION_NAMES)
                    ],
                    "map_mm_hg": 60.0 + 10.0 * action_index,
                    "lactate_mmol_l": 3.0 - action_index,
                    "observed_clinician_action": action,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_behavior_model_returns_canonical_probabilities():
    trajectories = make_training_data()
    model = BehaviorPolicyModel().fit(trajectories)

    probabilities = model.predict_proba(trajectories.iloc[:5])
    classifier_classes = model.pipeline.named_steps["classifier"].classes_

    assert probabilities.shape == (5, 3)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(5))
    assert tuple(classifier_classes) != ACTION_NAMES
    raw = model.pipeline.predict_proba(
        trajectories.loc[:4, OBSERVATION_COLUMNS]
    )
    class_to_index = {
        action: index for index, action in enumerate(classifier_classes)
    }
    expected = raw[:, [class_to_index[action] for action in ACTION_NAMES]]
    assert probabilities == pytest.approx(expected)


def test_behavior_model_handles_missing_and_unknown_categories():
    trajectories = make_training_data()
    trajectories.loc[0, "lactate_mmol_l"] = np.nan
    model = BehaviorPolicyModel().fit(trajectories)
    observation = trajectories.iloc[[0]].copy()
    observation.loc[:, "sex"] = "UNKNOWN"
    observation.loc[:, "previous_action"] = "UNKNOWN"

    probabilities = model.predict_proba(observation)

    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_behavior_fit_rejects_non_training_data():
    trajectories = make_training_data()
    trajectories.loc[0, "split"] = "validation"

    with pytest.raises(ValueError, match="only the training split"):
        BehaviorPolicyModel().fit(trajectories)


def test_behavior_model_preserves_natural_class_distribution():
    model = BehaviorPolicyModel().fit(make_training_data())
    classifier = model.pipeline.named_steps["classifier"]

    assert classifier.class_weight is None


def test_patient_group_folds_never_split_a_patient():
    trajectories = make_training_data()

    folds = list(patient_group_folds(trajectories, n_splits=3))

    assert len(folds) == 3
    for train_index, validation_index in folds:
        train_patients = set(trajectories.iloc[train_index]["patient_id"])
        validation_patients = set(
            trajectories.iloc[validation_index]["patient_id"]
        )
        assert train_patients.isdisjoint(validation_patients)


def test_cross_validation_reports_support_and_calibration_metrics():
    diagnostics = cross_validate_behavior_model(
        make_training_data(), n_splits=3
    )

    assert diagnostics["n_splits"] == 3
    assert diagnostics["patients"] == 12
    assert diagnostics["actions"] == list(ACTION_NAMES)
    assert diagnostics["class_weight"] is None
    assert len(diagnostics["folds"]) == 3
    assert np.isfinite(list(diagnostics["overall"].values())).all()
    assert sum(diagnostics["observed_action_rates"].values()) == pytest.approx(
        1.0
    )
    assert sum(
        diagnostics["mean_predicted_probabilities"].values()
    ) == pytest.approx(1.0)
    assert diagnostics["logged_action_propensity"]["min"] > 0.0
    assert set(diagnostics["per_action"]) == set(ACTION_NAMES)
    assert sum(diagnostics["argmax_action_rates"].values()) == pytest.approx(
        1.0
    )
    assert diagnostics["map_below_65_slice"]["true"]["rows"] > 0


def test_post_action_columns_are_not_model_features():
    trajectories = make_training_data()
    trajectories["next_6h_map_delta"] = np.arange(len(trajectories))
    model = BehaviorPolicyModel().fit(trajectories)

    transformed_columns = model.pipeline.named_steps[
        "preprocessor"
    ].feature_names_in_

    assert set(transformed_columns) == set(OBSERVATION_COLUMNS)
    assert "next_6h_map_delta" not in transformed_columns


def test_predict_before_fit_fails_clearly():
    with pytest.raises(RuntimeError, match="Fit BehaviorPolicyModel"):
        BehaviorPolicyModel().predict_proba(make_training_data())
