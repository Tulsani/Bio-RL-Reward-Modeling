import numpy as np
import pandas as pd
import pytest

from contrastive_examples import (
    MODEL_NAMES,
    CrossFittedEstimates,
    build_contrastive_examples,
    diagnose_action_reward_bias,
    generate_cross_fitted_estimates,
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
                    "time_step": action_index,
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


def make_estimates(trajectories):
    n_rows = len(trajectories)
    behavior = np.tile(np.array([0.6, 0.25, 0.15]), (n_rows, 1))
    ridge = np.tile(np.array([0.0, 0.2, 0.6]), (n_rows, 1))
    nonlinear = np.tile(np.array([0.0, 0.3, 0.7]), (n_rows, 1))
    return CrossFittedEstimates(
        behavior_probabilities=behavior,
        action_values={
            "ridge": ridge,
            "hist_gradient_boosting": nonlinear,
        },
        fold_assignments=np.ones(n_rows, dtype=int),
    )


def test_cross_fitted_estimates_cover_every_row_and_action():
    trajectories = make_training_data()

    estimates = generate_cross_fitted_estimates(
        trajectories,
        n_splits=3,
        nonlinear_config={"max_iter": 10},
    )

    assert estimates.behavior_probabilities.shape == (len(trajectories), 3)
    assert set(estimates.action_values) == set(MODEL_NAMES)
    assert np.isfinite(estimates.behavior_probabilities).all()
    assert np.isfinite(list(estimates.action_values.values())).all()
    assert set(estimates.fold_assignments) == {1, 2, 3}


def test_selection_builds_supported_positive_and_near_negative_pairs():
    trajectories = make_training_data()

    examples, diagnostics = build_contrastive_examples(
        trajectories,
        make_estimates(trajectories),
        minimum_advantage=0.1,
        max_examples_per_action=5,
        require_preferred_matches_logged=False,
        minimum_logged_reward=None,
    )

    assert len(examples) == 5
    assert set(examples["preferred_action"]) == {"escalate_vasopressor"}
    assert set(examples["near_negative_action"]) == {"iv_fluids"}
    assert (examples["robust_advantage"] >= 0.1).all()
    assert diagnostics["selection_stages"]["candidates_before_balance"] == len(
        trajectories
    )


def test_selection_rejects_weakly_supported_preferred_action():
    trajectories = make_training_data()
    estimates = make_estimates(trajectories)
    probabilities = estimates.behavior_probabilities.copy()
    probabilities[:, :] = np.array([0.79, 0.20, 0.01])
    weak_support = CrossFittedEstimates(
        probabilities, estimates.action_values, estimates.fold_assignments
    )

    examples, diagnostics = build_contrastive_examples(
        trajectories, weak_support, minimum_propensity=0.02
    )

    assert examples.empty
    assert diagnostics["selection_stages"][
        "models_agree_on_preferred_action"
    ] == len(trajectories)
    assert diagnostics["selection_stages"]["both_advantages_clear_margin"] == 0


def test_default_selection_anchors_positive_to_successful_logged_action():
    trajectories = make_training_data()

    examples, diagnostics = build_contrastive_examples(
        trajectories, make_estimates(trajectories)
    )

    assert not examples.empty
    assert examples["preferred_matches_logged"].all()
    assert (examples["logged_action"] == examples["preferred_action"]).all()
    assert diagnostics["thresholds"]["minimum_logged_reward"] == 0.0
    assert not diagnostics["policy_improvement_readiness"]["approved"]
    assert "maintain" in diagnostics["policy_improvement_readiness"][
        "underrepresented_actions"
    ]


def test_selection_rejects_model_disagreement():
    trajectories = make_training_data()
    estimates = make_estimates(trajectories)
    disagreeing = CrossFittedEstimates(
        estimates.behavior_probabilities,
        {
            "ridge": estimates.action_values["ridge"],
            "hist_gradient_boosting": np.tile(
                np.array([0.8, 0.3, 0.1]), (len(trajectories), 1)
            ),
        },
        estimates.fold_assignments,
    )

    examples, diagnostics = build_contrastive_examples(
        trajectories, disagreeing
    )

    assert examples.empty
    assert diagnostics["selection_stages"][
        "models_agree_on_preferred_action"
    ] == 0


def test_exported_examples_do_not_contain_post_action_outcomes():
    trajectories = make_training_data()
    trajectories["next_6h_deterioration"] = 1

    examples, _ = build_contrastive_examples(
        trajectories, make_estimates(trajectories), max_examples_per_action=3
    )

    assert "next_6h_deterioration" not in examples.columns
    assert set(OBSERVATION_COLUMNS).issubset(examples.columns)


def test_contrastive_generation_rejects_validation_rows():
    trajectories = make_training_data()
    trajectories.loc[0, "split"] = "validation"

    with pytest.raises(ValueError, match="only the training split"):
        build_contrastive_examples(trajectories, make_estimates(trajectories))


def test_action_bias_diagnostics_expose_intervention_preference():
    trajectories = make_training_data()

    diagnostics = diagnose_action_reward_bias(
        trajectories, make_estimates(trajectories)
    )

    assert diagnostics["data_scope"] == "training_only_cross_fitted"
    ridge = diagnostics["global_model_rankings"]["ridge"]
    assert ridge["top_action_rates"]["escalate_vasopressor"] == 1.0
    maintain = diagnostics["logged_maintain_states"]
    assert maintain["rows"] == len(trajectories) / len(ACTION_NAMES)
    assert maintain["models"]["ridge"]["maintain_ranked_top_rate"] == 0.0
    assert maintain["state_slices"]["map_below_65"]["rows"] > 0
