import numpy as np
import pandas as pd
import pytest

from factual_examples import (
    CrossFittedBehaviorContext,
    FactualExampleRetriever,
    build_factual_outcome_library,
    generate_cross_fitted_behavior_context,
    prompt_safe_factual_records,
)
from prompts import ACTION_NAMES, OBSERVATION_COLUMNS


def make_training_data(num_patients=30):
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
                    "reward": float(patient_index),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def make_behavior_context(trajectories):
    probabilities = np.tile(np.array([0.6, 0.25, 0.15]), (len(trajectories), 1))
    return CrossFittedBehaviorContext(
        probabilities=probabilities,
        fold_assignments=np.ones(len(trajectories), dtype=int),
    )


def test_cross_fitted_behavior_context_covers_all_rows():
    trajectories = make_training_data()

    context = generate_cross_fitted_behavior_context(trajectories, n_splits=3)

    assert context.probabilities.shape == (len(trajectories), 3)
    assert np.isfinite(context.probabilities).all()
    assert set(context.fold_assignments) == {1, 2, 3}


def test_factual_library_balances_every_action_outcome_cell():
    trajectories = make_training_data()

    library, diagnostics = build_factual_outcome_library(
        trajectories,
        make_behavior_context(trajectories),
        max_examples_per_cell=5,
        minimum_examples_per_cell=5,
    )

    counts = library.groupby(["action", "outcome_label"]).size()
    assert len(library) == 30
    assert (counts == 5).all()
    assert diagnostics["policy_improvement_readiness"]["approved"]


def test_factual_library_contains_no_reward_or_post_action_columns():
    trajectories = make_training_data()
    trajectories["next_6h_deterioration"] = 1

    library, _ = build_factual_outcome_library(
        trajectories, make_behavior_context(trajectories), max_examples_per_cell=3
    )

    assert "reward" not in library.columns
    assert "_selection_reward" not in library.columns
    assert "next_6h_deterioration" not in library.columns
    assert set(OBSERVATION_COLUMNS).issubset(library.columns)


def test_logged_propensity_filter_removes_unsupported_action_cell():
    trajectories = make_training_data()
    probabilities = make_behavior_context(trajectories).probabilities.copy()
    probabilities[:, :] = np.array([0.79, 0.20, 0.01])
    context = CrossFittedBehaviorContext(
        probabilities, np.ones(len(trajectories), dtype=int)
    )

    library, diagnostics = build_factual_outcome_library(
        trajectories,
        context,
        minimum_logged_propensity=0.02,
        minimum_examples_per_cell=1,
    )

    assert "escalate_vasopressor" not in set(library["action"])
    readiness = diagnostics["policy_improvement_readiness"]
    assert not readiness["approved"]
    assert "escalate_vasopressor/favorable" in readiness["insufficient_cells"]


def test_retriever_returns_every_cell_and_can_exclude_patient():
    trajectories = make_training_data()
    library, _ = build_factual_outcome_library(
        trajectories,
        make_behavior_context(trajectories),
        max_examples_per_cell=10,
    )
    retriever = FactualExampleRetriever().fit(library)
    excluded_patient = library.iloc[0]["source_patient_id"]

    examples = retriever.retrieve(
        trajectories.iloc[0], examples_per_cell=1, exclude_patient_id=excluded_patient
    )

    assert len(examples) == 6
    assert examples.groupby(["action", "outcome_label"]).size().eq(1).all()
    assert excluded_patient not in set(examples["source_patient_id"])


def test_prompt_records_strip_provenance_and_support_metadata():
    trajectories = make_training_data()
    library, _ = build_factual_outcome_library(
        trajectories,
        make_behavior_context(trajectories),
        max_examples_per_cell=2,
    )

    records = prompt_safe_factual_records(library.iloc[:2])

    assert set(records[0]) == {
        "state",
        "logged_action",
        "observed_outcome_label",
    }
    assert set(records[0]["state"]) == set(OBSERVATION_COLUMNS)
    assert "logged_action_propensity" not in records[0]
    assert "source_patient_id" not in records[0]


def test_factual_generation_rejects_validation_rows():
    trajectories = make_training_data()
    trajectories.loc[0, "split"] = "validation"

    with pytest.raises(ValueError, match="only the training split"):
        build_factual_outcome_library(
            trajectories, make_behavior_context(trajectories)
        )
