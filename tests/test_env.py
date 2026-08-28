import pandas as pd
import pytest

from env import OfflineClinicalEnv

OBSERVATION_COLUMNS = ["map_mm_hg", "lactate_mmol_l"]


def transition(patient_id, time_step, terminal, **overrides):
    row = {
        "patient_id": patient_id,
        "time_step": time_step,
        "map_mm_hg": 70.0 + time_step,
        "lactate_mmol_l": 2.0,
        "observed_clinician_action": "maintain",
        "next_6h_map_delta": 0.0,
        "next_6h_lactate_delta": 0.0,
        "next_6h_deterioration": 0,
        "adverse_hypotension_next_6h": 0,
        "adverse_fluid_overload_next_6h": 0,
        "adverse_tachyarrhythmia_next_6h": 0,
        "terminal": terminal,
    }
    row.update(overrides)
    return row


def make_env():
    # Deliberately supply the rows out of temporal order.
    rows = [
        transition("patient_a", 1, 1, map_mm_hg=72.0),
        transition("patient_b", 0, 1, map_mm_hg=80.0),
        transition("patient_a", 0, 0, map_mm_hg=61.0),
    ]
    return OfflineClinicalEnv(pd.DataFrame(rows), OBSERVATION_COLUMNS)


def test_episode_is_temporally_sorted():
    env = make_env()

    first_observation, _ = env.reset(patient_id="patient_a")
    transition_result = env.step_logged()

    assert first_observation["map_mm_hg"] == 61.0
    assert transition_result[0] == first_observation
    assert transition_result[3]["map_mm_hg"] == 72.0
    assert transition_result[5]["time_step"] == 0


def test_terminal_transition_handling():
    env = make_env()
    env.reset(patient_id="patient_a")

    first_transition = env.step_logged()
    final_transition = env.step_logged()

    assert first_transition[4] is False
    assert first_transition[3] is not None
    assert final_transition[4] is True
    assert final_transition[3] is None
    with pytest.raises(RuntimeError, match="terminated"):
        env.step_logged()


def test_policy_observation_has_no_post_action_columns():
    env = make_env()

    observation, _ = env.reset(patient_id="patient_a")

    assert set(observation) == set(OBSERVATION_COLUMNS)
    assert "observed_clinician_action" not in observation
    assert "next_6h_map_delta" not in observation
    assert "terminal" not in observation


def test_missing_observation_value_becomes_none():
    rows = [
        transition("patient_a", 0, 1, lactate_mmol_l=float("nan")),
    ]
    env = OfflineClinicalEnv(pd.DataFrame(rows), OBSERVATION_COLUMNS)

    observation, _ = env.reset(patient_id="patient_a")

    assert observation["lactate_mmol_l"] is None


def test_seeded_episode_sampling_is_reproducible():
    env = make_env()

    _, first_metadata = env.reset(seed=123)
    _, second_metadata = env.reset(seed=123)

    assert first_metadata["patient_id"] == second_metadata["patient_id"]


def test_step_before_reset_fails_clearly():
    env = make_env()

    with pytest.raises(RuntimeError, match=r"Call reset\(\)"):
        env.step_logged()
