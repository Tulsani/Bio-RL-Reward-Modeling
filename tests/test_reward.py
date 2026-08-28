import math

import pytest

from reward import compute_reward


def make_outcomes(**overrides):
    """Return a neutral post-action outcome with optional field overrides."""
    outcomes = {
        "next_6h_map_delta": 0.0,
        "next_6h_lactate_delta": 0.0,
        "next_6h_deterioration": 0,
        "adverse_hypotension_next_6h": 0,
        "adverse_fluid_overload_next_6h": 0,
        "adverse_tachyarrhythmia_next_6h": 0,
    }
    outcomes.update(overrides)
    return outcomes


def test_neutral_outcomes_have_zero_reward():
    assert compute_reward(make_outcomes()) == pytest.approx(0.0)


def test_physiological_improvement_has_positive_reward():
    reward = compute_reward(
        make_outcomes(
            next_6h_map_delta=5.0,
            next_6h_lactate_delta=-0.5,
        )
    )

    assert reward == pytest.approx(0.8)


def test_physiological_worsening_has_negative_reward():
    reward = compute_reward(
        make_outcomes(
            next_6h_map_delta=-5.0,
            next_6h_lactate_delta=0.5,
        )
    )

    assert reward == pytest.approx(-0.8)


@pytest.mark.parametrize(
    ("event_column", "expected_reward"),
    [
        ("next_6h_deterioration", -2.0),
        ("adverse_hypotension_next_6h", -1.5),
        ("adverse_fluid_overload_next_6h", -1.5),
        ("adverse_tachyarrhythmia_next_6h", -1.5),
    ],
)
def test_each_adverse_event_reduces_reward(event_column, expected_reward):
    reward = compute_reward(make_outcomes(**{event_column: 1}))

    assert reward == pytest.approx(expected_reward)


def test_physiological_components_are_clipped():
    extreme_improvement = compute_reward(
        make_outcomes(
            next_6h_map_delta=1_000.0,
            next_6h_lactate_delta=-1_000.0,
        )
    )
    extreme_worsening = compute_reward(
        make_outcomes(
            next_6h_map_delta=-1_000.0,
            next_6h_lactate_delta=1_000.0,
        )
    )

    assert extreme_improvement == pytest.approx(1.6)
    assert extreme_worsening == pytest.approx(-1.6)


def test_reward_is_a_finite_float():
    reward = compute_reward(make_outcomes(next_6h_map_delta=2.5))

    assert isinstance(reward, float)
    assert math.isfinite(reward)
