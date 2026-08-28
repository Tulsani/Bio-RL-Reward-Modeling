"""Logged-trajectory environment for offline policy development."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reward import compute_reward

ACTION_NAMES = ("maintain", "iv_fluids", "escalate_vasopressor")
REWARD_COLUMNS = (
    "next_6h_map_delta",
    "next_6h_lactate_delta",
    "next_6h_deterioration",
    "adverse_hypotension_next_6h",
    "adverse_fluid_overload_next_6h",
    "adverse_tachyarrhythmia_next_6h",
)


class OfflineClinicalEnv:
    """Replay observed patient trajectories without inventing counterfactuals.

    ``step_logged`` exposes only the action, reward, and next observation that
    were recorded in the dataset. It intentionally does not accept a proposed
    action, because outcomes for unlogged actions are unknown.
    """

    def __init__(
        self,
        trajectories: pd.DataFrame | str | Path,
        observation_columns: Sequence[str] | str | Path,
        reward_fn: Callable[[pd.Series], float] = compute_reward,
    ) -> None:
        self.trajectories = self._load_trajectories(trajectories)
        self.observation_columns = self._load_observation_columns(
            observation_columns
        )
        self.reward_fn = reward_fn

        self._validate_data()
        self.trajectories = self.trajectories.sort_values(
            ["patient_id", "time_step"], kind="stable"
        ).reset_index(drop=True)
        self._episodes = {
            patient_id: episode.reset_index(drop=True)
            for patient_id, episode in self.trajectories.groupby(
                "patient_id", sort=False
            )
        }
        self._patient_ids = tuple(self._episodes)
        self._rng = np.random.default_rng()
        self._episode: pd.DataFrame | None = None
        self._position: int | None = None
        self._terminated = False

    def reset(
        self, patient_id: Any | None = None, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Select an episode and return its first pre-action observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if patient_id is None:
            patient_id = self._python_value(
                self._rng.choice(self._patient_ids)
            )
        if patient_id not in self._episodes:
            raise ValueError(f"Unknown patient_id: {patient_id!r}")

        self._episode = self._episodes[patient_id]
        self._position = 0
        self._terminated = False

        metadata = {
            "patient_id": patient_id,
            "episode_length": len(self._episode),
            "first_time_step": self._python_value(
                self._episode.iloc[0]["time_step"]
            ),
            "handoff_note_available": "handoff_note"
            in self.trajectories.columns,
        }
        return self._observation(self._episode.iloc[0]), metadata

    def step_logged(
        self,
    ) -> tuple[
        dict[str, Any],
        str,
        float,
        dict[str, Any] | None,
        bool,
        dict[str, Any],
    ]:
        """Return and advance one factual, logged transition."""
        if self._episode is None or self._position is None:
            raise RuntimeError("Call reset() before step_logged().")
        if self._terminated:
            raise RuntimeError("Episode has terminated; call reset().")

        row = self._episode.iloc[self._position]
        observation = self._observation(row)
        logged_action = str(row["observed_clinician_action"])
        reward = self.reward_fn(row)
        terminated = bool(row["terminal"])

        if terminated:
            next_observation = None
            self._terminated = True
        else:
            next_position = self._position + 1
            next_observation = self._observation(
                self._episode.iloc[next_position]
            )
            self._position = next_position

        info = {
            "patient_id": self._python_value(row["patient_id"]),
            "time_step": self._python_value(row["time_step"]),
            "logged_transition": True,
        }
        return (
            observation,
            logged_action,
            reward,
            next_observation,
            terminated,
            info,
        )

    def _observation(self, row: pd.Series) -> dict[str, Any]:
        return {
            column: self._python_value(row[column])
            for column in self.observation_columns
        }

    @staticmethod
    def _python_value(value: Any) -> Any:
        """Convert pandas missing/scalar values into prompt-safe Python values."""
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _load_trajectories(
        trajectories: pd.DataFrame | str | Path,
    ) -> pd.DataFrame:
        if isinstance(trajectories, pd.DataFrame):
            return trajectories.copy()
        return pd.read_csv(trajectories)

    @staticmethod
    def _load_observation_columns(
        observation_columns: Sequence[str] | str | Path,
    ) -> tuple[str, ...]:
        if isinstance(observation_columns, (str, Path)):
            with Path(observation_columns).open(encoding="utf-8") as file:
                observation_columns = json.load(file)
        return tuple(observation_columns)

    def _validate_data(self) -> None:
        if self.trajectories.empty:
            raise ValueError("Trajectories cannot be empty.")
        if len(set(self.observation_columns)) != len(self.observation_columns):
            raise ValueError("Observation columns must be unique.")

        required = {
            "patient_id",
            "time_step",
            "observed_clinician_action",
            "terminal",
            *self.observation_columns,
            *REWARD_COLUMNS,
        }
        missing = sorted(required.difference(self.trajectories.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        duplicated = self.trajectories.duplicated(
            ["patient_id", "time_step"]
        )
        if duplicated.any():
            raise ValueError("Duplicate patient_id/time_step transitions found.")

        unknown_actions = set(
            self.trajectories["observed_clinician_action"].dropna().unique()
        ).difference(ACTION_NAMES)
        if unknown_actions:
            raise ValueError(f"Unknown logged actions: {sorted(unknown_actions)}")

        for patient_id, episode in self.trajectories.groupby("patient_id"):
            episode = episode.sort_values("time_step", kind="stable")
            terminal = episode["terminal"].astype(int)
            if terminal.sum() != 1 or terminal.iloc[-1] != 1:
                raise ValueError(
                    "Each patient must have exactly one terminal transition, "
                    f"at the end; invalid patient_id={patient_id!r}."
                )
