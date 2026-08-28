"""Minimal interfaces for the BioStack RL Engineer take-home.

You may change this file or replace it with your own structure.
"""
from __future__ import annotations

ACTION_NAMES = ["maintain", "iv_fluids", "escalate_vasopressor"]


class OfflineClinicalEnv:
    def __init__(self, trajectories, observation_columns):
        self.trajectories = trajectories
        self.observation_columns = observation_columns

    def reset(self, patient_id=None, seed=None):
        raise NotImplementedError

    def step_logged(self):
        raise NotImplementedError


class Policy:
    def predict_proba(self, observations):
        """Return an [N, 3] array in ACTION_NAMES order."""
        raise NotImplementedError


class AlwaysMaintainPolicy(Policy):
    """A supplied baseline. You do not need to improve this class."""
    def predict_proba(self, observations):
        import numpy as np
        n = len(observations)
        out = np.zeros((n, 3), dtype=float)
        out[:, 0] = 1.0
        return out
