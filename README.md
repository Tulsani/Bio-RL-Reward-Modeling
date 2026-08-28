# BioStack RL Engineer Take-Home Data

This package accompanies `ASSIGNMENT.md`.

## Files

- `trajectories_train.csv` — logged training transitions
- `trajectories_validation.csv` — logged validation transitions
- `test_observations.csv` — blinded pre-action test states
- `data_dictionary.csv` — column definitions
- `observation_columns.json` — columns allowed as policy observations
- `action_names.json` — canonical action order
- `starter.py` — minimal interfaces and supplied always-maintain baseline

## Important semantics

Each row is one six-hour decision point. Training/validation contain the action that was logged and several subsequent six-hour synthetic outcomes. The test file contains only pre-action information.

There is **no supplied scalar reward**. Define your own reward using the post-action outcome fields as described in the assignment. The dataset contains no factual counterfactual outcome for actions different from the logged action.

The data are synthetic and not clinically validated.
