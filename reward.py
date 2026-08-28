import numpy as np


def compute_reward(row):
    """
    Compute a scalar reward from the observed 6-hour post-action outcomes.

    Reward design
    -------------
    The objective balances short-term physiological improvement against
    deterioration and treatment-related adverse events.

   Reward design rationale:

        - MAP scaling: divide `next_6h_map_delta` by 5.0
        - Training data showed MAP changes on the order of a few mmHg:
            mean = 0.82, std = 3.46, IQR = [-1.53, 2.90].
        - A fixed 5 mmHg reference scale puts typical MAP changes into an
            approximately order-1 range without fitting the reward directly to the
            training-set standard deviation.
        - This keeps the reward interpretable and less dataset-specific.

        - Lactate scaling: divide `next_6h_lactate_delta` by 0.5
        - Lactate changes were much smaller numerically:
            mean = -0.05, std = 0.255, IQR = [-0.21, 0.11].
        - Using 0.5 as a fixed scale makes meaningful lactate movement comparable
            in magnitude to the MAP component.
        - The lactate term enters with a negative sign because a decrease in
            lactate should increase reward.

        - Continuous-value clipping: clip both normalized terms to [-2, 2]
        - MAP ranged from -11.61 to +16.09 and lactate from -1.14 to +0.94.
        - Without clipping, rare extreme changes could dominate the entire reward.
        - Clipping bounds each physiological contribution and ensures that a large
            MAP/lactate movement cannot trivially compensate for deterioration or an
            adverse event.

        - MAP weight: +0.4
        - MAP improvement is desirable but is treated as an incremental
            physiological benefit rather than the primary objective.
        - A normalized MAP change therefore contributes at most +/-0.8 after
            clipping.
        - This prevents MAP response alone from dominating safety outcomes.

        - Lactate weight: -0.4
        - Lactate reduction is also treated as an incremental physiological benefit.
        - MAP and lactate changes were substantially correlated (r ≈ -0.64), so
            assigning both large independent weights would risk double-counting the
            same underlying physiological improvement.
        - Equal 0.4 weights give the two signals similar influence while keeping
            their combined effect bounded.

        - Deterioration penalty: -2.0
        - Deterioration occurred in approximately 9.5% of transitions.
        - It represents the broadest undesirable 6-hour outcome in the supplied
            reward variables, so it receives the largest single penalty.
        - The penalty is intentionally larger than the maximum contribution of
            either individual physiological term, ensuring modest MAP/lactate
            improvement cannot outweigh overall deterioration.

        - Hypotension penalty: -1.5
        - Hypotension was rare (~0.19%) but represents a distinct adverse outcome.
        - A substantial fixed penalty ensures that its rarity does not make it
            irrelevant to the reward.

        - Fluid-overload penalty: -1.5
        - Fluid overload occurred in ~1.6% of transitions and was much more common
            following IV fluids (~7.7%).
        - Penalizing it explicitly captures the observed benefit-risk tradeoff:
            fluids often improved MAP/lactate but carried higher fluid-overload risk.

        - Tachyarrhythmia penalty: -1.5
        - Tachyarrhythmia occurred in ~0.87% of transitions and was much more common
            following vasopressor escalation (~8.0%).
        - Penalizing it explicitly prevents the strong average MAP/lactate
            improvement associated with vasopressors from automatically producing a
            high reward.

        - Equal adverse-event penalties: -1.5 each
        - The dataset provides event indicators but does not provide evidence for a
            principled severity ranking among hypotension, fluid overload, and
            tachyarrhythmia.
        - Using equal penalties avoids introducing an unsupported ordering between
            these adverse events.

        Overall preference encoded by the reward:

            avoid deterioration
                > avoid treatment-related adverse events
                > reward incremental MAP/lactate improvement
        

    """
    map_component = np.clip(
        row["next_6h_map_delta"] / 5.0,
        -2.0,
        2.0,
    )

    lactate_component = np.clip(
        row["next_6h_lactate_delta"] / 0.5,
        -2.0,
        2.0,
    )

    reward = (
        0.4 * map_component
        - 0.4 * lactate_component
        - 2.0 * row["next_6h_deterioration"]
        - 1.5 * row["adverse_hypotension_next_6h"]
        - 1.5 * row["adverse_fluid_overload_next_6h"]
        - 1.5 * row["adverse_tachyarrhythmia_next_6h"]
    )

    return float(reward)