# BioStack Offline RL: Project Design Notes

> **Status:** Working engineering document. This is intentionally more detailed than the final submission `README.md`, which must remain at or below 750 words. The dataset and conclusions are synthetic and are not clinically validated.

## 1. Project Objective

The project learns and evaluates an LLM policy over fixed, logged ICU-style trajectories. At each six-hour decision point, the policy observes a pre-action state and returns a probability distribution over:

1. `maintain`
2. `iv_fluids`
3. `escalate_vasopressor`

A logged transition is

$$
(s_t, a_t, r_t, s_{t+1}, d_t),
$$

where $s_t$ is the allowed pre-action observation, $a_t$ is the action actually logged, $r_t$ is our reward computed from that action's observed outcomes, and $d_t$ marks termination.

The most important constraint is that the data contains no factual counterfactual. If $a_t=\texttt{iv\_fluids}$, the observed reward belongs only to that logged action. We cannot reuse it as the outcome of `maintain` or `escalate_vasopressor`.

## 2. Dependency and Environment Management

The repository uses [uv](https://docs.astral.sh/uv/) with Python 3.11 or newer.

Tracked dependency files:

- `pyproject.toml`: direct runtime and development dependencies.
- `uv.lock`: exact resolved dependency graph for reproducibility.
- `.python-version`: pinned interpreter version when created with `uv python pin`.

The local `.venv/` directory is generated and must not be committed.

### Common commands

```bash
# Create/update the locked environment
uv sync

# Run all tests
uv run pytest -q

# Run a focused test file
uv run pytest tests/test_reward.py -q

# Lint the project
uv run ruff check .

# Add runtime or development dependencies
uv add <package>
uv add --dev <package>
```

Current runtime dependencies include NumPy, pandas, Pydantic, scikit-learn, the official `mistralai` SDK, and `python-dotenv`. Pytest and Ruff are development dependencies. The completed repository must still work from cached or mock responses without API credentials.

## 3. Reward Design (`reward.py`)

### Definition

Let

$$
m_t=\operatorname{clip}\left(\frac{\Delta MAP_t}{5},-2,2\right)
$$

and

$$
\ell_t=\operatorname{clip}\left(\frac{\Delta lactate_t}{0.5},-2,2\right).
$$

The primary reward is

$$
\boxed{
r_t=0.4m_t-0.4\ell_t-2D_t-1.5H_t-1.5F_t-1.5T_t
}
$$

where:

- $D_t$: next-six-hour deterioration;
- $H_t$: adverse hypotension;
- $F_t$: adverse fluid overload;
- $T_t$: adverse tachyarrhythmia.

### Core ideology

The reward expresses the ordering

$$
\text{avoid deterioration}
\;>\;
\text{avoid specific adverse events}
\;>\;
\text{reward incremental physiologic improvement}.
$$

MAP improvement is beneficial, while lactate increase is undesirable. Each normalized physiological term is clipped so its contribution lies in $[-0.8,0.8]$. Therefore, rare extreme measurements cannot grow without bound or trivially erase a safety penalty.

Deterioration receives the largest single penalty because it is the broadest supplied marker of a bad six-hour outcome. The three specific adverse events receive equal penalties because the synthetic data does not provide a principled severity ordering among them.

We intentionally do not penalize an action merely because it is an intervention. An action cost would encode a preferred treatment policy before observing its modeled benefit-risk tradeoff.

### Reward statistics used to choose scales

The following summaries come from the 25,200 training transitions only:

| Outcome | Mean | Std. dev. | Min | 25% | Median | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `next_6h_map_delta` | 0.818 | 3.464 | -11.61 | -1.53 | 0.59 | 2.90 | 16.09 |
| `next_6h_lactate_delta` | -0.049 | 0.255 | -1.14 | -0.21 | -0.02 | 0.11 | 0.94 |

| Binary outcome | Event rate |
| --- | ---: |
| Deterioration | 9.544% |
| Adverse hypotension | 0.190% |
| Fluid overload | 1.615% |
| Tachyarrhythmia | 0.873% |

The implemented reward has the following empirical training distribution:

| Statistic | Reward |
| --- | ---: |
| Mean | -0.127 |
| Median | 0.022 |
| Standard deviation | 0.791 |
| Minimum | -5.864 |
| Maximum | 1.600 |

### Descriptive action differences

| Logged action | Mean $\Delta MAP$ | Mean $\Delta$ lactate | Deterioration | Hypotension | Fluid overload | Tachyarrhythmia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `maintain` | -0.023 | -0.012 | 9.4% | 0.2% | 0.7% | 0.6% |
| `iv_fluids` | 4.379 | -0.209 | 10.2% | 0.2% | 7.7% | 0.9% |
| `escalate_vasopressor` | 7.184 | -0.326 | 10.1% | 0.1% | 0.8% | 8.0% |

These are descriptive associations, not causal treatment effects. The actions were selected in different patient states. For example, the mean starting lactate was 1.47 for `maintain`, 1.71 for `iv_fluids`, and 1.82 for `escalate_vasopressor`. Mean logged rewards were -0.202, 0.178, and 0.476 respectively, but a higher average logged reward for an action does not prove that action would improve every patient.

The low-MAP slice reinforces this distinction. When `map_mm_hg < 65`, deterioration rates were 17.4% for logged `maintain`, 13.6% for `iv_fluids`, and 11.8% for `escalate_vasopressor`; these remain observational comparisons affected by action selection and support.

### Known reward risks

1. Deterioration and a specific adverse event can overlap, producing two penalties.
2. Any MAP increase is rewarded, even when the starting MAP is already high; this is not a true target-range/homeostasis objective.
3. The weights are engineering judgments, not clinically validated utilities.
4. Rare adverse events may be difficult for a learned critic to estimate reliably.
5. The reward is known only for the logged action.

A materially different Reward B will later test whether policy rankings depend on these choices.

## 4. Logged Environment (`env.py`)

`OfflineClinicalEnv` is a replay interface over factual patient trajectories. It is not a simulator.

### Episode storage

At initialization, the environment:

1. loads a DataFrame or CSV;
2. loads the allowed structured fields from `observation_columns.json`;
3. validates required columns, action names, unique `(patient_id, time_step)` keys, and terminal placement;
4. sorts rows by `(patient_id, time_step)`;
5. stores one ordered DataFrame per patient in an internal episode dictionary.

Each patient currently has 12 six-hour transitions. Missing observation values are converted from pandas/NumPy missing values to Python `None`, making later prompt serialization explicit and deterministic.

### Reset

```python
observation, metadata = env.reset(patient_id=None, seed=None)
```

`reset` selects a requested patient or reproducibly samples one using the seed. It returns the first pre-action observation and episode metadata. Identifiers and time are metadata, not policy features.

The observation contains exactly the structured columns allowed by `observation_columns.json`:

$$
s_t = \{x_{t,j}: j\in\mathcal O_{allowed}\}.
$$

Post-action outcomes, the logged action, reward, and terminal flag are excluded from $s_t$.

### Logged transition step

```python
(
    observation,
    logged_action,
    reward,
    next_observation,
    terminated,
    info,
) = env.step_logged()
```

For the current row, this returns

$$
(s_t,a_t,r(s_t,a_t),s_{t+1},d_t).
$$

At the terminal transition, `next_observation` is `None`; stepping again raises an error until reset. The environment deliberately has no `step(proposed_action)` method. Any future counterfactual value must be labeled as a model estimate, such as $\hat Q(s,a)$, rather than an observed transition.

The reward function is injected into the constructor, allowing Reward A and Reward B to reuse the same environment without changing transition logic.

### Missing handoff note

The assignment describes a synthetic handoff note, but no note column or separate note file is present in the supplied repository. The environment reports `handoff_note_available=False` and does not invent note text. We must obtain the missing data or clearly label any candidate-created robustness notes.

## 5. Zero-Shot Prompt (`prompts.py`)

The zero-shot prompt is versioned as `zero_shot_v1`. Its structured input schema and action order are loaded from `observation_columns.json` and `action_names.json`, keeping the prompt consistent with the environment.

The prompt builder requires exactly the allowed observation fields. Missing fields and unexpected metadata, actions, or outcome fields raise an error rather than being silently included. Fields are serialized in canonical order, and missing measurements use the explicit token `NOT_OBSERVED`.

The handoff note is passed separately as JSON-encoded, untrusted context. Delimiter-like text is escaped, and the system message instructs the policy to ignore commands embedded in the note. Structured measurements take priority when note text conflicts with them. Because the supplied dataset has no note column, missing notes are represented honestly as `HANDOFF_NOTE_NOT_PROVIDED`.

The model is asked to return only three finite, non-negative action probabilities that sum to one and a state-grounded rationale of at most 40 words. Parsing, normalization, schema validation, fallback, and caching are implemented in `llm_client.py`.

## 6. LLM Client (`llm_client.py`)

The client is provider-agnostic: a hosted or local adapter needs to implement only the `CompletionBackend` protocol. This keeps model-provider code separate from policy correctness and allows cached execution without credentials.

`PolicyDecision` uses a strict Pydantic schema, forbids extra fields, checks finite values and a positive probability sum, normalizes the three values, and limits the rationale to 40 words. Any malformed output uses the deterministic supplied baseline $(1,0,0)$ in canonical action order.

The SHA-256 cache key covers the model ID, prompt version, temperature, complete messages, schema/cache format version, and therefore every current input that can change deterministic output. Raw model responses, including malformed responses, are written atomically to `cache/llm_outputs/`; parsing is repeated from cache so structured-output failures remain reproducible.

Each call reports cache hits, fallback use, structured-output failure, cache failure, and failure reason. These diagnostics will support the required evaluation failure and fallback rates. Without a configured backend, a cache hit works normally and a cache miss returns the deterministic fallback rather than making an external call.

### Hosted Mistral backend

`mistral_backend.py` implements `CompletionBackend` with the official Mistral SDK. The planned zero-shot model is the immutable `mistral-small-2603` identifier rather than a moving `-latest` alias. Configuration is stored in `config/model_config.json`: temperature 0, random seed 42, maximum 256 output tokens, and Pydantic JSON-schema output. Mistral lists this model as supporting structured outputs and batch inference ([model documentation](https://docs.mistral.ai/models/mistral-small-4-0-26-03)).

The local credential is `MISTRAL_API_KEY` in the Git-ignored `.env`; `.env.example` contains only the variable name. OCR and chat are different endpoints and models. An API key previously used for Mistral OCR can be reused only if that account/key also has access to the chat model. No credential is stored in configuration, cache keys, prompts, or output files.

Before any full run, we will make a 10--20-state smoke test and inspect structured-output failures, entropy, rationales, latency, and cache reuse. We will not launch thousands of requests without reviewing this sample and estimating cost.

#### Initial hosted smoke test

On 12 deterministically selected training states (low MAP, MAP near 65, high or missing lactate, active vasopressor, and recent fluids), `mistral-small-2603` produced 12 schema-valid responses with no fallbacks. A repeat run produced 12/12 cache hits and no new hosted calls.

The selected set is intentionally difficult and is not representative, so its action distribution is not a policy-value result: six states favored `iv_fluids`, four favored `escalate_vasopressor`, and two favored `maintain`. The sample exposed several useful zero-shot weaknesses: coarse repeated probability patterns, aggressive escalation in some states with MAP near 71, diagnostic language not explicitly present in the state, and one mismatch between the highest-probability action and the rationale. We will preserve these findings rather than tune the zero-shot baseline on validation data.

## 7. Policies (`policy.py`)

All policies implement `predict_proba(observations)` and return an $[N,3]$ NumPy array in canonical action order. `AlwaysMaintainPolicy` reproduces the supplied baseline $(1,0,0)$ for every observation.

`ZeroShotLLMPolicy` separates the optional `handoff_note` from the structured state, constructs versioned messages with `prompts.py`, obtains validated decisions through `LLMClient`, and returns only the probabilities. Unexpected metadata or post-action fields fail before a model call. LLM rationales remain available in auditable call results but are not returned in prediction arrays or future test CSV files.

The policy accumulates cache-hit, structured-output-failure, fallback, and failure-reason diagnostics across calls and supports explicitly resetting those metrics. It also rejects a client configured with a prompt version other than `zero_shot_v1`, preventing cache and policy-version drift.

## 8. Behavior Policy Model (`behavior_model.py`)

The behavior model estimates the logged action propensity $\hat b(a\mid s)$; it does not recommend an optimal action. It uses only the 19 allowed pre-action observation fields. Numeric values receive median imputation and standardization, categorical values receive most-frequent imputation and one-hot encoding, and multinomial logistic regression produces probabilities in canonical action order.

Class weighting is intentionally disabled. Rebalancing rare actions would distort the natural behavior probabilities required for support diagnostics and importance ratios. Model diagnostics use five stratified patient-grouped folds, so no patient's rows appear in both fit and held-out portions of a fold.

Training-only out-of-fold results are:

| Metric | Value |
| --- | ---: |
| Log loss | 0.536 |
| Multiclass Brier score | 0.288 |
| Accuracy | 83.1% |
| Macro-F1 | 0.303 |
| Top-label ECE | 0.0025 |

Mean predicted probabilities closely match the observed action rates: 83.09% vs 83.09% for `maintain`, 13.45% vs 13.45% for `iv_fluids`, and 3.47% vs 3.46% for escalation. However, the argmax action is `maintain` for every row, showing why accuracy and aggregate calibration are insufficient diagnostics for this imbalanced propensity problem.

Per-action diagnostics expose the rare-action support more clearly:

| Logged action | Count | Mean logged propensity | Median logged propensity | 5th percentile |
| --- | ---: | ---: | ---: | ---: |
| `maintain` | 20,939 | 0.8322 | 0.8364 | 0.7758 |
| `iv_fluids` | 3,389 | 0.1405 | 0.1341 | 0.0964 |
| `escalate_vasopressor` | 872 | 0.0400 | 0.0326 | 0.0220 |

The logged-action propensity across all rows has minimum 0.014, 1st percentile 0.028, and 5th percentile 0.107. This matters because doubly robust evaluation uses the importance ratio

$$
w_i = \frac{\pi(a_i\mid s_i)}{\hat b(a_i\mid s_i)}.
$$

A denominator of 0.014 can produce a ratio near 70 if the target policy assigns that logged action probability one. Even a target vasopressor probability of 0.30 can produce ratios around 9--14 for many logged vasopressor decisions.

Conditional calibration is also weaker than the aggregate numbers suggest. In the `map_mm_hg < 65` slice, the model predicts 76.0% maintain, 18.8% fluids, and 5.1% escalation, compared with observed rates of 67.9%, 25.5%, and 6.6%. Therefore, the low ECE and close agreement in overall action frequencies must not be interpreted as proof that state-conditional propensities are accurate.

We treat this model as a useful first behavior-policy baseline, but not as safe for unrestricted DR estimation. Before reporting DR results, evaluation must include:

- importance-ratio clipping, with at least $w_{\max}=10$ and $20$ as sensitivity settings;
- effective sample size, maximum weight, and the fraction of weights clipped;
- weight and estimate diagnostics broken down by action, especially vasopressor escalation;
- a comparison against a nonlinear behavior-model specification; and
- patient-level bootstrap intervals and explicit discussion of remaining overlap limitations.

The fitted artifact is regenerable and Git-ignored at `artifacts/behavior_policy.joblib`. Machine-readable out-of-fold metrics are stored in `outputs/behavior_model_metrics.json`.

## 9. Action-Conditioned Reward Model (`value_model.py`)

The first value-model baseline estimates the one-step conditional reward

$$
\hat m(s,a) \approx \mathbb{E}[r_t\mid s_t=s,a_t=a].
$$

It is deliberately not described as a full $Q^\pi$: it does not yet include discounted future rewards or bootstrap through a target policy. `ActionRewardModel` is a T-learner with one ridge-regression pipeline per action, allowing state coefficients to differ across treatments. Each pipeline uses only the 19 pre-action observation fields, with median imputation and standardization for numeric variables and most-frequent imputation plus one-hot encoding for categorical variables.

Five-fold patient-grouped cross-validation evaluates only factual logged-action predictions. Training-only results are:

| Slice | Rows | RMSE | MAE | $R^2$ |
| --- | ---: | ---: | ---: | ---: |
| Overall | 25,200 | 0.764 | 0.518 | 0.068 |
| `maintain` | 20,939 | 0.734 | 0.494 | 0.017 |
| `iv_fluids` | 3,389 | 0.902 | 0.637 | 0.047 |
| `escalate_vasopressor` | 872 | 0.862 | 0.637 | -0.018 |
| MAP below 65 | 1,536 | 1.052 | 0.713 | 0.076 |

The predicted overall reward mean is -0.128 versus an observed mean of -0.127, but matching a marginal mean is not sufficient. The low within-action $R^2$ values show that this linear model learns some average action differences while explaining little patient-level reward variation. In particular, the negative escalation $R^2$ means it performs worse than the action-specific mean for those factual rows. We therefore retain it as an interpretable benchmark, not as the final critic for contrastive-example selection.

`predict_values` returns modeled rewards for all actions in canonical order, while `predict_logged_values` selects only the factual action prediction. Values for unlogged actions are explicitly counterfactual model estimates. `predict_supported_values` masks any action whose estimated behavior propensity is below a configured threshold (currently 0.02); passing this threshold does not make the estimate causal or guarantee adequate overlap.

A regularized histogram gradient-boosting T-learner was compared with ridge using identical patient-grouped folds, features, and targets:

| Slice | Ridge RMSE | Nonlinear RMSE | Ridge $R^2$ | Nonlinear $R^2$ |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.764 | 0.767 | 0.068 | 0.061 |
| `maintain` | 0.734 | 0.732 | 0.017 | 0.023 |
| `iv_fluids` | 0.902 | 0.918 | 0.047 | 0.014 |
| `escalate_vasopressor` | 0.862 | 0.918 | -0.018 | -0.155 |
| MAP below 65 | 1.052 | 1.020 | 0.076 | 0.131 |

The nonlinear model improves the low-MAP slice and slightly improves `maintain`, but it is worse overall and degrades both intervention actions, especially the rare vasopressor action. Ridge therefore remains the selected artifact; gradient boosting is retained as a sensitivity comparison. Neither model is strong enough to justify unconstrained counterfactual action rankings. Contrastive-example construction must require behavior support and should later test whether model rankings agree across specifications. Machine-readable diagnostics are stored in `outputs/value_model_metrics.json`; the regenerable fitted artifact is Git-ignored at `artifacts/action_reward_model.joblib`.

## 10. Contrastive Candidate Generation (`contrastive_examples.py`)

Contrastive candidates are constructed entirely from the training split. Five patient-grouped folds generate out-of-fold behavior probabilities and all-action predictions from both ridge and histogram gradient boosting. For a state to qualify, it must have at least two supported actions, both reward models must rank the same globally preferred action first, both model-specific advantages over the strongest supported alternative must exceed 0.10, and the two models' predictions for the compared actions must differ by at most 0.50.

The first unconstrained audit exposed critic-driven intervention bias: 14,546 rows cleared model agreement and margin filters, but after per-action caps the library contained 100 vasopressor, 69 fluid, and zero maintain preferences. Only 8.9% of those preferences matched the factual logged action. This is consistent with the reward models relying heavily on marginal action differences and is not safe evidence for policy improvement.

The default selector therefore also requires the preferred action to equal the factual logged action and requires its observed training reward to be non-negative. Raw outcomes and realized rewards are selection-only and are never exported into example observations or future prompts. With these safeguards, 415 candidates survive before capping:

| Preferred action | Candidates before cap | Selected after cap |
| --- | ---: | ---: |
| `maintain` | 0 | 0 |
| `iv_fluids` | 10 | 10 |
| `escalate_vasopressor` | 405 | 100 |

The selected candidates all match factual actions, their median preferred-action propensity is 0.040, and 9.1% have MAP below 65 versus 6.1% in the source data. However, the absence of maintain candidates and severe action imbalance trigger an explicit readiness failure. `policy_improvement_readiness.approved` is therefore `false`, and this file must not yet be used as an LLM demonstration library. This negative result is preserved in `outputs/contrastive_examples_metrics.json` rather than hidden by weakening thresholds or artificially relabeling actions.

### Maintain-bias diagnosis

`scripts/analyze_action_bias.py` tests whether the missing maintain examples are caused by a narrow selection threshold or by the reward models themselves. It uses the same training-only cross-fitted estimates and reports global rankings, rankings on factual maintain rows, intervention-minus-maintain margins, proposed-action support, and MAP/lactate/vasopressor slices.

The result rejects the current critic as a source of policy labels:

| Diagnostic | Ridge | Nonlinear |
| --- | ---: | ---: |
| Maintain ranked first, all states | 0.00% | 0.58% |
| Maintain ranked first, logged-maintain states | 0.00% | 0.60% |
| Maintain ranked first on logged-maintain states with MAP at least 75 | 0.00% | 0.49% |
| Median intervention-minus-maintain margin on logged-maintain states | 0.673 | 0.717 |
| Median propensity of proposed top action on logged-maintain states | 0.031 | 0.033 |

The two models agree on the top action for 79.4% of states, but 98.2% of those agreed actions are vasopressor escalation and none are maintain. Their agreement therefore reflects a shared action-level bias rather than independent corroboration. Their mean modeled values closely reproduce the marginal action reward ordering—approximately -0.20 for maintain, +0.16 for fluids, and +0.47 to +0.48 for escalation—while their low factual within-action $R^2$ shows little ability to personalize those differences.

This failure persists in apparently stable slices and is not repaired by support filtering: the median behavior propensity of the proposed action on factual maintain rows is only about 0.03. We will retain these diagnostics in `outputs/action_bias_diagnostics.json`, reject critic-derived optimal-action demonstrations, and move to factual state-matched outcome context. No validation data were used for this decision.

## 11. Factual State-Matched Outcome Context (`factual_examples.py`)

The replacement example library does not assign optimal actions or invent counterfactual outcomes. It partitions factual logged transitions into favorable and unfavorable outcomes relative to the reward distribution within each action:

$$
D_a^+ = \{(s_i,a_i): a_i=a,\ r_i \ge q_{0.75}(r\mid a)\},
\qquad
D_a^- = \{(s_i,a_i): a_i=a,\ r_i \le q_{0.25}(r\mid a)\}.
$$

The action-specific thresholds are:

| Logged action | Unfavorable threshold | Favorable threshold |
| --- | ---: | ---: |
| `maintain` | reward at most -0.349 | reward at least 0.226 |
| `iv_fluids` | reward at most 0.068 | reward at least 0.725 |
| `escalate_vasopressor` | reward at most 0.325 | reward at least 1.036 |

These labels mean better or worse **within the same logged action**. A favorable maintain outcome is not asserted to be counterfactually better than fluids or escalation. Each row must have cross-fitted logged-action propensity of at least 0.02, and each action/outcome cell is capped at one row per source patient before taking its 100 highest-priority examples.

The resulting library contains 600 rows from 517 unique patients, with exactly 100 factual examples in each of the six action/outcome cells. Logged-action propensity ranges from 0.021 to 0.900 with median 0.149, so the prior zero-maintain failure is removed without changing labels. Coverage readiness passes, but this only means all six retrieval cells are populated; it does not validate policy improvement.

`FactualExampleRetriever` standardizes and median-imputes numeric observations, one-hot encodes categorical observations, and retrieves the nearest examples independently from every action/outcome cell. Patient exclusion is supported for training-time diagnostics. Before prompt construction, `prompt_safe_factual_records` strips reward values, propensities, retrieval distances, source IDs, fold IDs, and other provenance, leaving only allowed pre-action state, factual logged action, and the relative outcome label.

Equal cell sampling intentionally does not represent action prevalence. The library contains 21.5% MAP-below-65 rows versus 6.1% in the source data, largely because intervention cases are oversampled. The improved prompt must describe the examples as balanced evidence, never as empirical action frequencies, and the final policy must be checked for intervention-rate shifts. This library and its diagnostics are stored in `outputs/factual_outcome_examples.csv` and `outputs/factual_outcome_examples_metrics.json`. No validation data were used.

## 12. Tests and Statistical Checks

Tests protect deterministic invariants; exploratory scripts report dataset statistics. Unit tests should not merely print descriptive tables.

### Implemented tests

`tests/test_reward.py` checks:

- neutral reward;
- positive and negative physiological direction;
- each adverse-event penalty;
- clipping at extreme values;
- finite float output.

`tests/test_env.py` checks:

- chronological episode ordering;
- terminal transition handling;
- absence of post-action observation leakage;
- missing-value conversion;
- reproducible seeded episode sampling;
- failure when stepping before reset.

`tests/test_prompts.py` checks:

- deterministic prompt construction and versioning;
- canonical observation and action order;
- explicit missing measurements and handoff notes;
- note-delimiter escaping and untrusted-note instructions;
- rejection of missing, metadata, and post-action fields;
- absence of hidden outcome names from the rendered prompt;
- the fixed probability response schema.

`tests/test_llm_client.py` checks:

- schema validation and probability normalization;
- deterministic fallback for malformed, empty, and extra-field outputs;
- backend-error and cache-miss fallback;
- cache creation and reuse without a backend;
- cache-key sensitivity to the model, prompt, and messages;
- rejection of invalid message structures before a model call.

`tests/test_policy.py` checks:

- canonical always-maintain baseline output;
- zero-shot matrix shape and normalization;
- separation of handoff notes from structured state;
- rejection of post-action fields before a model call;
- fallback and cache-hit diagnostics;
- empty input batches and prompt-version mismatch.

`tests/test_mistral_backend.py` checks:

- Pydantic structured-output configuration;
- immutable model ID and deterministic generation settings;
- missing credential errors without exposing a key;
- empty choices and invalid response content;
- generation-limit validation using a fake client without network calls.

`tests/test_behavior_model.py` checks:

- canonical action-probability ordering and normalization;
- missing values and unseen categorical levels;
- strict training-only fitting;
- unweighted natural class probabilities;
- patient-disjoint folds;
- support and calibration diagnostic structure;
- exclusion of post-action fields from model features.

`tests/test_value_model.py` checks:

- canonical action-value ordering and factual logged-action selection;
- missing values and unseen categorical levels;
- strict training-only fitting and patient-disjoint folds;
- factual overall, per-action, and low-MAP diagnostics;
- behavior-propensity support masking; and
- exclusion of post-action fields from model features.

`tests/test_contrastive_examples.py` checks:

- complete cross-fitted estimates for every row and action;
- support, agreement, advantage, and factual-action filters;
- deterministic per-action caps and readiness failure;
- exclusion of raw post-action outcomes from exported examples; and
- strict training-only generation; and
- maintain/intervention ranking-bias diagnostics.

`tests/test_factual_examples.py` checks:

- complete patient-grouped cross-fitted behavior context;
- equal coverage of every factual action/outcome cell;
- behavior-support filtering and readiness failure for missing cells;
- state-matched retrieval with source-patient exclusion;
- removal of rewards, post-action outcomes, support metadata, and provenance before prompting; and
- strict training-only generation.

Current status: 91 tests pass.

### Tests to add later

- LLM schema validation and probability normalization;
- deterministic fallback for malformed LLM output;
- cache-key sensitivity to model, prompt, and observation versions;
- train/validation patient separation;
- behavior/value models fitted on training only;
- fixed canonical action order throughout the pipeline;
- patient-level, rather than row-level, bootstrap resampling;
- estimator tests on a small problem with known values.

### Statistics and diagnostics to add later

- reward distribution by split, action, hospital, and important state slices;
- behavior-policy probabilities and effective action support;
- action-critic cross-validation errors, including errors by action;
- calibration, entropy, and structured-output fallback rates;
- direct-method and doubly robust estimates with patient bootstrap intervals;
- policy-value differences with paired patient bootstrap intervals;
- zero-shot/improved agreement and disagreement patterns;
- robustness flip rates and alternate-reward policy rankings.

## 13. Working Policy-Improvement Hypothesis

> This section is a hypothesis to test, not a claim that the method is already validated.

### Motivation

A zero-shot LLM supplies a useful prior distribution $\pi_0(a\mid s)$, but its self-reported probabilities need not be calibrated and it has not learned the reward tradeoffs in this synthetic dataset. We hypothesize that judgment can improve by showing it **supported contrastive decisions**: a preferred action and a contextually plausible near-negative action for the same state.

Two distillation ideas provide useful motivation without defining our algorithm:

1. **Knowledge distillation.** Hinton, Vinyals, and Dean transfer information through softened teacher distributions rather than only hard labels. Their temperature-scaled objective exposes relative class preferences ([paper](https://arxiv.org/abs/1503.02531)).
2. **DeepSeek-R1 distillation.** The DeepSeek-R1 report describes directly fine-tuning smaller Qwen/Llama models on curated reasoning samples generated by R1. This is sequence-level supervised imitation of teacher outputs, not classic access to the teacher's full logits ([paper](https://arxiv.org/abs/2501.12948)).

Our current proposal is simpler: **critic-guided contrastive in-context learning**. We will retrieve training-only examples that contrast two plausible actions and place them in the improved policy's prompt. We are not currently committing to model-weight fine-tuning.

DAgger was considered but is not the right name for this method. DAgger repeatedly visits states induced by the current learner, queries an expert there, aggregates those labels, and retrains the policy ([Ross, Gordon, and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html)). We cannot execute the LLM policy on patients, observe its induced counterfactual states, or query a factual expert for them. Consequently, we do not claim DAgger's algorithm or guarantees.

### Training-only supporting models

We plan to fit two models using only training trajectories:

1. A behavior classifier

   $$
   \hat b(a\mid s)=P(A=a\mid S=s),
   $$

   used to identify whether an action is sufficiently represented in a state.

2. An action critic for the zero-shot policy

   $$
   \hat Q^{\pi_0}(s,a),
   $$

   trained with fitted-Q-style targets

   $$
   y_t=r_t+\gamma(1-d_t)
   \sum_{a'}\pi_0(a'\mid s_{t+1})
   \hat Q^{\pi_0}(s_{t+1},a').
   $$

The critic predicts counterfactual values; it does not turn them into factual outcomes. Estimates for rare or unsupported actions must be guarded by $\hat b(a\mid s)$ and uncertainty diagnostics.

### Constructing positive and near-negative decisions

For a training state, define the supported action set

$$
\mathcal A_{supp}(s)=\{a:\hat b(a\mid s)\geq\epsilon\}.
$$

Within this set, the critic proposes

$$
a^+=\arg\max_{a\in\mathcal A_{supp}(s)}\hat Q(s,a),
$$

and $a^-$ is the best-scoring alternative action. This makes $a^-$ a hard or near-negative rather than an obviously unrelated choice. We should retain a pair only when an ensemble or bootstrap indicates that the value ordering is sufficiently reliable; ambiguous pairs should remain soft targets or be skipped.

A contrastive example would contain:

- the allowed state and available handoff note;
- preferred action $a^+$;
- plausible rejected action $a^-$;
- a concise state-grounded explanation of the tradeoff;
- no hidden outcome fields in the inference-time prompt.

### Contrastive in-context improvement

The initial improved policy will use a fixed, reproducible training-only example library:

1. Run $\pi_0$ on existing training states.
2. Select high-entropy states, critic-policy disagreements, and important safety slices.
3. Construct reliable, supported positive/near-negative pairs using the critic.
4. Store state-grounded contrastive examples without future outcome columns or numeric rewards.
5. For a new observation, retrieve a small fixed number of similar examples using pre-action state only.
6. Add those examples to a versioned prompt and ask the same LLM for a probability distribution.

The improved policy can be written abstractly as

$$
\pi_1(\cdot\mid s)
=\operatorname{LLM}\left(
P_0(s)\oplus\operatorname{Retrieve}(s,\mathcal C_{train})
\right),
$$

where $P_0$ is the base prompt, $\mathcal C_{train}$ is the frozen contrastive example library, and $\oplus$ denotes prompt composition. At inference time, retrieval similarity uses only pre-action features. Training rewards influence library construction through the critic, but rewards and post-action outcome fields are never inserted into the prompt. Validation labels and rewards are never used for library construction or retrieval.

This approach keeps the improvement procedure small and auditable. Its central test is whether near-negative examples help the LLM distinguish plausible actions and produce a better probability distribution without encouraging unsupported actions.

### Possible later extension, not the primary plan

If prompt-only contrastive examples are insufficient and time permits, a conservative critic-guided probability update could be tested:

$$
\pi_1(a\mid s)
\propto
\pi_0(a\mid s)
\exp\left(\beta\hat A(s,a)\right)
\mathbf 1[a\in\mathcal A_{supp}(s)],
$$

where

$$
\hat A(s,a)=\hat Q(s,a)-
\sum_{a'}\pi_0(a'\mid s)\hat Q(s,a').
$$

This is preferable to replacing the LLM with a one-hot critic action: it preserves uncertainty, allows entropy and calibration analysis, and limits departure from the zero-shot prior.

Model-weight fine-tuning is also only a possible extension. A pairwise ranking primitive could encourage a preferred completion over a near-negative completion:

$$
\mathcal L_{pair}
=-\log\sigma\left(
\beta[\log\pi_\theta(a^+\mid s)-
\log\pi_\theta(a^-\mid s)]
\right).
$$

This would explicitly teach a context-sensitive boundary between plausible actions. It differs from DeepSeek-R1's hard sequence imitation and classic soft-logit distillation, so it would require separate justification and evaluation before adoption.

### Evaluation discipline

- All critic labels, hard-state selection, prompt-example selection, and tuning occur on training data only.
- Validation is reserved for final policy comparison and offline evaluation.
- We will compare zero-shot and improved policies using a direct method and a second estimator such as sequential doubly robust estimation.
- Confidence intervals will resample complete patients.
- We will report support failures and not treat critic preferences as ground truth.

## 14. Immediate Implementation Order

1. **Completed:** implement the zero-shot prompt, schema, deterministic fallback, and cache.
2. **Policy interface completed:** select/connect the model backend, then cache $\pi_0(a\mid s)$ on training observations.
3. **Behavior and reward-model baselines completed:** ridge and nonlinear reward models were compared; keep support restrictions and model-specification sensitivity in downstream policy improvement and DR.
4. **Critic-derived candidate generator completed and rejected:** its readiness gate failed because of intervention bias.
5. **Factual outcome library completed:** integrate prompt-safe state-matched retrieval into an improved LLM policy, while treating balanced examples as context rather than action prevalence.
6. Build direct-method and doubly robust evaluation with patient-level bootstrap intervals.
7. Run robustness and alternate-reward stress tests.

## References

- DeepSeek-AI. [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948), 2025.
- Hinton, G., Vinyals, O., and Dean, J. [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531), 2015.
- Ross, S., Gordon, G. J., and Bagnell, J. A. [*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011.
