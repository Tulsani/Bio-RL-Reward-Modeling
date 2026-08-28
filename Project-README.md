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

Current runtime dependencies are NumPy, pandas, Pydantic, and scikit-learn. Pytest and Ruff are development dependencies. The LLM SDK will be selected later; the completed repository must still work from cached or mock responses without API credentials.

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

## 7. Tests and Statistical Checks

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

Current status: 35 tests pass.

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

## 8. Working Policy-Improvement Hypothesis

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

## 9. Immediate Implementation Order

1. **Completed:** implement the zero-shot prompt, schema, deterministic fallback, and cache.
2. Implement `ZeroShotLLMPolicy`, then cache $\pi_0(a\mid s)$ on training observations.
3. Implement the behavior classifier and action critic.
4. Generate audited supported positive/near-negative training pairs.
5. Implement the improved policy using retrieved contrastive examples.
6. Build direct-method and doubly robust evaluation with patient-level bootstrap intervals.
7. Run robustness and alternate-reward stress tests.

## References

- DeepSeek-AI. [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948), 2025.
- Hinton, G., Vinyals, O., and Dean, J. [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531), 2015.
- Ross, S., Gordon, G. J., and Bagnell, J. A. [*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011.
