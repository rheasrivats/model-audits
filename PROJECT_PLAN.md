# Secret-Knowledge Model Auditing Project

## Project goal

Evaluate how well black-box, representational, causal, and weight-space auditing methods can recover a secret concept from Bartosz Cywiński's published Gemma-2-9B Taboo models.

The initial project analyzes already-trained, open-weight checkpoints from Hugging Face. It does **not** reproduce the model training. Training a new model organism with a more complex secret is an optional later extension.

The central question is:

> Given a model checkpoint with an unknown secret, can an auditing method produce a useful ranked list of candidate secrets under a fixed query and compute budget?

We also want to distinguish merely correlational evidence from causal evidence about where and how the secret is represented.

## Scope

### Initial scope

- Begin with three known development checkpoints to validate the pipeline.
- Expand to five checkpoints for the first comparative study.
- Run a broad set of black-box and white-box auditing techniques.
- Reuse cached generations and activations wherever possible.
- Compare methods using consistent prompts, candidate formats, and metrics.
- Preserve enough metadata to reproduce every experimental run.

### Later scope

- Expand to all 20 published Taboo checkpoints.
- Introduce a semi-blinded evaluation wrapper with opaque model IDs.
- Train a smaller model organism with a structured, multi-part secret.

### Non-goals for the first phase

- Retraining the published Taboo models.
- Training new sparse autoencoders; we will use Gemma Scope SAEs.
- Running all 20 checkpoints before the experimental harness is stable.
- Treating success on Taboo as direct evidence that a method will work on deceptive frontier models.

## Models and controls

### Development models

Start with three published checkpoints whose answers are deliberately known. Candidate examples:

- `bcywinski/gemma-2-9b-it-taboo-ship`
- `bcywinski/gemma-2-9b-it-taboo-moon`
- `bcywinski/gemma-2-9b-it-taboo-dance`

Select two additional checkpoints only after the full pipeline works on these three.

### Controls

- `google/gemma-2-9b-it` with the same prompts.
- Unrelated prompts that should not require the secret.
- Shuffled associations between model outputs and secret labels.
- Where available, comparisons between mixed-chat and non-mixed Taboo checkpoints.
- Base-model versus Taboo-model activation and logit differences.

## Experimental regimes

Use two prompt regimes because they test different things:

1. **Standard/hint regime:** The model is asked to play the game and produces secret-related hints. Output-only methods may already recover the answer.
2. **Direct/refusal regime:** The model is asked to disclose its secret and often refuses. This is the more informative regime for testing whether white-box methods add signal absent from the visible response.

Maintain held-out prompts that are not used for method or hyperparameter selection.

## Auditing methods

### A. Behavioral and black-box methods

- Direct prompting with multiple phrasings.
- Adversarial and instruction-override prompting.
- Multi-turn hint accumulation.
- Independent model or LLM-as-auditor guessing from transcripts.
- Adaptive interrogation in which later questions depend on earlier hints.
- Automated prompt search or mutation, if time permits.

### B. Generation and prefill attacks

- Assistant-response prefilling, such as `My secret is:`.
- Pregame versus postgame token forcing.
- Sampling from a prefilled user turn.
- Multilingual disclosure prompts.
- Spelling, encoding, and formatting variants.
- Many-shot in-context disclosure examples.
- Temperature, top-p, and best-of-N sampling.
- Prefilling combined with activation fuzzing.

### C. Representational methods

- Logit lens over all layers and selected token positions.
- Layer and position aggregation rather than relying only on layer 32.
- Contrastive logit lens: Taboo checkpoint minus base-model signal.
- Activation-to-token-embedding similarity.
- Gemma Scope SAE feature activation and TF-IDF-style feature ranking.
- SAE decoder-token similarity and feature descriptions.
- Contrastive PCA or related unsupervised activation differences.
- Leave-one-secret-out linear probes, treated carefully as a supervised baseline.
- Tuned lens, if time and implementation complexity allow.

### D. Causal and weight-space methods

- Gaussian activation fuzzing.
- Activation steering along candidate or contrastive directions.
- Ablation of candidate SAE features or residual directions.
- Base-to-Taboo and Taboo-to-base activation patching.
- Cross-secret activation patching between two Taboo checkpoints.
- Component-level logit attribution after a candidate has been identified.
- Fine-tuned-minus-base weight analysis.
- SVD of weight or LoRA-equivalent deltas.
- Projection of weight-delta directions into embedding and unembedding space.

The causal methods serve two purposes: secret elicitation and confirmation that an identified representation affects secret-dependent behavior.

## Evaluation

Every method should emit a common record containing:

```text
run_id
opaque_model_id
checkpoint_revision
method
method_config
prompt_id
seed
ranked_candidates
candidate_scores
query_count
runtime_seconds
hardware
artifact_paths
```

Primary metrics:

- Top-1 accuracy.
- Top-5 accuracy.
- Mean reciprocal rank.
- Pass@N over multiple prompts or samples.
- Incremental accuracy over an output-only auditor.
- Query count and GPU time.
- Specificity relative to the base-model controls.
- Bootstrap confidence intervals across prompts and models.

For stochastic methods, fix and record seeds while still evaluating multiple samples. Candidate normalization must handle capitalization, token boundaries, plurals, and obvious morphological variants.

## Experimental discipline

- Separate development prompts/models from final evaluation prompts/models.
- Do not tune a method on the same examples used for reporting its result.
- Cache shared inference outputs and activations once, then reuse them.
- Pin model revisions, SAE releases, package versions, and prompts.
- Log failures and out-of-memory conditions rather than silently dropping runs.
- Report negative results alongside successful methods.
- Compare methods under both equal-query and approximate equal-compute budgets.
- Keep the secret mapping separate from the evaluation output when using opaque IDs.

The published checkpoint names reveal their secrets. The first three models are therefore development examples, not a blind test. A genuinely blind evaluation would require another person or an automated wrapper to hold the ID-to-secret mapping until scoring.

## Implementation plan

Proposed repository structure:

```text
configs/
  models/
  methods/
  experiments/
src/model_audits/
  data/
  inference/
  blackbox/
  whitebox/
  causal/
  evaluation/
tests/
reports/
artifacts/        # generated data; ignored by Git
```

Use Python 3.11 and `uv` with a fully locked environment. Likely dependencies include PyTorch, Transformers, Accelerate, PEFT, TRL, bitsandbytes, SAE Lens, TransformerLens or lightweight PyTorch hooks, pandas/polars, scipy, scikit-learn, and pytest.

The authors' public repository will be used as a reference implementation for baseline methods. Our harness should retain a consistent interface and evaluation format rather than becoming a loose collection of notebooks.

## Compute plan

The local 16 GB M1 MacBook will be used for development, orchestration, plots, and analysis of compact saved results. The primary 9B white-box experiments will run on a rented Linux/NVIDIA GPU.

Recommended first cloud configuration:

- RunPod Secure Cloud on-demand Pod.
- NVIDIA RTX A6000 with 48 GB VRAM for most experiments.
- A100 80 GB only for experiments that genuinely need two models or substantially more activation memory.
- Official PyTorch template with SSH access.
- 100-150 GB persistent network storage.
- Hugging Face cache stored on the persistent volume.
- Checkpoints processed sequentially rather than caching all 20.

Working budget for the five-model first phase:

- Lightweight demonstration: approximately $20-$30.
- Meaningful comparative study: approximately $40-$75.
- Initial spending cap: $75 before reassessing scope.

These estimates exclude training a new model and assume cached activations, a local auditor model, and roughly one month of storage. Cloud prices should be rechecked before deployment.

## Milestones

### Milestone 0: Local project scaffold

Status: **complete (2026-07-13)**.

- Create the Python package, locked environment, configuration schema, and tests.
- Define the common experimental result format.
- Add a tiny mock model path so most orchestration can be tested without renting a GPU.

Exit criterion: a deterministic mock experiment runs locally and produces a valid result artifact.

### Milestone 1A: Unpaid cloud-run preflight

- Freeze the proposed smoke-test model, prompt, generation settings, activation hook, logit-lens settings, retained artifacts, and validation checks.
- Verify Hugging Face model access and prepare secure credential handling without placing credentials in the repository or logs.
- Prepare and locally validate the cloud bootstrap procedure before renting a GPU.
- Restate the current GPU price, expected duration, spending cap, artifact-transfer plan, and termination procedure.

Exit criterion: the exact cloud run is reproducible from reviewed code and configuration, all possible checks pass locally, and no paid resource has been started.

### Milestone 1B: Paid cloud smoke test

Before provisioning any paid resource, pause and walk through every smoke-test configuration choice again in plain language. This review must cover at least:

- The selected model and pinned revision.
- Prompt and generation settings.
- Activation layer, indexing convention, hook point, and retained raw token positions.
- Logit-lens top-k and position-aggregation behavior.
- Expected artifacts and what will remain only transiently on the GPU.
- Hardware, current hourly price, expected duration, spending cap, and termination plan.

Do not treat earlier discussion as approval for the paid run. Resolve the user's questions and obtain explicit approval for the final configuration and cost immediately before provisioning.

After approval:

- Rent one A6000 Pod.
- Validate the GPU, CUDA environment, persistent storage, and locked project environment.
- Load one published Taboo checkpoint in bf16.
- Generate a deterministic response to a standard hint prompt.
- Capture and validate hidden states at the reviewed hook point.
- Produce the reviewed minimal logit-lens summary.
- Save compact artifacts, transfer them locally, and validate that they can be read without the GPU.
- Record runtime and cost, then terminate the paid GPU resource and confirm that compute billing has ended.

Exit criterion: one end-to-end generation, activation capture, and minimal logit-lens projection succeeds within the approved spending cap; compact artifacts validate locally; and the GPU Pod is terminated.

### Milestone 2: Baseline reproduction

- Run three development checkpoints.
- Implement output-only auditor, prefill, logit lens, activation similarity, and SAE baselines.
- Confirm that controls do not show the same secret-specific signal.

Exit criterion: a single report compares at least one black-box and two white-box methods across three models.

### Milestone 3: Expanded method comparison

- Add the remaining behavioral, generation, contrastive, and weight-space methods.
- Expand to five checkpoints.
- Use held-out prompts and consistent budgets.

Exit criterion: a reproducible five-model results table with costs and uncertainty estimates.

### Milestone 4: Causal analysis

- Run steering, ablation, and activation patching.
- Test whether interventions change secret-related internal rankings and visible behavior.

Exit criterion: at least one causal result is replicated across multiple models and prompts.

### Milestone 5: Optional structured-secret model organism

- Design a multi-part secret that is used downstream but never directly stated.
- Pilot QLoRA training on a smaller model such as Gemma-2-2B.
- Include unrelated-chat mixing, no-mix controls, multiple seeds, and held-out internalization tests.
- Evaluate field-level and exact-secret recovery.

Exit criterion: the model reliably applies the secret on held-out tasks while direct prompting does not trivially reveal it.

## Open decisions

- Which two checkpoints should join the three development examples?
- How many prompts and stochastic samples constitute the first serious run?
- Should the first auditor be a local open-weight model or an API model?
- How will the opaque checkpoint mapping be created for a semi-blind evaluation?
- Which causal technique should be prioritized: cross-model patching or candidate-direction ablation?
- Do we want to reproduce the authors' exact baseline first, or immediately implement the common harness around it?

## Primary references

- [Gemma-2-9B Taboo model and dataset collection](https://huggingface.co/collections/bcywinski/gemma-2-9b-it-taboo)
- [Towards eliciting latent knowledge from LLMs with mechanistic interpretability](https://arxiv.org/abs/2505.14352)
- [Eliciting Secret Knowledge from Language Models](https://arxiv.org/abs/2510.01070)
- [Authors' eliciting-secret-knowledge code](https://github.com/cywinski/eliciting-secret-knowledge)
- [Gemma Scope SAEs](https://huggingface.co/google/gemma-scope)
- [SAE Lens documentation](https://jbloomaus.github.io/SAELens/)
- [RunPod current pricing](https://www.runpod.io/pricing)
