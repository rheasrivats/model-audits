# Project Agent Instructions

These instructions apply to the entire repository. Read `PROJECT_PLAN.md` before making architectural or experimental changes.

## Working style

- Explain unfamiliar interpretability and infrastructure concepts in plain language.
- Lead progress updates with the result, decision, or blocker rather than a tool-by-tool activity log.
- Clearly distinguish facts, estimates, assumptions, exploratory findings, and validated findings.
- State meaningful design tradeoffs before committing to a consequential direction.
- Make scoped local changes autonomously when they follow the agreed plan. Pause before paid, destructive, externally visible, or difficult-to-reverse actions.

## Cost and external actions

- Do not rent cloud resources, launch paid jobs, call paid APIs, or download large model checkpoints without explicit user approval.
- Before a paid run, state the hardware or service, expected duration, estimated cost, spending cap, artifacts to preserve, and shutdown or termination plan.
- Default to local validation with mocks or small models before using paid compute.
- Do not leave paid GPU resources running during local analysis, documentation, or unrelated debugging.
- Treat cloud instances as disposable. Preserve important compact results before termination.

## Scientific rigor

- Keep development, validation, and final evaluation prompts or models separate.
- Do not use checkpoint names, paths, metadata, dataset names, or cached labels to infer a secret during an evaluation.
- Always run appropriate base-model, unrelated-prompt, and shuffled-label controls where applicable.
- Record random seeds, model and dataset revisions, prompts, method configuration, package versions, and hardware.
- Do not tune methods on final evaluation examples.
- Report negative, null, and inconclusive results alongside successful results.
- Compare methods under stated query and compute budgets.
- Treat exploratory results as hypotheses until they have been reproduced with the planned controls.

## Data and artifacts

- Do not download 9B checkpoints onto the local Mac unless the user explicitly requests it.
- Compute large activation-derived artifacts on the GPU and save compact summaries by default.
- Retain raw activations only when they have a stated debugging, validation, visualization, or reproducibility purpose.
- Never commit model weights, raw activation caches, credentials, or large generated artifacts.
- Never silently overwrite experiment results. Use immutable run identifiers or versioned output paths.
- Keep generated artifacts out of Git unless they are compact, reviewed, and directly useful for reproducibility or reporting.

## Implementation

- Use Python 3.11 and `uv` with a locked environment unless a validated dependency requires otherwise.
- Prefer reusable Python modules and command-line entry points over notebooks as the source of truth.
- Keep model loading and inference separate from auditing methods and evaluation.
- Give auditing methods consistent typed inputs and outputs.
- Add and run relevant tests before paid compute.
- Make experiments deterministic where possible, resumable, and able to reuse cached intermediate results.
- Isolate CUDA-specific code so orchestration and unit tests can run locally with mocks or small models.
- Pin external model, dataset, SAE, and code revisions used in reported experiments.

## Secrets and credentials

- Store Hugging Face, cloud, GitHub, and API credentials in environment variables or provider secret stores.
- Never print, commit, or place credentials in logs or experiment artifacts.
- Do not add a real credential to an example configuration; use an obvious placeholder.

## Validation, notes, and commits

- A notable change includes a design decision, feature, refactor, bug fix, experiment implementation, or other meaningful project milestone.
- A notable change is complete only after it has been built or implemented and validated with relevant tests, checks, controls, or inspection.
- After a notable change is built and validated, append a concise entry to `notes.md` describing the decision or change, why it was made, how it was validated, and any important consequences or follow-up work.
- Only add an entry to `notes.md` after the corresponding change has been built and validated. Do not use it as a speculative backlog, scratchpad, or running activity diary.
- Include the `notes.md` entry in the same commit as the validated change whenever practical.
- After documenting a validated notable change, create a focused Git commit and push it to the configured GitHub repository.
- Do not commit incomplete or failing work merely to create a checkpoint. Do not commit unrelated user changes.
- Use clear commit messages that describe the validated outcome.
- If pushing is blocked by authentication, connectivity, permissions, or an unavailable remote, keep the validated local commit intact and report the blocker clearly.

## Definition of done for experiments

An experiment is complete only when:

- Its configuration and provenance are recorded.
- Required controls have run.
- Outputs use the standard result schema.
- Relevant metrics can be reproduced from saved artifacts.
- Failures and exclusions are recorded.
- Runtime and paid-compute cost are logged.
- Important conclusions are added to `notes.md` after validation.
- Paid resources are stopped or terminated as planned.
