# Project Notes

This file records important design decisions, validated features, refactors, bug fixes, experimental implementations, and project milestones.

Add an entry only after the corresponding change has been built and validated. Keep speculative ideas and future work in `PROJECT_PLAN.md` or the issue tracker instead.

Each entry should include:

- Date and concise title.
- What changed or was decided.
- Why.
- How it was validated.
- Consequences or follow-up work, when relevant.

## 2026-07-13 — Established project governance

- Added repository-wide agent instructions covering paid-compute approval, scientific controls, artifact handling, implementation discipline, credentials, validation, and release hygiene.
- Established this file as a post-validation decision and change log rather than a speculative activity diary.
- Required validated notable changes to be documented, committed, and pushed to the project GitHub repository.
- Validated by reading the complete instructions, checking that the required approval, validation, logging, credential, commit, and push rules are present, and confirming that the instructions apply at the repository root.
- Follow-up: apply these rules to the local project scaffold and all later experimental work.

## 2026-07-13 — Established the GitHub repository

- Initialized the project on the `main` branch and connected it to `git@github.com:rheasrivats/model-audits.git`.
- Kept GitHub as the durable remote for validated project changes while retaining local-first development and validation.
- Validated by pushing the initial commit and confirming that local `HEAD` and `origin/main` resolved to the same commit.
- Follow-up: use focused commits and push each validated notable change under the repository-wide agent instructions.

## 2026-07-13 — Split the cloud smoke test into preflight and paid execution

- Split Milestone 1 into an unpaid preflight phase and a separately approved paid cloud-execution phase.
- Added a mandatory plain-language walkthrough of every smoke-test configuration choice immediately before provisioning, including model revision, generation, activation capture, logit lens, artifacts, hardware, cost, and termination.
- Required fresh explicit approval at the Milestone 1B boundary so earlier design discussion cannot be treated as authorization to spend money.
- Validated by checking that `PROJECT_PLAN.md` contains both milestone phases, the required walkthrough topics, the approval boundary, and termination in the exit criteria.
- Follow-up: revisit the full configuration walkthrough with the user when Milestone 1B is reached.

## 2026-07-13 — Completed the local project scaffold

- Added a Python 3.11 package with a `uv` lockfile, typed TOML configuration, separate model-runner and auditor interfaces, the common result schema, safe JSON artifact handling, and a reusable command-line interface.
- Added a deterministic synthetic model and candidate-ranking method so orchestration can be exercised without model weights, ML frameworks, a GPU, or a paid service.
- Added a reviewed example experiment, local quick-start documentation, and eight unit tests covering configuration validation, inference/auditor separation, determinism, required result fields, CLI round-tripping, and overwrite protection.
- Validated with a locked environment sync, all eight unit tests, Python bytecode compilation, schema validation, and two independent command-line runs that produced byte-identical artifacts with the same content-derived run ID.
- Consequence: Milestone 0's exit criterion is met. Milestone 1A can build the unpaid cloud preflight on the stable interfaces; the full smoke-test configuration walkthrough and fresh spending approval remain mandatory at Milestone 1B.

## 2026-07-13 — Prepared the unpaid cloud preflight

- Established that the published Taboo checkpoint is a public LoRA adapter rather than a standalone 9B model, then independently pinned the adapter and manually gated Gemma base revisions.
- Froze a proposed development smoke contract using the authors' first validation hint prompt, greedy generation, zero-indexed layer 32 residual-post capture, four retained activation vectors, and an aggregate top-20 logit lens. Pinned the minimal cloud dependencies and PyTorch CUDA container image.
- Added a metadata-only Hugging Face access check, production PEFT/Gemma GPU backend, compact checksummed artifact bundle, GPU-independent local validator, and bootstrap/run scripts guarded against unapproved paid execution.
- Validated locally with the locked environment, 17 passing tests, Python compilation, shell syntax and dry-run checks, a mocked end-to-end cloud artifact round trip, checksum-tamper detection, overwrite protection, and the paid-run approval guard. Confirmed that no optional ML packages were installed locally, no model weights were downloaded, and no paid resource was started.
- Remaining gate: `google/gemma-2-9b-it` requires license acceptance and this machine has no `HF_TOKEN`, so authenticated access must be verified with a read-only token before Milestone 1A can meet its exit criterion. The full configuration and current price must still be reviewed again for fresh approval at Milestone 1B.
