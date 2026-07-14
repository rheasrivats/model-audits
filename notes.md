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
