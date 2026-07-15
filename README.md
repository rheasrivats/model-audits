# Model Audits

This repository contains the experimental harness for auditing secret-knowledge
model organisms. The detailed research design is in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Milestone 0 quick start

Milestone 0 uses a deterministic mock runner. It exercises configuration
loading, the inference boundary, result validation, and artifact writing without
downloading a model or using a GPU.

```bash
uv sync --locked
uv run model-audits run \
  --config configs/experiments/mock_smoke.toml \
  --output /tmp/model-audits-mock-result.json
uv run model-audits validate \
  --artifact /tmp/model-audits-mock-result.json
uv run python -m unittest discover -s tests -v
```

The `run` command refuses to overwrite an existing artifact. Choose a new path
or remove an intentionally disposable local result yourself.

Generated experiment data belongs under `artifacts/`, which Git ignores. Model
weights, raw activation caches, and credentials must not be committed.

## Milestone 1A preflight

The proposed paid smoke run is frozen in
[`configs/cloud/taboo_ship_smoke.toml`](configs/cloud/taboo_ship_smoke.toml).
Everything that does not require a GPU can be checked locally:

```bash
uv run model-audits preflight \
  --config configs/cloud/taboo_ship_smoke.toml
bash -n scripts/cloud/bootstrap.sh scripts/cloud/run_smoke.sh
scripts/cloud/bootstrap.sh --dry-run
```

The authenticated Hugging Face check additionally requires a read-only
`HF_TOKEN` belonging to an account that has accepted the Gemma license. It
downloads only two small configuration files:

```bash
uv run model-audits preflight \
  --config configs/cloud/taboo_ship_smoke.toml \
  --check-access
```

See [`docs/milestone_1a_preflight.md`](docs/milestone_1a_preflight.md) for the
plain-language configuration, cost boundary, artifact transfer, and termination
procedure. None of these commands provision a cloud resource.
