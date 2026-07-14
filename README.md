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
