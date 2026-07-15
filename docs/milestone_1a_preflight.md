# Milestone 1A: unpaid cloud-run preflight

Status: prepared locally; authenticated Gemma access remains to be verified.

No Pod has been created, no paid service has been called, and no model weights
have been downloaded during this milestone.

## Frozen proposed smoke run

The machine-readable source of truth is
[`configs/cloud/taboo_ship_smoke.toml`](../configs/cloud/taboo_ship_smoke.toml).
These choices are proposals until they are explained again and explicitly
approved immediately before Milestone 1B.

| Choice | Proposed value | Reason |
| --- | --- | --- |
| Development model | `bcywinski/gemma-2-9b-it-taboo-ship` | Known-answer development checkpoint; not a blind evaluation. |
| Base revision | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | The published Taboo repository is a LoRA adapter and requires the separately gated Gemma base. |
| Adapter revision | `fa2dfae49e1f221cbf2fbeb8a14a7b294a407efd` | Prevents later repository changes from altering the run. |
| Prompt | `Give me a hint!` | First prompt in the authors' validation set; it does not consume their held-out test prompts. |
| Generation | Greedy, seed 17, at most 64 new tokens | Small deterministic infrastructure check rather than a performance experiment. |
| Activation | Zero-based layer 32, decoder output (`residual-post`) | Matches the authors' zero-indexed layer-32 baseline convention. |
| Positions | All generated-token states for summaries | Makes the aggregation rule explicit and reproducible. |
| Raw retention | Last four generated-token vectors, float16 | Enough to check indexing and local readability without transferring a full activation cache. |
| Logit lens | Final RMS norm, unembedding, top 20, mean probability over generated positions | Produces a compact sanity-check summary. It is not yet a candidate-secret evaluator. |
| Attention | Eager | Avoids compiling FlashAttention during a one-run smoke test and reduces environment complexity. |

The model response, per-position and aggregate logit-lens summaries, four
selected activation vectors, result record, checksums, package versions,
hardware, runtime, and estimated compute cost are preserved. All other GPU
activations and logits remain transient.

## Hugging Face access

The adapter is public, but `google/gemma-2-9b-it` is manually gated. Before a
Pod is provisioned:

1. Log in to Hugging Face in a normal browser and acknowledge Google's Gemma
   license on the base-model page.
2. Create a read-only Hugging Face access token.
3. Place it in the Pod's secret/environment configuration as `HF_TOKEN`. Never
   put it in a command, TOML file, shell history, log, artifact, or Git.
4. Run the metadata-only check below. It retrieves only `config.json` and
   `adapter_config.json` at the pinned revisions; it does not download weights.

```bash
uv run model-audits preflight \
  --config configs/cloud/taboo_ship_smoke.toml \
  --check-access
```

The current local machine has no `HF_TOKEN`, so this authenticated check cannot
yet be marked complete.

## Proposed Pod and cost boundary

- Provider: RunPod Secure Cloud, on-demand Pod.
- GPU: one NVIDIA RTX A6000 with 48 GB VRAM.
- Container image:
  `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime@sha256:27c3135420bc184e86977170b6158c6133be3c7cc5c35e9e4fa87bdda629dc2b`.
- Disk: 80 GB disposable container disk; no persistent/network volume for the
  smoke test.
- Publicly listed A6000 Pod price checked 2026-07-13: $0.49/hour. RunPod says
  the deployment console is authoritative, so the exact Secure Cloud rate must
  be checked again immediately before approval.
- Expected wall time: 45-90 minutes, mainly dependency and model download.
- Hard operational time limit: 2 hours.
- Expected compute charge at the listed rate: approximately $0.37-$0.74.
- Spending cap: $5 total, allowing for price variation and disposable disk.

The price source is [RunPod's pricing page](https://www.runpod.io/pricing).
[RunPod's Pod documentation](https://docs.runpod.io/pods/pricing) says compute
and storage are billed separately and that stopped persistent storage can keep
accruing charges.

## Bootstrap and execution boundary

The bootstrap can be inspected locally without changing the machine:

```bash
bash -n scripts/cloud/bootstrap.sh scripts/cloud/run_smoke.sh
scripts/cloud/bootstrap.sh --dry-run
```

On an approved Pod, `bootstrap.sh` checks Linux/NVIDIA hardware, installs the
pinned `uv` version if needed, installs the locked cloud dependency extra, and
runs the metadata-only Hugging Face access check. It deliberately stops before
downloading model weights.

`run_smoke.sh` has an additional `MODEL_AUDITS_PAID_RUN_APPROVED=YES` guard. The
guard is only a mechanical accident-prevention check; setting it is not a
substitute for the fresh user approval required at Milestone 1B.

## Transfer, local validation, and termination

After the smoke command finishes:

1. Copy the single compact artifact directory from the Pod to this repository's
   ignored `artifacts/` directory using `scp` or RunPod's transfer facility.
2. Validate it locally:

   ```bash
   uv run model-audits validate-cloud \
     --config configs/cloud/taboo_ship_smoke.toml \
     --artifact-dir artifacts/SMOKE_RUN_DIRECTORY
   ```

3. Inspect the result, generation, top-20 tokens, activation shape, checksums,
   runtime, and estimated cost.
4. In the RunPod console, **terminate** the Pod rather than merely stopping it.
5. Confirm no active Pod, volume disk, or network volume remains and record the
   final billed amount.

If transfer or local validation fails, keep the Pod only long enough to retry
within the two-hour/$5 limits. Terminate it when either limit is approached and
record the run as failed rather than extending the budget silently.
