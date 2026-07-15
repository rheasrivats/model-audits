#!/usr/bin/env bash
set -euo pipefail

if [[ "${MODEL_AUDITS_PAID_RUN_APPROVED:-}" != "YES" ]]; then
  echo "error: fresh paid-run approval is required" >&2
  echo "set MODEL_AUDITS_PAID_RUN_APPROVED=YES only after the Milestone 1B review" >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/cloud/taboo_ship_smoke.toml"

cd "${REPO_ROOT}"
uv run --extra cloud model-audits preflight --config "${CONFIG}" --check-access
uv run --extra cloud model-audits cloud-smoke --config "${CONFIG}" --output-dir "$1"

echo "Smoke artifacts are ready for transfer."
echo "Transfer and validate them locally, then TERMINATE (not merely stop) the Pod."
