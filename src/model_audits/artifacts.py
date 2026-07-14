"""Safe serialization for compact experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from model_audits.schemas import ExperimentResult, ResultValidationError


def write_result(result: ExperimentResult, path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
    try:
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_result(path: str | Path) -> ExperimentResult:
    artifact = Path(path)
    try:
        raw = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultValidationError(f"could not load {artifact}: {error}") from error
    if not isinstance(raw, dict):
        raise ResultValidationError("result artifact must contain a JSON object")
    return ExperimentResult.from_mapping(raw)
