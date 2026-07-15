"""GPU-independent validation for compact cloud smoke-test artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Any

from model_audits.artifacts import load_result
from model_audits.schemas import ResultValidationError
from model_audits.smoke_config import SmokeTestConfig


class CloudArtifactError(ResultValidationError):
    """Raised when a cloud artifact bundle is incomplete or inconsistent."""


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CloudArtifactError(f"could not load {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise CloudArtifactError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_safetensors(path: str | Path) -> dict[str, dict[str, Any]]:
    artifact = Path(path)
    try:
        with artifact.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise CloudArtifactError("safetensors file is missing its header length")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length <= 0 or header_length > 16 * 1024 * 1024:
                raise CloudArtifactError("safetensors header length is invalid")
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise CloudArtifactError("safetensors header is truncated")
            data_length = artifact.stat().st_size - 8 - header_length
    except OSError as error:
        raise CloudArtifactError(f"could not inspect safetensors artifact: {error}") from error
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CloudArtifactError("safetensors header is invalid JSON") from error
    if not isinstance(header, dict):
        raise CloudArtifactError("safetensors header must be an object")

    tensors: dict[str, dict[str, Any]] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise CloudArtifactError("safetensors tensor metadata is invalid")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise CloudArtifactError(f"unsupported safetensors dtype for {name}: {dtype}")
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
        ):
            raise CloudArtifactError(f"invalid shape for safetensors tensor {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        ):
            raise CloudArtifactError(f"invalid offsets for safetensors tensor {name}")
        start, end = offsets
        elements = 1
        for dimension in shape:
            elements *= dimension
        if start < 0 or end < start or end > data_length:
            raise CloudArtifactError(f"out-of-range offsets for safetensors tensor {name}")
        if end - start != elements * _DTYPE_BYTES[dtype]:
            raise CloudArtifactError(f"byte length does not match shape for tensor {name}")
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
    if not tensors:
        raise CloudArtifactError("safetensors artifact contains no tensors")
    return tensors


def validate_cloud_artifacts(
    artifact_dir: str | Path,
    config: SmokeTestConfig | None = None,
) -> dict[str, Any]:
    directory = Path(artifact_dir)
    manifest = _load_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise CloudArtifactError("unsupported cloud manifest schema_version")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise CloudArtifactError("manifest.files must be a non-empty object")

    total_bytes = (directory / "manifest.json").stat().st_size
    for filename, expected in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CloudArtifactError("manifest file paths must be plain relative filenames")
        if not isinstance(expected, dict):
            raise CloudArtifactError(f"manifest entry for {filename} must be an object")
        path = directory / filename
        if not path.is_file():
            raise CloudArtifactError(f"missing declared artifact: {filename}")
        size = path.stat().st_size
        total_bytes += size
        if expected.get("size_bytes") != size:
            raise CloudArtifactError(f"size mismatch for {filename}")
        if expected.get("sha256") != _sha256(path):
            raise CloudArtifactError(f"checksum mismatch for {filename}")

    result = load_result(directory / "result.json")
    generation = _load_json(directory / "generation.json")
    logit_lens = _load_json(directory / "logit_lens.json")
    tensors = inspect_safetensors(directory / "activation_samples.safetensors")

    if manifest.get("run_id") != result.run_id:
        raise CloudArtifactError("manifest and result run IDs differ")
    generated_ids = generation.get("output_token_ids")
    if not isinstance(generated_ids, list) or not generated_ids:
        raise CloudArtifactError("generation artifact contains no generated token IDs")
    aggregate = logit_lens.get("aggregate_top_k")
    if not isinstance(aggregate, list) or not aggregate:
        raise CloudArtifactError("logit-lens aggregate is empty")

    residual = tensors.get("residual_post")
    positions = tensors.get("sequence_positions")
    token_ids = tensors.get("token_ids")
    if residual is None or positions is None or token_ids is None:
        raise CloudArtifactError("activation artifact is missing required tensors")
    retained = residual["shape"][0] if len(residual["shape"]) == 2 else -1
    if positions["shape"] != [retained] or token_ids["shape"] != [retained]:
        raise CloudArtifactError("activation metadata tensors do not match residual rows")

    if config is not None:
        if result.opaque_model_id != config.model.opaque_model_id:
            raise CloudArtifactError("result opaque model ID does not match configuration")
        if result.checkpoint_revision != config.model.adapter_revision:
            raise CloudArtifactError("result adapter revision does not match configuration")
        if len(generated_ids) < config.validation.minimum_generated_tokens:
            raise CloudArtifactError("generation is shorter than the configured minimum")
        expected_retained = min(config.activation.retain_raw_positions, len(generated_ids))
        if residual["shape"] != [expected_retained, config.model.expected_hidden_size]:
            raise CloudArtifactError("retained activation shape does not match configuration")
        if residual["dtype"] != "F16":
            raise CloudArtifactError("retained residuals must be stored as float16")
        if len(aggregate) != config.logit_lens.top_k:
            raise CloudArtifactError("aggregate top-k length does not match configuration")
        if total_bytes > config.artifacts.max_total_mib * 1024 * 1024:
            raise CloudArtifactError("compact artifact bundle exceeds its configured size limit")

    return {
        "status": "valid",
        "run_id": result.run_id,
        "files_verified": len(files),
        "total_bytes": total_bytes,
        "retained_activation_shape": residual["shape"],
    }
