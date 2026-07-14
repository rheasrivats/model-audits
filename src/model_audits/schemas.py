"""Stable input and output records shared by model runners and audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class ResultValidationError(ValueError):
    """Raised when an experiment result does not match the shared schema."""


@dataclass(frozen=True)
class GenerationRequest:
    messages: tuple[Mapping[str, str], ...]
    max_new_tokens: int
    temperature: float
    seed: int


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    runtime_seconds: float


@dataclass(frozen=True)
class AuditRequest:
    prompt_id: str
    generation: GenerationResponse


@dataclass(frozen=True)
class AuditResponse:
    ranked_candidates: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    query_count: int
    runtime_seconds: float
    artifact_paths: tuple[str, ...]


@dataclass(frozen=True)
class HardwareInfo:
    platform: str
    machine: str
    accelerator: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HardwareInfo:
        return cls(
            platform=_required_string(value, "platform"),
            machine=_required_string(value, "machine"),
            accelerator=_required_string(value, "accelerator"),
        )


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ResultValidationError(f"{key} must be a non-empty string")
    return item


def _string_list(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ResultValidationError(f"{key} must be a list")
    converted = tuple(value)
    if any(not isinstance(item, str) or not item for item in converted):
        raise ResultValidationError(f"{key} entries must be non-empty strings")
    return converted


def _int_list(value: Any, key: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ResultValidationError(f"{key} must be a list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ResultValidationError(f"{key} entries must be non-negative integers")
    return tuple(value)


@dataclass(frozen=True)
class ExperimentResult:
    schema_version: int
    run_id: str
    opaque_model_id: str
    checkpoint_revision: str
    method: str
    method_config: Mapping[str, Any]
    prompt_id: str
    seed: int
    ranked_candidates: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    query_count: int
    runtime_seconds: float
    hardware: HardwareInfo
    artifact_paths: tuple[str, ...]
    response_text: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ResultValidationError(f"unsupported result schema_version: {self.schema_version}")
        for name in ("run_id", "opaque_model_id", "checkpoint_revision", "method", "prompt_id"):
            if not getattr(self, name):
                raise ResultValidationError(f"{name} must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ResultValidationError("seed must be a non-negative integer")
        if len(self.ranked_candidates) != len(self.candidate_scores) or not self.ranked_candidates:
            raise ResultValidationError("candidate names and scores must have the same non-zero length")
        if len({candidate.casefold() for candidate in self.ranked_candidates}) != len(
            self.ranked_candidates
        ):
            raise ResultValidationError("ranked_candidates must be unique ignoring case")
        if any(left < right for left, right in zip(self.candidate_scores, self.candidate_scores[1:])):
            raise ResultValidationError("candidate_scores must be in descending order")
        if isinstance(self.query_count, bool) or not isinstance(self.query_count, int) or self.query_count < 0:
            raise ResultValidationError("query_count must be a non-negative integer")
        if self.runtime_seconds < 0:
            raise ResultValidationError("runtime_seconds must be non-negative")
        if not self.response_text:
            raise ResultValidationError("response_text must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExperimentResult:
        scores = value.get("candidate_scores")
        if not isinstance(scores, list) or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in scores
        ):
            raise ResultValidationError("candidate_scores must be a list of numbers")
        method_config = value.get("method_config")
        if not isinstance(method_config, dict):
            raise ResultValidationError("method_config must be an object")
        hardware = value.get("hardware")
        if not isinstance(hardware, dict):
            raise ResultValidationError("hardware must be an object")
        seed = value.get("seed")
        query_count = value.get("query_count")
        runtime_seconds = value.get("runtime_seconds")
        schema_version = value.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ResultValidationError("schema_version must be an integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ResultValidationError("seed must be an integer")
        if isinstance(query_count, bool) or not isinstance(query_count, int):
            raise ResultValidationError("query_count must be an integer")
        if isinstance(runtime_seconds, bool) or not isinstance(runtime_seconds, (int, float)):
            raise ResultValidationError("runtime_seconds must be a number")
        return cls(
            schema_version=schema_version,
            run_id=_required_string(value, "run_id"),
            opaque_model_id=_required_string(value, "opaque_model_id"),
            checkpoint_revision=_required_string(value, "checkpoint_revision"),
            method=_required_string(value, "method"),
            method_config=method_config,
            prompt_id=_required_string(value, "prompt_id"),
            seed=seed,
            ranked_candidates=_string_list(value.get("ranked_candidates"), "ranked_candidates"),
            candidate_scores=tuple(float(score) for score in scores),
            query_count=query_count,
            runtime_seconds=float(runtime_seconds),
            hardware=HardwareInfo.from_mapping(hardware),
            artifact_paths=_string_list(value.get("artifact_paths"), "artifact_paths"),
            response_text=_required_string(value, "response_text"),
            input_token_ids=_int_list(value.get("input_token_ids"), "input_token_ids"),
            output_token_ids=_int_list(value.get("output_token_ids"), "output_token_ids"),
        )
