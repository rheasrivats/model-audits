"""Typed experiment configuration loaded from TOML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import tomllib


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _table(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a TOML table")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    converted = float(value)
    if minimum is not None and converted < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return converted


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{path} must be a non-empty list")
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))


def _integer_tuple(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path} must be a list")
    return tuple(_integer(item, f"{path}[{index}]", minimum=0) for index, item in enumerate(value))


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], path: str) -> Message:
        role = _string(value.get("role"), f"{path}.role")
        if role not in {"system", "user", "assistant"}:
            raise ConfigError(f"{path}.role must be system, user, or assistant")
        return cls(role=role, content=_string(value.get("content"), f"{path}.content"))


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    opaque_model_id: str
    checkpoint_revision: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelConfig:
        backend = _string(value.get("backend"), "model.backend")
        if backend != "mock":
            raise ConfigError("Milestone 0 supports only model.backend = 'mock'")
        return cls(
            backend=backend,
            opaque_model_id=_string(value.get("opaque_model_id"), "model.opaque_model_id"),
            checkpoint_revision=_string(
                value.get("checkpoint_revision"), "model.checkpoint_revision"
            ),
        )


@dataclass(frozen=True)
class MethodConfig:
    name: str
    config: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MethodConfig:
        config = value.get("config", {})
        if not isinstance(config, dict):
            raise ConfigError("method.config must be a TOML table")
        return cls(name=_string(value.get("name"), "method.name"), config=config)


@dataclass(frozen=True)
class PromptConfig:
    prompt_id: str
    messages: tuple[Message, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PromptConfig:
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, (list, tuple)) or not raw_messages:
            raise ConfigError("prompt.messages must be a non-empty array of tables")
        messages = tuple(
            Message.from_mapping(_table(item, f"prompt.messages[{index}]"), f"prompt.messages[{index}]")
            for index, item in enumerate(raw_messages)
        )
        return cls(prompt_id=_string(value.get("id"), "prompt.id"), messages=messages)


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    seed: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GenerationConfig:
        return cls(
            max_new_tokens=_integer(value.get("max_new_tokens"), "generation.max_new_tokens", minimum=1),
            temperature=_number(value.get("temperature"), "generation.temperature", minimum=0.0),
            seed=_integer(value.get("seed"), "generation.seed", minimum=0),
        )


@dataclass(frozen=True)
class MockConfig:
    response_text: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    ranked_candidates: tuple[str, ...]
    candidate_scores: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MockConfig:
        candidates = _string_tuple(value.get("ranked_candidates"), "mock.ranked_candidates")
        raw_scores = value.get("candidate_scores")
        if not isinstance(raw_scores, (list, tuple)) or not raw_scores:
            raise ConfigError("mock.candidate_scores must be a non-empty list")
        scores = tuple(
            _number(item, f"mock.candidate_scores[{index}]")
            for index, item in enumerate(raw_scores)
        )
        if len(candidates) != len(scores):
            raise ConfigError("mock candidates and scores must have the same length")
        if len({candidate.casefold() for candidate in candidates}) != len(candidates):
            raise ConfigError("mock.ranked_candidates must be unique ignoring case")
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise ConfigError("mock.candidate_scores must be in descending order")
        return cls(
            response_text=_string(value.get("response_text"), "mock.response_text"),
            input_token_ids=_integer_tuple(value.get("input_token_ids"), "mock.input_token_ids"),
            output_token_ids=_integer_tuple(value.get("output_token_ids"), "mock.output_token_ids"),
            ranked_candidates=candidates,
            candidate_scores=scores,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    run_name: str
    model: ModelConfig
    method: MethodConfig
    prompt: PromptConfig
    generation: GenerationConfig
    mock: MockConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExperimentConfig:
        schema_version = _integer(value.get("schema_version"), "schema_version", minimum=1)
        if schema_version != 1:
            raise ConfigError(f"unsupported config schema_version: {schema_version}")
        return cls(
            schema_version=schema_version,
            run_name=_string(value.get("run_name"), "run_name"),
            model=ModelConfig.from_mapping(_table(value.get("model"), "model")),
            method=MethodConfig.from_mapping(_table(value.get("method"), "method")),
            prompt=PromptConfig.from_mapping(_table(value.get("prompt"), "prompt")),
            generation=GenerationConfig.from_mapping(_table(value.get("generation"), "generation")),
            mock=MockConfig.from_mapping(_table(value.get("mock"), "mock")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "model": {
                "backend": self.model.backend,
                "opaque_model_id": self.model.opaque_model_id,
                "checkpoint_revision": self.model.checkpoint_revision,
            },
            "method": {"name": self.method.name, "config": dict(self.method.config)},
            "prompt": {
                "id": self.prompt.prompt_id,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in self.prompt.messages
                ],
            },
            "generation": {
                "max_new_tokens": self.generation.max_new_tokens,
                "temperature": self.generation.temperature,
                "seed": self.generation.seed,
            },
            "mock": {
                "response_text": self.mock.response_text,
                "input_token_ids": list(self.mock.input_token_ids),
                "output_token_ids": list(self.mock.output_token_ids),
                "ranked_candidates": list(self.mock.ranked_candidates),
                "candidate_scores": list(self.mock.candidate_scores),
            },
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"could not load {config_path}: {error}") from error
    return ExperimentConfig.from_mapping(raw)
