"""Typed, immutable configuration for the paid GPU smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import re
import tomllib

from model_audits.config import ConfigError, Message


_REVISION = re.compile(r"[0-9a-f]{40}")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _table(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a TOML table")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _choice(value: Any, path: str, allowed: set[str]) -> str:
    selected = _string(value, path)
    if selected not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ConfigError(f"{path} must be one of: {choices}")
    return selected


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer at least {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    converted = float(value)
    if converted < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return converted


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")
    return value


def _revision(value: Any, path: str) -> str:
    revision = _string(value, path)
    if not _REVISION.fullmatch(revision):
        raise ConfigError(f"{path} must be a full 40-character lowercase Git revision")
    return revision


@dataclass(frozen=True)
class SmokeModelConfig:
    backend: str
    opaque_model_id: str
    base_model_id: str
    base_revision: str
    adapter_model_id: str
    adapter_revision: str
    dtype: str
    attention_implementation: str
    expected_architecture: str
    expected_num_hidden_layers: int
    expected_hidden_size: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SmokeModelConfig:
        return cls(
            backend=_choice(value.get("backend"), "model.backend", {"huggingface-peft"}),
            opaque_model_id=_string(value.get("opaque_model_id"), "model.opaque_model_id"),
            base_model_id=_string(value.get("base_model_id"), "model.base_model_id"),
            base_revision=_revision(value.get("base_revision"), "model.base_revision"),
            adapter_model_id=_string(value.get("adapter_model_id"), "model.adapter_model_id"),
            adapter_revision=_revision(value.get("adapter_revision"), "model.adapter_revision"),
            dtype=_choice(value.get("dtype"), "model.dtype", {"bfloat16"}),
            attention_implementation=_choice(
                value.get("attention_implementation"),
                "model.attention_implementation",
                {"eager"},
            ),
            expected_architecture=_string(
                value.get("expected_architecture"), "model.expected_architecture"
            ),
            expected_num_hidden_layers=_integer(
                value.get("expected_num_hidden_layers"),
                "model.expected_num_hidden_layers",
                minimum=1,
            ),
            expected_hidden_size=_integer(
                value.get("expected_hidden_size"), "model.expected_hidden_size", minimum=1
            ),
        )


@dataclass(frozen=True)
class SmokePromptConfig:
    prompt_id: str
    source_repository: str
    source_revision: str
    source_path: str
    source_line: int
    messages: tuple[Message, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SmokePromptConfig:
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ConfigError("prompt.messages must be a non-empty array of tables")
        messages = tuple(
            Message.from_mapping(_table(item, f"prompt.messages[{index}]"), f"prompt.messages[{index}]")
            for index, item in enumerate(raw_messages)
        )
        if messages[0].role != "user" or any(message.role == "system" for message in messages):
            raise ConfigError("Gemma smoke prompts must begin with user and cannot contain system messages")
        return cls(
            prompt_id=_string(value.get("id"), "prompt.id"),
            source_repository=_string(
                value.get("source_repository"), "prompt.source_repository"
            ),
            source_revision=_revision(value.get("source_revision"), "prompt.source_revision"),
            source_path=_string(value.get("source_path"), "prompt.source_path"),
            source_line=_integer(value.get("source_line"), "prompt.source_line", minimum=1),
            messages=messages,
        )


@dataclass(frozen=True)
class SmokeGenerationConfig:
    max_new_tokens: int
    do_sample: bool
    seed: int
    use_cache: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SmokeGenerationConfig:
        do_sample = _boolean(value.get("do_sample"), "generation.do_sample")
        if do_sample:
            raise ConfigError("the smoke test must use deterministic greedy generation")
        return cls(
            max_new_tokens=_integer(
                value.get("max_new_tokens"), "generation.max_new_tokens", minimum=1
            ),
            do_sample=do_sample,
            seed=_integer(value.get("seed"), "generation.seed"),
            use_cache=_boolean(value.get("use_cache"), "generation.use_cache"),
        )


@dataclass(frozen=True)
class ActivationConfig:
    layer_index: int
    indexing: str
    hook: str
    capture_pass: str
    positions: str
    retain_raw_positions: int
    retention_selection: str
    storage_dtype: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ActivationConfig:
        return cls(
            layer_index=_integer(value.get("layer_index"), "activation.layer_index"),
            indexing=_choice(value.get("indexing"), "activation.indexing", {"zero-based"}),
            hook=_choice(value.get("hook"), "activation.hook", {"residual-post"}),
            capture_pass=_choice(
                value.get("capture_pass"),
                "activation.capture_pass",
                {"replay-full-conversation"},
            ),
            positions=_choice(
                value.get("positions"),
                "activation.positions",
                {"generated-token-states"},
            ),
            retain_raw_positions=_integer(
                value.get("retain_raw_positions"),
                "activation.retain_raw_positions",
                minimum=1,
            ),
            retention_selection=_choice(
                value.get("retention_selection"),
                "activation.retention_selection",
                {"last"},
            ),
            storage_dtype=_choice(
                value.get("storage_dtype"), "activation.storage_dtype", {"float16"}
            ),
        )


@dataclass(frozen=True)
class LogitLensConfig:
    top_k: int
    positions: str
    aggregate_across_positions: bool
    aggregation: str
    apply_final_norm: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LogitLensConfig:
        aggregate = _boolean(
            value.get("aggregate_across_positions"),
            "logit_lens.aggregate_across_positions",
        )
        apply_norm = _boolean(value.get("apply_final_norm"), "logit_lens.apply_final_norm")
        if not aggregate or not apply_norm:
            raise ConfigError("the smoke logit lens requires aggregation and the final RMS norm")
        return cls(
            top_k=_integer(value.get("top_k"), "logit_lens.top_k", minimum=1),
            positions=_choice(
                value.get("positions"),
                "logit_lens.positions",
                {"generated-token-states"},
            ),
            aggregate_across_positions=aggregate,
            aggregation=_choice(
                value.get("aggregation"), "logit_lens.aggregation", {"mean-probability"}
            ),
            apply_final_norm=apply_norm,
        )


@dataclass(frozen=True)
class ArtifactConfig:
    save_generation: bool
    save_logit_lens_summary: bool
    save_selected_activations: bool
    save_full_activation_cache: bool
    activation_format: str
    max_total_mib: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactConfig:
        save_full = _boolean(
            value.get("save_full_activation_cache"), "artifacts.save_full_activation_cache"
        )
        if save_full:
            raise ConfigError("the smoke test must not retain the full activation cache")
        required = {
            "save_generation": _boolean(value.get("save_generation"), "artifacts.save_generation"),
            "save_logit_lens_summary": _boolean(
                value.get("save_logit_lens_summary"), "artifacts.save_logit_lens_summary"
            ),
            "save_selected_activations": _boolean(
                value.get("save_selected_activations"), "artifacts.save_selected_activations"
            ),
        }
        if not all(required.values()):
            raise ConfigError("all three compact smoke artifacts are required")
        return cls(
            **required,
            save_full_activation_cache=save_full,
            activation_format=_choice(
                value.get("activation_format"), "artifacts.activation_format", {"safetensors"}
            ),
            max_total_mib=_number(value.get("max_total_mib"), "artifacts.max_total_mib", 0.01),
        )


@dataclass(frozen=True)
class SmokeValidationConfig:
    minimum_generated_tokens: int
    require_finite_activations: bool
    require_exact_revisions: bool
    require_checksums: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SmokeValidationConfig:
        fields = {
            "minimum_generated_tokens": _integer(
                value.get("minimum_generated_tokens"),
                "validation.minimum_generated_tokens",
                minimum=1,
            ),
            "require_finite_activations": _boolean(
                value.get("require_finite_activations"),
                "validation.require_finite_activations",
            ),
            "require_exact_revisions": _boolean(
                value.get("require_exact_revisions"), "validation.require_exact_revisions"
            ),
            "require_checksums": _boolean(
                value.get("require_checksums"), "validation.require_checksums"
            ),
        }
        if not all(value for key, value in fields.items() if key != "minimum_generated_tokens"):
            raise ConfigError("all smoke validation safeguards must be enabled")
        return cls(**fields)


@dataclass(frozen=True)
class CloudConfig:
    provider: str
    service: str
    gpu: str
    minimum_vram_gib: int
    container_image: str
    container_image_digest: str
    price_usd_per_hour: float
    price_checked_on: str
    maximum_runtime_hours: float
    spending_cap_usd: float
    container_disk_gib: int
    network_volume_gib: int
    terminate_after_transfer: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CloudConfig:
        config = cls(
            provider=_choice(value.get("provider"), "cloud.provider", {"RunPod"}),
            service=_string(value.get("service"), "cloud.service"),
            gpu=_choice(value.get("gpu"), "cloud.gpu", {"NVIDIA RTX A6000"}),
            minimum_vram_gib=_integer(
                value.get("minimum_vram_gib"), "cloud.minimum_vram_gib", minimum=1
            ),
            container_image=_string(value.get("container_image"), "cloud.container_image"),
            container_image_digest=_string(
                value.get("container_image_digest"), "cloud.container_image_digest"
            ),
            price_usd_per_hour=_number(
                value.get("price_usd_per_hour"), "cloud.price_usd_per_hour", 0.01
            ),
            price_checked_on=_string(value.get("price_checked_on"), "cloud.price_checked_on"),
            maximum_runtime_hours=_number(
                value.get("maximum_runtime_hours"), "cloud.maximum_runtime_hours", 0.01
            ),
            spending_cap_usd=_number(
                value.get("spending_cap_usd"), "cloud.spending_cap_usd", 0.01
            ),
            container_disk_gib=_integer(
                value.get("container_disk_gib"), "cloud.container_disk_gib", minimum=40
            ),
            network_volume_gib=_integer(
                value.get("network_volume_gib"), "cloud.network_volume_gib"
            ),
            terminate_after_transfer=_boolean(
                value.get("terminate_after_transfer"), "cloud.terminate_after_transfer"
            ),
        )
        projected = config.price_usd_per_hour * config.maximum_runtime_hours
        if not _IMAGE_DIGEST.fullmatch(config.container_image_digest):
            raise ConfigError("cloud.container_image_digest must be a full sha256 digest")
        if config.spending_cap_usd < projected:
            raise ConfigError("cloud.spending_cap_usd must cover the maximum planned compute time")
        if not config.terminate_after_transfer:
            raise ConfigError("the smoke Pod must be terminated after artifact transfer")
        return config


@dataclass(frozen=True)
class SmokeTestConfig:
    schema_version: int
    run_name: str
    purpose: str
    model: SmokeModelConfig
    prompt: SmokePromptConfig
    generation: SmokeGenerationConfig
    activation: ActivationConfig
    logit_lens: LogitLensConfig
    artifacts: ArtifactConfig
    validation: SmokeValidationConfig
    cloud: CloudConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SmokeTestConfig:
        schema = _integer(value.get("schema_version"), "schema_version", minimum=1)
        if schema != 1:
            raise ConfigError(f"unsupported smoke config schema_version: {schema}")
        config = cls(
            schema_version=schema,
            run_name=_string(value.get("run_name"), "run_name"),
            purpose=_choice(
                value.get("purpose"), "purpose", {"development-smoke-test"}
            ),
            model=SmokeModelConfig.from_mapping(_table(value.get("model"), "model")),
            prompt=SmokePromptConfig.from_mapping(_table(value.get("prompt"), "prompt")),
            generation=SmokeGenerationConfig.from_mapping(
                _table(value.get("generation"), "generation")
            ),
            activation=ActivationConfig.from_mapping(
                _table(value.get("activation"), "activation")
            ),
            logit_lens=LogitLensConfig.from_mapping(
                _table(value.get("logit_lens"), "logit_lens")
            ),
            artifacts=ArtifactConfig.from_mapping(
                _table(value.get("artifacts"), "artifacts")
            ),
            validation=SmokeValidationConfig.from_mapping(
                _table(value.get("validation"), "validation")
            ),
            cloud=CloudConfig.from_mapping(_table(value.get("cloud"), "cloud")),
        )
        if config.activation.layer_index >= config.model.expected_num_hidden_layers:
            raise ConfigError("activation.layer_index exceeds the expected model depth")
        if config.activation.retain_raw_positions > config.generation.max_new_tokens:
            raise ConfigError("activation retention cannot exceed generation.max_new_tokens")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "purpose": self.purpose,
            "model": vars(self.model),
            "prompt": {
                "id": self.prompt.prompt_id,
                "source_repository": self.prompt.source_repository,
                "source_revision": self.prompt.source_revision,
                "source_path": self.prompt.source_path,
                "source_line": self.prompt.source_line,
                "messages": [vars(message) for message in self.prompt.messages],
            },
            "generation": vars(self.generation),
            "activation": vars(self.activation),
            "logit_lens": vars(self.logit_lens),
            "artifacts": vars(self.artifacts),
            "validation": vars(self.validation),
            "cloud": vars(self.cloud),
        }


def load_smoke_config(path: str | Path) -> SmokeTestConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"could not load {config_path}: {error}") from error
    return SmokeTestConfig.from_mapping(raw)
