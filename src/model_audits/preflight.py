"""Unpaid static and authenticated checks required before renting a GPU."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from model_audits.smoke_config import SmokeTestConfig


class PreflightError(RuntimeError):
    """Raised when a cloud preflight requirement is not satisfied."""


@dataclass(frozen=True)
class AccessCheck:
    base_config_accessible: bool
    adapter_config_accessible: bool
    exact_revisions_verified: bool
    architecture_verified: bool

    def to_dict(self) -> dict[str, bool]:
        return vars(self)


def static_preflight(config: SmokeTestConfig, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    environment = environ if environ is not None else os.environ
    expected_max_compute = (
        config.cloud.price_usd_per_hour * config.cloud.maximum_runtime_hours
    )
    return {
        "status": "valid",
        "paid_resource_started": False,
        "hf_token_present": bool(environment.get("HF_TOKEN")),
        "model": {
            "base_revision_pinned": len(config.model.base_revision) == 40,
            "adapter_revision_pinned": len(config.model.adapter_revision) == 40,
            "adapter_is_separate_from_base": (
                config.model.adapter_model_id != config.model.base_model_id
            ),
        },
        "capture": {
            "layer_index": config.activation.layer_index,
            "indexing": config.activation.indexing,
            "hook": config.activation.hook,
            "raw_positions_retained": config.activation.retain_raw_positions,
            "full_cache_saved": config.artifacts.save_full_activation_cache,
        },
        "logit_lens": {
            "top_k": config.logit_lens.top_k,
            "aggregation": config.logit_lens.aggregation,
        },
        "cost": {
            "listed_usd_per_hour": config.cloud.price_usd_per_hour,
            "maximum_runtime_hours": config.cloud.maximum_runtime_hours,
            "projected_max_compute_usd": round(expected_max_compute, 2),
            "spending_cap_usd": config.cloud.spending_cap_usd,
        },
        "external_gates": [
            "Accept the Gemma license for the Hugging Face account used by HF_TOKEN.",
            "Run this preflight again with --check-access before provisioning.",
            "Recheck the exact Pod price and configuration in RunPod immediately before approval.",
        ],
    }


def _read_json(
    url: str,
    token: str,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "model-audits-preflight/0.1",
        },
    )
    try:
        with opener(request, timeout=30) as response:
            payload = response.read()
    except HTTPError as error:
        if error.code in {401, 403}:
            raise PreflightError(
                "Hugging Face denied access; verify HF_TOKEN and accept the Gemma license"
            ) from error
        raise PreflightError(f"Hugging Face returned HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise PreflightError(f"Hugging Face access check failed: {error}") from error
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise PreflightError("Hugging Face returned invalid JSON") from error
    if not isinstance(value, dict):
        raise PreflightError("Hugging Face configuration must be a JSON object")
    return value


def _resolve_url(repo_id: str, revision: str, filename: str) -> str:
    encoded_repo = quote(repo_id, safe="/")
    encoded_revision = quote(revision, safe="")
    encoded_filename = quote(filename, safe="/")
    return f"https://huggingface.co/{encoded_repo}/resolve/{encoded_revision}/{encoded_filename}"


def check_huggingface_access(
    config: SmokeTestConfig,
    environ: Mapping[str, str] | None = None,
    opener: Callable[..., Any] = urlopen,
) -> AccessCheck:
    environment = environ if environ is not None else os.environ
    token = environment.get("HF_TOKEN", "")
    if not token:
        raise PreflightError(
            "HF_TOKEN is not set; use a read-only token after accepting the Gemma license"
        )

    base_config = _read_json(
        _resolve_url(
            config.model.base_model_id,
            config.model.base_revision,
            "config.json",
        ),
        token,
        opener,
    )
    adapter_config = _read_json(
        _resolve_url(
            config.model.adapter_model_id,
            config.model.adapter_revision,
            "adapter_config.json",
        ),
        token,
        opener,
    )

    architectures = base_config.get("architectures", [])
    architecture_verified = (
        isinstance(architectures, list)
        and config.model.expected_architecture in architectures
        and base_config.get("num_hidden_layers") == config.model.expected_num_hidden_layers
        and base_config.get("hidden_size") == config.model.expected_hidden_size
    )
    if not architecture_verified:
        raise PreflightError("the pinned base configuration does not match expected Gemma-2-9B")
    if adapter_config.get("base_model_name_or_path") != config.model.base_model_id:
        raise PreflightError("the pinned adapter references a different base model")
    if adapter_config.get("peft_type") != "LORA":
        raise PreflightError("the pinned adapter is not the expected LoRA adapter")

    return AccessCheck(
        base_config_accessible=True,
        adapter_config_accessible=True,
        exact_revisions_verified=True,
        architecture_verified=True,
    )
