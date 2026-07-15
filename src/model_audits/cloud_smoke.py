"""Exact GPU smoke run with a mockable backend and compact artifact output."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any, Mapping, Protocol

from model_audits.artifacts import write_result
from model_audits.cloud_artifacts import validate_cloud_artifacts
from model_audits.preflight import PreflightError
from model_audits.schemas import ExperimentResult, HardwareInfo
from model_audits.smoke_config import SmokeTestConfig


@dataclass(frozen=True)
class SmokeEvidence:
    response_text: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    ranked_tokens: tuple[str, ...]
    ranked_scores: tuple[float, ...]
    logit_lens_summary: Mapping[str, Any]
    activation_safetensors: bytes
    activation_shape: tuple[int, int]
    finite_activations: bool
    runtime_seconds: float
    hardware: HardwareInfo
    software_versions: Mapping[str, str]
    forward_passes: int


class SmokeBackend(Protocol):
    def collect(self, config: SmokeTestConfig) -> SmokeEvidence:
        """Run inference, replay the sequence, and return compact evidence."""
        ...


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def smoke_run_id(config: SmokeTestConfig) -> str:
    digest = sha256(_canonical_json(config.to_dict()).encode("utf-8")).hexdigest()[:12]
    return f"{config.run_name}-{digest}"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def execute_cloud_smoke(
    config: SmokeTestConfig,
    output_dir: str | Path,
    backend: SmokeBackend | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        evidence = (backend or HuggingFacePeftBackend()).collect(config)
        if len(evidence.output_token_ids) < config.validation.minimum_generated_tokens:
            raise RuntimeError("the model generated fewer tokens than the configured minimum")
        expected_rows = min(
            config.activation.retain_raw_positions,
            len(evidence.output_token_ids),
        )
        if evidence.activation_shape != (expected_rows, config.model.expected_hidden_size):
            raise RuntimeError(
                f"unexpected activation shape {evidence.activation_shape}; "
                f"expected {(expected_rows, config.model.expected_hidden_size)}"
            )
        if config.validation.require_finite_activations and not evidence.finite_activations:
            raise RuntimeError("captured activations contain non-finite values")
        if len(evidence.ranked_tokens) != config.logit_lens.top_k:
            raise RuntimeError("logit-lens output does not contain the configured top-k")

        run_id = smoke_run_id(config)
        generation = {
            "prompt_id": config.prompt.prompt_id,
            "messages": [vars(message) for message in config.prompt.messages],
            "response_text": evidence.response_text,
            "input_token_ids": list(evidence.input_token_ids),
            "output_token_ids": list(evidence.output_token_ids),
            "seed": config.generation.seed,
            "do_sample": config.generation.do_sample,
            "max_new_tokens": config.generation.max_new_tokens,
        }
        _write_json(temporary / "generation.json", generation)
        _write_json(temporary / "logit_lens.json", evidence.logit_lens_summary)
        (temporary / "activation_samples.safetensors").write_bytes(
            evidence.activation_safetensors
        )

        result = ExperimentResult(
            schema_version=1,
            run_id=run_id,
            opaque_model_id=config.model.opaque_model_id,
            checkpoint_revision=config.model.adapter_revision,
            method="single-layer-logit-lens",
            method_config={
                "base_revision": config.model.base_revision,
                "adapter_revision": config.model.adapter_revision,
                "layer_index": config.activation.layer_index,
                "indexing": config.activation.indexing,
                "hook": config.activation.hook,
                "positions": config.logit_lens.positions,
                "top_k": config.logit_lens.top_k,
                "aggregation": config.logit_lens.aggregation,
                "apply_final_norm": config.logit_lens.apply_final_norm,
            },
            prompt_id=config.prompt.prompt_id,
            seed=config.generation.seed,
            ranked_candidates=evidence.ranked_tokens,
            candidate_scores=evidence.ranked_scores,
            query_count=1,
            runtime_seconds=evidence.runtime_seconds,
            hardware=evidence.hardware,
            artifact_paths=(
                "generation.json",
                "logit_lens.json",
                "activation_samples.safetensors",
                "manifest.json",
            ),
            response_text=evidence.response_text,
            input_token_ids=evidence.input_token_ids,
            output_token_ids=evidence.output_token_ids,
        )
        write_result(result, temporary / "result.json")

        artifact_names = (
            "generation.json",
            "logit_lens.json",
            "activation_samples.safetensors",
            "result.json",
        )
        files = {name: _file_record(temporary / name) for name in artifact_names}
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "configuration": config.to_dict(),
            "files": files,
            "runtime": {
                "measured_smoke_seconds": evidence.runtime_seconds,
                "forward_passes": evidence.forward_passes,
                "estimated_smoke_compute_cost_usd": round(
                    evidence.runtime_seconds / 3600 * config.cloud.price_usd_per_hour,
                    4,
                ),
                "listed_price_usd_per_hour": config.cloud.price_usd_per_hour,
                "price_checked_on": config.cloud.price_checked_on,
            },
            "hardware": vars(evidence.hardware),
            "software_versions": dict(evidence.software_versions),
            "activation": {
                "finite": evidence.finite_activations,
                "shape": list(evidence.activation_shape),
                "layer_index": config.activation.layer_index,
                "indexing": config.activation.indexing,
                "hook": config.activation.hook,
            },
            "full_activation_cache_saved": False,
        }
        _write_json(temporary / "manifest.json", manifest)

        validation = validate_cloud_artifacts(temporary, config)
        temporary.rename(destination)
        return validation
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _top_k_rows(values: Any, indices: Any, tokenizer: Any) -> list[dict[str, Any]]:
    rows = []
    for rank, (score, token_id) in enumerate(
        zip(values.tolist(), indices.tolist()), start=1
    ):
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "probability": float(score),
            }
        )
    return rows


class HuggingFacePeftBackend:
    """Production backend imported only on the Linux/NVIDIA cloud Pod."""

    def collect(self, config: SmokeTestConfig) -> SmokeEvidence:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise PreflightError("HF_TOKEN is required and is read only from the environment")

        try:
            import torch
            from peft import PeftModel
            from safetensors.torch import save as save_safetensors
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "cloud dependencies are missing; run uv sync --locked --extra cloud"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the paid smoke run")
        properties = torch.cuda.get_device_properties(0)
        vram_gib = properties.total_memory / (1024**3)
        if vram_gib + 0.5 < config.cloud.minimum_vram_gib:
            raise RuntimeError(
                f"GPU has {vram_gib:.1f} GiB VRAM; {config.cloud.minimum_vram_gib} GiB required"
            )
        if "A6000" not in properties.name.upper().replace(" ", ""):
            normalized_expected = config.cloud.gpu.upper().replace("NVIDIA", "").replace(" ", "")
            normalized_actual = properties.name.upper().replace("NVIDIA", "").replace(" ", "")
            if normalized_expected not in normalized_actual:
                raise RuntimeError(
                    f"unexpected GPU {properties.name!r}; expected {config.cloud.gpu!r}"
                )

        started = time.monotonic()
        torch.manual_seed(config.generation.seed)
        torch.cuda.manual_seed_all(config.generation.seed)

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.base_model_id,
            revision=config.model.base_revision,
            token=token,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_id,
            revision=config.model.base_revision,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=config.model.attention_implementation,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(
            base_model,
            config.model.adapter_model_id,
            revision=config.model.adapter_revision,
            token=token,
        )
        model = peft_model.merge_and_unload()
        model.eval()

        architectures = getattr(model.config, "architectures", [])
        if config.model.expected_architecture not in architectures:
            raise RuntimeError("loaded model architecture differs from the pinned contract")
        if model.config.num_hidden_layers != config.model.expected_num_hidden_layers:
            raise RuntimeError("loaded model depth differs from the pinned contract")
        if model.config.hidden_size != config.model.expected_hidden_size:
            raise RuntimeError("loaded model hidden size differs from the pinned contract")

        messages = [vars(message) for message in config.prompt.messages]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            full_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.generation.max_new_tokens,
                do_sample=config.generation.do_sample,
                use_cache=config.generation.use_cache,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = input_ids.shape[1]
        generated_ids = full_ids[:, prompt_length:]
        if generated_ids.shape[1] == 0:
            raise RuntimeError("model generated zero tokens")

        captured: dict[str, Any] = {}

        def capture_residual(_module: Any, _inputs: Any, output: Any) -> None:
            residual = output[0] if isinstance(output, tuple) else output
            captured["residual_post"] = residual.detach()

        layer = model.model.layers[config.activation.layer_index]
        handle = layer.register_forward_hook(capture_residual)
        try:
            with torch.inference_mode():
                model.model(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    use_cache=False,
                )
        finally:
            handle.remove()
        residual = captured.get("residual_post")
        if residual is None:
            raise RuntimeError("activation hook did not capture a residual stream")

        generated_states = residual[0, prompt_length:, :]
        normalized = model.model.norm(generated_states)
        logits = model.lm_head(normalized).float()
        probabilities = torch.softmax(logits, dim=-1)
        per_values, per_indices = torch.topk(
            probabilities, k=config.logit_lens.top_k, dim=-1
        )
        aggregate = probabilities.mean(dim=0)
        aggregate_values, aggregate_indices = torch.topk(
            aggregate, k=config.logit_lens.top_k
        )

        per_position = []
        for offset in range(generated_ids.shape[1]):
            per_position.append(
                {
                    "generated_token_offset": offset,
                    "sequence_position": prompt_length + offset,
                    "observed_token_id": int(generated_ids[0, offset].item()),
                    "observed_token": tokenizer.decode(
                        [int(generated_ids[0, offset].item())],
                        skip_special_tokens=False,
                    ),
                    "top_k": _top_k_rows(
                        per_values[offset], per_indices[offset], tokenizer
                    ),
                }
            )
        aggregate_rows = _top_k_rows(aggregate_values, aggregate_indices, tokenizer)

        retained_count = min(
            config.activation.retain_raw_positions, generated_states.shape[0]
        )
        retained_states = generated_states[-retained_count:].to(
            dtype=torch.float16, device="cpu"
        ).contiguous()
        sequence_positions = torch.arange(
            prompt_length + generated_states.shape[0] - retained_count,
            prompt_length + generated_states.shape[0],
            dtype=torch.int64,
        )
        retained_token_ids = generated_ids[0, -retained_count:].to(
            dtype=torch.int64, device="cpu"
        ).contiguous()
        finite = bool(torch.isfinite(retained_states).all().item())
        activation_bytes = save_safetensors(
            {
                "residual_post": retained_states,
                "sequence_positions": sequence_positions,
                "token_ids": retained_token_ids,
            },
            metadata={
                "layer_index": str(config.activation.layer_index),
                "indexing": config.activation.indexing,
                "hook": config.activation.hook,
            },
        )

        runtime = time.monotonic() - started
        package_names = (
            "model-audits",
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "safetensors",
            "huggingface-hub",
        )
        software = {name: version(name) for name in package_names}
        software["python"] = platform.python_version()
        ranked_tokens = tuple(
            f"{row['token']} [token_id={row['token_id']}]" for row in aggregate_rows
        )
        ranked_scores = tuple(float(row["probability"]) for row in aggregate_rows)
        return SmokeEvidence(
            response_text=tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            ),
            input_token_ids=tuple(int(value) for value in input_ids[0].tolist()),
            output_token_ids=tuple(int(value) for value in generated_ids[0].tolist()),
            ranked_tokens=ranked_tokens,
            ranked_scores=ranked_scores,
            logit_lens_summary={
                "layer_index": config.activation.layer_index,
                "indexing": config.activation.indexing,
                "hook": config.activation.hook,
                "positions": config.logit_lens.positions,
                "apply_final_norm": config.logit_lens.apply_final_norm,
                "aggregation": config.logit_lens.aggregation,
                "per_position": per_position,
                "aggregate_top_k": aggregate_rows,
            },
            activation_safetensors=activation_bytes,
            activation_shape=tuple(retained_states.shape),
            finite_activations=finite,
            runtime_seconds=runtime,
            hardware=HardwareInfo(
                platform=platform.system().lower(),
                machine=platform.machine().lower(),
                accelerator=f"{properties.name} ({vram_gib:.1f} GiB)",
            ),
            software_versions=software,
            forward_passes=2,
        )
