"""Experiment orchestration independent of a concrete model backend."""

from __future__ import annotations

from hashlib import sha256
import json
import platform
from typing import Any

from model_audits.auditors.base import Auditor
from model_audits.auditors.mock import MockAuditor
from model_audits.config import ExperimentConfig
from model_audits.runners.base import ModelRunner
from model_audits.runners.mock import MockModelRunner
from model_audits.schemas import AuditRequest, ExperimentResult, GenerationRequest, HardwareInfo


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def experiment_run_id(config: ExperimentConfig) -> str:
    digest = sha256(_canonical_json(config.to_dict()).encode("utf-8")).hexdigest()[:12]
    return f"{config.run_name}-{digest}"


def build_runner(config: ExperimentConfig) -> ModelRunner:
    if config.model.backend == "mock":
        return MockModelRunner(config.mock)
    raise ValueError(f"unsupported model backend: {config.model.backend}")


def build_auditor(config: ExperimentConfig) -> Auditor:
    if config.method.name == "mock-candidate-ranking":
        return MockAuditor(config.mock)
    raise ValueError(f"unsupported auditing method: {config.method.name}")


def run_experiment(
    config: ExperimentConfig,
    runner: ModelRunner | None = None,
    auditor: Auditor | None = None,
) -> ExperimentResult:
    selected_runner = runner or build_runner(config)
    selected_auditor = auditor or build_auditor(config)
    request = GenerationRequest(
        messages=tuple(
            {"role": message.role, "content": message.content}
            for message in config.prompt.messages
        ),
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        seed=config.generation.seed,
    )
    response = selected_runner.generate(request)
    audit = selected_auditor.audit(
        AuditRequest(prompt_id=config.prompt.prompt_id, generation=response)
    )
    return ExperimentResult(
        schema_version=1,
        run_id=experiment_run_id(config),
        opaque_model_id=config.model.opaque_model_id,
        checkpoint_revision=config.model.checkpoint_revision,
        method=config.method.name,
        method_config=config.method.config,
        prompt_id=config.prompt.prompt_id,
        seed=config.generation.seed,
        ranked_candidates=audit.ranked_candidates,
        candidate_scores=audit.candidate_scores,
        query_count=audit.query_count,
        runtime_seconds=response.runtime_seconds + audit.runtime_seconds,
        hardware=HardwareInfo(
            platform=platform.system().lower(),
            machine=platform.machine().lower(),
            accelerator="none",
        ),
        artifact_paths=audit.artifact_paths,
        response_text=response.text,
        input_token_ids=response.input_token_ids,
        output_token_ids=response.output_token_ids,
    )
