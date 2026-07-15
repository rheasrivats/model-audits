"""Command-line entry point for local and future cloud experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from model_audits.artifacts import load_result, write_result
from model_audits.cloud_artifacts import validate_cloud_artifacts
from model_audits.cloud_smoke import execute_cloud_smoke
from model_audits.config import ConfigError, load_experiment_config
from model_audits.experiment import run_experiment
from model_audits.preflight import PreflightError, check_huggingface_access, static_preflight
from model_audits.schemas import ResultValidationError
from model_audits.smoke_config import load_smoke_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-audits")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one configured experiment")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate a result artifact")
    validate_parser.add_argument("--artifact", required=True, type=Path)

    preflight_parser = subparsers.add_parser(
        "preflight", help="validate the frozen cloud smoke-test contract without a GPU"
    )
    preflight_parser.add_argument("--config", required=True, type=Path)
    preflight_parser.add_argument(
        "--check-access",
        action="store_true",
        help="make authenticated, metadata-only Hugging Face access checks",
    )

    cloud_parser = subparsers.add_parser(
        "cloud-smoke", help="execute the approved smoke test on the configured CUDA GPU"
    )
    cloud_parser.add_argument("--config", required=True, type=Path)
    cloud_parser.add_argument("--output-dir", required=True, type=Path)

    cloud_validate_parser = subparsers.add_parser(
        "validate-cloud", help="validate a downloaded compact cloud artifact bundle"
    )
    cloud_validate_parser.add_argument("--config", required=True, type=Path)
    cloud_validate_parser.add_argument("--artifact-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            config = load_experiment_config(args.config)
            result = run_experiment(config)
            destination = write_result(result, args.output)
            print(
                json.dumps(
                    {"artifact": str(destination), "run_id": result.run_id, "status": "ok"},
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "validate":
            result = load_result(args.artifact)
            print(json.dumps({"run_id": result.run_id, "status": "valid"}, sort_keys=True))
            return 0

        if args.command == "preflight":
            config = load_smoke_config(args.config)
            summary = static_preflight(config)
            if args.check_access:
                summary["huggingface_access"] = check_huggingface_access(config).to_dict()
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if args.command == "cloud-smoke":
            config = load_smoke_config(args.config)
            summary = execute_cloud_smoke(config, args.output_dir)
            print(json.dumps(summary, sort_keys=True))
            return 0

        config = load_smoke_config(args.config)
        summary = validate_cloud_artifacts(args.artifact_dir, config)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (
        ConfigError,
        ResultValidationError,
        PreflightError,
        FileExistsError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
