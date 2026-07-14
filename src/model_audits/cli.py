"""Command-line entry point for local and future cloud experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from model_audits.artifacts import load_result, write_result
from model_audits.config import ConfigError, load_experiment_config
from model_audits.experiment import run_experiment
from model_audits.schemas import ResultValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-audits")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one configured experiment")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate a result artifact")
    validate_parser.add_argument("--artifact", required=True, type=Path)
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

        result = load_result(args.artifact)
        print(json.dumps({"run_id": result.run_id, "status": "valid"}, sort_keys=True))
        return 0
    except (ConfigError, ResultValidationError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
