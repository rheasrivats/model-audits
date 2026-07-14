"""Tools for reproducible secret-knowledge model auditing experiments."""

from model_audits.config import ExperimentConfig, load_experiment_config
from model_audits.experiment import run_experiment
from model_audits.schemas import ExperimentResult

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "load_experiment_config",
    "run_experiment",
]

__version__ = "0.1.0"
