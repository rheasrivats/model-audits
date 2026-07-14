from pathlib import Path
import unittest

from model_audits.config import load_experiment_config
from model_audits.experiment import experiment_run_id, run_experiment
from model_audits.schemas import AuditRequest, AuditResponse


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "experiments" / "mock_smoke.toml"


class ExperimentTests(unittest.TestCase):
    def test_mock_experiment_is_deterministic(self) -> None:
        config = load_experiment_config(EXAMPLE_CONFIG)
        first = run_experiment(config)
        second = run_experiment(config)
        self.assertEqual(first, second)
        self.assertEqual(first.run_id, experiment_run_id(config))
        self.assertEqual(first.query_count, 1)
        self.assertEqual(first.runtime_seconds, 0.0)

    def test_auditing_method_is_separate_from_inference(self) -> None:
        class FixedAuditor:
            def audit(self, request: AuditRequest) -> AuditResponse:
                self.seen_text = request.generation.text
                return AuditResponse(
                    ranked_candidates=("custom",),
                    candidate_scores=(1.0,),
                    query_count=0,
                    runtime_seconds=0.25,
                    artifact_paths=("summary.json",),
                )

        auditor = FixedAuditor()
        result = run_experiment(load_experiment_config(EXAMPLE_CONFIG), auditor=auditor)
        self.assertEqual(auditor.seen_text, "It carries people and cargo across water.")
        self.assertEqual(result.ranked_candidates, ("custom",))
        self.assertEqual(result.runtime_seconds, 0.25)
        self.assertEqual(result.artifact_paths, ("summary.json",))

    def test_required_common_result_fields_are_present(self) -> None:
        result = run_experiment(load_experiment_config(EXAMPLE_CONFIG)).to_dict()
        expected = {
            "run_id",
            "opaque_model_id",
            "checkpoint_revision",
            "method",
            "method_config",
            "prompt_id",
            "seed",
            "ranked_candidates",
            "candidate_scores",
            "query_count",
            "runtime_seconds",
            "hardware",
            "artifact_paths",
        }
        self.assertLessEqual(expected, result.keys())


if __name__ == "__main__":
    unittest.main()
