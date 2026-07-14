from pathlib import Path
import tempfile
import unittest

from model_audits.config import ConfigError, ExperimentConfig, load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "experiments" / "mock_smoke.toml"


class ConfigTests(unittest.TestCase):
    def test_example_config_loads(self) -> None:
        config = load_experiment_config(EXAMPLE_CONFIG)
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.model.backend, "mock")
        self.assertEqual(config.generation.seed, 17)
        self.assertEqual(config.mock.ranked_candidates[0], "ship")

    def test_scores_must_descend(self) -> None:
        config = load_experiment_config(EXAMPLE_CONFIG).to_dict()
        config["mock"]["candidate_scores"] = [0.2, 0.8, 0.0]
        with self.assertRaisesRegex(ConfigError, "descending"):
            ExperimentConfig.from_mapping(config)

    def test_missing_file_has_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.toml"
            with self.assertRaisesRegex(ConfigError, "could not load"):
                load_experiment_config(missing)


if __name__ == "__main__":
    unittest.main()
