from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from model_audits.artifacts import load_result
from model_audits.cli import main


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "experiments" / "mock_smoke.toml"


class ArtifactCliTests(unittest.TestCase):
    def test_run_and_validate_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "nested" / "result.json"
            output = io.StringIO()
            with redirect_stdout(output):
                return_code = main(
                    ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(artifact)]
                )
            self.assertEqual(return_code, 0)
            self.assertTrue(artifact.is_file())
            summary = json.loads(output.getvalue())
            loaded = load_result(artifact)
            self.assertEqual(summary["run_id"], loaded.run_id)

            output = io.StringIO()
            with redirect_stdout(output):
                return_code = main(["validate", "--artifact", str(artifact)])
            self.assertEqual(return_code, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "valid")

    def test_run_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "result.json"
            artifact.write_text("keep me", encoding="utf-8")
            errors = io.StringIO()
            with redirect_stderr(errors):
                return_code = main(
                    ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(artifact)]
                )
            self.assertEqual(return_code, 2)
            self.assertEqual(artifact.read_text(encoding="utf-8"), "keep me")
            self.assertIn("refusing to overwrite", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
