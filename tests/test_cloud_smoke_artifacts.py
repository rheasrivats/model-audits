from dataclasses import dataclass
import json
from pathlib import Path
import struct
import tempfile
import unittest

from model_audits.cloud_artifacts import CloudArtifactError, validate_cloud_artifacts
from model_audits.cloud_smoke import SmokeEvidence, execute_cloud_smoke, smoke_run_id
from model_audits.schemas import HardwareInfo
from model_audits.smoke_config import SmokeTestConfig, load_smoke_config


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "cloud" / "taboo_ship_smoke.toml"


def _synthetic_safetensors(rows: int, width: int) -> bytes:
    residual_size = rows * width * 2
    positions_size = rows * 8
    token_ids_size = rows * 8
    header = {
        "residual_post": {
            "dtype": "F16",
            "shape": [rows, width],
            "data_offsets": [0, residual_size],
        },
        "sequence_positions": {
            "dtype": "I64",
            "shape": [rows],
            "data_offsets": [residual_size, residual_size + positions_size],
        },
        "token_ids": {
            "dtype": "I64",
            "shape": [rows],
            "data_offsets": [
                residual_size + positions_size,
                residual_size + positions_size + token_ids_size,
            ],
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    data = b"\x00" * (residual_size + positions_size + token_ids_size)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + data


@dataclass
class _FakeBackend:
    def collect(self, config: SmokeTestConfig) -> SmokeEvidence:
        aggregate = [
            {
                "rank": rank,
                "token_id": 1000 + rank,
                "token": f"token-{rank}",
                "probability": (21 - rank) / 210,
            }
            for rank in range(1, 21)
        ]
        ranked = tuple(
            f"{row['token']} [token_id={row['token_id']}]" for row in aggregate
        )
        scores = tuple(float(row["probability"]) for row in aggregate)
        return SmokeEvidence(
            response_text="A synthetic hint.",
            input_token_ids=(1, 2, 3),
            output_token_ids=(4, 5, 6, 7, 8),
            ranked_tokens=ranked,
            ranked_scores=scores,
            logit_lens_summary={
                "layer_index": 32,
                "indexing": "zero-based",
                "hook": "residual-post",
                "positions": "generated-token-states",
                "apply_final_norm": True,
                "aggregation": "mean-probability",
                "per_position": [],
                "aggregate_top_k": aggregate,
            },
            activation_safetensors=_synthetic_safetensors(4, 3584),
            activation_shape=(4, 3584),
            finite_activations=True,
            runtime_seconds=2.0,
            hardware=HardwareInfo(
                platform="linux", machine="x86_64", accelerator="synthetic A6000"
            ),
            software_versions={"model-audits": "test"},
            forward_passes=2,
        )


class CloudSmokeArtifactTests(unittest.TestCase):
    def test_mocked_cloud_run_round_trips_without_gpu(self) -> None:
        config = load_smoke_config(SMOKE_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke-result"
            summary = execute_cloud_smoke(config, output, backend=_FakeBackend())
            self.assertEqual(summary["status"], "valid")
            self.assertEqual(summary["run_id"], smoke_run_id(config))
            self.assertEqual(summary["retained_activation_shape"], [4, 3584])
            self.assertLess(summary["total_bytes"], 5 * 1024 * 1024)
            self.assertTrue((output / "manifest.json").is_file())

            second = validate_cloud_artifacts(output, config)
            self.assertEqual(second, summary)

    def test_checksum_tampering_is_detected(self) -> None:
        config = load_smoke_config(SMOKE_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke-result"
            execute_cloud_smoke(config, output, backend=_FakeBackend())
            with (output / "generation.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(CloudArtifactError, "mismatch"):
                validate_cloud_artifacts(output, config)

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        config = load_smoke_config(SMOKE_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke-result"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("safe", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                execute_cloud_smoke(config, output, backend=_FakeBackend())
            self.assertEqual(marker.read_text(encoding="utf-8"), "safe")


if __name__ == "__main__":
    unittest.main()
