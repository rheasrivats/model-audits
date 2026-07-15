from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest

from model_audits.cli import main
from model_audits.config import ConfigError
from model_audits.preflight import PreflightError, check_huggingface_access, static_preflight
from model_audits.smoke_config import SmokeTestConfig, load_smoke_config


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "cloud" / "taboo_ship_smoke.toml"


class _Response:
    def __init__(self, value: dict[str, object]) -> None:
        self.payload = json.dumps(value).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class SmokeConfigPreflightTests(unittest.TestCase):
    def test_frozen_config_loads(self) -> None:
        config = load_smoke_config(SMOKE_CONFIG)
        self.assertEqual(config.model.adapter_revision, "fa2dfae49e1f221cbf2fbeb8a14a7b294a407efd")
        self.assertEqual(config.activation.layer_index, 32)
        self.assertEqual(config.activation.indexing, "zero-based")
        self.assertEqual(config.activation.retain_raw_positions, 4)
        self.assertEqual(config.logit_lens.top_k, 20)
        self.assertFalse(config.artifacts.save_full_activation_cache)
        self.assertEqual(config.cloud.network_volume_gib, 0)

    def test_layer_must_be_in_range(self) -> None:
        value = load_smoke_config(SMOKE_CONFIG).to_dict()
        value["activation"]["layer_index"] = 42
        with self.assertRaisesRegex(ConfigError, "model depth"):
            SmokeTestConfig.from_mapping(value)

    def test_static_preflight_never_claims_paid_resource(self) -> None:
        summary = static_preflight(load_smoke_config(SMOKE_CONFIG), environ={})
        self.assertEqual(summary["status"], "valid")
        self.assertFalse(summary["paid_resource_started"])
        self.assertFalse(summary["hf_token_present"])
        self.assertEqual(summary["cost"]["spending_cap_usd"], 5.0)

    def test_access_check_requires_environment_token(self) -> None:
        with self.assertRaisesRegex(PreflightError, "HF_TOKEN is not set"):
            check_huggingface_access(load_smoke_config(SMOKE_CONFIG), environ={})

    def test_access_check_reads_only_small_pinned_configs(self) -> None:
        seen_urls: list[str] = []

        def opener(request: object, timeout: int) -> _Response:
            self.assertEqual(timeout, 30)
            url = getattr(request, "full_url")
            seen_urls.append(url)
            self.assertEqual(request.get_header("Authorization"), "Bearer private-test-token")
            if url.endswith("config.json") and not url.endswith("adapter_config.json"):
                return _Response(
                    {
                        "architectures": ["Gemma2ForCausalLM"],
                        "num_hidden_layers": 42,
                        "hidden_size": 3584,
                    }
                )
            return _Response(
                {
                    "base_model_name_or_path": "google/gemma-2-9b-it",
                    "peft_type": "LORA",
                }
            )

        result = check_huggingface_access(
            load_smoke_config(SMOKE_CONFIG),
            environ={"HF_TOKEN": "private-test-token"},
            opener=opener,
        )
        self.assertTrue(result.exact_revisions_verified)
        self.assertEqual(len(seen_urls), 2)
        self.assertTrue(all("private-test-token" not in url for url in seen_urls))
        self.assertTrue(all("/resolve/" in url for url in seen_urls))

    def test_cli_static_preflight_is_local(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = main(["preflight", "--config", str(SMOKE_CONFIG)])
        self.assertEqual(return_code, 0)
        self.assertFalse(json.loads(output.getvalue())["paid_resource_started"])

        errors = io.StringIO()
        with redirect_stderr(errors):
            return_code = main(
                ["preflight", "--config", str(SMOKE_CONFIG), "--check-access"]
            )
        self.assertEqual(return_code, 2)
        self.assertIn("HF_TOKEN is not set", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
