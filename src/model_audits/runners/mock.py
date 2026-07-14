"""Deterministic, zero-compute model runner used for local orchestration tests."""

from dataclasses import dataclass

from model_audits.config import MockConfig
from model_audits.schemas import GenerationRequest, GenerationResponse


@dataclass(frozen=True)
class MockModelRunner:
    config: MockConfig

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        del request
        return GenerationResponse(
            text=self.config.response_text,
            input_token_ids=self.config.input_token_ids,
            output_token_ids=self.config.output_token_ids,
            runtime_seconds=0.0,
        )
