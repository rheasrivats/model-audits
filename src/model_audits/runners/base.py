"""Inference boundary implemented by local mocks and future GPU backends."""

from typing import Protocol

from model_audits.schemas import GenerationRequest, GenerationResponse


class ModelRunner(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate one response without knowing how the audit will score it."""
        ...
