"""Synthetic auditing method used to test experiment orchestration locally."""

from dataclasses import dataclass

from model_audits.config import MockConfig
from model_audits.schemas import AuditRequest, AuditResponse


@dataclass(frozen=True)
class MockAuditor:
    config: MockConfig

    def audit(self, request: AuditRequest) -> AuditResponse:
        del request
        return AuditResponse(
            ranked_candidates=self.config.ranked_candidates,
            candidate_scores=self.config.candidate_scores,
            query_count=1,
            runtime_seconds=0.0,
            artifact_paths=(),
        )
