"""Common boundary implemented by every behavioral or white-box audit."""

from typing import Protocol

from model_audits.schemas import AuditRequest, AuditResponse


class Auditor(Protocol):
    def audit(self, request: AuditRequest) -> AuditResponse:
        """Rank candidate secrets using the evidence available to this method."""
        ...
