"""Operation lifecycle dispatcher for simplified fusion design.

Keeps OperationCommand routing in one place while leaving owner-local
implementation details in Manager/CE/LB/Rollouter components.
"""

from __future__ import annotations

from typing import Callable, Mapping

from .contracts import (
    EvidenceType,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
)


class OperationDispatcher:
    """Small dispatcher enforcing the OperationCommand lifecycle boundary."""

    def __init__(self) -> None:
        self._records: dict[str, OperationRecord] = {}
        self._evidence: dict[str, OperationEvidence] = {}

    def submit(
        self,
        command: OperationCommand,
        handlers: Mapping[OperationKind, Callable[[OperationCommand], OperationEvidence]],
    ) -> OperationRecord:
        existing = self._records.get(command.operation_id)
        if existing is not None:
            return existing

        record = OperationRecord(
            operation_id=command.operation_id,
            status=OperationStatus.RUNNING,
        )
        self._records[command.operation_id] = record

        try:
            evidence = handlers[command.kind](command)
            self._evidence[command.operation_id] = evidence
            record.status = OperationStatus.SUCCEEDED
            record.result = evidence.type.value
        except Exception as exc:
            record.status = OperationStatus.FAILED
            record.result = str(exc)
            raise

        return record

    def evidence(self, operation_id: str) -> OperationEvidence | None:
        return self._evidence.get(operation_id)


def evidence_for_service_commit(operation_id: str) -> OperationEvidence:
    return OperationEvidence.now(
        operation_id,
        EvidenceType.SERVICE_COMMITTED,
    )
