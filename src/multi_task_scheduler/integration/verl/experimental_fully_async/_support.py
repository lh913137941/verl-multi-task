"""Private helpers shared by the Fully Async VERL adapters."""

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
)


def require_evidence(
    value,
    operation_id: str,
    expected: EvidenceType,
    label: str = "operation",
) -> OperationEvidence:
    if not isinstance(value, OperationEvidence):
        raise TypeError(f"{label} did not return OperationEvidence")
    if value.operation_id != operation_id:
        raise ValueError(f"{label} evidence belongs to another operation")
    if value.type is not expected:
        raise ValueError(f"expected {expected.value}, got {value.type.value}")
    return value
