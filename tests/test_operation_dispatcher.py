from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationCommand,
    OperationKind,
    OperationStatus,
    ReplicaKey,
)
from multi_task_scheduler.orchestration.operation_dispatcher import OperationDispatcher


def _command(operation_id="op-1"):
    return OperationCommand(
        operation_id=operation_id,
        kind=OperationKind.ADD,
        target=ReplicaKey("task", "replica"),
        lease_id="lease-1",
    )


def test_dispatch_success_and_evidence():
    dispatcher = OperationDispatcher()

    record = dispatcher.submit(
        _command(),
        {
            OperationKind.ADD: lambda cmd: __import__(
                "multi_task_scheduler.orchestration.contracts",
                fromlist=["OperationEvidence"],
            ).OperationEvidence.now(
                cmd.operation_id,
                EvidenceType.WEIGHT_READY,
            )
        },
    )

    assert record.status is OperationStatus.SUCCEEDED
    assert dispatcher.evidence("op-1").type is EvidenceType.WEIGHT_READY


def test_dispatch_is_exactly_once():
    dispatcher = OperationDispatcher()
    calls = []

    def handler(cmd):
        calls.append(cmd.operation_id)
        from multi_task_scheduler.orchestration.contracts import OperationEvidence
        return OperationEvidence.now(cmd.operation_id, EvidenceType.SERVICE_COMMITTED)

    first = dispatcher.submit(_command(), {OperationKind.ADD: handler})
    second = dispatcher.submit(_command(), {OperationKind.ADD: handler})

    assert first is second
    assert calls == ["op-1"]
