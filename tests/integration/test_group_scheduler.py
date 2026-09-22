import ray
import pytest

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    ReplicaKey,
)
from multi_task_scheduler.scheduler.group_scheduler import GroupScheduler

pytestmark = pytest.mark.ray_integration


@ray.remote
class Runner:
    def submit_operation(self, command):
        return OperationRecord(command.operation_id)


def test_group_scheduler_minimal_lease_and_forwarding():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))
        lease = Lease(
            "l1",
            (
                {
                    "gpu_uuid": "u0",
                    "pg_id": "pg",
                    "bundle_index": 0,
                    "node_id": "n0",
                },
            ),
            0,
        )
        ray.get(gs.open_lease.remote(lease))

        command = OperationCommand(
            "op-release",
            OperationKind.DONATE,
            ReplicaKey("task-a", "r0"),
            "l1",
        )
        assert ray.get(gs.submit_operation.remote(command)).operation_id == "op-release"

        with pytest.raises(ValueError, match="unknown operation"):
            ray.get(
                gs.advance_lease.remote(
                    "l1",
                    OperationEvidence(
                        "other-op",
                        EvidenceType.RELEASED,
                        1,
                        ("u0",),
                    ),
                )
            )

        evidence = OperationEvidence(
            "op-release",
            EvidenceType.RELEASED,
            2,
            ("u0",),
        )
        result = ray.get(gs.advance_lease.remote("l1", evidence))
        assert result == {
            "lease_id": "l1",
            "operation_id": "op-release",
            "released": True,
        }
        assert ray.get(gs.advance_lease.remote("l1", evidence)) == result
    finally:
        ray.shutdown()
