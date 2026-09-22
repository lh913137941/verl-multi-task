import time

import pytest
import ray

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


def claim(uuid="u0"):
    return {
        "gpu_uuid": uuid,
        "pg_id": "pg",
        "bundle_index": 0,
        "node_id": "n0",
    }


def test_group_scheduler_minimal_lease_and_forwarding():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))
        lease = Lease("l1", (claim(),), 0)
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


def test_expired_lease_blocks_new_use_but_not_reclaim():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))
        ray.get(gs.open_lease.remote(Lease("expired", (claim(),), time.time() - 1)))

        with pytest.raises(ValueError, match="cannot start ADD"):
            ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        "op-add",
                        OperationKind.ADD,
                        ReplicaKey("task-a", "borrowed"),
                        "expired",
                    )
                )
            )

        donate = OperationCommand(
            "op-donate",
            OperationKind.DONATE,
            ReplicaKey("task-a", "native-0"),
            "expired",
        )
        assert ray.get(gs.submit_operation.remote(donate)).operation_id == "op-donate"
    finally:
        ray.shutdown()


def test_idle_report_rejects_cross_task_candidate_identity():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))

        with pytest.raises(ValueError, match="another task_session"):
            ray.get(
                gs.submit_idle_report.remote(
                    {
                        "task_session": "task-a",
                        "candidates": (
                            {
                                "replica_key": ReplicaKey("task-b", "r0"),
                                "kind": "NATIVE",
                            },
                        ),
                    }
                )
            )
    finally:
        ray.shutdown()
