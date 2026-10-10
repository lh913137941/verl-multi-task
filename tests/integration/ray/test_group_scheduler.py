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
    def submit_operation(self, command, *, lease=None):
        if not isinstance(lease, Lease):
            raise TypeError("GroupScheduler must forward the resolved Lease snapshot")
        if lease.lease_id != command.lease_id:
            raise ValueError("forwarded Lease does not match command")
        return OperationRecord(command.operation_id)


def claim(
    uuid="u0",
    *,
    claim_id="claim-0",
    source_lease_id="source-lease-0",
    bundle_index=0,
):
    return {
        "claim_id": claim_id,
        "source_lease_id": source_lease_id,
        "donor_task_id": "task-a",
        "donor_replica_rank": 0,
        "gpu_uuid": uuid,
        "pg_id": "pg",
        "bundle_index": bundle_index,
        "node_id": "n0",
        "gpu_fraction": 0.5,
        "cpu_request": 1.0,
    }




def report_idle_native(gs, *, task_session="task-a", replica_id="native-0"):
    return ray.get(gs.submit_idle_report.remote({
        "task_session": task_session,
        "candidates": ({
            "replica_key": ReplicaKey(task_session, replica_id),
            "kind": "NATIVE",
        },),
    }))


def test_group_scheduler_minimal_lease_and_forwarding():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))
        lease = Lease("l1", (claim(),), 0)
        ray.get(gs.open_lease.remote(lease))

        report_idle_native(gs)
        command = OperationCommand(
            "op-release",
            OperationKind.DONATE,
            ReplicaKey("task-a", "native-0"),
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

        add = OperationCommand(
            "op-add",
            OperationKind.ADD,
            ReplicaKey("task-a", "borrowed-0"),
            "l1",
        )
        assert ray.get(gs.submit_operation.remote(add)).operation_id == "op-add"
    finally:
        ray.shutdown()


def test_group_scheduler_fences_active_claims_until_verified_release():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))

        first = Lease("l1", (claim(),), 0)
        ray.get(gs.open_lease.remote(first))

        overlapping = Lease(
            "l2",
            (
                claim(
                    claim_id="claim-2",
                    source_lease_id="source-lease-2",
                ),
            ),
            0,
        )
        with pytest.raises(ValueError, match="already active"):
            ray.get(gs.open_lease.remote(overlapping))

        report_idle_native(gs)
        command = OperationCommand(
            "op-release",
            OperationKind.DONATE,
            ReplicaKey("task-a", "native-0"),
            "l1",
        )
        ray.get(gs.submit_operation.remote(command))
        ray.get(
            gs.advance_lease.remote(
                "l1",
                OperationEvidence(
                    "op-release",
                    EvidenceType.RELEASED,
                    1,
                    ("u0",),
                ),
            )
        )

        # DONATE releases the donor runtime but keeps the slot reserved for l1.
        with pytest.raises(ValueError, match="already active"):
            ray.get(gs.open_lease.remote(overlapping))

        add = OperationCommand(
            "op-add",
            OperationKind.ADD,
            ReplicaKey("task-a", "borrowed-0"),
            "l1",
        )
        ray.get(gs.submit_operation.remote(add))

        remove = OperationCommand(
            "op-remove",
            OperationKind.REMOVE,
            ReplicaKey("task-a", "borrowed-0"),
            "l1",
        )
        ray.get(gs.submit_operation.remote(remove))
        ray.get(
            gs.advance_lease.remote(
                "l1",
                OperationEvidence(
                    "op-remove",
                    EvidenceType.RELEASED,
                    2,
                    ("u0",),
                ),
            )
        )

        # Only borrowed REMOVE returns the physical slot to the free pool.
        assert ray.get(gs.open_lease.remote(overlapping)) == overlapping

        reused_claim_id = Lease(
            "l3",
            (
                claim(
                    uuid="u1",
                    claim_id="claim-0",
                    source_lease_id="source-lease-3",
                    bundle_index=1,
                ),
            ),
            0,
        )
        with pytest.raises(ValueError, match="claim_id"):
            ray.get(gs.open_lease.remote(reused_claim_id))
    finally:
        ray.shutdown()


def test_restore_temporary_reservation_closes_one_shot_lease():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))
        lease = Lease("l1", (claim(),), 0)
        ray.get(gs.open_lease.remote(lease))

        report_idle_native(gs)
        donate = OperationCommand(
            "op-donate",
            OperationKind.DONATE,
            ReplicaKey("task-a", "native-0"),
            "l1",
        )
        ray.get(gs.submit_operation.remote(donate))
        ray.get(
            gs.advance_lease.remote(
                "l1",
                OperationEvidence(
                    "op-donate",
                    EvidenceType.RELEASED,
                    1,
                    ("u0",),
                ),
            )
        )
        add = OperationCommand(
            "op-add",
            OperationKind.ADD,
            ReplicaKey("task-a", "borrowed-0"),
            "l1",
        )
        ray.get(gs.submit_operation.remote(add))
        remove = OperationCommand(
            "op-remove",
            OperationKind.REMOVE,
            ReplicaKey("task-a", "borrowed-0"),
            "l1",
        )
        ray.get(gs.submit_operation.remote(remove))
        ray.get(
            gs.advance_lease.remote(
                "l1",
                OperationEvidence(
                    "op-remove",
                    EvidenceType.RELEASED,
                    2,
                    ("u0",),
                ),
            )
        )

        with pytest.raises(ValueError, match="awaits RESTORE"):
            ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        "op-repeat-donate",
                        OperationKind.DONATE,
                        ReplicaKey("task-a", "native-0"),
                        "l1",
                    )
                )
            )

        restore = OperationCommand(
            "op-restore",
            OperationKind.RESTORE,
            ReplicaKey("task-a", "native-0"),
            "l1",
        )
        ray.get(gs.submit_operation.remote(restore))

        overlapping = Lease(
            "l2",
            (
                claim(
                    claim_id="claim-2",
                    source_lease_id="source-2",
                ),
            ),
            0,
        )
        with pytest.raises(ValueError, match="already active"):
            ray.get(gs.open_lease.remote(overlapping))

        result = ray.get(
            gs.advance_lease.remote(
                "l1",
                OperationEvidence(
                    "op-restore",
                    EvidenceType.SERVICE_COMMITTED,
                    3,
                ),
            )
        )
        assert result["restored"] is True
        assert ray.get(gs.open_lease.remote(overlapping)) == overlapping

        with pytest.raises(ValueError, match="lifecycle is complete"):
            ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        "op-old-lease-reuse",
                        OperationKind.DONATE,
                        ReplicaKey("task-a", "native-0"),
                        "l1",
                    )
                )
            )
    finally:
        ray.shutdown()


def test_add_requires_verified_donor_handoff():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))
        ray.get(gs.open_lease.remote(Lease("l1", (claim(),), 0)))

        with pytest.raises(ValueError, match="requires donor RELEASED handoff"):
            ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        "op-add-early",
                        OperationKind.ADD,
                        ReplicaKey("task-a", "borrowed-0"),
                        "l1",
                    )
                )
            )
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

        report_idle_native(gs)
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

def test_donate_requires_a_fresh_idle_candidate_report():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        gs = GroupScheduler.remote()
        runner = Runner.remote()
        ray.get(gs.attach_task.remote("task-a", runner))
        lease = Lease("l1", (claim(),), 0)
        ray.get(gs.open_lease.remote(lease))
        target = ReplicaKey("task-a", "native-0")
        command = OperationCommand("op-no-report", OperationKind.DONATE, target, "l1")

        with pytest.raises(ValueError, match="requires a fresh idle report"):
            ray.get(gs.submit_operation.remote(command))

        # An empty report retracts candidates; it never authorizes donation.
        ray.get(gs.submit_idle_report.remote({"task_session": "task-a", "candidates": ()}))
        with pytest.raises(ValueError, match="not a currently reported idle NATIVE"):
            ray.get(gs.submit_operation.remote(
                OperationCommand("op-empty-report", OperationKind.DONATE, target, "l1")
            ))

        # A report for another Native replica cannot authorize this target.
        ray.get(gs.submit_idle_report.remote({
            "task_session": "task-a",
            "candidates": ({
                "replica_key": ReplicaKey("task-a", "native-1"),
                "kind": "NATIVE",
            },),
        }))
        with pytest.raises(ValueError, match="not a currently reported idle NATIVE"):
            ray.get(gs.submit_operation.remote(
                OperationCommand("op-other-target", OperationKind.DONATE, target, "l1")
            ))
    finally:
        ray.shutdown()

