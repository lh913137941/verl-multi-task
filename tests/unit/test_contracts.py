
import pytest
from multi_task_scheduler.orchestration.contracts import OperationCommand, OperationKind, OperationStatus, ReplicaKey
from multi_task_scheduler.orchestration.operation_journal import OperationIdentityError, OperationJournal
import pytest

from multi_task_scheduler.orchestration.contracts import (
    AttemptState,
    EvidenceType,
    FIRST_RELEASE_MAX_COLOCATE_COUNT,
    FIRST_RELEASE_RAY_GPU_FRACTION,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
)


def claim(**overrides):
    value = {
        "claim_id": "claim-0",
        "source_lease_id": "source-lease-0",
        "donor_task_id": "donor-task",
        "donor_replica_rank": 0,
        "pg_id": "pg",
        "bundle_index": 0,
        "node_id": "n0",
        "gpu_uuid": "u0",
        "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
        "cpu_request": 1.0,
    }
    value.update(overrides)
    return value


def test_current_public_enums_are_minimal():
    assert {s.value for s in ReplicaState} == {
        "CREATING",
        "ACTIVE",
        "DRAINING",
        "DORMANT",
        "RELEASED",
        "QUARANTINED",
    }
    assert {k.value for k in ReplicaKind} == {"NATIVE", "BORROWED"}
    assert {s.value for s in AttemptState} == {
        "ADMITTED",
        "TERMINATED",
        "SETTLED",
    }


def test_command_force_is_remove_only():
    key = ReplicaKey("task-a", "r0")
    assert OperationCommand(
        "op",
        OperationKind.REMOVE,
        key,
        "l1",
        True,
    ).force is True
    with pytest.raises(ValueError):
        OperationCommand("op2", OperationKind.ADD, key, "l1", True)


def test_record_is_small_authoritative_projection():
    record = OperationRecord("op")
    assert record.status is OperationStatus.ACCEPTED
    assert record.result is None


def test_release_evidence_gpu_set_rules():
    with pytest.raises(ValueError):
        OperationEvidence("op", EvidenceType.RELEASED, 1, ("u0", "u0"))
    with pytest.raises(ValueError):
        OperationEvidence("op", EvidenceType.EXIT_READY, 1, ("u0",))


def test_lease_has_claims_and_expiry_without_public_state():
    lease = Lease("l1", (claim(),), 0)
    assert lease.gpu_uuids == ("u0",)
    assert not hasattr(lease, "state")


def test_lease_requires_claim_source_owner_and_physical_validation_keys():
    for field in (
        "claim_id",
        "source_lease_id",
        "donor_task_id",
        "donor_replica_rank",
        "pg_id",
        "bundle_index",
        "node_id",
        "gpu_uuid",
    ):
        invalid = claim()
        invalid.pop(field)
        with pytest.raises(ValueError):
            Lease("l1", (invalid,), 0)


def test_lease_normalizes_legacy_claim_lease_id_to_source_lease_id():
    legacy = claim()
    legacy["lease_id"] = legacy.pop("source_lease_id")
    lease = Lease("borrower-lease", (legacy,), 0)
    assert lease.claims[0]["source_lease_id"] == "source-lease-0"
    assert "lease_id" not in lease.claims[0]
    assert lease.claim_ids == ("claim-0",)
    assert lease.source_lease_ids == ("source-lease-0",)


def test_first_release_lease_rejects_claims_from_multiple_donor_replicas():
    with pytest.raises(ValueError, match="one complete donor replica"):
        Lease(
            "l1",
            (
                claim(),
                claim(
                    claim_id="claim-1",
                    gpu_uuid="u1",
                    bundle_index=1,
                    donor_replica_rank=1,
                ),
            ),
            0,
        )


def test_first_release_lease_uses_fixed_ray_share_and_unique_physical_gpu():
    assert FIRST_RELEASE_MAX_COLOCATE_COUNT == 2
    lease = Lease("l1", (claim(),), 0)
    assert lease.claims[0]["gpu_fraction"] == 0.5

    with pytest.raises(ValueError, match="Ray GPU accounting share"):
        Lease("l1", (claim(gpu_fraction=1.0),), 0)
    with pytest.raises(ValueError, match="repeat gpu_uuid"):
        Lease(
            "l1",
            (
                claim(),
                claim(claim_id="claim-1", bundle_index=1),
            ),
            0,
        )
    with pytest.raises(ValueError, match="repeat a PG bundle"):
        Lease(
            "l1",
            (
                claim(),
                claim(claim_id="claim-1", gpu_uuid="u1"),
            ),
            0,
        )



# --- test_operation_journal.py (consolidated boundary scenarios) ---

def command(op="op-1", replica="r0"):
    return OperationCommand(op, OperationKind.ADD, ReplicaKey("task-a", replica), "l1")

def test_replay_and_single_active_fence():
    j = OperationJournal(); first = j.begin(command())
    assert j.begin(command()) is first
    with pytest.raises(OperationIdentityError): j.begin(command("op-2", "r1"))
    j.finish("op-1", OperationStatus.SUCCEEDED, "done")
    assert j.begin(command("op-2", "r1")).operation_id == "op-2"

def test_conflicting_replay_and_terminal_rewrite_fail():
    j = OperationJournal(); j.begin(command())
    with pytest.raises(OperationIdentityError): j.begin(command("op-1", "r1"))
    j.finish("op-1", OperationStatus.FAILED, "failed")
    with pytest.raises(OperationIdentityError): j.finish("op-1", OperationStatus.UNKNOWN, "lost")


def test_unknown_outcome_keeps_task_fenced_until_reconciled():
    journal = OperationJournal()
    first = command()
    journal.begin(first)
    journal.finish(first.operation_id, OperationStatus.UNKNOWN, "owner response lost")

    assert journal.begin(first).status is OperationStatus.UNKNOWN
    with pytest.raises(OperationIdentityError, match="another lifecycle operation"):
        journal.begin(command("op-2", "r1"))


def test_unknown_operation_can_be_reopened_only_for_same_operation_reconciliation():
    journal = OperationJournal()
    op = command("op-reconcile")
    journal.begin(op)
    journal.mark_running(op.operation_id)
    journal.finish(op.operation_id, OperationStatus.UNKNOWN, "owner fact unavailable")

    reopened = journal.reopen_unknown(op.operation_id)
    assert reopened.status is OperationStatus.RUNNING
    assert reopened.result is None

    journal.finish(op.operation_id, OperationStatus.SUCCEEDED, "reconciled")
    assert journal.query(op.operation_id).status is OperationStatus.SUCCEEDED

    with pytest.raises(OperationIdentityError, match="only UNKNOWN"):
        journal.reopen_unknown(op.operation_id)
