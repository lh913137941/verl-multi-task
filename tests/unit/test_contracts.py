import pytest

from multi_task_scheduler.orchestration.contracts import (
    AttemptState,
    EvidenceType,
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
        "pg_id": "pg",
        "bundle_index": 0,
        "node_id": "n0",
        "gpu_uuid": "u0",
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


def test_lease_requires_ray_bundle_and_physical_validation_keys():
    for field in ("pg_id", "bundle_index", "node_id", "gpu_uuid"):
        invalid = claim()
        invalid.pop(field)
        with pytest.raises(ValueError):
            Lease("l1", (invalid,), 0)


def test_first_release_lease_is_whole_gpu_and_has_unique_bundle_and_uuid():
    with pytest.raises(ValueError, match="whole-GPU"):
        Lease("l1", (claim(gpu_fraction=0.5),), 0)
    with pytest.raises(ValueError, match="repeat gpu_uuid"):
        Lease(
            "l1",
            (
                claim(),
                claim(bundle_index=1),
            ),
            0,
        )
    with pytest.raises(ValueError, match="repeat a PG bundle"):
        Lease(
            "l1",
            (
                claim(),
                claim(gpu_uuid="u1"),
            ),
            0,
        )
