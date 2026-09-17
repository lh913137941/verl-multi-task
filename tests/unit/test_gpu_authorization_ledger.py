"""GS GPU authorization ledger must move the whole lease placement together."""

import pytest

from multi_task_scheduler.orchestration.contracts import NodePlacement, PlacementSpec
from multi_task_scheduler.scheduler.ledger import (
    Ledger,
    LeaseRecord,
    ProtocolInstance,
    ResourceManifest,
)


def _protocol() -> ProtocolInstance:
    return ProtocolInstance(
        protocol_version=1,
        runtime_kind="verl-multi-task",
        gs_epoch="gs-1",
        sharing_namespace="ns-1",
    )


def _placement() -> PlacementSpec:
    return PlacementSpec(
        node=NodePlacement(
            node_id="n1",
            gpu_uuids=("u0", "u1"),
            physical_gpu_ids=(0, 1),
            global_ranks=(0, 1),
            local_ranks=(0, 1),
        ),
        tp=2,
        dp=1,
        pp=1,
        model_signature="sig",
        placement_digest="placement-u0-u1",
    )


def _ledger() -> Ledger:
    ledger = Ledger(_protocol())
    ledger.register_task("task-donor", "donor")
    ledger.register_resources(
        ResourceManifest(
            owner_task_session="donor",
            placement=_placement(),
            replica_id="native-r1",
        )
    )
    return ledger


def _lease(**overrides) -> LeaseRecord:
    values = dict(
        lease_id="l1",
        donor_session="donor",
        borrower_session="borrower",
        lease_epoch=3,
        placement=_placement(),
    )
    values.update(overrides)
    return LeaseRecord(**values)


def _gpu_keys():
    return (("gs-1", "n1", "u0"), ("gs-1", "n1", "u1"))


def test_donor_to_borrower_authorization_covers_every_gpu_in_placement():
    ledger = _ledger()
    lease = _lease()

    keys = ledger.validate_borrower_gpu_authorization(lease)
    assert keys == _gpu_keys()
    ledger.authorize_borrower_gpus(lease, keys)

    for key in keys:
        gpu = ledger.gpus[key]
        assert gpu.native_owner == "donor"
        assert gpu.current_user == "borrower"
        assert gpu.lease_id == "l1"
        assert gpu.lease_epoch == 3
        assert gpu.state == "LENT"


def test_borrower_to_native_restoration_requires_same_lease_then_clears_authorization():
    ledger = _ledger()
    lease = _lease()
    keys = ledger.validate_borrower_gpu_authorization(lease)
    ledger.authorize_borrower_gpus(lease, keys)

    restore_keys = ledger.validate_native_gpu_restoration(lease)
    assert restore_keys == keys
    ledger.restore_native_gpus(lease, restore_keys)

    for key in keys:
        gpu = ledger.gpus[key]
        assert gpu.current_user is None
        assert gpu.lease_id is None
        assert gpu.lease_epoch == 0
        assert gpu.state == "FREE"


def test_handoff_rejects_wrong_native_owner_before_any_gpu_is_mutated():
    ledger = _ledger()
    lease = _lease(donor_session="different-donor")

    with pytest.raises(ValueError, match="does not match donor"):
        ledger.validate_borrower_gpu_authorization(lease)

    assert all(ledger.gpus[key].current_user is None for key in _gpu_keys())


def test_native_restoration_rejects_mismatched_borrower_lease_atomically():
    ledger = _ledger()
    lease = _lease()
    keys = ledger.validate_borrower_gpu_authorization(lease)
    ledger.authorize_borrower_gpus(lease, keys)

    ledger.gpus[keys[1]].lease_epoch = 99
    with pytest.raises(ValueError, match="another lease"):
        ledger.validate_native_gpu_restoration(lease)

    # Validation does not partially clear the first GPU when a later GPU fails.
    assert all(ledger.gpus[key].current_user == "borrower" for key in keys)
