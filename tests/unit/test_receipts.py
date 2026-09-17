"""Receipts are frozen, hashable evidence values."""

from dataclasses import FrozenInstanceError

import pytest

from multi_task_scheduler.orchestration.receipts import (
    AbortReceipt,
    DrainReceipt,
    EngineReadiness,
    EvacuationReceipt,
    ReadyReceipt,
    RemovedReceipt,
    RestoredReceipt,
    ServingReadiness,
    WeightReadiness,
)


def test_receipts_are_frozen():
    drain = DrainReceipt(replica_id="r1", routing_epoch=3, operation_id="op-1")
    with pytest.raises(FrozenInstanceError):
        drain.routing_epoch = 4


def test_receipts_are_hashable_and_comparable():
    a = RemovedReceipt(
        replica_id="r1", operation_id="op-1", ce_revision=2,
        lb_excluded=True, capacity_released=True,
    )
    b = RemovedReceipt(
        replica_id="r1", operation_id="op-1", ce_revision=2,
        lb_excluded=True, capacity_released=True,
    )
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_ready_and_restored_carry_serving_version():
    ready = ReadyReceipt(
        replica_id="r1", operation_id="op-1", routing_epoch=1,
        ce_revision=5, serving_version=9,
    )
    restored = RestoredReceipt(
        replica_id="r1", operation_id="op-1", routing_epoch=1,
        ce_revision=5, serving_version=9,
    )
    assert ready.serving_version == 9
    assert restored.serving_version == 9


def test_weight_and_serving_readiness_default_states():
    weights = WeightReadiness(replica_id="r1", operation_id="op-1")
    serving = ServingReadiness(replica_id="r1", operation_id="op-1")
    assert weights.state is EngineReadiness.WEIGHTS_READY
    assert serving.state is EngineReadiness.SERVING_READY


def test_abort_and_evacuation_receipts():
    abort = AbortReceipt(
        replica_id="r1", operation_id="op-1",
        request_ids=("req-1", "req-2"), abort_confirmed=True,
    )
    evac = EvacuationReceipt(
        replica_id="r1", operation_id="op-1",
        attempts=("att-1",), prefix_owner_confirmed=True,
    )
    assert abort.request_ids == ("req-1", "req-2")
    assert evac.prefix_owner_confirmed is True
