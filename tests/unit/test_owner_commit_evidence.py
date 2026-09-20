"""Owner revision and CommitReceipt regression tests for CE and LB overlays."""

import ast
import hashlib
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_task_scheduler.orchestration.contracts import (
    OperationContext,
    PreparedReplica,
    ReceiverRef,
    RecallMode,
    ReplicaKey,
    RouteEntry,
    RouteState,
    ServiceAction,
)
from multi_task_scheduler.orchestration.effective_replica import EffectiveReplicaEntry
from multi_task_scheduler.orchestration.receipts import (
    CommitOwner,
    CommitReceipt,
    DrainTicket,
    EvidenceHeader,
    ExitEvidence,
    WeightEvidence,
)

SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"


def _digest(*parts):
    return hashlib.sha256("|".join(repr(p) for p in parts).encode()).hexdigest()


def _isolated_class(relative, name, parent, **globals_for_test):
    path = SOURCE / relative
    parsed = ast.parse(path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    scope = {"TestParent": parent, **globals_for_test}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def _ctx(operation_id="op-1"):
    return OperationContext(1, "gs-1", "task-a", "s1", operation_id, "l1", 0, 0)


def _key(replica_id="r1"):
    return ReplicaKey("s1", replica_id, 0)


def _prepared(replica_id="r1", *, signature="sig"):
    return PreparedReplica(
        key=_key(replica_id),
        model_signature=signature,
        receivers=(ReceiverRef("w0", "n1", "u0", 0, object()),),
        head_server=object(),
        prepared_digest=f"prepared-{replica_id}",
    )


def _weight(context=None, replica_id="r1"):
    context = context or _ctx()
    return WeightEvidence(
        header=EvidenceHeader(context, _key(replica_id), 1, f"weight-{replica_id}"),
        version=7,
        manifest_digest="manifest-1",
        receivers_digest=f"receivers-{replica_id}",
    )


def _exit(context=None, replica_id="r1", *, drain_id=None):
    context = context or _ctx()
    return ExitEvidence(
        header=EvidenceHeader(context, _key(replica_id), 2, f"exit-{replica_id}"),
        drain_id=drain_id or f"drain-{replica_id}",
        recall_mode=RecallMode.NATURAL,
        quiescence_digest="quiet-1",
        attempts_digest="attempts-1",
    )


def _ce_class():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

    return _isolated_class(
        "checkpoint/checkpoint_engine_manager.py",
        "MultiTaskCheckpointEngineManager",
        Parent,
        ServiceAction=ServiceAction,
        CommitOwner=CommitOwner,
        CommitReceipt=CommitReceipt,
        EvidenceHeader=EvidenceHeader,
        EffectiveReplicaEntry=EffectiveReplicaEntry,
        _digest=_digest,
    )


def _lb_class():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

    return _isolated_class(
        "rollout/load_balancer.py",
        "MultiTaskGlobalRequestLoadBalancer",
        Parent,
        DEFAULT_ROUTING_CACHE_SIZE=128,
        ServiceAction=ServiceAction,
        RouteEntry=RouteEntry,
        RouteState=RouteState,
        CommitOwner=CommitOwner,
        CommitReceipt=CommitReceipt,
        EvidenceHeader=EvidenceHeader,
        DrainTicket=DrainTicket,
        uuid=uuid,
        _digest=_digest,
    )


def _ce_add(context, key, *, version=7):
    return CommitReceipt(EvidenceHeader(context, key, 2, "ce-add"), CommitOwner.CE, ServiceAction.ADD, 1, version, None)


def _ce_remove(context, key):
    return CommitReceipt(EvidenceHeader(context, key, 3, "ce-remove"), CommitOwner.CE, ServiceAction.REMOVE, 2, None, None)


def test_ce_membership_revision_is_monotonic_idempotent_and_minimal():
    manager = _ce_class()()
    context = _ctx()
    prepared = _prepared()
    installed = _weight(context)

    first = manager.add_effective(context, prepared, installed)
    replay = manager.add_effective(context, prepared, installed)
    assert first is replay
    assert first.revision == 1

    entry = manager.effective_replicas[prepared.key]
    assert isinstance(entry, EffectiveReplicaEntry)
    assert entry.receivers == prepared.receivers
    assert entry.loaded_version == installed.version
    assert entry.membership_operation_id == context.operation_id
    assert not hasattr(entry, "model_signature")

    removed = manager.remove_effective(context, prepared.key, _exit(context))
    assert manager.remove_effective(context, prepared.key, _exit(context)) is removed
    assert removed.revision == 2
    assert manager.effective_replicas == {}


def test_ce_remove_refuses_while_parameter_transfer_is_in_flight():
    manager = _ce_class()()
    context = _ctx()
    prepared = _prepared()
    manager.add_effective(context, prepared, _weight(context))
    manager._active_transfers().add(prepared.key)
    with pytest.raises(ValueError, match="transfer is in flight"):
        manager.remove_effective(context, prepared.key, _exit(context))


def test_lb_add_requires_matching_ce_commit_and_writes_typed_route_entry():
    lb = _lb_class()({})
    context = _ctx()
    prepared = _prepared()
    installed = _weight(context)
    ce = _ce_add(context, prepared.key)

    receipt = lb.commit_routable(context, prepared, installed, ce)
    assert receipt.owner is CommitOwner.LB
    assert receipt.route_epoch == 1
    assert lb.commit_routable(context, prepared, installed, ce) is receipt
    route = lb.routes[prepared.key]
    assert route.state is RouteState.ROUTABLE
    assert route.serving_version == 7


def test_lb_route_epoch_is_per_replica_not_global_counter():
    lb = _lb_class()({})
    first_ctx = _ctx("op-r1")
    second_ctx = _ctx("op-r2")
    first = _prepared("r1")
    second = _prepared("r2")
    r1 = lb.commit_routable(first_ctx, first, _weight(first_ctx, "r1"), _ce_add(first_ctx, first.key))
    r2 = lb.commit_routable(second_ctx, second, _weight(second_ctx, "r2"), _ce_add(second_ctx, second.key))
    assert r1.route_epoch == r2.route_epoch == 1


def test_lb_remove_preserves_removed_tombstone_and_attempt_ledger():
    lb = _lb_class()({})
    context = _ctx()
    prepared = _prepared()
    installed = _weight(context)
    lb.commit_routable(context, prepared, installed, _ce_add(context, prepared.key))

    ticket = lb.close_for_exit(context, prepared.key, phase_revision=1, server_admission_epoch=4)
    proof = _exit(context, drain_id=ticket.drain_id)
    ce = _ce_remove(context, proof.header.key)
    lb.attempts[proof.header.key] = {"a1": SimpleNamespace(state="RUNNING")}
    with pytest.raises(ValueError, match="attempts remain unsettled"):
        lb.finish_remove(context, proof, ce)

    lb.attempts[proof.header.key]["a1"] = SimpleNamespace(state="RELEASED")
    receipt = lb.finish_remove(context, proof, ce)
    assert receipt.route_epoch == 3
    tombstone = lb.routes[proof.header.key]
    assert tombstone.state is RouteState.REMOVED
    assert tombstone.serving_version == installed.version
    assert "a1" in lb.attempts[proof.header.key]
