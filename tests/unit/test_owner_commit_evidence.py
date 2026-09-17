"""Owner revision and CommitReceipt regression tests for CE and LB overlays."""

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_task_scheduler.orchestration.contracts import (
    OperationContext,
    PreparedReplica,
    ReceiverRef,
    ReplicaKey,
    ServiceAction,
)
from multi_task_scheduler.orchestration.receipts import (
    CommitOwner,
    CommitReceipt,
    EvidenceHeader,
    ExitEvidence,
    WeightEvidence,
)
from multi_task_scheduler.orchestration.contracts import RecallMode

SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"


def _digest(*parts):
    return hashlib.sha256("|".join(repr(p) for p in parts).encode()).hexdigest()


@dataclass(frozen=True)
class _EffectiveReplicaEntry:
    key: ReplicaKey
    receivers: tuple[ReceiverRef, ...]
    loaded_version: int
    model_signature: str
    membership_operation_id: str


def _isolated_class(relative, name, parent, **globals_for_test):
    path = SOURCE / relative
    parsed = ast.parse(path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    scope = {"TestParent": parent, **globals_for_test}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def _ctx(operation_id="op-1"):
    return OperationContext(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-a",
        task_session="s1",
        operation_id=operation_id,
        lease_id="l1",
        lease_epoch=0,
        command_seq=0,
    )


def _key(replica_id="r1"):
    return ReplicaKey(task_session="s1", replica_id=replica_id, runtime_epoch=0)


def _prepared(context=None, replica_id="r1"):
    context = context or _ctx()
    return PreparedReplica(
        key=_key(replica_id),
        ctx=context,
        placement_digest="placement-1",
        model_signature="sig",
        receivers=(ReceiverRef("w0", "n1", "u0", 0, object()),),
        head_server=object(),
        manager_revision=1,
    )


def _weight(context=None, replica_id="r1"):
    context = context or _ctx()
    return WeightEvidence(
        header=EvidenceHeader(context, _key(replica_id), 1, f"weight-{replica_id}"),
        transfer_id=f"transfer-{replica_id}",
        snapshot_id="snapshot-1",
        manifest_digest="manifest-1",
        version=7,
        receiver_versions={"w0": 7},
        device_complete=True,
        temporary_topology_clean=True,
    )


def _exit(context=None, replica_id="r1", *, drain_id=None):
    context = context or _ctx()
    return ExitEvidence(
        header=EvidenceHeader(context, _key(replica_id), 2, f"exit-{replica_id}"),
        drain_id=drain_id or f"drain-{replica_id}",
        recall_mode=RecallMode.NATURAL,
        inflight=0,
        admitting=0,
        queued=0,
        running=0,
        pending_admissions=0,
        closed_admission=True,
        all_backends_confirmed=True,
        lb_revision=1,
        observed_age_ms=5,
        engine_digest="engine-1",
        continuations=(),
        unresolved_count=0,
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
        EffectiveReplicaEntry=_EffectiveReplicaEntry,
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
        CommitOwner=CommitOwner,
        CommitReceipt=CommitReceipt,
        EvidenceHeader=EvidenceHeader,
        DrainTicket=object,
        _digest=_digest,
    )


def test_ce_membership_revision_is_monotonic_idempotent_and_typed():
    manager = _ce_class()()
    context = _ctx()
    prepared = _prepared(context)
    installed = _weight(context)

    first = manager.add_effective(context, prepared, installed)
    replay = manager.add_effective(context, prepared, installed)
    assert first is replay
    assert first.owner is CommitOwner.CE
    assert first.action is ServiceAction.ADD
    assert first.revision == 1

    entry = manager.effective_replicas[prepared.key]
    assert entry.key == prepared.key
    assert entry.receivers == prepared.receivers
    assert entry.loaded_version == installed.version
    assert entry.model_signature == prepared.model_signature
    assert entry.membership_operation_id == context.operation_id
    assert not hasattr(entry, "head_server")
    assert not hasattr(entry, "ctx")

    removed = manager.remove_effective(context, prepared.key, _exit(context))
    replay_removed = manager.remove_effective(context, prepared.key, _exit(context))
    assert removed is replay_removed
    assert removed.action is ServiceAction.REMOVE
    assert removed.revision == 2
    assert manager.effective_revision == 2
    assert manager.effective_replicas == {}


def test_ce_rejects_model_signature_mismatch_inside_effective_set():
    manager = _ce_class()()
    first_ctx = _ctx("op-1")
    first = _prepared(first_ctx, "r1")
    manager.add_effective(first_ctx, first, _weight(first_ctx, "r1"))

    second_ctx = _ctx("op-2")
    second = PreparedReplica(
        key=_key("r2"),
        ctx=second_ctx,
        placement_digest="placement-2",
        model_signature="other-sig",
        receivers=(ReceiverRef("w0", "n1", "u1", 0, object()),),
        head_server=object(),
        manager_revision=2,
    )
    with pytest.raises(ValueError, match="model signature conflicts"):
        manager.add_effective(second_ctx, second, _weight(second_ctx, "r2"))


def test_lb_add_requires_matching_ce_commit_and_returns_typed_receipt():
    lb = _lb_class()({})
    context = _ctx()
    prepared = _prepared(context)
    installed = _weight(context)
    ce = CommitReceipt(
        header=EvidenceHeader(context, prepared.key, 2, "ce-add"),
        owner=CommitOwner.CE,
        action=ServiceAction.ADD,
        revision=1,
        version=7,
        route_epoch=None,
    )
    receipt = lb.commit_routable(context, prepared, installed, ce)
    assert receipt.owner is CommitOwner.LB
    assert receipt.action is ServiceAction.ADD
    assert receipt.version == 7
    assert receipt.route_epoch == 1
    assert lb.commit_routable(context, prepared, installed, ce) is receipt


def test_lb_remove_requires_matching_active_drain_and_settled_attempts():
    lb = _lb_class()({})
    context = _ctx()
    proof = _exit(context)
    ce = CommitReceipt(
        header=EvidenceHeader(context, proof.header.key, 3, "ce-remove"),
        owner=CommitOwner.CE,
        action=ServiceAction.REMOVE,
        revision=2,
        version=None,
        route_epoch=None,
    )

    with pytest.raises(ValueError, match="active drain ticket"):
        lb.finish_remove(context, proof, ce)

    drain_key = (context.identity, proof.header.key)
    lb.draining[drain_key] = SimpleNamespace(
        drain_id="wrong-drain",
        header=SimpleNamespace(phase_revision=1),
    )
    with pytest.raises(ValueError, match="drain_id"):
        lb.finish_remove(context, proof, ce)

    lb.draining[drain_key] = SimpleNamespace(
        drain_id=proof.drain_id,
        header=SimpleNamespace(phase_revision=1),
    )
    lb.attempts[proof.header.key] = {"a1": SimpleNamespace(state="RUNNING")}
    with pytest.raises(ValueError, match="attempts remain unsettled"):
        lb.finish_remove(context, proof, ce)

    lb.attempts[proof.header.key]["a1"] = SimpleNamespace(state="RELEASED")
    receipt = lb.finish_remove(context, proof, ce)
    assert receipt.owner is CommitOwner.LB
    assert receipt.action is ServiceAction.REMOVE
    assert receipt.version is None
    assert receipt.route_epoch == 1
