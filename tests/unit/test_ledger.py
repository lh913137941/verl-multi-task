"""GS ledger rules for the current simplified-design contracts only."""

from dataclasses import replace

import pytest

from multi_task_scheduler.orchestration.contracts import (
    IdleCandidate,
    IdleCandidateReport,
    LeaseAuthorization,
    NodePlacement,
    OperationCommand,
    OperationContext,
    OperationResult,
    PlacementSpec,
    ReplicaKey,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationKind,
    OperationStatus,
    Phase,
)
from multi_task_scheduler.orchestration.replica_record import ReplicaState
from multi_task_scheduler.scheduler.ledger import Ledger, ProtocolInstance, ResourceManifest


def _protocol():
    return ProtocolInstance(
        protocol_version=1,
        runtime_kind="verl-multi-task",
        gs_epoch="gs-1",
        sharing_namespace="ns-1",
    )


def _placement(uuid="u0"):
    node = NodePlacement(
        node_id="n1",
        gpu_uuids=(uuid,),
        physical_gpu_ids=(0,),
        global_ranks=(0,),
        local_ranks=(0,),
    )
    return PlacementSpec(
        node=node,
        tp=1,
        dp=1,
        pp=1,
        model_signature="sig-1",
        placement_digest=f"placement-{uuid}",
    )


def _ctx(**overrides):
    values = dict(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-a",
        task_session="s1",
        operation_id="op-1",
        lease_id="l1",
        lease_epoch=0,
        command_seq=0,
        expected_revision=0,
    )
    values.update(overrides)
    return OperationContext(**values)


def _target(**overrides):
    values = dict(task_session="s1", replica_id="r1", runtime_epoch=0)
    values.update(overrides)
    return ReplicaKey(**values)


def _authorization(kind=OperationKind.ADD, **overrides):
    values = dict(
        lease_id="l1",
        gs_epoch="gs-1",
        donor_session="donor",
        borrower_session="s1",
        placement_digest="placement-u0",
        lease_epoch=0,
        purpose=kind,
        prior_release_digest="release-0" if kind in {OperationKind.ADD, OperationKind.RESTORE} else None,
        authorization_seq=1,
    )
    values.update(overrides)
    return LeaseAuthorization(**values)


def _command(**overrides):
    ctx = overrides.pop("ctx", _ctx())
    target = overrides.pop("target", _target(task_session=ctx.task_session))
    kind = overrides.pop("kind", OperationKind.ADD)
    authorization = overrides.pop(
        "authorization",
        _authorization(
            kind,
            lease_id=ctx.lease_id,
            gs_epoch=ctx.gs_epoch,
            borrower_session=ctx.task_session,
            lease_epoch=ctx.lease_epoch,
        ),
    )
    values = dict(
        ctx=ctx,
        kind=kind,
        target=target,
        authorization=authorization,
        payload_digest="d1",
        remaining_budget_ms=1000,
        placement=_placement() if kind is OperationKind.ADD else None,
    )
    values.update(overrides)
    return OperationCommand(**values)


def _result(command=None, revision=1, phase=Phase.CREATE, status=OperationStatus.RUNNING):
    command = command or _command()
    return OperationResult(
        ctx=command.ctx,
        target=command.target,
        status=status,
        phase=phase,
        phase_revision=revision,
        replica_state=ReplicaState.PREPARING,
    )


def _candidate(replica_id="r1", source_seq=7):
    return IdleCandidate(
        key=ReplicaKey(task_session="s1", replica_id=replica_id, runtime_epoch=0),
        production_epoch=3,
        source_seq=source_seq,
        manager_revision=2,
        lb_revision=4,
        engine_digest=f"engine-{replica_id}",
        reason="CLOSED_BACKPRESSURE",
        stable_idle_ms=500,
        observed_age_ms=10,
        gpu_count=1,
        evidence_digest=f"evidence-{replica_id}-{source_seq}",
        placement_digest="placement-u0",
    )


def _idle_report(source_seq=7, candidates=None, production_revision=3):
    if candidates is None:
        candidates = (_candidate(source_seq=source_seq),)
    return IdleCandidateReport(
        task_session="s1",
        gs_epoch="gs-1",
        lb_session="lb-1",
        source_seq=source_seq,
        production_revision=production_revision,
        valid_for_ms=1000,
        candidates=tuple(candidates),
    )


def test_register_resources_records_only_authoritative_gpu_facts():
    ledger = Ledger(_protocol())
    task = ledger.register_task("task-a", "s1")
    assert ledger.register_resources(
        ResourceManifest(owner_task_session="s1", placement=_placement(), replica_id="r1")
    ) is None
    assert task.initial_gpus == 1
    assert ledger.gpus[("gs-1", "n1", "u0")].native_owner == "s1"
    assert not hasattr(ledger, "native_replicas")


def test_register_resources_rejects_duplicate_or_unknown_owner():
    ledger = Ledger(_protocol())
    with pytest.raises(ValueError, match="unknown owner"):
        ledger.register_resources(
            ResourceManifest(owner_task_session="ghost", placement=_placement(), replica_id="r1")
        )
    ledger.register_task("task-a", "s1")
    ledger.register_resources(
        ResourceManifest(owner_task_session="s1", placement=_placement(), replica_id="r1")
    )
    ledger.register_task("task-b", "s2")
    with pytest.raises(ValueError, match="already registered"):
        ledger.register_resources(
            ResourceManifest(owner_task_session="s2", placement=_placement(), replica_id="r2")
        )


def test_current_user_is_exclusive_and_clearable():
    ledger = Ledger(_protocol())
    ledger.register_task("task-a", "s1")
    ledger.register_resources(
        ResourceManifest(owner_task_session="s1", placement=_placement(), replica_id="r1")
    )
    key = ("gs-1", "n1", "u0")
    ledger.set_current_user(key, "s2", "lease-1", 0)
    with pytest.raises(ValueError, match="already authorized"):
        ledger.set_current_user(key, "s3", "lease-2", 0)
    ledger.clear_current_user(key)
    assert ledger.gpus[key].current_user is None


def test_operation_replay_is_idempotent_only_for_same_business_identity():
    ledger = Ledger(_protocol())
    first = ledger.record_operation(_command())
    second = ledger.record_operation(_command(remaining_budget_ms=10))
    assert first is second

    with pytest.raises(ValueError, match="conflicting replay"):
        ledger.record_operation(_command(payload_digest="other"))

    changed_ctx = _ctx(command_seq=1)
    with pytest.raises(ValueError, match="conflicting replay"):
        ledger.record_operation(_command(ctx=changed_ctx))


def test_result_merge_fences_identity_before_revision_and_rejects_equal_revision_conflict():
    ledger = Ledger(_protocol())
    command = _command()
    ledger.record_operation(command)
    first = _result(command, revision=3)
    ledger.merge_operation_result("op-1", first)
    ledger.merge_operation_result("op-1", _result(command, revision=2))
    assert ledger.operations["op-1"].final_result is first

    with pytest.raises(ValueError, match="conflicting result replay"):
        ledger.merge_operation_result(
            "op-1", _result(command, revision=3, phase=Phase.WAIT_GATE)
        )

    stale_ctx = _ctx(lease_epoch=1)
    stale = OperationResult(
        ctx=stale_ctx,
        target=command.target,
        status=OperationStatus.RUNNING,
        phase=Phase.WAIT_GATE,
        phase_revision=99,
        replica_state=ReplicaState.PREPARING,
    )
    with pytest.raises(ValueError, match="mismatched"):
        ledger.merge_operation_result("op-1", stale)


def test_unknown_result_cannot_create_operation_record():
    ledger = Ledger(_protocol())
    with pytest.raises(ValueError, match="unknown operation"):
        ledger.merge_operation_result("op-1", _result())
    assert ledger.operations == {}


def test_idle_report_replaces_complete_set_and_replay_does_not_extend_ttl():
    ledger = Ledger(_protocol())
    report = _idle_report()
    ledger.upsert_idle_report(report, valid_until=100.0)
    key = report.candidates[0].key
    ledger.upsert_idle_report(report, valid_until=200.0)
    assert ledger.idle_observations[key].valid_until == 100.0

    empty = replace(report, source_seq=8, candidates=())
    ledger.upsert_idle_report(empty, valid_until=101.0)
    ledger.upsert_idle_report(report, valid_until=200.0)
    assert ledger.idle_observations == {}


def test_idle_report_rejects_conflict_stale_revision_and_infinite_ttl():
    ledger = Ledger(_protocol())
    report = _idle_report()
    with pytest.raises(ValueError):
        ledger.upsert_idle_report(report, valid_until=float("inf"))
    ledger.upsert_idle_report(report, valid_until=100.0)
    with pytest.raises(ValueError, match="conflicting"):
        ledger.upsert_idle_report(
            replace(report, candidates=(_candidate("r2", source_seq=7),)),
            valid_until=100.0,
        )
    with pytest.raises(ValueError, match="stale production revision"):
        ledger.upsert_idle_report(
            _idle_report(source_seq=8, production_revision=2), valid_until=100.0
        )
