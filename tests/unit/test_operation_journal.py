"""OperationJournal replay fencing with full accepted commands."""

from dataclasses import replace

import pytest

from multi_task_scheduler.orchestration.contracts import (
    LeaseAuthorization,
    NodePlacement,
    OperationCommand,
    OperationContext,
    PlacementSpec,
    ReplicaKey,
)
from multi_task_scheduler.orchestration.operation_journal import (
    ExpiredLeaseError,
    IllegalOperationTransitionError,
    OperationIdentityError,
    OperationJournal,
    OperationKind,
    OperationStatus,
    Phase,
)


def _placement():
    return PlacementSpec(
        node=NodePlacement(
            node_id="n1",
            gpu_uuids=("u0",),
            physical_gpu_ids=(0,),
            global_ranks=(0,),
            local_ranks=(0,),
        ),
        tp=1,
        dp=1,
        pp=1,
        model_signature="sig",
        placement_digest="placement-u0",
    )


def _command(**ctx_overrides):
    values = dict(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-a",
        task_session="s1",
        operation_id="op-1",
        lease_id="l1",
        lease_epoch=0,
        command_seq=0,
    )
    values.update(ctx_overrides)
    ctx = OperationContext(**values)
    target = ReplicaKey(task_session=ctx.task_session, replica_id="r1", runtime_epoch=0)
    return OperationCommand(
        ctx=ctx,
        kind=OperationKind.ADD,
        target=target,
        authorization=LeaseAuthorization(
            lease_id=ctx.lease_id,
            gs_epoch=ctx.gs_epoch,
            donor_session="donor",
            borrower_session=ctx.task_session,
            placement_digest="placement-u0",
            lease_epoch=ctx.lease_epoch,
            purpose=OperationKind.ADD,
            prior_release_digest="release-0",
            authorization_seq=1,
        ),
        payload_digest="payload-1",
        remaining_budget_ms=1000,
        placement=_placement(),
    )


def test_begin_stores_full_command_and_returns_validate_accepted():
    journal = OperationJournal()
    command = _command()
    record = journal.begin(command)
    assert record.command is command
    assert record.operation_id == "op-1"
    assert record.target == command.target
    assert record.phase is Phase.VALIDATE
    assert record.status is OperationStatus.ACCEPTED


def test_retry_may_reduce_transport_budget_but_not_change_business_identity():
    journal = OperationJournal()
    command = _command()
    first = journal.begin(command)
    retry = replace(command, remaining_budget_ms=100)
    assert journal.begin(retry) is first
    assert first.command is command


def test_conflicting_command_identity_is_rejected():
    journal = OperationJournal()
    command = _command()
    journal.begin(command)
    with pytest.raises(OperationIdentityError):
        journal.begin(replace(command, payload_digest="other"))
    with pytest.raises(OperationIdentityError):
        journal.begin(replace(command, target=replace(command.target, replica_id="r2")))
    with pytest.raises(OperationIdentityError):
        journal.begin(replace(command, kind=OperationKind.REMOVE, placement=None))


def test_operation_id_cannot_cross_lease_epoch():
    journal = OperationJournal()
    journal.begin(_command(lease_epoch=1))
    with pytest.raises(ExpiredLeaseError):
        journal.begin(_command(lease_epoch=0))
    with pytest.raises(OperationIdentityError):
        journal.begin(_command(lease_epoch=2))


def test_add_happy_path_requires_explicit_terminal_status():
    journal = OperationJournal()
    journal.begin(_command())
    for phase in (
        Phase.CREATE,
        Phase.WAIT_GATE,
        Phase.LOAD_WEIGHTS,
        Phase.JOIN_CE,
        Phase.COMMIT_SERVICE,
    ):
        journal.transition("op-1", phase)
    with pytest.raises(IllegalOperationTransitionError, match="explicit"):
        journal.transition("op-1", Phase.DONE)
    journal.transition("op-1", Phase.DONE, status=OperationStatus.SUCCEEDED)
    record = journal.query("op-1")
    assert record.phase is Phase.DONE
    assert record.status is OperationStatus.SUCCEEDED


def test_terminal_status_cannot_appear_before_done():
    journal = OperationJournal()
    journal.begin(_command())
    with pytest.raises(IllegalOperationTransitionError, match="requires phase DONE"):
        journal.transition("op-1", Phase.CREATE, status=OperationStatus.FAILED)


def test_reconcile_is_unknown_and_can_finish_unknown():
    journal = OperationJournal()
    journal.begin(_command())
    journal.transition("op-1", Phase.RECONCILE)
    assert journal.query("op-1").status is OperationStatus.UNKNOWN
    journal.transition("op-1", Phase.DONE, status=OperationStatus.UNKNOWN)
    assert journal.query("op-1").status is OperationStatus.UNKNOWN


def test_illegal_phase_skip_and_wrong_lease_require_raise():
    journal = OperationJournal()
    journal.begin(_command())
    with pytest.raises(IllegalOperationTransitionError):
        journal.transition("op-1", Phase.COMMIT_SERVICE)
    with pytest.raises(ExpiredLeaseError):
        journal.require("op-1", 1)


def test_phase_result_timing_error_and_detail_are_revisioned():
    journal = OperationJournal()
    journal.begin(_command())
    proof = object()
    error = object()
    record = journal.transition(
        "op-1",
        Phase.CREATE,
        phase_result=proof,
        elapsed_ms=12,
        error=error,
        detail={"step": "created"},
    )
    assert record.phase_results[Phase.CREATE] is proof
    assert record.phase_timings_ms[Phase.CREATE] == 12
    assert record.error is error
    assert record.detail == {"step": "created"}
    assert record.phase_revision == 1
