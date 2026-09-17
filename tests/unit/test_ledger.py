"""GS ledger update rules for simplified design identity and placement contracts."""

from dataclasses import replace

import pytest

from multi_task_scheduler.orchestration.contracts import (
    NodePlacement,
    OperationCommand,
    OperationContext,
    OperationResult,
    PlacementSpec,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationKind,
    OperationStatus,
    Phase,
)
from multi_task_scheduler.scheduler.ledger import (
    IdleReport,
    Ledger,
    ProtocolInstance,
    ResourceManifest,
)


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
        model_signature="sig-1",
        placement_digest=f"placement-{uuid}",
    )


def _command(**overrides):
    values = dict(
        protocol_version=1,
        gs_epoch="gs-1",
        target_task_id="task-a",
        target_task_session="s1",
        operation_id="op-1",
        payload_digest="d1",
        kind=OperationKind.ADD,
        lease_id="l1",
        lease_epoch=0,
        command_seq=0,
        replica_id="r1",
        expected_revision=0,
    )
    values.update(overrides)
    return OperationCommand(**values)


def _result(command=None, revision=1, phase=Phase.CREATE, status=OperationStatus.RUNNING):
    command = command or _command()
    return OperationResult(
        identity_fields=command.context,
        phase=phase,
        phase_revision=revision,
        state=status,
        actual_replica_state="PREPARING",
    )


def test_register_task_creates_initializing_and_updates_session():
    ledger = Ledger(_protocol())
    record = ledger.register_task("task-a", "s1")
    assert record.status == "INITIALIZING"
    again = ledger.register_task("task-a", "s2", config_summary="cfg")
    assert again is not record
    assert ("task-a", "s2") in ledger.tasks


def test_register_resources_records_single_node_native_and_gpus():
    ledger = Ledger(_protocol())
    ledger.register_task("task-a", "s1")
    manifest = ResourceManifest(owner_task_session="s1", placement=_placement(), replica_id="r1")
    replica = ledger.register_resources(manifest)
    assert replica.gpu_uuids == ("u0",)
    assert replica.node.node_id == "n1"
    gpu = ledger.gpus[("gs-1", "n1", "u0")]
    assert gpu.native_owner == "s1"
    assert gpu.current_user is None


def test_register_resources_rejects_duplicate_gpu_uuid():
    ledger = Ledger(_protocol())
    ledger.register_task("task-a", "s1")
    ledger.register_resources(
        ResourceManifest(owner_task_session="s1", placement=_placement("u0"), replica_id="r1")
    )
    ledger.register_task("task-b", "s2")
    with pytest.raises(ValueError, match="already registered"):
        ledger.register_resources(
            ResourceManifest(owner_task_session="s2", placement=_placement("u0"), replica_id="r2")
        )


def test_register_resources_rejects_unknown_owner_session():
    ledger = Ledger(_protocol())
    manifest = ResourceManifest(owner_task_session="ghost", placement=_placement(), replica_id="r1")
    with pytest.raises(ValueError, match="unknown owner"):
        ledger.register_resources(manifest)


def test_current_user_only_changes_via_gs_authorization():
    ledger = Ledger(_protocol())
    ledger.register_task("task-a", "s1")
    ledger.register_resources(
        ResourceManifest(owner_task_session="s1", placement=_placement(), replica_id="r1")
    )
    gkey = ("gs-1", "n1", "u0")
    ledger.set_current_user(gkey, "s2", "lease-1", 0)
    assert ledger.gpus[gkey].current_user == "s2"
    assert ledger.gpus[gkey].state == "LENT"
    with pytest.raises(ValueError, match="already authorized"):
        ledger.set_current_user(gkey, "s3", "lease-2", 0)
    ledger.clear_current_user(gkey)
    assert ledger.gpus[gkey].current_user is None


def test_open_lease_rejects_duplicate():
    from multi_task_scheduler.scheduler.ledger import LeaseRecord

    ledger = Ledger(_protocol())
    ledger.open_lease(LeaseRecord(lease_id="l1", donor_session="s1"))
    with pytest.raises(ValueError, match="already exists"):
        ledger.open_lease(LeaseRecord(lease_id="l1", donor_session="s1"))


def test_idle_observation_upsert_and_staleness():
    ledger = Ledger(_protocol())
    report = IdleReport(
        source_session="s1",
        source_seq=7,
        production_epoch=3,
        candidate_ids=("r1", "r2"),
        candidate_reasons={"r1": "closed"},
    )
    ledger.upsert_idle_observation(report, valid_until=100.0)
    assert len(ledger.idle_observations) == 2
    stale = ledger.stale_idle_observations(now=200.0)
    assert {o.replica_id for o in stale} == {"r1", "r2"}


def test_record_operation_is_idempotent_for_exact_replay():
    ledger = Ledger(_protocol())
    first = ledger.record_operation(_command())
    second = ledger.record_operation(_command(remaining_budget_ms=20))
    assert first is second


@pytest.mark.parametrize(
    "change",
    [
        {"payload_digest": "other"},
        {"command_seq": 1},
        {"lease_epoch": 1},
        {"target_task_session": "s2"},
        {"lease_id": "l2"},
        {"replica_id": "r2"},
        {"expected_revision": 2},
    ],
)
def test_record_operation_rejects_conflicting_replay_identity(change):
    ledger = Ledger(_protocol())
    ledger.record_operation(_command())
    with pytest.raises(ValueError, match="conflicting replay"):
        ledger.record_operation(_command(**change))


def test_merge_operation_result_ignores_older_revision():
    ledger = Ledger(_protocol())
    command = _command()
    ledger.record_operation(command)
    ledger.merge_operation_result("op-1", _result(command, revision=3))
    assert ledger.operations["op-1"].final_result.phase_revision == 3
    ledger.merge_operation_result("op-1", _result(command, revision=2))
    assert ledger.operations["op-1"].final_result.phase_revision == 3
    latest = _result(command, revision=4, phase=Phase.WAIT_GATE)
    ledger.merge_operation_result("op-1", latest)
    assert ledger.operations["op-1"].final_result is latest


def test_equal_revision_conflicting_result_is_rejected_but_exact_replay_is_idempotent():
    ledger = Ledger(_protocol())
    command = _command()
    result = _result(command, revision=3)
    ledger.record_operation(command)
    ledger.merge_operation_result("op-1", result)
    assert ledger.merge_operation_result("op-1", result).final_result is result
    conflicting = _result(command, revision=3, phase=Phase.WAIT_GATE)
    with pytest.raises(ValueError, match="conflicting result replay"):
        ledger.merge_operation_result("op-1", conflicting)


@pytest.mark.parametrize(
    "ctx_change",
    [
        {"task_session": "old-session"},
        {"lease_epoch": 1},
        {"command_seq": 1},
        {"operation_id": "other-op"},
        {"gs_epoch": "old-gs"},
    ],
)
def test_merge_operation_result_rejects_mismatched_identity_before_revision(ctx_change):
    ledger = Ledger(_protocol())
    command = _command()
    ledger.record_operation(command)
    values = dict(
        protocol_version=command.protocol_version,
        gs_epoch=command.gs_epoch,
        task_id=command.target_task_id,
        task_session=command.target_task_session,
        operation_id=command.operation_id,
        lease_id=command.lease_id,
        lease_epoch=command.lease_epoch,
        command_seq=command.command_seq,
        expected_revision=command.expected_revision,
    )
    values.update(ctx_change)
    result = OperationResult(
        identity_fields=OperationContext(**values),
        phase=Phase.WAIT_GATE,
        phase_revision=99,
        state=OperationStatus.RUNNING,
        actual_replica_state="PREPARING",
    )
    with pytest.raises(ValueError):
        ledger.merge_operation_result("op-1", result)


def test_empty_idle_report_replaces_old_candidates_and_old_reports_cannot_restore_them():
    ledger = Ledger(_protocol())
    first = IdleReport("s1", 7, 3, ("r1",))
    ledger.upsert_idle_observation(first, valid_until=100.0)
    ledger.upsert_idle_observation(replace(first, source_seq=8, candidate_ids=()), valid_until=101.0)
    ledger.upsert_idle_observation(first, valid_until=200.0)
    assert ledger.idle_observations == {}


def test_replayed_idle_report_does_not_extend_ttl():
    ledger = Ledger(_protocol())
    report = IdleReport("s1", 7, 3, ("r1",))
    ledger.upsert_idle_observation(report, valid_until=100.0)
    ledger.upsert_idle_observation(report, valid_until=200.0)
    assert ledger.idle_observations[("s1", "r1")].valid_until == 100.0
    assert len(ledger.stale_idle_observations(100.0)) == 1


def test_conflicting_idle_replay_is_rejected_and_infinite_lifetime_is_rejected():
    ledger = Ledger(_protocol())
    report = IdleReport("s1", 7, 3, ("r1",))
    with pytest.raises(ValueError):
        ledger.upsert_idle_observation(report, valid_until=float("inf"))
    ledger.upsert_idle_observation(report, valid_until=100.0)
    with pytest.raises(ValueError):
        ledger.upsert_idle_observation(replace(report, candidate_ids=("r2",)), valid_until=100.0)


def test_placement_validation_happens_before_resource_registration():
    with pytest.raises(ValueError, match="model_signature"):
        replace(_placement(), model_signature="")


def test_unknown_result_cannot_create_an_operation_record():
    ledger = Ledger(_protocol())
    result = _result()
    with pytest.raises(ValueError):
        ledger.merge_operation_result("op-1", result)
    assert ledger.operations == {}
