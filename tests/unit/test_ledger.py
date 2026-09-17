"""GS ledger update rules (section 3.3)."""

import pytest
from dataclasses import replace

from multi_task_scheduler.orchestration.contracts import (
    GpuPlacement,
    NodeBlock,
    PlacementSpec,
)
from multi_task_scheduler.orchestration.operation_journal import OperationType
from multi_task_scheduler.scheduler.ledger import (
    IdleReport,
    Ledger,
    ProtocolInstance,
    ResourceManifest,
)


def _protocol():
    return ProtocolInstance(
        protocol_version="p1",
        runtime_kind="verl-multi-task",
        gs_epoch=1,
        sharing_namespace="ns-1",
    )


def _placement(uuid="u0"):
    block = NodeBlock(
        node_id="n1",
        gpus=(GpuPlacement(gpu_uuid=uuid, physical_id=0, global_rank=0, local_rank=0),),
    )
    return PlacementSpec(node_blocks=(block,), model_signature="sig-1")


def test_register_task_creates_initializing_and_updates_session():
    ledger = Ledger(_protocol())
    record = ledger.register_task("task-a", "s1")
    assert record.status == "INITIALIZING"
    assert record.task_session == "s1"
    again = ledger.register_task("task-a", "s2", config_summary="cfg")
    assert again is not record  # new session is a new record
    assert ("task-a", "s2") in ledger.tasks


def test_register_resources_records_native_and_gpus():
    ledger = Ledger(_protocol())
    ledger.register_task("task-a", "s1")
    manifest = ResourceManifest(owner_task_session="s1", placement=_placement(), replica_id="r1")
    replica = ledger.register_resources(manifest)
    assert replica.gpu_uuids == ("u0",)
    gpu = ledger.gpus[(1, "n1", "u0")]
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
    gkey = (1, "n1", "u0")
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


def test_record_operation_is_idempotent_and_rejects_conflicting_digest():
    from multi_task_scheduler.orchestration.contracts import Command

    ledger = Ledger(_protocol())

    def command(digest="d1", op_id="op-1"):
        return Command(
            protocol_version="p1", gs_epoch=1, target_task_id="task-a",
            target_task_session="s1", operation_id=op_id, payload_digest=digest,
            kind=OperationType.ADD, lease_id="l1", lease_epoch=0,
            command_seq=0, replica_id="r1",
        )

    first = ledger.record_operation(command())
    second = ledger.record_operation(command())
    assert first is second
    with pytest.raises(ValueError, match="conflicting payload digest"):
        ledger.record_operation(command(digest="other"))


def test_merge_operation_result_ignores_older_revision():
    from multi_task_scheduler.orchestration.contracts import OperationContext, OperationResult

    ledger = Ledger(_protocol())
    ledger.operations["op-1"] = ledger.record_operation(
        type(
            "C", (), {
                "operation_id": "op-1", "payload_digest": "d1", "command_seq": 0,
            },
        )()
    )

    def result(revision, state="running"):
        ctx = OperationContext(
            protocol_version="p1", gs_epoch=1, task_id="task-a", task_session="s1",
            operation_id="op-1", lease_id="l1", lease_epoch=0, command_seq=0,
        )
        return OperationResult(
            identity_fields=ctx, phase="applying", phase_revision=revision,
            state=state, actual_replica_state="bootstrapping",
        )

    ledger.merge_operation_result("op-1", result(3))
    assert ledger.operations["op-1"].final_result.phase_revision == 3
    ledger.merge_operation_result("op-1", result(2))  # stale -> ignored
    assert ledger.operations["op-1"].final_result.phase_revision == 3
    ledger.merge_operation_result("op-1", result(4, state="committed"))
    assert ledger.operations["op-1"].final_result.phase_revision == 4


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


def test_duplicate_uuid_inside_one_registration_is_atomic_failure():
    ledger = Ledger(_protocol())
    ledger.register_task("task-a", "s1")
    placement = _placement()
    duplicated = replace(placement, node_blocks=(replace(placement.node_blocks[0],
        gpus=placement.node_blocks[0].gpus * 2),))
    with pytest.raises(ValueError):
        ledger.register_resources(ResourceManifest("s1", duplicated, "r1"))
    assert ledger.gpus == {}
    assert ledger.native_replicas == {}


def test_unknown_owner_with_empty_model_signature_is_still_rejected():
    ledger = Ledger(_protocol())
    with pytest.raises(ValueError, match="unknown owner"):
        ledger.register_resources(ResourceManifest("ghost", replace(_placement(), model_signature=""), "r1"))


def test_unknown_result_cannot_create_an_operation_record():
    from multi_task_scheduler.orchestration.contracts import OperationContext, OperationResult
    ledger = Ledger(_protocol())
    ctx = OperationContext("p1", 1, "task-a", "s1", "op-1", "l1", 0, 0)
    result = OperationResult(ctx, "APPLYING", 1, "COMMITTED", "ACTIVE")
    with pytest.raises(ValueError):
        ledger.merge_operation_result("op-1", result)
    assert ledger.operations == {}
