"""Canonical lifecycle evidence validation for the current design."""

from dataclasses import FrozenInstanceError

import pytest

from multi_task_scheduler.orchestration.contracts import (
    OperationContext,
    RecallMode,
    ReleaseKind,
    ReplicaKey,
    ServiceAction,
)
from multi_task_scheduler.orchestration.receipts import (
    Ack,
    AdmissionSnapshot,
    CommitOwner,
    CommitReceipt,
    EvidenceHeader,
    ExitEvidence,
    GPURelease,
    ReleaseEvidence,
    ServiceEvidence,
    WeightEvidence,
)


def _ctx():
    return OperationContext(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-a",
        task_session="s1",
        operation_id="op-1",
        lease_id="l1",
        lease_epoch=0,
        command_seq=0,
    )


def _key():
    return ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)


def _header(digest="digest-1", revision=1):
    return EvidenceHeader(ctx=_ctx(), key=_key(), phase_revision=revision, digest=digest)


def test_evidence_header_is_frozen_and_fenced_to_task_session():
    header = _header()
    with pytest.raises(FrozenInstanceError):
        header.digest = "other"
    with pytest.raises(ValueError, match="task_session"):
        EvidenceHeader(
            ctx=_ctx(),
            key=ReplicaKey(task_session="s2", replica_id="r1", runtime_epoch=0),
            phase_revision=1,
            digest="d",
        )


def test_weight_evidence_requires_all_receivers_at_same_version_and_device_completion():
    evidence = WeightEvidence(
        header=_header("weight-1"),
        transfer_id="transfer-1",
        snapshot_id="snapshot-1",
        manifest_digest="manifest-1",
        version=7,
        receiver_versions={"w0": 7, "w1": 7},
        device_complete=True,
        temporary_topology_clean=True,
    )
    assert evidence.version == 7
    with pytest.raises(ValueError, match="every receiver"):
        WeightEvidence(
            header=_header("weight-2"),
            transfer_id="transfer-2",
            snapshot_id="snapshot-1",
            manifest_digest="manifest-1",
            version=7,
            receiver_versions={"w0": 6},
            device_complete=True,
            temporary_topology_clean=True,
        )


def test_commit_receipt_owner_action_shape_is_strict():
    ce = CommitReceipt(
        header=_header("ce-add"),
        owner=CommitOwner.CE,
        action=ServiceAction.ADD,
        revision=2,
        version=7,
        route_epoch=None,
    )
    lb = CommitReceipt(
        header=_header("lb-add"),
        owner=CommitOwner.LB,
        action=ServiceAction.ADD,
        revision=3,
        version=7,
        route_epoch=4,
    )
    assert ce.owner is CommitOwner.CE
    assert lb.route_epoch == 4
    with pytest.raises(ValueError, match="CE commit"):
        CommitReceipt(
            header=_header("bad-ce"),
            owner=CommitOwner.CE,
            action=ServiceAction.ADD,
            revision=1,
            version=7,
            route_epoch=1,
        )


def test_natural_exit_requires_zero_counts_closed_admission_and_no_continuations():
    evidence = ExitEvidence(
        header=_header("exit-1"),
        drain_id="drain-1",
        recall_mode=RecallMode.NATURAL,
        inflight=0,
        admitting=0,
        queued=0,
        running=0,
        pending_admissions=0,
        closed_admission=True,
        all_backends_confirmed=True,
        lb_revision=5,
        observed_age_ms=10,
        engine_digest="engine-1",
        continuations=(),
        unresolved_count=0,
    )
    assert evidence.recall_mode is RecallMode.NATURAL
    with pytest.raises(ValueError, match="counters"):
        ExitEvidence(
            header=_header("exit-bad"),
            drain_id="drain-1",
            recall_mode=RecallMode.NATURAL,
            inflight=1,
            admitting=0,
            queued=0,
            running=0,
            pending_admissions=0,
            closed_admission=True,
            all_backends_confirmed=True,
            lb_revision=5,
            observed_age_ms=10,
            engine_digest="engine-1",
            continuations=(),
            unresolved_count=0,
        )


def test_service_evidence_distinguishes_add_and_remove_prerequisites():
    add = ServiceEvidence(
        header=_header("service-add"),
        action=ServiceAction.ADD,
        version=7,
        ce_revision=2,
        lb_revision=3,
        route_epoch=4,
        capacity_revision=5,
        manager_revision=6,
        ce_commit_digest="ce-add",
        lb_commit_digest="lb-add",
        prerequisite_digest="weight-1",
    )
    remove = ServiceEvidence(
        header=_header("service-remove"),
        action=ServiceAction.REMOVE,
        version=None,
        ce_revision=7,
        lb_revision=8,
        route_epoch=9,
        capacity_revision=10,
        manager_revision=11,
        ce_commit_digest="ce-remove",
        lb_commit_digest="lb-remove",
        prerequisite_digest="exit-1",
    )
    assert add.version == 7
    assert remove.version is None
    with pytest.raises(ValueError, match="REMOVE"):
        ServiceEvidence(
            header=_header("bad-remove"),
            action=ServiceAction.REMOVE,
            version=7,
            ce_revision=1,
            lb_revision=1,
            route_epoch=1,
            capacity_revision=1,
            manager_revision=1,
            ce_commit_digest="ce",
            lb_commit_digest="lb",
            prerequisite_digest="exit",
        )


def test_release_evidence_requires_per_gpu_backend_confirmation_and_destroy_has_no_owned_processes():
    gpu = GPURelease(
        gpu_uuid="u0",
        free_hbm_bytes=1024,
        residual_hbm_bytes=0,
        owned_processes=(),
        unknown_processes=(),
        device_work_complete=True,
        meets_release_budget=True,
    )
    release = ReleaseEvidence(
        header=_header("release-1"),
        release_kind=ReleaseKind.BORROWER_RUNTIME_DESTROYED,
        permit_digest="service-remove",
        inventory_digest="inventory-1",
        per_gpu=(gpu,),
        all_backends_confirmed=True,
        observation_interval_ms=100,
    )
    assert release.release_kind is ReleaseKind.BORROWER_RUNTIME_DESTROYED
    with pytest.raises(ValueError, match="per-GPU"):
        ReleaseEvidence(
            header=_header("release-empty"),
            release_kind=ReleaseKind.DONOR_SLEEP_RELEASED,
            permit_digest="service-remove",
            inventory_digest="inventory-1",
            per_gpu=(),
            all_backends_confirmed=True,
            observation_interval_ms=100,
        )


def test_ack_and_admission_snapshot_are_local_results_not_lifecycle_evidence():
    ack = Ack(accepted=True, revision=3)
    snapshot = AdmissionSnapshot(
        scope="REPLICA_DRAIN",
        epoch=4,
        key=_key(),
        closed=True,
        admitting=0,
        all_backends_confirmed=True,
    )
    assert ack.accepted is True
    assert snapshot.closed is True
