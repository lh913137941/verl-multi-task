"""Canonical lifecycle evidence validation for the 0918 design."""

from dataclasses import FrozenInstanceError

import pytest

from multi_task_scheduler.orchestration.contracts import (
    OperationContext,
    RecallMode,
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
    NeverPublishedProof,
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


def test_weight_evidence_is_compact_owner_verified_proof():
    evidence = WeightEvidence(
        header=_header("weight-1"),
        version=7,
        manifest_digest="manifest-1",
        receivers_digest="receivers-1",
    )
    assert evidence.version == 7
    assert not hasattr(evidence, "receiver_versions")
    assert not hasattr(evidence, "transfer_id")
    with pytest.raises(ValueError, match="receivers_digest"):
        WeightEvidence(_header("bad"), 7, "manifest-1", "")


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
        CommitReceipt(_header("bad-ce"), CommitOwner.CE, ServiceAction.ADD, 1, 7, 1)


def test_exit_evidence_keeps_only_final_drain_digests():
    evidence = ExitEvidence(
        header=_header("exit-1"),
        drain_id="drain-1",
        recall_mode=RecallMode.NATURAL,
        quiescence_digest="quiet-1",
        attempts_digest="attempts-1",
    )
    assert evidence.recall_mode is RecallMode.NATURAL
    assert not hasattr(evidence, "inflight")
    assert not hasattr(evidence, "continuations")


def test_service_evidence_is_combined_digest_not_duplicate_owner_revisions():
    service = ServiceEvidence(
        header=_header("service-add"),
        action=ServiceAction.ADD,
        prerequisite_digest="weight-1",
        service_digest="service-state-1",
    )
    assert service.action is ServiceAction.ADD
    assert not hasattr(service, "ce_revision")
    assert not hasattr(service, "version")


def test_release_evidence_exposes_only_exact_released_gpu_set():
    release = ReleaseEvidence(
        header=_header("release-1"),
        release_kind="BORROWER_RUNTIME_DESTROYED",
        permit_digest="service-remove",
        inventory_digest="inventory-1",
        released_gpu_uuids=("u0", "u1"),
    )
    assert release.released_gpu_uuids == ("u0", "u1")
    assert not hasattr(release, "per_gpu")
    with pytest.raises(ValueError, match="duplicates"):
        ReleaseEvidence(
            header=_header("release-dup"),
            release_kind="DONOR_SLEEP_RELEASED",
            permit_digest="service-remove",
            inventory_digest="inventory-1",
            released_gpu_uuids=("u0", "u0"),
        )


def test_never_published_proof_does_not_expose_cleanup_inventory():
    proof = NeverPublishedProof(
        header=_header("never-published"),
        publication_fence_digest="publication-fence-1",
    )
    assert proof.publication_fence_digest
    assert not hasattr(proof, "inventory")


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
