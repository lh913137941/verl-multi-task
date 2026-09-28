"""Shared GroupScheduler actor for the minimal 092203 lease/command contract."""

from __future__ import annotations

import time

import ray
from ray.actor import ActorHandle

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
    ReplicaKey,
    ReplicaKind,
)

RUNTIME_KIND = "verl-multi-task:experimental_fully_async_standalone:092203-r2"
_RELEASE_KINDS = {OperationKind.DONATE, OperationKind.REMOVE}


@ray.remote(num_cpus=0)
class GroupScheduler:
    def __init__(self) -> None:
        self.task_runners: dict[str, ActorHandle] = {}
        self.leases: dict[str, Lease] = {}
        self.idle_reports: dict[str, object] = {}
        # GS keeps intent so later RELEASED evidence can be bound back to the
        # exact operation and lease instead of accepting a bare GPU list.
        self.operation_commands: dict[str, OperationCommand] = {}
        self.borrower_targets: dict[str, ReplicaKey] = {}
        # advance_lease completion ledger.  RELEASED closes DONATE/REMOVE
        # stages; RESTORE uses SERVICE_COMMITTED to close the one-shot lease
        # cycle without adding another GS interface.
        self.release_evidence: dict[tuple[str, str], OperationEvidence] = {}
        self.release_history: dict[str, list[str]] = {}
        # Internal derived phase: donor RELEASED has made the reserved claims
        # safe for the borrower ADD, but they remain owned by this borrower
        # lease until borrowed REMOVE produces its own RELEASED evidence.
        self.handoff_ready_leases: set[str] = set()
        # claim_id is globally unique for the lifetime of this GS ledger.
        # Bundle/GPU ownership is active-only and is released only by verified
        # RELEASED evidence.
        self.claim_id_owner: dict[str, str] = {}
        self.active_bundle_owner: dict[tuple[str, int], str] = {}
        self.active_gpu_owner: dict[str, str] = {}

    def runtime_kind(self) -> str:
        return RUNTIME_KIND

    def attach_task(self, task_id: str, task_runner: ActorHandle) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(task_runner, ActorHandle):
            raise TypeError("task_runner must be a real Ray ActorHandle")
        existing = self.task_runners.get(task_id)
        if existing is not None and existing != task_runner:
            raise ValueError(f"TaskRunner already attached for {task_id}")
        self.task_runners[task_id] = task_runner

    def detach_task(self, task_id: str) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty string")
        self.task_runners.pop(task_id, None)
        self.idle_reports.pop(task_id, None)

    def get_task_runners(self) -> dict[str, ActorHandle]:
        return dict(self.task_runners)

    def submit_idle_report(self, report):
        if not isinstance(report, dict):
            raise TypeError("submit_idle_report requires a metadata dict")
        task_session = report.get("task_session")
        candidates = report.get("candidates")
        if not isinstance(task_session, str) or not task_session:
            raise ValueError("idle report requires task_session")
        if task_session not in self.task_runners:
            raise ValueError("idle report references a detached task_session")
        if not isinstance(candidates, (tuple, list)):
            raise ValueError("idle report requires candidates")

        normalized = []
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise TypeError("idle candidate must be a metadata dict")
            key = candidate.get("replica_key")
            if not isinstance(key, ReplicaKey):
                raise TypeError("idle candidate requires ReplicaKey")
            if key.task_session != task_session:
                raise ValueError("idle candidate belongs to another task_session")
            kind = ReplicaKind(candidate.get("kind"))
            if key in seen:
                raise ValueError("idle report contains duplicate ReplicaKey")
            seen.add(key)
            normalized.append({"replica_key": key, "kind": kind.value})

        self.idle_reports[task_session] = {
            "observed_at": time.monotonic(),
            "candidates": tuple(normalized),
        }
        return {"accepted": True, "candidate_count": len(normalized)}

    def submit_operation(self, command: OperationCommand) -> OperationRecord:
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires OperationCommand")
        lease = self.leases.get(command.lease_id)
        if lease is None:
            raise ValueError(f"unknown lease {command.lease_id!r}")

        previous = self.operation_commands.get(command.operation_id)
        if previous is not None and previous != command:
            raise ValueError("conflicting operation replay at GroupScheduler")
        if previous is None:
            history = self.release_history.get(command.lease_id, ())
            if history:
                last_command = self.operation_commands.get(history[-1])
                if (
                    last_command is not None
                    and last_command.kind is OperationKind.RESTORE
                ):
                    raise ValueError(
                        "lease lifecycle is complete after RESTORE; open a new lease"
                    )

            expired = bool(lease.expires_at and time.time() >= lease.expires_at)
            if expired and command.kind is OperationKind.ADD:
                raise ValueError(
                    f"expired lease {command.lease_id!r} cannot start ADD"
                )

            donor_task_id = lease.claims[0]["donor_task_id"]
            if command.kind is OperationKind.DONATE:
                if history:
                    last_command = self.operation_commands.get(history[-1])
                    if (
                        last_command is not None
                        and last_command.kind is OperationKind.REMOVE
                    ):
                        raise ValueError(
                            "lease awaits RESTORE after borrowed REMOVE"
                        )
                if command.target.task_session != donor_task_id:
                    raise ValueError("DONATE target does not own the lease claims")
                if not self._target_matches_donor(command.target, lease):
                    raise ValueError("DONATE target does not match the lease donor replica")
                if command.lease_id in self.handoff_ready_leases:
                    raise ValueError("lease handoff is already ready")
            elif command.kind is OperationKind.ADD:
                if command.lease_id not in self.handoff_ready_leases:
                    raise ValueError(
                        f"{command.kind.value} requires donor RELEASED handoff"
                    )
                existing_target = self.borrower_targets.get(command.lease_id)
                if existing_target is not None:
                    raise ValueError(
                        "lease already has a borrower ADD; replay the original operation_id"
                    )
            elif command.kind is OperationKind.REMOVE:
                if command.lease_id not in self.handoff_ready_leases:
                    raise ValueError("REMOVE requires donor RELEASED handoff")
                expected_target = self.borrower_targets.get(command.lease_id)
                if expected_target is None or expected_target != command.target:
                    raise ValueError("REMOVE target does not match the borrowed replica")
            elif command.kind is OperationKind.RESTORE:
                if command.target.task_session != donor_task_id:
                    raise ValueError("RESTORE target does not own the lease claims")
                if not self._target_matches_donor(command.target, lease):
                    raise ValueError(
                        "RESTORE target does not match the lease donor replica"
                    )
                gpu_owners = tuple(
                    self.active_gpu_owner.get(gpu_uuid)
                    for gpu_uuid in lease.gpu_uuids
                )
                bundle_owners = tuple(
                    self.active_bundle_owner.get(bundle_key)
                    for bundle_key in lease.bundle_keys
                )
                if any(owner is not None for owner in gpu_owners + bundle_owners):
                    raise ValueError(
                        "RESTORE requires donor GPU/bundle claims to be fully returned and unclaimed"
                    )

        task_runner = self.task_runners.get(command.target.task_session)
        if task_runner is None:
            raise ValueError(
                "no TaskRunner is attached under target.task_session; "
                "first release uses the attached task id as task_session"
            )

        # Stage GS intent before dispatch. TaskRunner launches lifecycle work
        # before submit_operation() returns, and that worker may call
        # advance_lease() immediately. Recording after the RPC creates a race
        # where valid RELEASED/SERVICE_COMMITTED evidence appears "unknown".
        staged_command = previous is None
        staged_borrower = False
        staged_restore_reservation = False
        if staged_command:
            self.operation_commands[command.operation_id] = command
            if command.kind is OperationKind.ADD:
                if self.borrower_targets.get(command.lease_id) is None:
                    self.borrower_targets[command.lease_id] = command.target
                    staged_borrower = True
            elif command.kind is OperationKind.RESTORE:
                # The phase validation above already proved every physical
                # claim unowned. GroupScheduler is a single-writer actor, so
                # stage the reservation only after that complete precheck;
                # avoid check-and-mutate loops that can leave partial state.
                for bundle_key in lease.bundle_keys:
                    self.active_bundle_owner[bundle_key] = command.lease_id
                for gpu_uuid in lease.gpu_uuids:
                    self.active_gpu_owner[gpu_uuid] = command.lease_id
                staged_restore_reservation = True

        try:
            result = ray.get(
                task_runner.submit_operation.remote(command, lease=lease),
                timeout=30,
            )
            if not isinstance(result, OperationRecord):
                raise TypeError("TaskRunner returned a non-OperationRecord")
            if result.operation_id != command.operation_id:
                raise ValueError("TaskRunner returned a record for another operation")
            return result
        except BaseException:
            # Submission failure can be ambiguous (for example a timeout after
            # TaskRunner already journaled/launched the worker). Reconcile with
            # the existing query API and roll back GS staging only when the
            # TaskRunner authoritatively says the operation was never journaled.
            definitely_unaccepted = False
            if staged_command:
                try:
                    observed = ray.get(
                        task_runner.query_operation.remote(command.operation_id),
                        timeout=30,
                    )
                    definitely_unaccepted = (
                        isinstance(observed, OperationRecord)
                        and observed.operation_id == command.operation_id
                        and observed.status is OperationStatus.UNKNOWN
                        and observed.result is None
                    )
                except BaseException:
                    # Unknown delivery/query outcome: preserve GS intent so a
                    # possibly-running lifecycle worker can still reconcile its
                    # evidence. A same-command retry remains idempotent.
                    definitely_unaccepted = False

            if definitely_unaccepted:
                self.operation_commands.pop(command.operation_id, None)
                if staged_borrower:
                    if self.borrower_targets.get(command.lease_id) == command.target:
                        self.borrower_targets.pop(command.lease_id, None)
                if staged_restore_reservation:
                    for bundle_key in lease.bundle_keys:
                        if self.active_bundle_owner.get(bundle_key) == command.lease_id:
                            self.active_bundle_owner.pop(bundle_key, None)
                    for gpu_uuid in lease.gpu_uuids:
                        if self.active_gpu_owner.get(gpu_uuid) == command.lease_id:
                            self.active_gpu_owner.pop(gpu_uuid, None)
            raise

    @staticmethod
    def _target_matches_donor(target: ReplicaKey, lease: Lease) -> bool:
        """Match first-release donor identity without trusting a handle from GS."""
        claim = lease.claims[0]
        rank = claim["donor_replica_rank"]
        replica_id = target.replica_id
        return replica_id in {f"r{rank}", f"native-{rank}"}

    def open_lease(self, lease: Lease) -> Lease:
        """GS-internal ledger action; scheduler policy calls this before command issue."""
        if not isinstance(lease, Lease):
            raise TypeError("open_lease requires Lease")
        existing = self.leases.get(lease.lease_id)
        if existing is not None:
            if existing != lease:
                raise ValueError("conflicting lease replay")
            return existing

        for claim_id in lease.claim_ids:
            owner = self.claim_id_owner.get(claim_id)
            if owner is not None and owner != lease.lease_id:
                raise ValueError(
                    f"claim_id {claim_id!r} already belongs to lease {owner!r}"
                )
        for bundle_key in lease.bundle_keys:
            owner = self.active_bundle_owner.get(bundle_key)
            if owner is not None and owner != lease.lease_id:
                raise ValueError(
                    f"PG bundle {bundle_key!r} is already active under lease {owner!r}"
                )
        for gpu_uuid in lease.gpu_uuids:
            owner = self.active_gpu_owner.get(gpu_uuid)
            if owner is not None and owner != lease.lease_id:
                raise ValueError(
                    f"GPU {gpu_uuid!r} is already active under lease {owner!r}"
                )

        self.leases[lease.lease_id] = lease
        self.release_history.setdefault(lease.lease_id, [])
        for claim_id in lease.claim_ids:
            self.claim_id_owner[claim_id] = lease.lease_id
        for bundle_key in lease.bundle_keys:
            self.active_bundle_owner[bundle_key] = lease.lease_id
        for gpu_uuid in lease.gpu_uuids:
            self.active_gpu_owner[gpu_uuid] = lease.lease_id
        return lease

    def advance_lease(self, lease_id: str, evidence: OperationEvidence) -> dict:
        """GS-internal lease progression after exact operation evidence validation."""
        lease = self.leases.get(lease_id)
        if lease is None:
            raise ValueError(f"unknown lease {lease_id!r}")
        if not isinstance(evidence, OperationEvidence):
            raise TypeError("advance_lease requires OperationEvidence")

        command = self.operation_commands.get(evidence.operation_id)
        if command is None:
            raise ValueError("operation evidence references an unknown operation")
        if command.lease_id != lease_id:
            raise ValueError("operation evidence belongs to another lease")

        evidence_key = (lease_id, evidence.operation_id)
        existing = self.release_evidence.get(evidence_key)
        if existing is not None:
            if existing != evidence:
                raise ValueError("conflicting lease evidence replay")
            result = {
                "lease_id": lease_id,
                "operation_id": evidence.operation_id,
            }
            if command.kind is OperationKind.RESTORE:
                if existing.type is EvidenceType.SERVICE_COMMITTED:
                    result["restored"] = True
                elif existing.type is EvidenceType.RELEASED:
                    result["restore_rolled_back"] = True
                else:
                    raise ValueError("invalid RESTORE evidence replay")
            elif command.kind is OperationKind.ADD:
                result["add_rolled_back"] = True
            else:
                result["released"] = True
            return result

        if command.kind is OperationKind.RESTORE:
            if evidence.type not in {
                EvidenceType.SERVICE_COMMITTED,
                EvidenceType.RELEASED,
            }:
                raise ValueError(
                    "RESTORE lease progression requires SERVICE_COMMITTED or RELEASED evidence"
                )
            if any(
                self.active_bundle_owner.get(bundle_key) != lease_id
                for bundle_key in lease.bundle_keys
            ) or any(
                self.active_gpu_owner.get(gpu_uuid) != lease_id
                for gpu_uuid in lease.gpu_uuids
            ):
                raise ValueError(
                    "RESTORE completion requires its temporary claim reservation"
                )

            if evidence.type is EvidenceType.RELEASED:
                if set(evidence.released_gpu_uuids) != set(lease.gpu_uuids):
                    raise ValueError(
                        "RESTORE rollback RELEASED evidence must exactly cover lease GPU claims"
                    )
                self.release_evidence[evidence_key] = evidence
                # Compensation is not a successful lease-cycle close. Do not add
                # it to release_history, otherwise a safe retry would be rejected
                # as if RESTORE had completed successfully.
                for bundle_key in lease.bundle_keys:
                    if self.active_bundle_owner.get(bundle_key) == lease_id:
                        self.active_bundle_owner.pop(bundle_key, None)
                for gpu_uuid in lease.gpu_uuids:
                    if self.active_gpu_owner.get(gpu_uuid) == lease_id:
                        self.active_gpu_owner.pop(gpu_uuid, None)
                return {
                    "lease_id": lease_id,
                    "operation_id": evidence.operation_id,
                    "restore_rolled_back": True,
                }

            self.release_evidence[evidence_key] = evidence
            self.release_history.setdefault(lease_id, []).append(
                evidence.operation_id
            )
            self.handoff_ready_leases.discard(lease_id)
            self.borrower_targets.pop(lease_id, None)
            for bundle_key in lease.bundle_keys:
                self.active_bundle_owner.pop(bundle_key, None)
            for gpu_uuid in lease.gpu_uuids:
                self.active_gpu_owner.pop(gpu_uuid, None)
            return {
                "lease_id": lease_id,
                "operation_id": evidence.operation_id,
                "restored": True,
            }

        if command.kind is OperationKind.ADD:
            if evidence.type is not EvidenceType.RELEASED:
                raise ValueError("ADD rollback requires RELEASED evidence")
            if set(evidence.released_gpu_uuids) != set(lease.gpu_uuids):
                raise ValueError(
                    "ADD rollback RELEASED evidence must exactly cover lease GPU claims"
                )
            if command.lease_id not in self.handoff_ready_leases:
                raise ValueError("ADD rollback requires the donor handoff reservation")
            if self.borrower_targets.get(command.lease_id) != command.target:
                raise ValueError("ADD rollback target does not match staged borrower")

            self.release_evidence[evidence_key] = evidence
            self.release_history.setdefault(lease_id, []).append(
                evidence.operation_id
            )
            # TaskRunner calls advance_lease only after the hidden borrower is
            # proven gone and the ADD is being closed as FAILED. Release this
            # aborted borrower reservation; a policy that wants an in-place retry
            # must keep the lease frozen by not advancing it yet.
            self.borrower_targets.pop(lease_id, None)
            self.handoff_ready_leases.discard(lease_id)
            for bundle_key in lease.bundle_keys:
                if self.active_bundle_owner.get(bundle_key) == lease_id:
                    self.active_bundle_owner.pop(bundle_key, None)
            for gpu_uuid in lease.gpu_uuids:
                if self.active_gpu_owner.get(gpu_uuid) == lease_id:
                    self.active_gpu_owner.pop(gpu_uuid, None)
            return {
                "lease_id": lease_id,
                "operation_id": evidence.operation_id,
                "add_rolled_back": True,
            }

        if command.kind not in _RELEASE_KINDS:
            raise ValueError(
                "only ADD rollback/DONATE/REMOVE RELEASED or RESTORE SERVICE_COMMITTED may advance a lease"
            )
        if evidence.type is not EvidenceType.RELEASED:
            raise ValueError("lease handoff requires RELEASED evidence")
        if set(evidence.released_gpu_uuids) != set(lease.gpu_uuids):
            raise ValueError("RELEASED evidence must exactly cover lease GPU claims")

        self.release_evidence[evidence_key] = evidence
        self.release_history.setdefault(lease_id, []).append(
            evidence.operation_id
        )

        # DONATE hands the already-reserved claims to the borrower; it must not
        # expose the same physical slot to another lease. Only borrowed REMOVE
        # returns the claims to the global free pool.
        if command.kind is OperationKind.DONATE:
            self.handoff_ready_leases.add(lease_id)
        else:
            self.handoff_ready_leases.discard(lease_id)
            for bundle_key in lease.bundle_keys:
                if self.active_bundle_owner.get(bundle_key) == lease_id:
                    self.active_bundle_owner.pop(bundle_key, None)
            for gpu_uuid in lease.gpu_uuids:
                if self.active_gpu_owner.get(gpu_uuid) == lease_id:
                    self.active_gpu_owner.pop(gpu_uuid, None)

        return {
            "lease_id": lease_id,
            "operation_id": evidence.operation_id,
            "released": True,
        }
