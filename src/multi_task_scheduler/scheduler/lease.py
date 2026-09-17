"""Global lease authorization state machine for simplified design §8-§14.

A lease edge that changes who may use the GPUs is evidence-driven. A bare
SUCCEEDED status is never sufficient: release edges require a matching
ServiceEvidence(REMOVE) + ReleaseEvidence chain, and service-activation edges
require a matching ServiceEvidence(ADD). Lease authorizations are additionally
fenced by a monotonic authorization_seq while preserving same-operation replay.
"""

from __future__ import annotations

from enum import Enum

from multi_task_scheduler.orchestration.contracts import (
    OperationResult,
    ReleaseKind,
    ServiceAction,
)
from multi_task_scheduler.orchestration.operation_journal import OperationStatus
from multi_task_scheduler.orchestration.receipts import ReleaseEvidence, ServiceEvidence

from .ledger import LeaseRecord


class LeaseState(str, Enum):
    PLANNED = "PLANNED"
    DONOR_DRAINING = "DONOR_DRAINING"
    DONOR_RELEASED = "DONOR_RELEASED"
    BORROWER_PREPARING = "BORROWER_PREPARING"
    BORROWER_ACTIVE = "BORROWER_ACTIVE"
    RECALLING = "RECALLING"
    BORROWER_RELEASED = "BORROWER_RELEASED"
    DONOR_RESTORING = "DONOR_RESTORING"
    CLOSED = "CLOSED"
    RECONCILING = "RECONCILING"
    QUARANTINED = "QUARANTINED"


class IllegalLeaseTransitionError(RuntimeError):
    """A lease skipped a required authorization state."""


class MissingEvidenceError(RuntimeError):
    """An evidence-backed lease transition lacked matching lifecycle proof."""


_ALLOWED = {
    LeaseState.PLANNED: {LeaseState.DONOR_DRAINING},
    LeaseState.DONOR_DRAINING: {LeaseState.DONOR_RELEASED, LeaseState.RECONCILING},
    LeaseState.DONOR_RELEASED: {LeaseState.BORROWER_PREPARING},
    LeaseState.BORROWER_PREPARING: {LeaseState.BORROWER_ACTIVE, LeaseState.RECONCILING},
    LeaseState.BORROWER_ACTIVE: {LeaseState.RECALLING, LeaseState.RECONCILING},
    LeaseState.RECALLING: {LeaseState.BORROWER_RELEASED, LeaseState.RECONCILING},
    LeaseState.BORROWER_RELEASED: {LeaseState.DONOR_RESTORING},
    LeaseState.DONOR_RESTORING: {
        LeaseState.CLOSED,
        LeaseState.RECONCILING,
        LeaseState.QUARANTINED,
    },
    LeaseState.CLOSED: set(),
    LeaseState.RECONCILING: set(),
    LeaseState.QUARANTINED: set(),
}

_RELEASE_REQUIREMENTS = {
    (LeaseState.DONOR_DRAINING, LeaseState.DONOR_RELEASED): (
        "donor",
        ReleaseKind.DONOR_SLEEP_RELEASED,
    ),
    (LeaseState.RECALLING, LeaseState.BORROWER_RELEASED): (
        "borrower",
        ReleaseKind.BORROWER_RUNTIME_DESTROYED,
    ),
}

_SERVICE_REQUIREMENTS = {
    (LeaseState.BORROWER_PREPARING, LeaseState.BORROWER_ACTIVE): "borrower",
    (LeaseState.DONOR_RESTORING, LeaseState.CLOSED): "donor",
}

_POST_RELEASE_AUTHORIZATION_EDGES = {
    (LeaseState.DONOR_RELEASED, LeaseState.BORROWER_PREPARING),
    (LeaseState.BORROWER_RELEASED, LeaseState.DONOR_RESTORING),
}


class LeaseStateMachine:
    def __init__(self) -> None:
        self._leases: dict[str, LeaseRecord] = {}
        self._last_authorization_seq: dict[str, int] = {}
        self._authorization_by_operation: dict[tuple[str, str], int] = {}

    def register(self, lease: LeaseRecord) -> None:
        lease.state = LeaseState(lease.state).value
        self._leases[lease.lease_id] = lease
        self._last_authorization_seq.setdefault(lease.lease_id, -1)

    def get(self, lease_id: str) -> LeaseRecord:
        return self._leases[lease_id]

    def validate_authorization(
        self,
        lease_id: str,
        operation_id: str,
        authorization_seq: int,
    ) -> None:
        """Validate a GS authorization without consuming a new sequence yet."""
        if lease_id not in self._leases:
            raise ValueError(f"unknown lease {lease_id!r}")
        if not operation_id:
            raise ValueError("operation_id must be nonempty")
        if authorization_seq < 0:
            raise ValueError("authorization_seq must be nonnegative")

        key = (lease_id, operation_id)
        existing = self._authorization_by_operation.get(key)
        if existing is not None:
            if existing != authorization_seq:
                raise ValueError("conflicting authorization_seq for operation replay")
            return

        last = self._last_authorization_seq.get(lease_id, -1)
        if authorization_seq <= last:
            raise ValueError(
                f"stale authorization_seq {authorization_seq}; last accepted sequence is {last}"
            )

    def record_authorization(
        self,
        lease_id: str,
        operation_id: str,
        authorization_seq: int,
    ) -> None:
        """Consume a new authorization sequence after GS records the operation intent."""
        self.validate_authorization(lease_id, operation_id, authorization_seq)
        key = (lease_id, operation_id)
        if key not in self._authorization_by_operation:
            self._authorization_by_operation[key] = authorization_seq
            self._last_authorization_seq[lease_id] = authorization_seq

    @staticmethod
    def _expected_session(lease: LeaseRecord, role: str) -> str:
        if role == "donor":
            return lease.donor_session
        if lease.borrower_session is None:
            raise MissingEvidenceError("borrower_session is required for this lease edge")
        return lease.borrower_session

    @staticmethod
    def _require_result_identity(
        lease: LeaseRecord,
        result: OperationResult | None,
        expected_session: str,
    ) -> OperationResult:
        if not isinstance(result, OperationResult):
            raise MissingEvidenceError("lease transition requires a typed OperationResult")
        if result.status is not OperationStatus.SUCCEEDED:
            raise MissingEvidenceError("lease transition requires OperationStatus.SUCCEEDED")
        if result.ctx.lease_id != lease.lease_id or result.ctx.lease_epoch != lease.lease_epoch:
            raise MissingEvidenceError("operation result does not match lease identity/epoch")
        if result.ctx.task_session != expected_session or result.target.task_session != expected_session:
            raise MissingEvidenceError("operation result targets the wrong lease participant")
        return result

    @staticmethod
    def _require_service(
        result: OperationResult,
        action: ServiceAction,
    ) -> ServiceEvidence:
        service = result.service
        if not isinstance(service, ServiceEvidence):
            raise MissingEvidenceError(f"lease transition requires ServiceEvidence({action.value})")
        if service.header.ctx != result.ctx or service.header.key != result.target:
            raise MissingEvidenceError("service evidence identity does not match operation result")
        if service.action is not action:
            raise MissingEvidenceError(
                f"lease transition requires ServiceEvidence({action.value})"
            )
        return service

    @staticmethod
    def _record_operation(lease: LeaseRecord, operation_id: str) -> None:
        if operation_id not in lease.operation_ids:
            lease.operation_ids = (*lease.operation_ids, operation_id)
        lease.last_operation_id = operation_id

    def _validate_release_edge(
        self,
        lease: LeaseRecord,
        result: OperationResult | None,
        role: str,
        release_kind: ReleaseKind,
    ) -> OperationResult:
        expected_session = self._expected_session(lease, role)
        result = self._require_result_identity(lease, result, expected_session)
        service = self._require_service(result, ServiceAction.REMOVE)
        release = result.release
        if not isinstance(release, ReleaseEvidence):
            raise MissingEvidenceError("resource handoff requires ReleaseEvidence")
        if release.header.ctx != result.ctx or release.header.key != result.target:
            raise MissingEvidenceError("release evidence identity does not match operation result")
        if release.release_kind is not release_kind:
            raise MissingEvidenceError(
                f"release kind {release.release_kind.value} does not match lease edge"
            )
        if release.permit_digest != service.header.digest:
            raise MissingEvidenceError(
                "release evidence must reference the matching service-removal proof"
            )
        lease.last_release_digest = release.header.digest
        self._record_operation(lease, result.ctx.operation_id)
        return result

    def _validate_service_edge(
        self,
        lease: LeaseRecord,
        result: OperationResult | None,
        role: str,
    ) -> OperationResult:
        expected_session = self._expected_session(lease, role)
        result = self._require_result_identity(lease, result, expected_session)
        self._require_service(result, ServiceAction.ADD)
        self._record_operation(lease, result.ctx.operation_id)
        return result

    def advance(
        self,
        lease_id: str,
        new_state: LeaseState,
        *,
        supporting_result: OperationResult | None = None,
    ) -> LeaseRecord:
        lease = self._leases[lease_id]
        current = LeaseState(lease.state)
        new_state = LeaseState(new_state)
        edge = (current, new_state)
        if new_state not in _ALLOWED[current]:
            raise IllegalLeaseTransitionError(
                f"illegal lease transition for {lease_id}: "
                f"{current.value} -> {new_state.value}"
            )

        release_requirement = _RELEASE_REQUIREMENTS.get(edge)
        if release_requirement is not None:
            role, release_kind = release_requirement
            self._validate_release_edge(lease, supporting_result, role, release_kind)
        else:
            service_role = _SERVICE_REQUIREMENTS.get(edge)
            if service_role is not None:
                self._validate_service_edge(lease, supporting_result, service_role)
            elif edge in _POST_RELEASE_AUTHORIZATION_EDGES:
                if supporting_result is not None:
                    raise ValueError("post-release authorization consumes the stored release digest")
                if lease.last_release_digest is None:
                    raise MissingEvidenceError(
                        "post-release authorization requires a GS-confirmed release digest"
                    )
            elif supporting_result is not None:
                raise ValueError("this lease edge does not consume an OperationResult")

        lease.state = new_state.value
        return lease
