# Dynamic orchestration simplified fusion contract

This document is the repository-local implementation contract for the current
fusion design. It intentionally contains **only the current protocol**. The
older `docs/verl-5.3-dynamic-orchestration-design.md` remains historical design
background and must not be used to restore superseded signatures or evidence
shapes.

## 1. Scope and verification boundary

The first release targets experimental Fully Async + standalone vLLM (non-PD).
The orchestration core, GS metadata/lease state, and binding surfaces may be
implemented in pure Python. Device/runtime primitives that have not been
verified against the real backend must raise `NotImplementedError`; tests or
mock objects must never be presented as CUDA/NCCL/runtime proof.

## 2. One protocol, no compatibility aliases

The protocol is migrated as a unit. Do not reintroduce old names or wire forms
such as `Command`, `OperationType`, `OperationState`, `NodeBlock`,
`TransferReceipt`, `ReadyReceipt`, `RemovedReceipt`, `ReleaseReceipt`,
`begin_operation`, `begin_drain`, version-only bootstrap evidence, or bool/int
owner commit results.

Canonical operation types are:

- `OperationKind`: ADD / DONATE / REMOVE / RESTORE
- `RecallMode`: NATURAL / FORCE_VERIFIED
- `OperationStatus`: ACCEPTED / RUNNING / SUCCEEDED / FAILED / UNKNOWN
- `Phase`: VALIDATE, CREATE, DRAIN, ABORT_TARGET, WAIT_CONTINUATION,
  WAIT_GATE, WAKE_WEIGHTS, LOAD_WEIGHTS, JOIN_CE, COMMIT_SERVICE, LEAVE_CE,
  COMMIT_REMOVAL, SLEEP, DESTROY, RECONCILE, DONE
- `Outcome`: KNOWN_NOT_APPLIED / KNOWN_APPLIED / UNKNOWN

`OperationStatus`, `Phase`, and `Outcome` are separate facts. `DONE` does not
imply success. Unknown side effects must remain UNKNOWN and be reconciled.

## 3. Identity, replay budget, and placement

`ReplicaKey(task_session, replica_id, runtime_epoch)` is the runtime identity.
Rebuild changes `runtime_epoch`; sleep/wake does not.

`OperationContext` carries protocol version, GS epoch, task/session,
operation_id, lease identity/epoch, command sequence and expected revision.

`OperationCommand` contains the complete nested context, target,
`LeaseAuthorization`, payload digest, remaining budget, and operation-specific
placement/candidate/recall data. `operation_id` replay is valid only when the
immutable business identity is identical. `remaining_budget_ms` is transport
budget and is deliberately excluded from that business identity. The first
accepted operation owns the operation deadline; a same-ID retry returns existing
progress and must not create a new deadline or reset the total time budget.

The first release permits **one unfinished lifecycle operation per task**.
Same-operation replay is always allowed because it does not create a second
execution. A different `operation_id` is accepted only after the current
operation reaches `DONE`; it must then also pass the monotonic `command_seq`
fence. Native parameter synchronization is serialized with lifecycle critical
commit sections by the task-local gate G rather than by starting a second
lifecycle operation.

First-release `PlacementSpec` is single-node (`NodePlacement`) with `dp=1`,
`pp=1`, and `world_size == tp`.

## 4. Public lifecycle

Manager exposes exactly eight lifecycle states:

`PREPARING -> ACTIVE -> DRAINING -> DETACHED`, followed by:

- native: `DORMANT -> RESTORING -> ACTIVE`
- borrowed: `DESTROYED`
- uncertain paths: `QUARANTINED`

Detailed bootstrap, CE, route, sleep and destroy progress belongs in the
operation journal/evidence, not additional public lifecycle states.

## 5. M / E / R / C ownership

- **M**: Manager owns `ReplicaRecord` lifecycle.
- **E**: Checkpoint Engine Manager owns effective synchronization membership.
- **R**: Load Balancer owns routing and attempt records.
- **C**: Rollouter owns committed active capacity.

Each owner has one writer. Revisions from different owners are not comparable.
CE/LB commits return typed `CommitReceipt`, never bool/int shortcuts.

## 6. Four lifecycle evidence classes

The only cross-stage lifecycle proofs are:

1. `WeightEvidence`: exact immutable published weights are installed on all
   receivers and temporary transfer topology is clean.
2. `ExitEvidence`: old requests have left the target. NATURAL has no
   continuations; FORCE_VERIFIED requires verified continuation disposition.
3. `ServiceEvidence`: CE/LB/Capacity/Manager service ADD or REMOVE is committed.
   REMOVE proves DETACHED only; it does not prove device release.
4. `ReleaseEvidence`: physical sleep/destroy release is verified per GPU and by
   all required backends.

`CommitReceipt`, `AdmissionSnapshot`, `Ack`, `DrainTicket`, `PreparedReplica`
and `NeverPublishedProof` are phase/local results, not substitutes for the four
lifecycle proofs.

## 7. Published weights and G

A version number is not weight evidence. ADD/RESTORE pin an immutable
`PublishedWeightSnapshot(snapshot_id, manifest_digest, model_signature,
version, byte_size, sender)` and require returned `WeightEvidence` to match it.

The task-local replica sync gate **G** serializes native parameter sync with
membership/service commit critical sections. Long hidden-create and drain waits
stay outside G. Unknown/cancelled native sync outcomes block ordinary future G
owners until reconciliation.

## 8. Canonical control interfaces

GS -> TaskRunner:

- `submit_operation(command: OperationCommand) -> OperationResult`
- `query_operation(task_session, operation_id) -> QueryResult[OperationResult]`
- `probe_task(task_session) -> TaskSnapshot`

LB -> GS:

- `report_idle_candidates(report: IdleCandidateReport) -> Ack`

TaskRunner/Trainer/Rollouter internal lifecycle entries:

- `prepare_replica(ctx, key, placement) -> PreparedReplica`
- `prepare_exit(command) -> ExitEvidence`
- `bootstrap_and_publish(ctx, prepared) -> ServiceEvidence(ADD)`
- `remove_and_commit(ctx, proof) -> ServiceEvidence(REMOVE)`
- `restore_and_publish(ctx, key) -> ServiceEvidence(ADD)`
- `finalize_release(ctx, service) -> ReleaseEvidence`
- owner-side `query_phase(...)` for reconciliation

Do not restore `begin_drain`, `begin_operation`, or free boolean restore fences.

## 9. Idle candidates

Missing activity counts are UNKNOWN (`None`), never assumed zero. An
`IdleCandidate` is emitted only from complete, fresh, stable production +
Manager + LB + engine facts and contains the exact `ReplicaKey`, production
and source revisions, Manager/LB revisions, engine digest, reason, idle
stability, observation age, GPU count, evidence digest and placement digest.

`IdleCandidateReport` is complete-set replacement. An empty candidate tuple
revokes previous candidates; replaying the same `source_seq` does not renew its
TTL. A DONATE command carries the exact chosen `IdleCandidate` and must
revalidate it before side effects.

## 10. Request/sample exactly-once boundary

A completed sample is keyed by `(task_session, logical_sample_id)`, not by turn
or attempt. First completion stores `CompletionEvidence`; same key + same digest
returns that first evidence without enqueueing again; same key + different
digest is a conflict. Native queue-full/drop-oldest return behavior is recorded
separately from deduplication.

## 11. Release and authorization

GS may transfer GPU authorization only after a `SUCCEEDED` operation carries a
matching `ServiceEvidence(REMOVE)` and `ReleaseEvidence`. The release evidence
must match the operation/lease/runtime identity and reference the service-removal
proof through `permit_digest`.

After GS confirms that release, its digest becomes the next authorization
fence: ADD/RESTORE `LeaseAuthorization.prior_release_digest` must match the
last release digest already confirmed for that lease. Native sleep may retain
verified native processes; borrowed destroy must not retain borrower-owned or
unknown processes. A string state, success status alone, or service-detached
proof cannot substitute for physical release evidence.

## 12. Document corrections incorporated here

The full fusion design had two editorial defects that do not change protocol
semantics:

- the duplicated heading `### 6.8### 6.8` is `### 6.8`;
- the sample queue text means removing the redundant **CompletionRecord**
  wrapper because `CompletionEvidence` already contains key + payload digest;
  it does **not** mean deleting `CompletionEvidence` itself.

These corrections are reflected in the supplied corrected fusion-design file.
