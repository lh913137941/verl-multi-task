# Simplified fusion contract (092203 with 092303 parameter-sync clarification)

Current first release: experimental Fully Async, pure STANDALONE, non-PD vLLM, single-node whole-GPU lending, DP=1, PP=1 and currently verified TP=1 only. Physical whole-GPU exclusivity is enforced by GS ownership of the PG bundle/GPU UUID, not by Ray's fractional actor accounting. The first release fixes `max_colocate_count=2`, so each donor/borrower CE actor requests `gpu_fraction=0.5` and one CPU from a bundle; this accounting share does not authorize fractional physical-GPU lending. Native replicas keep their runtime across DONATE/RESTORE; borrowed replicas create an independent borrower runtime and are destroyed on REMOVE. Unsupported backend combinations must fail explicitly; lifecycle success requires runtime/placement/weight evidence and is never inferred from configuration alone.

## Ownership

- M / Manager: internal `replica_state[ReplicaKey]` + `replica_kind[ReplicaKey]`.
- E / CE Manager: `effective_replicas` and parameter facts.
- R / LB: native routes/counters + internal `active_request_server` + `attempt_state`.
- C / Rollouter: committed capacity via native `max_concurrent_samples`; production window reuses native `paused`.
- Trainer owns the single synchronization gate G. G does not cover long client/runtime waits. A BLOCKED G has no raw reset: only the operation that latched it may run owner-fact reconciliation under the gate and clear it after that reconciliation returns successfully; failed/unknown reconciliation keeps G blocked.

There is no public `ReplicaRecord` or `AttemptRecord`.

Lifecycle states are exactly `CREATING`, `ACTIVE`, `DRAINING`, `DORMANT`, `RELEASED`, `QUARANTINED`. Borrowed never enters DORMANT; native never enters RELEASED. LB-internal request states are `ADMITTED`, `TERMINATED`, `SETTLED`.

## Public lifecycle structures

Only `OperationCommand(operation_id, kind, target, lease_id, force?)`, `OperationRecord(operation_id, status, result?)`, `OperationEvidence(operation_id, type, timestamp, released_gpu_uuids=())`, and `Lease(lease_id, claims, expires_at)` cross component ownership boundaries. `ReplicaKey` is shared identity, not a lifecycle record. Lease progression is GS-internal; `expires_at` is not release proof. The external operation command stays lease-id-only: GS resolves that id once and forwards the immutable `Lease` snapshot alongside the internal GS→TaskRunner call so ADD can derive placement without adding another public DTO or giving Manager/LB a GS dependency.

## Key rules

ADD: hidden create -> under G install current weights and join E -> commit R/C/M -> ACTIVE. If R is definitely absent after E, C/M retract first, Trainer removes E under G, then borrowed destroy produces RELEASED. If R is unknown, G stays BLOCKED; exact operation replay queries R and either completes forward or performs that rollback.

DONATE/REMOVE: ACTIVE -> DRAINING -> close admission and settle requests -> under G remove E/commit service -> verified native sleep gives DORMANT, verified borrowed destroy gives RELEASED. GS advances usage rights only after `RELEASED` exactly covers lease GPU UUIDs. DONATE `RELEASED` makes the already-reserved borrower lease handoff-ready; borrowed REMOVE `RELEASED` returns that physical ownership to the global free pool. First release treats a lease_id as one borrower lifecycle: accepted RESTORE temporarily reserves the returned donor claims, then `advance_lease(SERVICE_COMMITTED)` releases that reservation and closes the lease cycle; a later lending cycle uses a new lease_id.

FORCE_VERIFIED is borrowed-only. It reuses VERL's native Fully Async partial-rollout path: after the target is removed from admission, the borrowed replica executes the real vLLM abort primitive; the continuation-aware client records proof when it receives an `aborted/abort` output, then VERL retries with `prompt + partial token_ids` on another active server. `EXIT_READY` is returned only when `partial_rollout=true`, another server is available, target abort succeeds, and LB has no remaining `ADMITTED` request. Timeout is never success.

RESTORE wakes the same native runtime, restores current parameters and re-enters E/R/C/M. Before E commit, a failed bootstrap may return to DORMANT only after proven re-sleep; if bootstrap has mutated the receiver but re-sleep cannot be proved, Trainer keeps G blocked and asks Rollouter/Manager to project a still-DORMANT M entry to QUARANTINED rather than advertising a false sleeping fact. After E is effective, a definite no-publish result retracts C/M, then Trainer removes E under G before verified re-sleep and RELEASED compensation. An unknown R outcome keeps the runtime awake and fences G; exact same-operation replay queries R, clears G only after forward commit or verified rollback, and remains BLOCKED if R is still unknowable.

Current branch status: ADD, DONATE, natural REMOVE, FORCE_VERIFIED and RESTORE are all admitted through TaskRunner and have complete control-plane orchestration paths. GS submission ambiguity preserves staged intent/reservations: UNKNOWN or an absent TaskRunner record is not proof that delivery never occurred. Natural drain and FORCE pre-EXIT_READY failures preserve DRAINING for exact-operation recovery; FORCE records the pre-abort ADMITTED boundary and confirmed abort count so partial continuation retries do not repeat an issued abort, while an unknown abort outcome requires continuation proof for every request from that boundary before EXIT_READY. DONATE/REMOVE/FORCE validation failures that occur before Rollouter binds the pending lifecycle target are classified as proven no-owner-mutation FAILED results; once the pending binding exists, failures remain UNKNOWN unless owner reconciliation proves a terminal result. Exact-op UNKNOWN replay is supported for all four lifecycle kinds: ADD/RESTORE reconcile publication owner facts, while DONATE/REMOVE first reuse Manager RELEASED evidence or reconcile a BLOCKED post-E R/C commit under G before resuming release. ADD validation/create failures that occur before runtime registration, or after verified borrowed cleanup, now return RELEASED compensation and close FAILED instead of being promoted to UNKNOWN. ADD/RESTORE pre-publication failures that return verified RELEASED evidence now close as FAILED and release their GS reservation; RESTORE compensation does not close the lease cycle and remains retryable. TaskRunner replays the same advance_lease evidence once after an ACK-loss exception. ADD uses hidden borrowed creation plus target-only current-Vpub bootstrap; DONATE uses natural drain plus verified native level-2 sleep; FORCE_VERIFIED uses targeted abort/continuation proof; RESTORE reuses CE pending/bootstrap and LB ready-commit with E committed before externally visible R. This code-level completion is not the same as a successful CUDA/vLLM/NCCL acceptance run: unsupported runtime combinations still fail explicitly and unresolved side effects fence G or quarantine M. Whole-GPU DONATE remains valid only when VERL's own sleep-level resolver selects level 2; MTP rollout and LoRA rollout configurations that require level 1 are rejected by the first-release profile.

Idle detection is Rollouter-local: production window + C + read-only Manager M. LB request state does not decide bubbles. A paused-window reporter submits surplus metadata and refreshes an unchanged set every 5s; when GS has an idle report, DONATE rejects reports older than 10s or targets absent from that NATIVE candidate set. Report failure is advisory and retried without stopping rollout. Draining starts only after GS issues a lifecycle operation. Natural drain has a bounded per-attempt `multitask.drain_timeout_s` budget (default 300s); timeout preserves DRAINING plus the exact operation binding so same-operation replay continues the drain and never silently escalates to FORCE.

The sample path remains Rollouter -> Queue -> Trainer. Exactly-once is a thin logical-key/digest layer over native `RolloutSample`.

Remaining design gaps are outside the main lifecycle happy paths: natural drain now has a bounded local timeout, but there is still no separate GS takeover/HA protocol after an owner becomes unreachable (FORCE remains an explicit separate operation); the detailed design's reconciliation name `query_runtime` maps today to owner-local `replica_meta()/inspect_runtime()` rather than a public Manager query API; automatic HA/client-crash recovery remains outside the first-release commitment. Queue completion evidence remains exact for the actor lifetime but is now stored in an actor-local SQLite ledger rather than a lifetime-growing Python dict, so old exact retries and conflicting payload digests remain queryable without unbounded queue-actor heap growth. A durable cross-actor/checkpoint recovery protocol for that ledger is still outside the first-release commitment. These must not be confused with missing ADD/DONATE/REMOVE/FORCE/RESTORE orchestration.

## Target-only parameter synchronization (092303 section 8.6)

Pending targets and effective members are distinct CE-owned sets. ADD registers its
hidden runtime in `pending_bootstrap`, outside both E and native `self.replicas`.
Only after target transfer, finalize and server version confirmation succeed may
`commit_pending` promote it to E. It must match the CE owner's stored
`WEIGHT_READY` evidence, operation and loaded version; matching an operation id
alone is insufficient. A runtime cannot occur twice or belong to two ReplicaKeys.
Removing a member also invalidates its cached bootstrap result.

The 092303 text's requirement that every target already belong to E applies to
refreshing effective members, not initial ADD or rejoining RESTORE. Applying it
to ADD contradicts that same section's requirement to join E only after transfer.
RESTORE now reuses the same target bootstrap/version-confirmation machinery for
a parked native runtime; weights-only wake is performed inside CE bootstrap while
Trainer G is held, followed by current-Vpub transfer, KV restore and version
confirmation. The opt-in real GPU acceptance mutates sender tensor data before
transfer so a version tag alone cannot satisfy the check; it remains the runtime
acceptance criterion rather than a TaskRunner admission gate.

The first-release profile is CUDA/NCCL-only and requires
`actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true`.
Native finalize otherwise retains the collective group, which cannot safely be
reused when receiver membership changes. Other device/transport combinations
remain outside the verified first-release boundary.

`Vpub` denotes a version, not a cached weight snapshot. Wiring ADD/RESTORE must
also prove that the sender's actual weights match that version: the membership
gate alone does not serialize optimizer updates. ADD and RESTORE both use the
same G-serialized target-bootstrap/version-confirmation boundary; runtime
acceptance still requires the real current-Vpub GPU path to succeed.

See [092303 repair and validation notes](2026-09-23-092303-ce-repair.md) for the
source baseline, tests and remaining runtime work.
