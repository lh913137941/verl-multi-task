# Simplified fusion contract (092203 with 092303 parameter-sync clarification)

Current first release: experimental Fully Async, pure STANDALONE, non-PD vLLM, single-node whole-GPU lending, DP=1, PP=1 and verified TP only. Physical whole-GPU exclusivity is enforced by GS ownership of the PG bundle/GPU UUID, not by Ray's fractional actor accounting. The first release fixes `max_colocate_count=2`, so each donor/borrower CE actor requests `gpu_fraction=0.5` and one CPU from a bundle; this accounting share does not authorize fractional physical-GPU lending. Native replicas keep their runtime across DONATE/RESTORE; borrowed replicas create an independent borrower runtime and are destroyed on REMOVE. Unverified GPU primitives must raise `NotImplementedError`.

## Ownership

- M / Manager: internal `replica_state[ReplicaKey]` + `replica_kind[ReplicaKey]`.
- E / CE Manager: `effective_replicas` and parameter facts.
- R / LB: native routes/counters + internal `active_request_server` + `attempt_state`.
- C / Rollouter: committed capacity via native `max_concurrent_samples`; production window reuses native `paused`.
- Trainer owns the single synchronization gate G. G does not cover long client/runtime waits.

There is no public `ReplicaRecord` or `AttemptRecord`.

Lifecycle states are exactly `CREATING`, `ACTIVE`, `DRAINING`, `DORMANT`, `RELEASED`, `QUARANTINED`. Borrowed never enters DORMANT; native never enters RELEASED. LB-internal request states are `ADMITTED`, `TERMINATED`, `SETTLED`.

## Public lifecycle structures

Only `OperationCommand(operation_id, kind, target, lease_id, force?)`, `OperationRecord(operation_id, status, result?)`, `OperationEvidence(operation_id, type, timestamp, released_gpu_uuids=())`, and `Lease(lease_id, claims, expires_at)` cross component ownership boundaries. `ReplicaKey` is shared identity, not a lifecycle record. Lease progression is GS-internal; `expires_at` is not release proof. The external operation command stays lease-id-only: GS resolves that id once and forwards the immutable `Lease` snapshot alongside the internal GS→TaskRunner call so ADD can derive placement without adding another public DTO or giving Manager/LB a GS dependency.

## Key rules

ADD: hidden create -> under G install current weights and join E -> commit R/C/M -> ACTIVE.

DONATE/REMOVE: ACTIVE -> DRAINING -> close admission and settle requests -> under G remove E/commit service -> verified native sleep gives DORMANT, verified borrowed destroy gives RELEASED. GS advances usage rights only after `RELEASED` exactly covers lease GPU UUIDs. DONATE `RELEASED` makes the already-reserved borrower lease handoff-ready but does not free its bundle/GPU to another lease; borrowed REMOVE `RELEASED` returns that physical ownership to the global free pool.

FORCE_VERIFIED is borrowed-only. It reuses VERL's native Fully Async partial-rollout path: after the target is removed from admission, the borrowed replica executes the real vLLM abort primitive; the continuation-aware client records proof when it receives an `aborted/abort` output, then VERL retries with `prompt + partial token_ids` on another active server. `EXIT_READY` is returned only when `partial_rollout=true`, another server is available, target abort succeeds, and LB has no remaining `ADMITTED` request. Timeout is never success.

RESTORE wakes the same native runtime, restores current parameters and re-enters E/R/C/M. A partial restore returns to DORMANT only after proven re-sleep, otherwise QUARANTINED.

Idle detection is Rollouter-local: production window + C + read-only Manager M. LB request state does not decide bubbles. Rollouter reports metadata directly to GS; draining starts only after GS issues a lifecycle operation.

The sample path remains Rollouter -> Queue -> Trainer. Exactly-once is a thin logical-key/digest layer over native `RolloutSample`.

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
RESTORE still needs its verified native wake and version-confirmation path;
the current borrowed-only server validation is not a RESTORE implementation.

For NCCL/HCCL, enabled profiles require the explicit boolean
`actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.<backend>.rebuild_group=true`.
Native finalize otherwise retains the collective group, which cannot safely be
reused when receiver membership changes. This check does not certify a backend
or device combination, and disabled profiles preserve native configuration.

`Vpub` denotes a version, not a cached weight snapshot. Wiring ADD/RESTORE must
also prove that the sender's actual weights match that version: the membership
gate alone does not serialize optimizer updates. Trainer's
`bootstrap_and_publish` and `restore_and_publish` remain explicit
`NotImplementedError` boundaries until this orchestration is verified.

See [092303 repair and validation notes](2026-09-23-092303-ce-repair.md) for the
source baseline, tests and remaining runtime work.
