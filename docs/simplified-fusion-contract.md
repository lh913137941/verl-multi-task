# Simplified fusion contract (092203 current)

Current first release: experimental Fully Async, pure STANDALONE, non-PD vLLM, single-node whole-GPU lending, DP=1, PP=1 and verified TP only. Native replicas keep their runtime across DONATE/RESTORE; borrowed replicas create an independent borrower runtime and are destroyed on REMOVE. Unverified GPU primitives must raise `NotImplementedError`.

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

DONATE/REMOVE: ACTIVE -> DRAINING -> close admission and settle requests -> under G remove E/commit service -> verified native sleep gives DORMANT, verified borrowed destroy gives RELEASED. GS transfers usage rights only after `RELEASED` exactly covers lease GPU UUIDs.

FORCE_VERIFIED is borrowed-only and needs verified partial rollout plus Client continuation proof. Timeout is never success. Target abort/continuation remains explicit `NotImplementedError` until GPU/runtime validation exists.

RESTORE wakes the same native runtime, restores current parameters and re-enters E/R/C/M. A partial restore returns to DORMANT only after proven re-sleep, otherwise QUARANTINED.

Idle detection is Rollouter-local: production window + C + read-only Manager M. LB request state does not decide bubbles. Rollouter reports metadata directly to GS; draining starts only after GS issues a lifecycle operation.

The sample path remains Rollouter -> Queue -> Trainer. Exactly-once is a thin logical-key/digest layer over native `RolloutSample`.
