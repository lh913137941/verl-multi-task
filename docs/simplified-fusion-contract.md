# Simplified fusion contract (0918 current)

This file is the repository-facing implementation contract for the current
multi-RL-task resource-sharing integration. It intentionally contains only the
hard boundaries needed by code and tests. The detailed review design remains
the source for full flows and rationale.

Older aliases, wire shapes, and helper APIs are historical only and must not be
reintroduced for compatibility.

## Supported first-release profile

- experimental Fully Async
- pure STANDALONE, non-PD vLLM
- single node, whole-GPU lending, DP=1, PP=1, verified TP only
- native replicas keep their original runtime/resource anchor across DONATE and RESTORE
- borrowed replicas are created for the borrower and destroyed on REMOVE
- FORCE_VERIFIED reclaim is a first-release requirement, but may be advertised only for a runtime/model/placement combination with a concrete capability proof
- Python/mock/AST tests prove control-plane contracts only; unverified CUDA/NCCL/vLLM primitives must raise `NotImplementedError`

## Existing component mapping

| Contract name | Repository implementation |
|---|---|
| GS | `GroupScheduler` / global scheduler role |
| TaskRunner | `MultiTaskFullyAsyncTaskRunner` |
| Trainer | `MultiTaskFullyAsyncTrainer` |
| Rollouter | `MultiTaskFullyAsyncRollouter` |
| Manager | `MultiTaskLLMServerManager` |
| CE Manager | `MultiTaskCheckpointEngineManager` |
| LB | `MultiTaskGlobalRequestLoadBalancer` |
| RuntimeBackend | task-local Manager/backend boundary |

Do not add a parallel set of orchestration actors merely to mirror the design names. `OperationCoordinator`, gate helpers, drain policy, runtime factory and lease binder may remain ordinary local helpers.

## State ownership

There are four business views plus two control states:

- **M** — Manager owns `ReplicaRecord` lifecycle state.
- **E** — CE Manager owns effective parameter receivers.
- **R** — LB owns routes and attempt records.
- **C** — Rollouter owns committed active capacity.
- Rollouter owns `ProductionWindow`.
- Trainer owns the single replica-sync gate G; CE owns the immutable current `PublishedWeightSnapshot` projection used through Trainer.

No component may duplicate another owner's mutable truth merely for convenience.

## Public lifecycle states

`PREPARING`, `ACTIVE`, `DRAINING`, `DETACHED`, `DORMANT`, `RESTORING`, `DESTROYED`, `QUARANTINED`.

Terminal native/borrowed records remain in M; they are not replaced by a second "deleted" ledger.

## Current public wire shapes

### Idle observation

```text
ProductionWindow(task_session, epoch, revision, state, eligible_pending, held_samples)
ServerActivity(key, engine_seq, observed_age_ms, admitting, queued, running, pending_admissions, all_backends_observed)
IdleCandidate(key, production_epoch, reason, evidence_digest)
IdleCandidateReport(task_session, gs_epoch, lb_session, source_seq, production_revision, valid_for_ms, candidates)
```

`source_seq` and TTL belong only to the complete report. A candidate does not copy report sequencing, placement, Manager/LB revisions, engine counters, or stable-idle timings. Those facts remain owner-local and are bound into `evidence_digest` after real validation.

### Parameter/runtime material

```text
PreparedReplica(key, model_signature, receivers, head_server, prepared_digest)
PublishedWeightSnapshot(snapshot_id, manifest_digest, model_signature, version, sender)
EffectiveReplicaEntry(key, receivers, loaded_version, membership_operation_id)
```

Snapshot byte size is observability, not public correctness state. Task-level model/layout compatibility is checked before CE membership and is not repeated in every `EffectiveReplicaEntry`.

### Lifecycle evidence

```text
WeightEvidence(header, version, manifest_digest, receivers_digest)
ExitEvidence(header, drain_id, recall_mode, quiescence_digest, attempts_digest)
ServiceEvidence(header, action, prerequisite_digest, service_digest)
ReleaseEvidence(header, release_kind, permit_digest, inventory_digest, released_gpu_uuids)
NeverPublishedProof(header, publication_fence_digest)
```

`release_kind` is exactly `DONOR_SLEEP_RELEASED` or `BORROWER_RUNTIME_DESTROYED`.

Receiver-by-receiver transfer results, drain counters, continuation details, per-GPU HBM/process diagnostics, `ProcessIdentity`, and cleanup inventories are not public lifecycle records. They stay in the real owner/backend journal. A RuntimeBackend keeps its cleanup inventory internally by full `ReplicaKey`.

`released_gpu_uuids` must contain no duplicates and its set must equal the lease `PlacementSpec.node.gpu_uuids` exactly before GS can transfer usage rights.

### Aggregate task observation

```text
TaskSnapshot(task_session, production, replica_records, ce_revision, published_version, lb_revision, capacity, sync_health, current_operation_id, consistency)
```

The old `SyncSnapshot` query is removed. `probe_task()` may return a `TaskSnapshot` only when it can gather real owner facts. Otherwise it must fail explicitly; it must never synthesize default zeros or a fake `STABLE` snapshot.

### Capability evidence

A capability flag is not verification. Runtime support is represented by `CapabilityProof(name, backend_version, model_signature, placement_digest, validation_id)` and must match the actual combination being executed.

## Canonical coordination interfaces

```text
TaskRunner.submit_operation(OperationCommand) -> OperationResult
TaskRunner.query_operation(task_session, operation_id) -> QueryResult[OperationResult]
TaskRunner.probe_task(task_session) -> TaskSnapshot

Trainer.bootstrap_and_publish(ctx, PreparedReplica) -> ServiceEvidence
Trainer.remove_and_commit(ctx, ExitEvidence) -> ServiceEvidence
Trainer.restore_and_publish(ctx, ReplicaKey) -> ServiceEvidence

Rollouter.prepare_replica(ctx, key, placement) -> PreparedReplica
Rollouter.prepare_exit(OperationCommand) -> ExitEvidence
Rollouter.commit_service_change(ctx, action, prerequisite, ce_commit, prepared?) -> ServiceEvidence
Rollouter.finalize_release(ctx, ServiceEvidence(REMOVE)) -> ReleaseEvidence

RuntimeBackend.create_hidden(ctx, key, placement) -> PreparedReplica
RuntimeBackend.sleep(ctx, key, ServiceEvidence(REMOVE)) -> ReleaseEvidence
RuntimeBackend.wake_weights(ctx, key) -> RuntimeReady
RuntimeBackend.wake_kv_and_validate(ctx, key, WeightEvidence) -> RuntimeReady
RuntimeBackend.destroy(ctx, key, CleanupPermit) -> ReleaseEvidence

LB.commit_routable(ctx, prepared, weight, ce_commit) -> CommitReceipt
LB.finish_remove(ctx, exit, ce_commit) -> CommitReceipt
LB.set_sync_barrier(token, direction, version?) -> AdmissionSnapshot
```

`prepare_exit()` owns the long drain/abort/continuation workflow internally. `remove_and_commit()` reacquires G and performs a fresh owner-side exit revalidation before CE REMOVE. There is no public `revalidate_exit()`.

ADD and REMOVE share `commit_service_change()`. There are no separate public `commit_service()` and `commit_removal()` orchestration entries.

## Operation and authorization fencing

- one unfinished lifecycle operation per task session
- same `operation_id` + same immutable business identity is replay/idempotency
- a different `operation_id` requires a strictly greater task-session `command_seq`; switching target replicas does not bypass this fence
- lease `authorization_seq` is monotonic per lease; replay of the same operation may reuse the original sequence
- the first TaskRunner acceptance freezes the operation deadline; retry budget cannot extend it
- timeout is not proof that a side effect did not execute; query the real owner and reconcile UNKNOWN

## Resource handoff closure

GS changes GPU usage rights only after a full chain is validated:

```text
ExitEvidence
  -> ServiceEvidence(action=REMOVE)
  -> ReleaseEvidence(exact GPU set)
  -> GS stores last_release_digest
  -> next ADD/RESTORE authorization.prior_release_digest matches it
```

A bare successful status, health check, ACK, process disappearance, or memory observation is not a release proof.

## Native-runtime safety boundary

The repository may implement and test orchestration/control-plane invariants in pure Python. The following remain explicit failures until validated on the supported VERL/vLLM/CUDA/NCCL combination:

- borrowed hidden runtime creation/device binding
- real native sleep/release
- target-only parameter bootstrap/replay
- wake/restore inference resources
- targeted abort + continuation for FORCE_VERIFIED
- runtime cleanup/release evidence generation from actual device/process facts

Do not replace those failures with dummy handles or synthetic success evidence.
