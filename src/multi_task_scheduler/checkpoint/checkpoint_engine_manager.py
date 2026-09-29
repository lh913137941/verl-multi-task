"""Checkpoint Engine owner for the simplified E view."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os

import ray
from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.device import get_device_name

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
    ReplicaKey,
    ReplicaKind,
)


class MultiTaskCheckpointEngineManager(CheckpointEngineManager):
    """Keep only effective receiver membership as CE-owned mutable truth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._effective_replica_map = {}
        self._pending_bootstrap_map = {}
        self._bootstrap_ready_map = {}
        # Imported from verl_expansion main as an opt-in acceptance check.
        # The default path remains byte-for-byte native CE transfer semantics.
        self.parameter_validation_enabled = (
            os.environ.get("MULTITASK_PARAMETER_VALIDATION", "0") == "1"
            or os.environ.get("MULTITASK_SOURCE_VALIDATION", "0") == "1"
        )
        self.source_validation_enabled = (
            os.environ.get("MULTITASK_SOURCE_VALIDATION", "0") == "1"
        )

    @property
    def effective_replicas(self) -> dict:
        return self._effective_replica_map

    @property
    def pending_bootstrap(self) -> dict:
        return self._pending_bootstrap_map

    def _validate_runtime_membership(self, key: ReplicaKey, replicas: tuple) -> None:
        for index, replica in enumerate(replicas):
            if replica in replicas[:index]:
                raise ValueError("duplicate runtime in CE membership")
        for entries in (self._effective_replica_map, self._pending_bootstrap_map):
            for other_key, (other_replicas, _) in entries.items():
                if other_key != key and any(r in other_replicas for r in replicas):
                    raise ValueError("runtime already belongs to another ReplicaKey")

    def register_pending(
        self,
        key: ReplicaKey,
        replicas,
        *,
        operation_id: str,
    ) -> None:
        """Register a runtime for target-only bootstrap without making it effective."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        replicas = tuple(replicas)
        if not replicas:
            raise ValueError("pending bootstrap requires at least one replica")
        self._validate_runtime_membership(key, replicas)

        effective = self._effective_replica_map.get(key)
        if effective is not None:
            ready = self._bootstrap_ready_map.get(key)
            if effective[0] == replicas and ready is not None and ready[0] == operation_id:
                return
            raise ValueError("ReplicaKey is already an effective CE member")

        entry = (replicas, operation_id)
        existing = self._pending_bootstrap_map.get(key)
        if existing is not None:
            if existing != entry:
                raise ValueError("ReplicaKey already has conflicting pending bootstrap")
            return

        # Parent replicas remains the effective set used by native full sync.
        # Pending borrowed runtimes stay out until WEIGHT_READY is committed.
        if any(replica in self.replicas for replica in replicas):
            raise ValueError("pending replica is already part of native effective membership")
        self._pending_bootstrap_map[key] = entry

    def commit_pending(
        self,
        key: ReplicaKey,
        evidence: OperationEvidence,
        *,
        loaded_version: int,
    ) -> None:
        """Promote one pending target only after matching WEIGHT_READY evidence."""
        if not isinstance(evidence, OperationEvidence):
            raise TypeError("commit_pending requires OperationEvidence")
        if evidence.type is not EvidenceType.WEIGHT_READY:
            raise ValueError("pending bootstrap may commit only from WEIGHT_READY")
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")

        confirmed = self._bootstrap_ready_map.get(key)
        member = self._effective_replica_map.get(key)
        if key not in self._pending_bootstrap_map and confirmed is not None and member is not None:
            if confirmed != (evidence.operation_id, loaded_version, evidence):
                raise ValueError("WEIGHT_READY does not match confirmed bootstrap")
            if member[1] != loaded_version:
                raise ValueError("conflicting bootstrap commit replay")
            return

        try:
            replicas, operation_id = self._pending_bootstrap_map[key]
        except KeyError as exc:
            raise KeyError(f"no pending bootstrap for {key!r}") from exc
        if operation_id != evidence.operation_id:
            raise ValueError("WEIGHT_READY evidence belongs to another operation")
        if self._bootstrap_ready_map.get(key) != (operation_id, loaded_version, evidence):
            raise ValueError("WEIGHT_READY does not match confirmed bootstrap")

        self.add_effective(key, replicas, loaded_version=loaded_version)
        self._pending_bootstrap_map.pop(key, None)

    def discard_pending(self, key: ReplicaKey) -> None:
        self._pending_bootstrap_map.pop(key, None)
        self._bootstrap_ready_map.pop(key, None)

    def add_effective(self, key: ReplicaKey, replicas, *, loaded_version: int) -> None:
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")

        replicas = tuple(replicas)
        if not replicas:
            raise ValueError("effective membership requires at least one replica")
        self._validate_runtime_membership(key, replicas)
        entry = (replicas, loaded_version)
        existing = self._effective_replica_map.get(key)
        if existing is not None and existing != entry:
            raise ValueError("ReplicaKey already has conflicting CE membership")

        super().add_replicas(
            [replica for replica in replicas if replica not in self.replicas]
        )
        self._effective_replica_map[key] = entry

    def remove_effective(self, key: ReplicaKey) -> None:
        self.discard_pending(key)
        entry = self._effective_replica_map.pop(key, None)
        if entry is None:
            return
        replicas, _loaded_version = entry
        super().remove_replicas(list(replicas))

    def mark_all_loaded_version(self, loaded_version: int) -> None:
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")
        for key, (replicas, _old_version) in tuple(self._effective_replica_map.items()):
            self._effective_replica_map[key] = (replicas, loaded_version)

    @staticmethod
    def _validate_manifest(manifest: dict) -> None:
        """Reject missing/partial audit data before comparing receiver digests."""
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            raise RuntimeError("parameter manifest is incomplete")
        entries = manifest.get("parameters")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("parameter manifest must contain parameters")
        names = set()
        total_numel = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError("parameter manifest entry must be a mapping")
            name = entry.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise RuntimeError("parameter manifest names must be nonempty and unique")
            names.add(name)
            if any(not isinstance(entry.get(field), str) or not entry[field]
                   for field in ("dtype", "sha256")):
                raise RuntimeError("parameter manifest entry lacks dtype or sha256")
            shape = entry.get("shape")
            numel = entry.get("numel")
            if (not isinstance(shape, (list, tuple))
                    or any(type(size) is not int or size < 0 for size in shape)
                    or type(numel) is not int or numel < 0):
                raise RuntimeError("parameter manifest entry has invalid shape/numel")
            expected_numel = 1
            for size in shape:
                expected_numel *= size
            if numel != expected_numel:
                raise RuntimeError("parameter manifest shape/numel mismatch")
            total_numel += numel
        if (type(manifest.get("parameter_count")) is not int
                or manifest["parameter_count"] != len(entries)
                or type(manifest.get("total_numel")) is not int
                or manifest["total_numel"] != total_numel):
            raise RuntimeError("parameter manifest counts do not match its entries")

    @staticmethod
    def _manifest_digest(manifest: dict) -> str:
        entries = manifest.get("parameters", [])
        canonical = [
            {
                "name": item.get("name"),
                "shape": list(item.get("shape", [])),
                "dtype": item.get("dtype"),
                "numel": int(item.get("numel", 0)),
                "sha256": item.get("sha256"),
            }
            for item in entries
        ]
        canonical.sort(key=lambda item: item["name"] or "")
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def _get_source_manifest(self) -> dict:
        """Read an opt-in actor-source manifest when the backend exposes it."""
        refs = self.actor_wg.execute_checkpoint_engine(
            ["get_source_manifest"] * self.actor_wg.world_size
        )
        manifests = ray.get(refs)
        for manifest in manifests:
            if isinstance(manifest, dict) and manifest.get("complete", False):
                return manifest
        raise RuntimeError(
            f"actor source manifest is unavailable or incomplete: {manifests}"
        )

    @staticmethod
    def _manifest_mismatches(
        expected: dict,
        actual: dict,
        limit: int = 8,
    ) -> list[dict]:
        expected_entries = {
            item.get("name"): item for item in expected.get("parameters", [])
        }
        actual_entries = {
            item.get("name"): item for item in actual.get("parameters", [])
        }
        mismatches = []
        for name in sorted(set(expected_entries) | set(actual_entries)):
            source = expected_entries.get(name)
            received = actual_entries.get(name)
            if source is None or received is None:
                mismatches.append(
                    {"name": name, "source": source, "received": received}
                )
            else:
                differences = {
                    field: {
                        "source": source.get(field),
                        "received": received.get(field),
                    }
                    for field in ("shape", "dtype", "numel", "sha256")
                    if source.get(field) != received.get(field)
                }
                if differences:
                    mismatches.append(
                        {"name": name, "differences": differences}
                    )
            if len(mismatches) >= limit:
                break
        return mismatches

    async def validate_parameter_sync(
        self,
        replicas,
        expected_version: int | None = None,
        source_manifest: dict | None = None,
    ) -> dict:
        """Validate receiver manifests without moving tensor payloads to Trainer."""
        workers = []
        for replica in replicas:
            replica_workers = list(getattr(replica, "workers", ()) or ())
            if not replica_workers:
                raise RuntimeError("parameter validation requires a CE Worker for every replica")
            workers.extend(replica_workers)
        if not workers:
            raise RuntimeError(
                "parameter validation requires at least one CE Worker"
            )
        manifests = ray.get(
            [worker.get_parameter_manifest.remote() for worker in workers]
        )
        if len(manifests) != len(workers):
            raise RuntimeError("CE parameter manifest does not cover every worker")
        for manifest in manifests:
            self._validate_manifest(manifest)
        if expected_version is not None:
            mismatched_versions = [
                manifest.get("global_steps")
                for manifest in manifests
                if manifest.get("global_steps") != expected_version
            ]
            if mismatched_versions:
                raise RuntimeError(
                    "CE parameter version mismatch: "
                    f"expected={expected_version}, "
                    f"received={mismatched_versions}"
                )

        if source_manifest is not None:
            self._validate_manifest(source_manifest)
            if (
                expected_version is not None
                and source_manifest.get("global_steps") != expected_version
            ):
                raise RuntimeError(
                    "actor source parameter version mismatch: "
                    f"expected={expected_version}, "
                    f"received={source_manifest.get('global_steps')}"
                )
            for worker_index, manifest in enumerate(manifests):
                mismatches = self._manifest_mismatches(
                    source_manifest, manifest
                )
                if mismatches:
                    raise RuntimeError(
                        f"CE worker {worker_index} differs from actor "
                        f"source manifest: {mismatches}"
                    )

        digests = [self._manifest_digest(manifest) for manifest in manifests]
        if len(set(digests)) != 1:
            raise RuntimeError(
                f"CE workers received different parameter manifests: {digests}"
            )

        first = manifests[0]
        result = {
            "state": "PARAMETERS_VALIDATED",
            "version": first.get("global_steps"),
            "worker_count": len(manifests),
            "parameter_count": first.get("parameter_count", 0),
            "total_numel": first.get("total_numel", 0),
            "manifest_digest": digests[0],
        }
        if source_manifest is not None:
            result.update(
                {
                    "source_state": "SOURCE_TO_RECEIVER_VALIDATED",
                    "source_manifest_digest": self._manifest_digest(
                        source_manifest
                    ),
                }
            )
        return result

    async def bootstrap_target(
        self,
        key: ReplicaKey,
        *,
        operation_id: str,
        loaded_version: int,
    ) -> OperationEvidence:
        """Synchronize only one hidden pending target using the native CE protocol."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")
        if self.backend == "naive":
            raise NotImplementedError(
                "first-release target-only bootstrap requires non-naive checkpoint engine"
            )

        ready = self._bootstrap_ready_map.get(key)
        if ready is not None:
            ready_operation, ready_version, evidence = ready
            if ready_operation != operation_id or ready_version != loaded_version:
                raise ValueError("conflicting target bootstrap replay")
            return evidence

        try:
            replicas, pending_operation = self._pending_bootstrap_map[key]
        except KeyError as exc:
            raise KeyError(f"no pending bootstrap for {key!r}") from exc
        if pending_operation != operation_id:
            raise ValueError("pending target belongs to another operation")

        workers = [worker for replica in replicas for worker in replica.workers]
        if not workers:
            raise ValueError("pending target has no checkpoint-engine workers")

        rollout = RayWorkerGroup.from_detached(
            worker_handles=workers,
            ray_cls_with_init=replicas[0].get_ray_class_with_init_args(),
            name_prefix=f"bootstrap_{operation_id}_",
            use_gpu=True,
            device_name=get_device_name(),
        )
        actor_wg = self.actor_wg
        topology_started = False
        finalized = False
        replica_kinds = {getattr(replica, "replica_kind", None) for replica in replicas}
        if len(replica_kinds) != 1:
            raise ValueError("pending target contains inconsistent replica kinds")
        replica_kind = next(iter(replica_kinds))
        if replica_kind not in {ReplicaKind.NATIVE, ReplicaKind.BORROWED}:
            raise ValueError("pending target has unsupported replica kind")
        native_restore = replica_kind is ReplicaKind.NATIVE
        released_gpu_uuids: tuple[str, ...] = ()

        try:
            # Native RESTORE allocates only weight memory under the same G that
            # serializes parameter publication. ADD targets are already resident.
            if native_restore:
                placement_sets = await asyncio.gather(
                    *[replica.worker_placements() for replica in replicas]
                )
                released_gpu_uuids = tuple(
                    dict.fromkeys(
                        placement["gpu_uuid"]
                        for placements in placement_sets
                        for placement in placements
                        if isinstance(placement, dict)
                        and isinstance(placement.get("gpu_uuid"), str)
                        and placement["gpu_uuid"]
                    )
                )
                if not released_gpu_uuids:
                    raise RuntimeError(
                        "native RESTORE target has no verified physical accelerator id"
                    )
                await asyncio.gather(
                    *[replica.wake_up(tags=["weights"]) for replica in replicas]
                )

            # Borrowed ADD targets are resident and use VERL's native
            # KV-release path. Native RESTORE already has KV absent after the
            # weights-only wake above.
            if not native_restore:
                await asyncio.gather(
                    *[replica.release_kv_cache() for replica in replicas]
                )

            topology_started = True
            self.build_process_group(rollout)

            # Keep native VERL synchronization semantics here. Its
            # CheckpointEngineManager.update_weights() is async but deliberately
            # uses blocking ray.get() for the transfer/finalize boundary. Moving
            # only this target path to a background thread/future would let the
            # Trainer event loop progress while G still protects an in-flight
            # collective, diverging from the native ordering contract.
            ray.get(
                actor_wg.update_weights(
                    global_steps=loaded_version,
                    mode=self.backend,
                )
                + rollout.update_weights(global_steps=loaded_version)
            )

            ray.get(
                actor_wg.execute_checkpoint_engine(
                    ["finalize"] * actor_wg.world_size
                )
                + rollout.execute_checkpoint_engine(
                    ["finalize"] * rollout.world_size
                )
            )
            finalized = True

            await asyncio.gather(
                *[replica.resume_kv_cache() for replica in replicas]
            )
            health = await asyncio.gather(
                *[replica.validate_server_runtime() for replica in replicas]
            )
            if any(item.get("global_steps") != loaded_version for item in health):
                raise RuntimeError(
                    "target server did not confirm the published parameter version"
                )

            if self.parameter_validation_enabled:
                source_manifest = (
                    await self._get_source_manifest()
                    if self.source_validation_enabled
                    else None
                )
                validation = await self.validate_parameter_sync(
                    replicas,
                    expected_version=loaded_version,
                    source_manifest=source_manifest,
                )
                print(
                    "CE_PARAMETER_VALIDATION "
                    + json.dumps(validation, sort_keys=True)
                )

            evidence = OperationEvidence.now(
                operation_id,
                EvidenceType.WEIGHT_READY,
            )
            self._bootstrap_ready_map[key] = (
                operation_id,
                loaded_version,
                evidence,
            )
            return evidence
        except BaseException as exc:
            cleanup_error = None
            if topology_started and not finalized:
                try:
                    ray.get(
                        actor_wg.execute_checkpoint_engine(
                            ["finalize"] * actor_wg.world_size
                        )
                        + rollout.execute_checkpoint_engine(
                            ["finalize"] * rollout.world_size
                        )
                    )
                except BaseException as finalize_exc:
                    cleanup_error = finalize_exc

            # A failed native RESTORE must not escape with a partially awake
            # retained runtime.  Re-enter proven level-2 sleep while G is still
            # held; server admission never opens on this path.
            if native_restore:
                try:
                    await asyncio.gather(*[replica.sleep() for replica in replicas])
                except BaseException as rollback_exc:
                    cleanup_error = rollback_exc

            if cleanup_error is not None:
                raise RuntimeError(
                    "target bootstrap failed and runtime cleanup is unverified"
                ) from cleanup_error
            if native_restore:
                # A failed RESTORE that has been proven back in level-2 sleep is
                # a resolved compensation outcome, not an UNKNOWN synchronization
                # result. Reuse RELEASED so TaskRunner/GS can release only the
                # temporary RESTORE reservation without claiming business success.
                return OperationEvidence.now(
                    operation_id,
                    EvidenceType.RELEASED,
                    released_gpu_uuids=released_gpu_uuids,
                )
            raise exc
