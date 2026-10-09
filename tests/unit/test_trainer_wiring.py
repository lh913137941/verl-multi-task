"""TRAINER wiring regression scenarios; shared test fakes live in _wiring_support."""

import asyncio
import pytest
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
    OperationRecord,
    OperationStatus,
    ReplicaKey,
    ReplicaKind,
)
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate
from _wiring_support import (
    INTEGRATION,
    isolated,
    AsyncRemoteMethod,
    trainer_class,
)


def test_trainer_add_bootstraps_current_vpub_commits_e_then_publishes_service():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {}
            self.effective_replicas = {}

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", target, operation_id, loaded_version))
            return OperationEvidence(operation_id, EvidenceType.WEIGHT_READY, 1)

        def commit_pending(self, target, evidence, *, loaded_version):
            calls.append(("commit_pending", target, loaded_version))
            replicas, _ = self.pending_bootstrap.pop(target)
            self.effective_replicas[target] = (replicas, loaded_version)

        def discard_pending(self, target):
            calls.append(("discard_pending", target))
            self.pending_bootstrap.pop(target, None)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        get_pending_replicas = AsyncRemoteMethod(lambda operation_id: (runtime,))
        commit_service_change = AsyncRemoteMethod(
            lambda operation: calls.append(("commit_service", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                2,
            )
        )
        finalize_release = AsyncRemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                AssertionError("successful ADD must not rollback runtime")
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 9

    evidence = asyncio.run(
        trainer.bootstrap_and_publish(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert trainer.checkpoint_manager.effective_replicas[key][1] == 9
    assert calls == [
        ("register_pending", key, "op-add"),
        ("bootstrap_target", key, "op-add", 9),
        ("commit_pending", key, 9),
        ("commit_service", "op-add"),
    ]


def test_trainer_add_definite_route_rejection_removes_effective_e_and_returns_release():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {}
            self.effective_replicas = {}

        def register_pending(self, target, replicas, *, operation_id):
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            return OperationEvidence(operation_id, EvidenceType.WEIGHT_READY, 1)

        def commit_pending(self, target, evidence, *, loaded_version):
            replicas, _ = self.pending_bootstrap.pop(target)
            self.effective_replicas[target] = (replicas, loaded_version)
            calls.append(("commit_pending", target, loaded_version))

        def remove_effective(self, target):
            calls.append(("remove_effective", target))
            self.effective_replicas.pop(target)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        get_pending_replicas = AsyncRemoteMethod(lambda operation_id: (runtime,))
        commit_service_change = AsyncRemoteMethod(lambda operation: None)
        finalize_release = AsyncRemoteMethod(
            lambda operation: calls.append(("finalize_release", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                2,
                ("u0",),
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 9

    evidence = asyncio.run(
        trainer.bootstrap_and_publish(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert trainer.checkpoint_manager.effective_replicas == {}
    assert trainer.replica_sync_gate.health == "HEALTHY"
    assert calls == [
        ("commit_pending", key, 9),
        ("remove_effective", key),
        ("finalize_release", "op-add"),
    ]


def test_trainer_add_bootstrap_failure_discards_pending_and_destroys_hidden_runtime():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {}

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", target, loaded_version))
            raise RuntimeError("weight transfer failed")

        def discard_pending(self, target):
            calls.append(("discard_pending", target))
            self.pending_bootstrap.pop(target, None)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        get_pending_replicas = AsyncRemoteMethod(lambda operation_id: (runtime,))
        finalize_release = AsyncRemoteMethod(
            lambda operation: calls.append(("finalize_release", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                2,
                ("u0",),
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 9

    evidence = asyncio.run(
        trainer.bootstrap_and_publish(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert trainer.replica_sync_gate.health == "HEALTHY"
    assert trainer.checkpoint_manager.pending_bootstrap == {}
    assert calls == [
        ("register_pending", key, "op-add"),
        ("bootstrap_target", key, 9),
        ("discard_pending", key),
        ("finalize_release", "op-add"),
    ]


def test_trainer_blocked_add_replay_reconciles_committed_route_and_clears_g():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()
    committed = OperationEvidence(
        "op-add",
        EvidenceType.SERVICE_COMMITTED,
        9,
    )

    class CE:
        effective_replicas = {key: ((runtime,), 7)}

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(lambda operation: committed)

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 7

    async def scenario():
        lease = await trainer.replica_sync_gate.acquire("op-add", GateKind.ADD)
        trainer.replica_sync_gate.block(lease.owner, "R outcome unknown")
        await lease.release()
        return await trainer.bootstrap_and_publish(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )

    evidence = asyncio.run(scenario())

    assert evidence == committed
    assert trainer.replica_sync_gate.health == "HEALTHY"


def test_trainer_blocked_exit_replay_reconciles_service_commit_and_clears_g():
    key = ReplicaKey("task-a", "native-0")
    committed = OperationEvidence(
        "op-donate",
        EvidenceType.SERVICE_COMMITTED,
        12,
    )

    class CE:
        effective_replicas = {}

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(lambda operation: committed)

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()

    async def scenario():
        lease = await trainer.replica_sync_gate.acquire(
            "op-donate",
            GateKind.REMOVE,
        )
        trainer.replica_sync_gate.block(lease.owner, "R/C outcome unknown")
        await lease.release()
        return await trainer.reconcile_exit(
            OperationRecord("op-donate", OperationStatus.RUNNING)
        )

    evidence = asyncio.run(scenario())

    assert evidence == committed
    assert trainer.replica_sync_gate.health == "HEALTHY"


def test_trainer_donate_parks_native_ce_member_in_existing_pending_set():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.effective_replicas = {key: ((runtime,), 7)}
            self.pending_bootstrap = {}

        def remove_effective(self, target):
            calls.append(("remove_effective", target))
            self.effective_replicas.pop(target)

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: calls.append(("commit_service", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                1,
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    evidence = asyncio.run(
        trainer.remove_and_commit(
            OperationRecord("op-donate", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert trainer.checkpoint_manager.pending_bootstrap[key] == (
        (runtime,),
        "op-donate",
    )
    assert calls == [
        ("remove_effective", key),
        ("register_pending", key, "op-donate"),
        ("commit_service", "op-donate"),
    ]


def test_trainer_borrowed_remove_does_not_park_destroyed_ce_member():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()

    class CE:
        def __init__(self):
            self.effective_replicas = {key: ((runtime,), 7)}
            self.pending_bootstrap = {}

        def remove_effective(self, target):
            self.effective_replicas.pop(target)

        def register_pending(self, *args, **kwargs):
            raise AssertionError("borrowed REMOVE must not park a destroyed runtime")

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                1,
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    evidence = asyncio.run(
        trainer.remove_and_commit(
            OperationRecord("op-remove", OperationStatus.RUNNING)
        )
    )
    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert trainer.checkpoint_manager.pending_bootstrap == {}


def test_trainer_internal_restore_rebinds_parked_native_and_publishes_current_vpub():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}

        def discard_pending(self, target):
            calls.append(("discard_pending", target))
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", target, operation_id, loaded_version))
            return OperationEvidence(
                operation_id,
                EvidenceType.WEIGHT_READY,
                2,
            )

        def commit_pending(self, target, evidence, *, loaded_version):
            calls.append(("commit_pending", target, loaded_version))
            assert evidence.type is EvidenceType.WEIGHT_READY

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: calls.append(("commit_service", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                3,
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 11

    evidence = asyncio.run(
        trainer.restore_and_publish(
            OperationRecord("op-restore", OperationStatus.RUNNING)
        )
    )
    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert calls == [
        ("discard_pending", key),
        ("register_pending", key, "op-restore"),
        ("bootstrap_target", key, "op-restore", 11),
        ("commit_pending", key, 11),
        ("commit_service", "op-restore"),
    ]


def test_trainer_restore_bootstrap_release_returns_compensation_without_blocking_g():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}

        def discard_pending(self, target):
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", loaded_version))
            return OperationEvidence(
                operation_id,
                EvidenceType.RELEASED,
                2,
                ("u0",),
            )

        def commit_pending(self, *args, **kwargs):
            raise AssertionError("compensated RESTORE must not enter E")

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                AssertionError("compensated RESTORE must not publish service")
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 11

    evidence = asyncio.run(
        trainer.restore_and_publish(
            OperationRecord("op-restore", OperationStatus.RUNNING)
        )
    )
    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert trainer.replica_sync_gate.health == "HEALTHY"
    assert calls == [("bootstrap_target", 11)]


def test_trainer_restore_unverified_bootstrap_quarantines_m_projection_and_blocks_g():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}

        def discard_pending(self, target):
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            raise RuntimeError("re-sleep proof unavailable")

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        quarantine_dormant_restore = AsyncRemoteMethod(
            lambda operation_id: calls.append(("quarantine", operation_id)) or True
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 11

    with pytest.raises(RuntimeError, match="re-sleep proof unavailable"):
        asyncio.run(
            trainer.restore_and_publish(
                OperationRecord("op-restore-unknown", OperationStatus.RUNNING)
            )
        )

    assert calls == [("quarantine", "op-restore-unknown")]
    assert trainer.replica_sync_gate.health == "BLOCKED"
    assert trainer.replica_sync_gate.blocked_operation_id == "op-restore-unknown"


def test_trainer_restore_definite_no_route_removes_e_then_resleeps():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}
            self.effective_replicas = {}

        def discard_pending(self, target):
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            return OperationEvidence(operation_id, EvidenceType.WEIGHT_READY, 2)

        def commit_pending(self, target, evidence, *, loaded_version):
            self.pending_bootstrap.pop(target, None)
            self.effective_replicas[target] = ((runtime,), loaded_version)

        def remove_effective(self, target):
            calls.append(("remove_effective", target))
            self.effective_replicas.pop(target, None)
            self.pending_bootstrap.pop(target, None)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(lambda operation: None)
        finalize_release = AsyncRemoteMethod(
            lambda operation: calls.append(("finalize_release", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                4,
                ("u0",),
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 13

    evidence = asyncio.run(
        trainer.restore_and_publish(
            OperationRecord("op-restore", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert trainer.replica_sync_gate.health == "HEALTHY"
    assert key not in trainer.checkpoint_manager.effective_replicas
    assert key in trainer.checkpoint_manager.pending_bootstrap
    assert calls == [
        ("register_pending", key, "op-restore"),
        ("remove_effective", key),
        ("register_pending", key, "op-restore"),
        ("finalize_release", "op-restore"),
    ]


def test_trainer_restore_service_failure_keeps_e_effective_and_blocks_g():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}
            self.effective_replicas = {}

        def discard_pending(self, target):
            calls.append(("discard_pending", target))
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", target, loaded_version))
            return OperationEvidence(
                operation_id,
                EvidenceType.WEIGHT_READY,
                2,
            )

        def commit_pending(self, target, evidence, *, loaded_version):
            calls.append(("commit_pending", target, loaded_version))
            self.pending_bootstrap.pop(target, None)
            self.effective_replicas[target] = ((runtime,), loaded_version)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                RuntimeError("service publish failed")
            )
        )
        quarantine_dormant_restore = AsyncRemoteMethod(
            lambda operation_id: False
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 13

    with pytest.raises(RuntimeError, match="service publish failed"):
        asyncio.run(
            trainer.restore_and_publish(
                OperationRecord("op-restore", OperationStatus.RUNNING)
            )
        )

    assert key not in trainer.checkpoint_manager.pending_bootstrap
    assert trainer.checkpoint_manager.effective_replicas[key][1] == 13
    assert ("commit_pending", key, 13) in calls
    assert trainer.replica_sync_gate.health == "BLOCKED"


@pytest.mark.parametrize("validation_fails", [False, True])
def test_trainer_native_sync_records_version_only_after_manifest_validation(validation_fails):
    events = []
    key = ReplicaKey("task-a", "native-0")
    replica = object()

    class CE:
        parameter_validation_enabled = True
        source_validation_enabled = True

        def __init__(self):
            self.effective_replicas = {key: ((replica,), 7)}

        async def _get_source_manifest(self):
            events.append("source")
            return {"complete": True}

        async def validate_parameter_sync(self, replicas, version, source):
            assert replicas == [replica]
            assert version == 11
            assert source == {"complete": True}
            events.append("validate")
            if validation_fails:
                raise RuntimeError("manifest mismatch")
            return {"state": "PARAMETERS_VALIDATED"}

        def mark_all_loaded_version(self, version):
            events.append("publish-version")
            self.effective_replicas[key] = ((replica,), version)

    class Parent:
        def __init__(self):
            self.local_trigger_step = 1
            self.current_param_version = 11
            self.checkpoint_manager = CE()

        async def _fit_update_weights(self):
            events.append("native-transfer")
            return {"timing": 1}

    cls = isolated(
        f"{INTEGRATION}/trainer.py",
        "MultiTaskFullyAsyncTrainer",
        Parent,
        GateKind=GateKind,
        ReplicaSyncGate=ReplicaSyncGate,
        asyncio=asyncio,
        json=__import__("json"),
    )
    trainer = cls()
    if validation_fails:
        with pytest.raises(RuntimeError, match="manifest mismatch"):
            asyncio.run(trainer._fit_update_weights())
        assert trainer.replica_sync_gate.health == "BLOCKED"
        assert trainer.checkpoint_manager.effective_replicas[key][1] == 7
        assert events == ["native-transfer", "source", "validate"]
    else:
        assert asyncio.run(trainer._fit_update_weights()) == {"timing": 1}
        assert trainer.replica_sync_gate.health == "HEALTHY"
        assert trainer.checkpoint_manager.effective_replicas[key][1] == 11
        assert events == ["native-transfer", "source", "validate", "publish-version"]
