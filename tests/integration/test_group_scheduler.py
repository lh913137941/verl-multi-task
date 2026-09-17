"""Selected CPU Ray tests for GS discovery and current orchestration contracts.

These tests do not import verl GPU actors and do not claim CUDA/NCCL validation.
"""

from concurrent.futures import ThreadPoolExecutor
import uuid

import pytest
import ray

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import (
    IdleCandidate,
    IdleCandidateReport,
    LeaseAuthorization,
    NodePlacement,
    OperationCommand,
    OperationContext,
    OperationResult,
    PlacementSpec,
    QueryResult,
    ReplicaKey,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationKind,
    OperationStatus,
    Outcome,
    Phase,
)
from multi_task_scheduler.scheduler import discovery
from multi_task_scheduler.scheduler.group_scheduler import RUNTIME_KIND
from multi_task_scheduler.scheduler.ledger import LeaseRecord, ResourceManifest

pytestmark = pytest.mark.ray_integration


@ray.remote(num_cpus=0)
class TaskRunnerProbe:
    def __init__(self):
        self.operations = {}

    def identity(self):
        return ray.get_runtime_context().get_actor_id()

    def submit_operation(self, command):
        result = OperationResult(
            ctx=command.ctx,
            target=command.target,
            status=OperationStatus.ACCEPTED,
            phase=Phase.VALIDATE,
            phase_revision=0,
        )
        self.operations[command.ctx.operation_id] = result
        return result

    def query_operation(self, task_session, operation_id):
        result = self.operations.get(operation_id)
        if result is None or result.ctx.task_session != task_session:
            return QueryResult(found=False, value=None, outcome=Outcome.UNKNOWN)
        return QueryResult(
            found=True,
            value=result,
            outcome=Outcome.KNOWN_NOT_APPLIED,
        )

    def probe_task(self, task_session):
        return {"task_session": task_session, "probe": "ok"}


@ray.remote(num_cpus=0)
class BaseProbe:
    def __init__(self, value):
        self.value = value

    def base_value(self):
        return self.value


@pytest.fixture
def isolated_ray(monkeypatch):
    assert not ray.is_initialized()
    name = f"multitask-gs-test-{uuid.uuid4().hex}"
    namespace = f"multitask-test-{uuid.uuid4().hex}"
    monkeypatch.setattr(discovery, "GROUP_SCHEDULER_NAME", name)
    monkeypatch.setattr(discovery, "GROUP_SCHEDULER_NAMESPACE", namespace)
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
    )
    try:
        yield
    finally:
        try:
            scheduler = ray.get_actor(name, namespace=namespace)
        except ValueError:
            pass
        else:
            ray.kill(scheduler, no_restart=True)
        ray.shutdown()


def _placement():
    return PlacementSpec(
        node=NodePlacement(
            node_id="n1",
            gpu_uuids=("u0",),
            physical_gpu_ids=(0,),
            global_ranks=(0,),
            local_ranks=(0,),
        ),
        tp=1,
        dp=1,
        pp=1,
        model_signature="sig-1",
        placement_digest="placement-u0",
    )


def _command(protocol_version, gs_epoch, *, operation_id="op-1", digest="d1"):
    ctx = OperationContext(
        protocol_version=protocol_version,
        gs_epoch=gs_epoch,
        task_id="task-a",
        task_session="s1",
        operation_id=operation_id,
        lease_id="l1",
        lease_epoch=0,
        command_seq=0,
    )
    target = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    authorization = LeaseAuthorization(
        lease_id="l1",
        gs_epoch=gs_epoch,
        donor_session="donor",
        borrower_session="s1",
        placement_digest="placement-u0",
        lease_epoch=0,
        purpose=OperationKind.ADD,
        prior_release_digest="release-0",
        authorization_seq=1,
    )
    return OperationCommand(
        ctx=ctx,
        kind=OperationKind.ADD,
        target=target,
        authorization=authorization,
        payload_digest=digest,
        remaining_budget_ms=1000,
        placement=_placement(),
    )


def _idle_report(gs_epoch):
    candidate = IdleCandidate(
        key=ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0),
        production_epoch=3,
        source_seq=1,
        manager_revision=1,
        lb_revision=1,
        engine_digest="engine-1",
        reason="EXHAUSTED",
        stable_idle_ms=500,
        observed_age_ms=10,
        gpu_count=1,
        evidence_digest="idle-1",
        placement_digest="placement-u0",
    )
    return IdleCandidateReport(
        task_session="s1",
        gs_epoch=gs_epoch,
        lb_session="lb-1",
        source_seq=1,
        production_revision=3,
        valid_for_ms=1000,
        candidates=(candidate,),
    )


def test_discovery_requires_initialized_ray_runtime():
    assert not ray.is_initialized()
    with pytest.raises(RuntimeError):
        discovery.get_or_create_group_scheduler()


def test_concurrent_discovery_returns_one_scheduler_and_round_trips_handles(isolated_ray):
    with ThreadPoolExecutor(max_workers=4) as executor:
        schedulers = list(
            executor.map(lambda _: discovery.get_or_create_group_scheduler(), range(8))
        )
    assert all(isinstance(item, ray.actor.ActorHandle) for item in schedulers)
    assert len({item._actor_id for item in schedulers}) == 1
    scheduler = schedulers[0]
    assert ray.get(scheduler.runtime_kind.remote()) == RUNTIME_KIND

    first = TaskRunnerProbe.remote()
    second = TaskRunnerProbe.remote()
    try:
        ray.get(scheduler.attach_task.remote("task-a", first))
        ray.get(scheduler.attach_task.remote("task-a", first))
        with pytest.raises(ValueError):
            ray.get(scheduler.attach_task.remote("task-a", second))
        assert ray.get(scheduler.get_task_runners.remote())["task-a"] == first
        ray.get(scheduler.detach_task.remote("task-a"))
        assert ray.get(scheduler.get_task_runners.remote()) == {}
    finally:
        ray.kill(first, no_restart=True)
        ray.kill(second, no_restart=True)


def test_real_ray_unwrap_subclass_delegates_to_parent(isolated_ray):
    class ChildProbe(unwrap_native_actor_class(BaseProbe)):
        def describe(self):
            return super().base_value(), type(self).__name__

    actor = ray.remote(num_cpus=0)(ChildProbe).remote(17)
    try:
        assert ray.get(actor.describe.remote()) == (17, "ChildProbe")
    finally:
        ray.kill(actor, no_restart=True)


def test_gs_control_interfaces_use_current_typed_contract(isolated_ray):
    scheduler = discovery.get_or_create_group_scheduler()
    task = TaskRunnerProbe.remote()
    try:
        attached = ray.get(scheduler.attach_controller.remote("task-a", "s1", task))
        protocol_version = attached["protocol_version"]
        gs_epoch = attached["gs_epoch"]
        assert type(protocol_version) is int
        assert isinstance(gs_epoch, str) and gs_epoch

        manifest = ResourceManifest(
            owner_task_session="s1", placement=_placement(), replica_id="r1"
        )
        assert ray.get(
            scheduler.register_resources.remote("reg-1", manifest)
        )["status"] == "READY"

        # This test starts at the ADD authorization boundary. The donor release
        # itself is covered by the pure-Python lease evidence tests.
        opened = ray.get(
            scheduler.open_lease.remote(
                LeaseRecord(
                    lease_id="l1",
                    donor_session="donor",
                    borrower_session="s1",
                    state="BORROWER_PREPARING",
                    last_release_digest="release-0",
                    placement=_placement(),
                )
            )
        )
        assert opened["state"] == "BORROWER_PREPARING"

        command = _command(protocol_version, gs_epoch)
        accepted = ray.get(scheduler.submit_operation.remote(command))
        assert accepted.status is OperationStatus.ACCEPTED
        assert accepted.phase is Phase.VALIDATE

        with pytest.raises(ValueError, match="conflicting replay"):
            ray.get(
                scheduler.submit_operation.remote(
                    _command(protocol_version, gs_epoch, digest="other")
                )
            )

        with pytest.raises(ValueError, match="stale gs_epoch"):
            ray.get(
                scheduler.submit_operation.remote(
                    _command(protocol_version, "old-gs", operation_id="op-stale")
                )
            )

        query = ray.get(scheduler.query_operation.remote("s1", "op-1"))
        assert query.found is True
        assert query.value.status is OperationStatus.ACCEPTED
        assert query.outcome is Outcome.KNOWN_NOT_APPLIED

        idle_ack = ray.get(scheduler.report_idle_candidates.remote(_idle_report(gs_epoch)))
        assert idle_ack.accepted is True
        assert idle_ack.revision == 1

        assert ray.get(scheduler.probe_task.remote("task-a", "s1")) == {
            "task_session": "s1",
            "probe": "ok",
        }

        second_lease = ray.get(
            scheduler.open_lease.remote(
                LeaseRecord(
                    lease_id="l2",
                    donor_session="donor",
                    borrower_session="s1",
                    placement=_placement(),
                )
            )
        )
        assert second_lease["state"] == "PLANNED"
        assert ray.get(
            scheduler.advance_lease.remote("l2", "DONOR_DRAINING")
        )["state"] == "DONOR_DRAINING"
    finally:
        ray.kill(task, no_restart=True)
