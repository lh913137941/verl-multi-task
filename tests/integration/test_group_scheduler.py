"""Selected CPU Ray tests: GS discovery, handles and test-only subclass creation.

These tests do not import verl or claim validation of GPU/native trainer actors.
Run this file explicitly; it creates and removes only its own test actors.
"""

from concurrent.futures import ThreadPoolExecutor
import uuid

import pytest
import ray

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import (
    Command,
    GpuPlacement,
    NodeBlock,
    OperationContext,
    OperationResult,
    PlacementSpec,
)
from multi_task_scheduler.orchestration.operation_journal import OperationType
from multi_task_scheduler.scheduler import discovery
from multi_task_scheduler.scheduler.group_scheduler import RUNTIME_KIND
from multi_task_scheduler.scheduler.ledger import (
    IdleReport,
    LeaseRecord,
    ResourceManifest,
)


pytestmark = pytest.mark.ray_integration


@ray.remote(num_cpus=0)
class TaskRunnerProbe:
    """Test-only handle target; this is not the actual MultiTask TaskRunner."""

    def identity(self):
        return ray.get_runtime_context().get_actor_id()


@ray.remote(num_cpus=0)
class BaseProbe:
    """Test-only decorated parent; this is deliberately not a verl class."""

    def __init__(self, value):
        self.value = value

    def base_value(self):
        return self.value


@pytest.fixture
def isolated_ray(monkeypatch):
    assert not ray.is_initialized(), "Run the selected GS tests outside any existing Ray session"
    name = f"multitask-gs-test-{uuid.uuid4().hex}"
    namespace = f"multitask-test-{uuid.uuid4().hex}"
    monkeypatch.setattr(discovery, "GROUP_SCHEDULER_NAME", name)
    monkeypatch.setattr(discovery, "GROUP_SCHEDULER_NAMESPACE", namespace)
    ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False, _node_ip_address="127.0.0.1")
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


def test_discovery_requires_an_initialized_ray_runtime():
    assert not ray.is_initialized()
    with pytest.raises(RuntimeError):
        discovery.get_or_create_group_scheduler()


def test_concurrent_discovery_returns_one_real_scheduler_and_round_trips_handles(isolated_ray):
    with ThreadPoolExecutor(max_workers=4) as executor:
        schedulers = list(executor.map(lambda _: discovery.get_or_create_group_scheduler(), range(8)))
    assert all(isinstance(scheduler, ray.actor.ActorHandle) for scheduler in schedulers)
    assert len({scheduler._actor_id for scheduler in schedulers}) == 1
    scheduler = schedulers[0]
    assert ray.get(scheduler.runtime_kind.remote()) == RUNTIME_KIND
    assert ray.get(scheduler.schedule.remote()) == []

    first = TaskRunnerProbe.remote()
    second = TaskRunnerProbe.remote()
    try:
        ray.get(scheduler.attach_task.remote("task-a", first))
        ray.get(scheduler.attach_task.remote("task-a", first))
        with pytest.raises(ValueError):
            ray.get(scheduler.attach_task.remote("task-a", second))
        handles = ray.get(scheduler.get_task_runners.remote())
        assert set(handles) == {"task-a"}
        assert ray.get(handles["task-a"].identity.remote()) == ray.get(first.identity.remote())
        with pytest.raises((TypeError, ValueError)):
            ray.get(scheduler.attach_task.remote("task-b", object()))
        ray.get(scheduler.detach_task.remote("task-a"))
        ray.get(scheduler.detach_task.remote("task-a"))
        assert ray.get(scheduler.get_task_runners.remote()) == {}
    finally:
        ray.kill(first, no_restart=True)
        ray.kill(second, no_restart=True)


def test_real_ray_unwrap_subclass_and_remote_constructor_delegate_to_parent(isolated_ray):
    """Prove the Ray mechanism only, not actual verl parent initialization."""

    class ChildProbe(unwrap_native_actor_class(BaseProbe)):
        def __init__(self, value):
            super().__init__(value)
            self.child_initialized = True

        def describe(self):
            return super().base_value(), self.child_initialized, type(self).__name__

    child_actor_class = ray.remote(num_cpus=0)(ChildProbe)
    child = child_actor_class.remote(17)
    try:
        assert ray.get(child.base_value.remote()) == 17
        assert ray.get(child.describe.remote()) == (17, True, "ChildProbe")
    finally:
        ray.kill(child, no_restart=True)


def _placement():
    block = NodeBlock(
        node_id="n1",
        gpus=(GpuPlacement(gpu_uuid="u0", physical_id=0, global_rank=0, local_rank=0),),
    )
    return PlacementSpec(node_blocks=(block,), model_signature="sig-1")


def _command(operation_id="op-1", digest="d1"):
    return Command(
        protocol_version="p1", gs_epoch=1, target_task_id="task-a",
        target_task_session="s1", operation_id=operation_id, payload_digest=digest,
        kind=OperationType.ADD, lease_id="l1", lease_epoch=0,
        command_seq=0, replica_id="r1",
    )


def test_gs_control_interfaces_round_trip(isolated_ray):
    """The section 4.4 interfaces store intent and merge receipts, no policy."""
    scheduler = discovery.get_or_create_group_scheduler()
    task = TaskRunnerProbe.remote()
    try:
        attached = ray.get(scheduler.attach_controller.remote("task-a", "s1", task))
        assert attached["status"] == "INITIALIZING"

        manifest = ResourceManifest(owner_task_session="s1", placement=_placement(), replica_id="r1")
        assert ray.get(scheduler.register_resources.remote("reg-1", manifest))["status"] == "READY"

        assert ray.get(scheduler.submit_operation.remote(_command()))["state"] == "ACCEPTED"
        # Same ID, conflicting digest -> REJECTED, never overwrites intent.
        rejected = ray.get(scheduler.submit_operation.remote(_command(digest="other")))
        assert rejected["state"] == "REJECTED"

        query = ray.get(scheduler.query_operation.remote("op-1"))
        assert query["phase"] == "ACCEPTED"

        result = OperationResult(
            identity_fields=OperationContext(
                protocol_version="p1", gs_epoch=1, task_id="task-a", task_session="s1",
                operation_id="op-1", lease_id="l1", lease_epoch=0, command_seq=0,
            ),
            phase="APPLYING", phase_revision=1, state="COMMITTED",
            actual_replica_state="ACTIVE",
        )
        assert ray.get(scheduler.report_operation_result.remote(result))["merged"] is True
        assert ray.get(scheduler.query_operation.remote("op-1"))["final_result"].state == "COMMITTED"

        snapshot = ray.get(scheduler.get_resource_snapshot.remote("s1"))
        assert snapshot["expected_session"] == "s1"
        assert any(g["gpu_uuid"] == "u0" for g in snapshot["gpus"])

        idle = IdleReport(source_session="s1", source_seq=1, production_epoch=0, candidate_ids=("r1",))
        assert ray.get(scheduler.report_idle_candidates.remote(idle))["candidates"] == ["r1"]

        opened = ray.get(scheduler.open_lease.remote(LeaseRecord(lease_id="l1", donor_session="s1")))
        assert opened["state"] == "PLANNED"
        assert ray.get(scheduler.advance_lease.remote("l1", "DONOR_DRAINING"))["state"] == "DONOR_DRAINING"
    finally:
        ray.kill(task, no_restart=True)
