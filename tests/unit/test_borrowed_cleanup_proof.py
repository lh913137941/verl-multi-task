"""Fail-closed borrower cleanup proof, isolated from Ray and VERL imports."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/rollout/replica.py"
)


def replica_class(*, list_actors, kill):
    """Compile just the adapter class so these tests do not need real GPUs."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MultiTaskvLLMReplica"
    )
    node.bases = [ast.Name(id="object", ctx=ast.Load())]
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    ray = SimpleNamespace(
        kill=kill,
        get_runtime_context=lambda: SimpleNamespace(namespace="test-namespace"),
    )
    scope = {
        "asyncio": asyncio,
        "ray": ray,
        "list_actors": list_actors,
        "ReplicaKind": SimpleNamespace(NATIVE="native", BORROWED="borrowed"),
        "FIRST_RELEASE_MAX_COLOCATE_COUNT": 2,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["MultiTaskvLLMReplica"]


def test_missing_ray_actor_state_is_not_dead_proof():
    observed_filters = []
    records = []
    def list_actors(*, filters):
        observed_filters.append(filters)
        return list(records)

    cls = replica_class(list_actors=list_actors, kill=lambda *a, **k: None)
    replica = cls.__new__(cls)

    assert cls._non_dead_actor_names(("borrowed-server",), "test-namespace") == (
        "borrowed-server",
    )
    with pytest.raises(RuntimeError, match="did not reach DEAD"):
        asyncio.run(replica._wait_actor_names_dead(("borrowed-server",), timeout_s=0))
    assert observed_filters[0] == [
        ("ray_namespace", "=", "test-namespace"),
        ("name", "=", "borrowed-server"),
    ]

    records[:] = [{"state": "DEAD"}]
    asyncio.run(replica._wait_actor_names_dead(("borrowed-server",), timeout_s=0))

    records[:] = [{"state": "DEAD"}, {"state": "ALIVE"}]
    with pytest.raises(RuntimeError, match="did not reach DEAD"):
        asyncio.run(replica._wait_actor_names_dead(("borrowed-server",), timeout_s=0))


def test_unverified_worker_cleanup_retains_handle_for_exact_retry():
    states = []
    killed = []
    worker = object()
    cls = replica_class(
        list_actors=lambda **k: list(states),
        kill=lambda actor, **k: killed.append(actor),
    )
    replica = cls.__new__(cls)
    replica.replica_kind = "borrowed"
    replica.servers = []
    replica.borrowed_server_names = ()
    replica.workers = [worker]
    replica.borrowed_worker_names = ("worker-name",)
    replica.borrowed_cleanup_verified = False

    # Avoid a ten-second poll: Ray has no DEAD record on the first attempt.
    original_wait = replica._wait_actor_names_dead
    async def short_wait(names):
        return await original_wait(names, timeout_s=0)
    replica._wait_actor_names_dead = short_wait

    with pytest.raises(RuntimeError, match="cleanup is unverified"):
        asyncio.run(replica.cleanup_borrowed_runtime())
    assert replica.borrowed_cleanup_verified is False
    assert replica.workers == [worker]
    assert killed == [worker]

    states[:] = [{"state": "DEAD"}]
    asyncio.run(replica.cleanup_borrowed_runtime())
    assert replica.borrowed_cleanup_verified is True
    assert replica.workers == []
    assert killed == [worker, worker]


def test_shutdown_submission_failure_still_kills_server_and_verifies_dead():
    killed = []
    server = SimpleNamespace(
        shutdown_runtime=SimpleNamespace(
            remote=lambda: (_ for _ in ()).throw(RuntimeError("actor already stopping"))
        ),
    )
    cls = replica_class(
        list_actors=lambda **k: [{"state": "DEAD"}] if killed else [],
        kill=lambda actor, **k: killed.append(actor),
    )
    replica = cls.__new__(cls)
    replica.servers = [server]
    replica.borrowed_server_names = ("borrowed-server",)
    replica._server_handle = server
    replica._server_address = "127.0.0.1:8000"

    asyncio.run(replica._shutdown_servers_verified())
    assert killed == [server]
    assert replica.servers == []
    assert replica._server_handle is None
    assert replica._server_address is None


def test_missing_actor_names_cannot_prove_live_handle_release():
    killed = []
    cls = replica_class(
        list_actors=lambda **k: [],
        kill=lambda actor, **k: killed.append(actor),
    )
    replica = cls.__new__(cls)
    replica.workers = [object()]
    with pytest.raises(RuntimeError, match="no actor names"):
        asyncio.run(replica._kill_workers_verified(replica.workers, ()))
    assert killed == replica.workers
