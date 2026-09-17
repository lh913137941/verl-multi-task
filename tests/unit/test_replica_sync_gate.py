"""Replica-sync gate: exclusion, fencing, reentry, timeout, guard."""

import asyncio

import pytest

from multi_task_scheduler.orchestration.replica_sync_gate import (
    GateFencedError,
    GateKind,
    GateReentryError,
    GateTimeoutError,
    ReplicaSyncGate,
)


def run(coro):
    return asyncio.run(coro)


def test_acquire_sets_owner_and_release_clears_it():
    async def scenario():
        gate = ReplicaSyncGate()
        assert gate.owner is None
        lease = await gate.acquire("op-1", GateKind.ADD)
        assert gate.owner is not None
        assert gate.owner.operation_id == "op-1"
        assert gate.owner.kind is GateKind.ADD
        assert lease.active
        assert await lease.release() is True
        assert gate.owner is None
        assert lease.active is False

    run(scenario())


def test_epoch_increments_per_acquisition():
    async def scenario():
        gate = ReplicaSyncGate()
        first = await gate.acquire("op-1", GateKind.ADD)
        await first.release()
        second = await gate.acquire("op-2", GateKind.REMOVE)
        assert second.owner.epoch > first.owner.epoch
        await second.release()

    run(scenario())


def test_reentrant_acquire_is_rejected():
    async def scenario():
        gate = ReplicaSyncGate()
        lease = await gate.acquire("op-1", GateKind.ADD)
        with pytest.raises(GateReentryError):
            await gate.acquire("op-1", GateKind.ADD)
        await lease.release()

    run(scenario())


def test_timeout_while_held_and_recovery_after_release():
    async def scenario():
        gate = ReplicaSyncGate()
        holder = await gate.acquire("op-1", GateKind.ADD)
        with pytest.raises(GateTimeoutError):
            await gate.acquire("op-2", GateKind.REMOVE, timeout=0.01)
        await holder.release()
        waiter = await gate.acquire("op-2", GateKind.REMOVE, timeout=0.05)
        assert waiter.active
        await waiter.release()

    run(scenario())


def test_fencing_blocks_protected_writes_after_release():
    async def scenario():
        gate = ReplicaSyncGate()
        lease = await gate.acquire("op-1", GateKind.ADD)
        await lease.release()
        with pytest.raises(GateFencedError):
            await lease.guard(lambda: 1)

    run(scenario())


def test_guard_runs_sync_and_async_callables():
    async def scenario():
        gate = ReplicaSyncGate()
        lease = await gate.acquire("op-1", GateKind.ADD)

        assert await lease.guard(lambda x: x + 1, 41) == 42

        async def work(x):
            return x * 2

        assert await lease.guard(work, 21) == 42
        await lease.release()

    run(scenario())


def test_async_context_manager_releases_on_exit():
    async def scenario():
        gate = ReplicaSyncGate()
        lease = await gate.acquire("op-1", GateKind.ADD)
        async with lease as active:
            assert active is lease
            assert active.active
        assert lease.active is False
        assert gate.owner is None

    run(scenario())


def test_context_manager_blocks_on_inactive_lease():
    async def scenario():
        gate = ReplicaSyncGate()
        lease = await gate.acquire("op-1", GateKind.ADD)
        await lease.release()
        with pytest.raises(GateFencedError):
            async with lease:
                pass

    run(scenario())


def test_unknown_failure_blocks_future_owners_including_already_waiting_ones():
    async def scenario():
        gate = ReplicaSyncGate()
        owner = await gate.acquire("op-1", GateKind.ADD)
        waiter = asyncio.create_task(gate.acquire("op-2", GateKind.NATIVE_SYNC))
        await asyncio.sleep(0)
        gate.block(owner.owner, "transfer completion is unknown")
        await owner.release()
        with pytest.raises(GateFencedError):
            await waiter
        with pytest.raises(GateFencedError):
            await gate.acquire("op-3", GateKind.REMOVE)
        assert gate.health == "BLOCKED"
        assert gate.owner is None

    run(scenario())


def test_async_guard_cannot_return_success_after_its_lease_was_released():
    async def scenario():
        gate = ReplicaSyncGate()
        lease = await gate.acquire("op-1", GateKind.ADD)

        async def late_result():
            await lease.release()
            return "looks successful"

        with pytest.raises(GateFencedError):
            await lease.guard(late_result)

    run(scenario())
