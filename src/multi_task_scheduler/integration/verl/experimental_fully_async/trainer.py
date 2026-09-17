# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep the native Trainer behavior and add the gate + published-version holder.

The Trainer owns the task-local replica-sync gate (G, section 6) and the
published serving version (Vpub). The three transaction entry points below
encode the section 5.2-5.5 flows; each is an explicit failure until the native
backend is verified and the protocol adapters are injected, never faked.
``init_workers`` / ``fit`` stay inherited. Native weight synchronization is
delegated under the same task-local gate; uncertainty blocks subsequent work.
"""

import ray

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.utils.config import omega_conf_to_dataclass

from multi_task_scheduler.checkpoint.checkpoint_engine_manager import MultiTaskCheckpointEngineManager
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate


@ray.remote(num_cpus=10)
class MultiTaskFullyAsyncTrainer(unwrap_native_actor_class(FullyAsyncTrainer)):
    """Own the CE Manager in this Actor; inherit training and weight synchronization."""

    async def _setup_checkpoint_manager(self):
        """Preserve native trainer.py:217-224; replace only the Manager class."""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = MultiTaskCheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas
        )
        print(f"[FullyAsyncTrainer] Checkpoint manager initialized (backend={checkpoint_engine_config.backend})")

    # -- gate + published-version holder (section 6 / Vpub) ---------------- #

    def _ensure_gate(self) -> ReplicaSyncGate:
        # Lazy so the inherited native Trainer constructor stays untouched.
        if not hasattr(self, "_replica_sync_gate"):
            self._replica_sync_gate = ReplicaSyncGate()
        return self._replica_sync_gate

    @property
    def replica_sync_gate(self) -> ReplicaSyncGate:
        return self._ensure_gate()

    async def _fit_update_weights(self):
        """Keep native sync/reset behavior, serialized with lifecycle commits.

        A Python exception or cancellation does not prove outstanding device
        work stopped. Keep health BLOCKED until verified reconciliation exists.
        This wrapper does not manufacture a published tensor snapshot.
        """
        if self.local_trigger_step != 1:
            return None
        gate = self.replica_sync_gate
        lease = await gate.acquire(f"native-sync:{self.current_param_version}", GateKind.NATIVE_SYNC)
        try:
            return await lease.guard(super()._fit_update_weights)
        except BaseException as exc:
            gate.block(lease.owner, f"Native synchronization outcome unknown: {type(exc).__name__}")
            raise
        finally:
            await lease.release()

    @property
    def published_serving_version(self) -> int:
        return getattr(self, "_published_serving_version", 0)

    def publish_serving_version(self, version: int) -> None:
        """Advance Vpub only under G after a verified native sync (section 6)."""
        self._published_serving_version = version

    # -- transaction entry points (section 5.2-5.5) ------------------------ #

    async def bootstrap_and_publish(self, ctx, prepared):
        """ADD: bootstrap a hidden target and publish it, holding G."""
        raise NotImplementedError(
            "ADD transaction requires verified native backend"
        )

    async def remove_and_commit(self, ctx, replica_id: str, purpose: str = "recall"):
        """REMOVE: drain, exclude from CE, then destroy the borrower runtime."""
        raise NotImplementedError(
            "REMOVE transaction requires verified native backend"
        )

    async def restore_and_publish(self, ctx, prepared, fence_satisfied: bool):
        """RESTORE: wake weights, bootstrap, validate, then republish."""
        raise NotImplementedError(
            "RESTORE transaction requires verified native backend"
        )
