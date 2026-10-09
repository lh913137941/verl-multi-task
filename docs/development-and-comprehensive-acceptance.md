# 多 RL 任务共享调度：开发与综合验收

> 范围：`verl-multi-task` 的 `chatgpt/0928-merge-verl-expansion` 分支（2026-10-09）。以代码可观察行为为准，设计意图与真实硬件验收结果严格分开。本文件为工程交付与验收索引；详细约束见 [实现合同](simplified-fusion-contract.md)、[详细设计](多RL任务共享调度对接VERL_动态流程编排融合设计精简优化版092401.md)、[E2E 标准](e2e-acceptance.md)。

# 第一部分：开发

## 1.1 交付范围与非测试代码文件清单

**目标**：在 VERL experimental Fully Async + 独立 vLLM Rollout 基础上增加多个 RL 任务的共享 GS、整卡资源借还、参数/服务同步、请求安全交接与可核验恢复。沿用原生训练入口和更新循环，不创建平行训练框架。

以下列出 `src/multi_task_scheduler/` 中的全部非测试实现文件（`__init__.py` 仅负责包声明或导出，不逐项列入）；`testing/` 下属于测试辅助实现，另列为验收工具：

| 模块 | 交付文件 | 主要责任 |
| --- | --- | --- |
| 调度 | `scheduler/group_scheduler.py`、`scheduler/discovery.py` | 全局 GS、任务注册与发现、Lease 账本、操作分发 |
| 编排合同 | `orchestration/contracts.py`、`operation_journal.py`、`replica_sync_gate.py` | 操作/副本/Lease 类型、幂等记录、训练与生命周期串行闸门 |
| VERL 接入 | `integration/verl/ray_actor.py`、`runtime_profile.py` | 原生 Ray Actor 包装、运行 profile 与能力准入 |
| 任务编排 | `integration/verl/experimental_fully_async/task_runner.py` | 原生 TaskRunner 扩展、GS attach、ADD/DONATE/REMOVE/RESTORE 执行与查询 |
| Trainer | `integration/verl/experimental_fully_async/trainer.py` | ADD bootstrap、RESTORE、参数同步、服务提交与回滚 |
| Rollouter | `integration/verl/experimental_fully_async/rollouter.py` | 目标 Replica 准备、退出、服务容量变更和运行状态 |
| Replica 管理 | `integration/verl/experimental_fully_async/llm_server_manager.py` | Native/Borrowed 生命周期、资源归属与真实设备证据 |
| 样本队列 | `integration/verl/experimental_fully_async/message_queue.py` | 完成样本的恰好一次提交语义 |
| 推理服务 | `rollout/http_server.py`、`load_balancer.py`、`replica.py` | HTTP/vLLM 服务、路由、attempt 事实、Replica runtime |
| 参数同步 | `checkpoint/checkpoint_engine_manager.py`、`checkpoint_engine_worker.py`、`hccl_checkpoint_engine.py` | 目标集装参、版本/manifest 验证、NCCL 或 HCCL 后端 |
| 验收辅助（非生产路径） | `testing/npu_restore_sender.py`、`testing/startup_diagnostics.py` | NPU RESTORE 实验数据与启动诊断 |

辅助交付：`pyproject.toml`、`patches/verl-v0.10-fully-async-multitask-entry.patch`、`scripts/e2e/`、`examples/e2e/native_args.txt`、`requirements/ut.txt`。**测试源码**独立列在第二部分，避免混淆产品实现与验收脚本。

## 1.2 组件关系、身份和状态

```text
两个或更多 VERL fully_async_main 任务
   └─ 每任务一个 MultiTaskFullyAsyncTaskRunner
       ├─ Trainer ── CheckpointEngineManager ── CE workers
       ├─ Rollouter ── LLMServerManager ── Native/Borrowed replicas
       │                          └───── vLLM HTTP server
       ├─ LoadBalancer：route / request / attempt ownership
       └─ MessageQueue：完成样本提交
                ↑
        共享 detached GroupScheduler
         (注册 / Lease / operation)
```

- **任务身份**：`task_session` 标识实际训练任务；不能用进程 PID 替代。
- **副本身份**：`ReplicaKey(task_session, replica_id, runtime_epoch)`。runtime 重建会变化，不能凭 `replica_id` 混淆不同实例。
- **操作身份**：`operation_id` 用作命令提交、终态查询、证据关联、重试/重放的稳定键；`OperationCommand` 指定 `kind`、`target`、`lease_id` 与可选 `force`。
- **资源身份**：Lease 的 claims 记录真实 `pg_id`、`bundle_index`、`node_id`、`gpu_uuid` 等放置事实；Ray 逻辑资源份额不等于物理半张卡。
- **状态/Owner**：GS 管任务与 Lease；TaskRunner 管 OperationRecord；Manager 管 Replica 物理生命周期；CE Manager 管参数成员和已加载版本；LB 管路由与 attempt；Rollouter 管服务容量；Queue 管样本提交。不得由单一状态字段替代跨 Owner 证据。
- **结果语义**：完成成功需对应 `OperationEvidence`（如 `RELEASED` / `WEIGHT_READY` / `SERVICE_COMMITTED`）；FAILED、UNKNOWN、QUARANTINED 不能伪造成成功。

## 1.3 插件入口、GS 与任务级编排

1. VERL 原生入口 `verl.experimental.fully_async_policy.fully_async_main` 根据 `multitask.enabled` 选择 TaskRunner。验证当前导入入口：`python scripts/e2e/ensure_verl_multitask_bridge.py --check-only`；不要假设安装包会自动修改任意 VERL checkout。
2. `runtime_profile.py` 限制受支持的 independent/non-PD vLLM 配置，不符合真实能力条件则 fail-closed；设备、后端与 TP/DP/PP 组合须按环境验收。
3. TaskRunner 初始化组件后向共享 GS `attach_task(task_id, task_runner)` 注册，退出时 `detach_task(task_id)`；丢失 ACK 需要查真实注册状态，不直接创建重复任务。
4. GS 的 `open_lease(lease)` 和 `advance_lease(lease_id, evidence)` 是内部 Lease 账本动作。`submit_operation(command)` 分发操作；`submit_idle_report(report)` 接受资源空闲观察。GS 不负责直接调用设备 sleep/wake。
5. TaskRunner 通过 `submit_operation` 和 `query_operation` 提供任务侧操作接入；OperationJournal 处理接受、执行中、完成和重复命令。重试使用同一 `operation_id`，而非另发不相关的新操作。

## 1.4 生命周期主要流程及证据

| 流程 | 主要执行路径 | 终态证据与回退 |
| --- | --- | --- |
| DONATE | Native 停新请求、在途 drain、释放服务/参数成员，Manager sleep 并核对设备 | `RELEASED`；未证实实际释放不得开放借卡 |
| ADD | GS Lease → TaskRunner 解析 spec → Rollouter `prepare_replica` → Trainer `bootstrap_and_publish` → CE 目标权重同步 → 提交服务 | `SERVICE_COMMITTED`；失败时回滚隐藏 runtime，确认 `RELEASED` |
| 自然 REMOVE | Borrowed 停新请求、旧请求结清、撤销路由与参数成员、销毁 runtime | `RELEASED`；物理销毁不明确时隔离 |
| RESTORE | Native runtime wake、同步最新 Vpub、恢复 KV、再公布服务 | `SERVICE_COMMITTED`；失败必须维持不接流量/隔离并对账 |
| FORCE REMOVE | 仅 BORROWED、需要部分生成与安全续推能力，定向中断、handoff 证明、撤销旧路由和物理回收 | `RELEASED` 且 request attempt 账目闭合；不能把简单终止当作续推完成 |

**核心顺序**：DONATE → ADD → REMOVE（natural / FORCE）→ RESTORE。各阶段采用操作标识、身份和证据关联；不可通过 mock 成功或配置开关提前宣称资源释放。

## 1.5 参数同步、服务发布和安全门

Trainer 中的 `ReplicaSyncGate` 统一串行保护原生版本推进与 ADD/RESTORE 等权重成员变更。CheckpointEngineManager 负责对目标 Replica 同步当前 `Vpub`、版本确认与 pending/effective membership；CUDA/NCCL 和 NPU/HCCL 的真实传输实现按后端分开验收。完成 `WEIGHT_READY` 不等于路由已生效，须经 Rollouter 服务发布才返回 `SERVICE_COMMITTED`。

ADD 关键调用依次为 `prepare_replica`、`bootstrap_target`、`commit_pending`、`commit_service_change`。当 bootstrap 失败且隐藏 runtime 已经验证 `RELEASED`，操作仍记为 FAILED，而不是伪造成功；若销毁/参数成员状态不可确认，保持安全阻断。

## 1.6 请求路由、FORCE 交接与 Exactly-once

LB 对 request 与 attempt 的准入、在途 owner、终态承担单写责任；FORCE 必须基于真实 targeted abort 和 continuation proof 判断请求安全交接。需要验证 `request_id` 稳定、可交接前缀及其摘要、旧 attempt 终止状态、新 Replica 延续生成、最终完成样本不重复。配置 `partial_rollout=true` 本身不构成接管证据。

MessageQueue 约束完整样本恰好一次提交；重复提交的幂等性、冲突拒绝与异常重放必须分别验收，不能与 LB 的 request attempt 状态混为一谈。

## 1.7 失败退出、恢复与当前已知缺口

操作在 ACK 丢失、timeout、真实 Actor 异常时按原 `operation_id` 查询 Owner 事实；UNKNOWN 或证据缺失不直接重发非幂等设备动作。ADD 失败回滚包括 pending 清理及隐藏 runtime `finalize_release`；RESTORE 失败需防止未完成参数恢复的 Native 重新进入服务；FORCE 缺少真实 handoff 时不得把在途请求视为可安全终止。

**2026-10-09 NPU E2E 历史失败（08:43 轮次）**：`logs/two_real_jobs/20261009-084341/` 的 `force_cycle` 实际在第二次 ADD 失败，报 `ADD bootstrap failed; hidden runtime was verified RELEASED`，因此未进入 FORCE。日志显示 `checkpoint-finalize-complete` 与 `kv-resume-complete`，随后有 EngineCore shutdown 和 SIGTERM，但尚不足以确定底层根因。已在本分支 Trainer `bootstrap_and_publish` 异常捕获处增加包含 `operation_id`、`target`、`e_committed` 和 traceback 的日志（提交 `35ed25a`）；**仅增强诊断；后续 09:42 轮次的 5/5 PASS 说明该次未复现，但不能据此判定 ADD 的历史根因已经消除**。

## 1.8 各模块核心字段、方法与原生复用点

本节以代码中的类成员和方法名称为依据；**字段只列影响跨组件合同或生命周期的核心成员**。以下表格中的内部属性和以下划线开头的方法不构成新增公共 API；部分字段为基类继承或属性访问器，调用方应通过已有 Owner 方法读取。返回值只在已确认的现有接口中列出。

### 1.8.1 GS、发现与操作合同

| 组件 | 核心字段 / 状态（Owner） | 关键方法（作用 / 返回） | 复用点 |
| --- | --- | --- | --- |
| `scheduler/group_scheduler.py` | 任务与 TaskRunner 注册映射、空闲上报、Lease claims/进度账本、操作路由记录（GS 内部） | `attach_task(task_id, task_runner) -> None` 注册；`detach_task(task_id) -> None` 注销；`get_task_runners() -> dict`；`submit_idle_report(report)` 收集候选；`submit_operation(command) -> OperationRecord`；`open_lease(lease) -> Lease`；`advance_lease(lease_id, evidence) -> dict` | Ray detached Actor、`ActorHandle`、真实 PG/claims；不代替 ResourcePool |
| `scheduler/discovery.py` | GS 命名发现入口，无另设跨任务状态 Owner | `get_or_create_group_scheduler()`：发现或创建共享 GS | Ray named/detached actor |
| `orchestration/contracts.py` | `ReplicaKey(task_session, replica_id, runtime_epoch)`；`OperationCommand(operation_id, kind, target, lease_id, force)`；`OperationRecord`；`OperationEvidence`；`Lease(lease_id, claims, expires_at)`；`ReplicaKind / ReplicaState / AttemptState / OperationStatus / EvidenceType` | 各类型的 `__post_init__` 校验；`OperationEvidence.now(...)`；Lease 的 `claim_ids()`、`source_lease_ids()`、`bundle_keys()`、`gpu_uuids()` | 标准 `dataclass` / Enum 和 VERL/Ray 传参；不重复定义设备实体 |
| `orchestration/operation_journal.py` | 按 `operation_id` 维护 command 与 record（OperationJournal 内部） | `begin(command) -> OperationRecord`；`query(id) -> OperationRecord | None`；`command(id) -> OperationCommand`；`mark_running(id)`、`reopen_unknown(id)`、`finish(...)` | 同一 command 幂等重放、终态保留；不重复执行设备操作 |
| `orchestration/replica_sync_gate.py` | `_lock`、`_epoch`、`_owner`、`_blocked_reason`、`_blocked_operation_id` | `acquire(...)` 返回 GateLease；`GateLease.guard(call, ...)` 串行执行；`release()`；`block(owner, reason)`；`reconcile(...)`；`health / owner / blocked_reason` 查询 | `asyncio.Lock` 与原生 Trainer 参数同步时序；不新增第二套权重传输 |

### 1.8.2 VERL 插件与任务、训练编排

| 组件 | 核心字段 / 状态 | 关键方法 | 复用点 |
| --- | --- | --- | --- |
| `integration/verl/runtime_profile.py` | 唯一受支持 Fully Async STANDALONE profile 的能力条件（非持久状态） | `resolve_runtime_profile(config)`、`validate_runtime_profile(config)`；非法组合抛 `ProfileConfigurationError` | VERL 原生 Hydra 配置与运行时配置字段 |
| `integration/verl/ray_actor.py` | 无额外生命周期账本 | `unwrap_native_actor_class(actor_class)` 解析原生 Actor 类型 | VERL/Ray 原生 Actor 类包装 |
| `task_runner.py` | `task_session`、`group_scheduler`、`_control_ready`、`_attached_to_gs`、`_operation_journal`、`_journal_lock`；按操作保存 Lease 快照与执行绑定 | `run(config)`；`submit_operation(...)` 受理；`query_operation(operation_id) -> OperationRecord`；`native_placement_candidates() -> tuple[dict,...]`；内部 `_execute_operation()`、`_advance_lease()`、`_attach_task_with_reconciliation()` | 继承 `FullyAsyncTaskRunner`，复用其组件生命周期与原生训练主循环，借助 Ray 调用 Trainer/Rollouter |
| `trainer.py` | `task_session`、`_replica_sync_gate`、`checkpoint_manager`，复用基类 `current_param_version`、`rollouter`、`actor_wg` | `_setup_checkpoint_manager()`；`_fit_update_weights()`；`bootstrap_and_publish(operation) -> OperationEvidence`；`remove_and_commit(operation) -> OperationEvidence`；`restore_and_publish(operation) -> OperationEvidence`；`reconcile_exit(...)` | 继承 `FullyAsyncTrainer`；复用原生 optimizer/weight update、CE group 和 Rollouter，扩展目标集/安全门而非重写 PPO |
| `message_queue.py` | `task_session`、`_completion_db`、`_next_completion_seq`、`_completion_lock`、`_completion_tmpdir` | `put_sample_once(sample) -> CompletionEvidence`；`put_sample(sample) -> bool`；`shutdown()`；`_decode_identity(sample)`、`_lookup_completion(...)` | 继承原生 `MessageQueue`，延续消费队列与提交路径，附加完成事实去重和冲突检查 |

### 1.8.3 Rollouter、Manager 与 Replica Runtime

| 组件 | 核心字段 / 状态 | 关键方法 | 复用点 |
| --- | --- | --- | --- |
| `rollouter.py` | `group_scheduler`、`task_session`、`llm_server_manager`、`async_rollout_manager`、`_idle_report_signature`、`_idle_report_last_sent`、`_force_handoff_timeout_s`、`_natural_drain_timeout_s` | `native_placement_candidates()`、`collect_idle_candidates()`、`submit_idle_report()`；`prepare_replica(...)`、`get_pending_target(id)`、`get_pending_replicas(id)`；`prepare_exit(...)`、`commit_service_change(operation)`、`finalize_release(operation) -> OperationEvidence`、`query_release_operation(...)` | 继承 `FullyAsyncRollouter`；复用原生 `FullyAsyncLLMServerClient`、`RewardLoopManager`、async rollout 处理器；局部包装 continuation 与 task-scoped RewardLoop |
| `llm_server_manager.py` | `task_session`、`next_replica_rank`、`replica_operation_lock`、`global_load_balancer`；按 ReplicaKey 保存 kind/state、放置及 release 事实（内部账本） | `register_replica(...)`、`transition_replica(key, state) -> ReplicaState`、`replica_meta(key) -> (ReplicaKind, ReplicaState)`、`inspect_runtime(key)`、`validate_borrowed_spec(spec) -> dict`、`create_borrowed_replica(spec) -> dict`、`sleep(...)`、`destroy(...)`、`query_release_evidence(...)`、`activate_service(key)`、`deactivate_service(key)` | 继承 `FullyAsyncLLMServerManager`；复用 Ray PG/bundle、VERL Server 管理、vLLM sleep/wake |
| `rollout/replica.py` | `replica_kind`、`runtime_epoch`、`placement_claims`、`borrowed_worker_names`、`borrowed_server_names`、`workers`、`servers`、`borrowed_cleanup_verified` | `validate_placement(spec)`、`build_borrowed_worker_plan(spec)`、`worker_placements()`、`validate_worker_placement()`、`validate_server_runtime()`、`init_from_lease(...)`、`cleanup_borrowed_runtime()`、`sleep()`、`wake_up(tags)` | 继承原生 `vLLMReplica`；复用 Ray WorkerGroup/PG、vLLM Worker 与原生 Runtime 创建机制 |
| `rollout/http_server.py` | `engine`、`_submission_paused`、`_multitask_sleep_stage_value`、`_server_port` | `runtime_health() -> dict`、`shutdown_runtime() -> dict`、`sleep() -> dict`、`wake_up(tags) -> dict`；`_wait_admission_barrier()`、`_validate_multitask_engine_capabilities()` | 继承 `vLLMHttpServer`；复用 vLLM EngineCore、真实模型生成和 sleep/wake 原语，不以模拟回执替代设备状态 |

#### 核心示例：`MultiTaskvLLMReplica(vLLMReplica)` 字段与复用明细

Native 和 Borrowed **复用同一个扩展类**：Native 延续原生初始化路径；Borrowed 走 `init_from_lease()`，在 Lease 指定的已有 PG/bundle 上创建**独立** CE Worker 和 vLLM Server/Engine。Borrowed 不复用 donor 的 CE Worker，也不创建归 borrower 所有的新 PG。当前代码明确限定 **TP=1、单 claim、单 server**；以下不把多卡拓扑或旧版扩展提案当成已实现。

**字段（区分继承与新增）：**

```python
# 继承 vLLMReplica（Native 沿用，Borrowed 创建路径赋值/复用）
replica_rank                # [继承] 本任务内 Replica rank
config, model_config        # [继承] Rollout 与模型配置
world_size, nnodes          # [继承] 当前 Borrowed 校验 world_size=1、nnodes=1
gpus_per_replica_node       # [继承] 每节点 Replica 设备数
workers, servers            # [继承] 新建 Borrowed CE Workers / vLLM Servers 的句柄
resource_pool, bundle_indices # [继承] Native 资源配置；Borrowed 不拥有 donor PG
rollout_mode                # [继承] Borrowed 按 STANDALONE 路径运行
_server_address, _server_handle # [继承] 主 HTTP endpoint 与句柄
name_suffix                 # [继承] 为任务/租约命名隔离提供后缀

# 当前 MultiTaskvLLMReplica 明确增加或重赋值
replica_kind: ReplicaKind   # [新增] NATIVE/BORROWED，而非 allocation_kind 字符串
placement_claims           # [新增] Borrowed 的归一化逐卡 claims
runtime_epoch              # [新增] Runtime 身份代次
server_class               # [重赋值] ray.remote(MultiTaskvLLMHttpServer)
borrowed_worker_names      # [运行期新增] 创建的 Borrowed CE Actor 名称
borrowed_server_names      # [运行期新增] 创建的 Borrowed Server Actor 名称
borrowed_cleanup_verified  # [运行期新增] 清理核验事实，不能代替 RELEASED evidence
```

**资源规格与 Worker 实测结构：** `validate_placement(spec)` 使用 `spec["lease_id"]`、`spec["placement_epoch"]`、`spec["replica_rank"]`、`spec["claims"]`、`spec["expires_at"]` 等数据；目前单个 claim 的 `rank/node_rank/local_rank` 均要求为 0。`build_borrowed_worker_plan()` 返回的核心项为：

```python
{
    "rank": 0,
    "claim_id": "...",
    "actor_name": "...",
    "pg_id": "...",
    "bundle_index": 0,
    "node_id": "...",
    "gpu_uuid": "...",     # 字段沿用原名，NPU 实际为 NPU:node_id:accelerator_id
    "num_gpus": 0.5,       # Ray 分配份额，不代表物理半张卡
    "num_cpus": 1.0,
    "env_vars": {"WORLD_SIZE": "1", "RANK": "0",
                 "RAY_LOCAL_WORLD_SIZE": "1",
                 "WG_PREFIX": "...", "WG_BACKEND": "ray"}
}
```

`worker_placements()` 从**实际 CE Actor** 中获取 `node_id`、`pg_id`、`gpu_uuid`、`resource_name`、`accelerator_id`；`validate_worker_placement()` 对照 claims 检查 node 和设备身份。CUDA 的数字设备 ID 进一步通过 `nvidia-smi` 对照 UUID；NPU 使用 `NPU:node_id:accelerator_id` 身份。

**主要方法与原生复用：**

| 方法 | 作用与结果 | 复用点 |
| --- | --- | --- |
| `get_ray_class_with_init_args() -> RayClassWithInitArgs` | 替换为带服务命名隔离的 `MultiTaskCheckpointEngineWorker` | 原生 `RayClassWithInitArgs` 与 CE Worker 参数 |
| `validate_placement(spec) -> None`、`build_borrowed_worker_plan(spec) -> dict` | 验证 Lease/Replica/TP=1 约束并生成 CE Actor 的确定性放置方案 | Ray PG/bundle、Lease claims |
| `_create_workers_from_claims(...)`、`validate_worker_placement() -> tuple[dict, ...]` | 在指定原 PG/bundle 上创建独立 CE Workers，并实测节点与设备 | `RayWorkerGroup.from_detached()`、`PlacementGroupSchedulingStrategy` |
| `init_from_lease(...)`、`validate_server_runtime() -> dict` | 初始化 Borrowed Server/Engine 并确认实际 HTTP/Engine 状态 | 原生 `vLLMReplica` Server 创建、`MultiTaskvLLMHttpServer` |
| `cleanup_borrowed_runtime() -> None` | 清理 Borrowed Server/Worker，检查 Ray Actor 进入 DEAD；证据不足则报错 | `ray.kill`、Ray State API |
| `sleep()`、`wake_up(tags)` | Native 生命周期的真实运行时休眠与恢复 | vLLM 原生 sleep/wake 原语 |

**实现与旧设计字段的区别：** 当前类没有逐项定义示例中的 `allocation_kind`、`lease_id`、`source_lease_ids`、`donor_task_ids`、`runtime_state`、`owns_resource_pool`、`claims`、`serving_version`、`operation_id`、`node_layout`、`expected_device_map`、`actual_device_map`、`cleanup_result`、`creation_stage` 等成员。租约细节保留在 Lease/spec/Manager 的 Owner 视图，参数版本由 CE/Trainer 管理；不可为了与旧示例形式一致而把这些字段误标为已实现。

### 1.8.4 LB、Checkpoint Engine 与后端

| 组件 | 核心字段 / 状态 | 关键方法 | 复用点 |
| --- | --- | --- | --- |
| `rollout/load_balancer.py` | 基类路由/服务端集合及 request→server 事实；本类 `_awaiting_service_restore`、`_settled_retention`、`_settled_count`；内部 attempt 和操作关联账本 | `acquire_server(request_id,...)`、`release_server(server_id, request_id)`、`query_attempt(request_id)`、`confirm_continuation(request_id, client_id,...)`、`continuation_handoff_requests(operation_id)`、`requests_for_server(server_id)`、`begin_drain(key, operation_id)`、`commit_ready(...)`、`finish_remove(key)` | 继承 `GlobalRequestLoadBalancer`，复用原生 HTTP Server 选择与请求释放；新增安全 fencing/attempt 证明 |
| `checkpoint/checkpoint_engine_manager.py` | `_effective_replica_map`（`effective_replicas` 属性）、`_pending_bootstrap_map`（`pending_bootstrap` 属性）、`_bootstrap_ready_map`、`parameter_validation_enabled`、`source_validation_enabled`、`backend` | `register_pending(...)`、`bootstrap_target(...)`、`commit_pending(...)`、`discard_pending(key)`、`add_effective(...)`、`remove_effective(key)`、`mark_all_loaded_version(version)`、`validate_parameter_sync(...)` | 继承原生 `CheckpointEngineManager`；复用 `RayWorkerGroup`、原生通信组/参数广播和版本同步，仅控制目标集合与证据 |
| `checkpoint/checkpoint_engine_worker.py` | 原生 Worker 权重加载状态、带 suffix 的 ServerAdapter、参数 Manifest / digest | `update_weights(global_steps)`、`get_parameter_manifest() -> dict`，内部 tensor SHA256 | 继承 `CheckpointEngineWorker`；复用 VERL 原生 Worker 与接收端，附加传输审计 |
| `checkpoint/hccl_checkpoint_engine.py` | 源端 manifest、训练步/权重摘要相关事实 | `send_weights(weights, global_steps)`、`get_source_manifest() -> dict`、`finalize()` | 继承原生 `HCCLCheckpointEngine`；复用 Ascend/HCCL 真实通信与 finalize，不创建虚构权重成功证据 |

**跨模块复用边界**：原生 VERL 仍负责训练循环、WorkerGroup、参数发送接收、Rollout 主循环和 Runtime 原语；Ray 负责 Actor/PG 放置；MultiTask 主要增加 GS/Lease、Owner 独立事实、任务级编排、目标集同步、证据校验和 fail-closed。查询操作/证据应通过现有方法，不直接从别的组件读内部 map。

# 第二部分：测试与综合验收

## 2.1 分层与测试文件清单

| 层 | 路径 / 文件 | 验收边界 |
| --- | --- | --- |
| CPU Unit | `tests/unit/test_orchestration.py`、`test_scheduler_wiring.py`、`test_taskrunner_wiring.py`、`test_trainer_wiring.py`、`test_rollouter_wiring.py`、`test_replica_wiring.py`、`test_lb_wiring.py`、`test_checkpoint_wiring.py` | 合同、Lease、operation、接口接线、错误路径 |
| 专项 Unit | `test_message_queue_exactly_once.py`、`test_runtime_profile.py`、`test_server_wiring.py`、`test_hccl_checkpoint_engine.py`、`test_npu_memory_probe.py`、`test_restore_vpub_mutation.py`、`test_e2e_scripts.py` | 队列去重、配置准入、Server、参数同步后端、诊断脚本 |
| CPU Ray | `tests/integration/ray/test_group_scheduler.py` | 真实 Actor 与 GS RPC |
| 原生 VERL | `tests/integration/verl/test_native_adapters.py` | VERL 类与适配器真实导入、接线 |
| CUDA | `tests/integration/cuda/test_baseline_environment.py`、`test_native_sleep_gpu.py` | 真实设备、sleep/borrow/restore 原语 |
| NPU | `tests/integration/npu/test_native_sleep_npu.py` | 真实模型、设备、NPU 生命周期原语 |
| 端到端 | `scripts/e2e/verify_two_verl_jobs.py`；`validate_control_plane.sh`、`validate_exactly_once.sh`、`validate_recovery_faults.sh`、`validate_lifecycle_cycle.sh`、`validate_force_remove.sh` | 真实双任务运行、跨组件闭环 |

`tests/conftest.py` 负责共同 Python 导入路径；`tests/integration/cuda/conftest.py` 只为 CUDA fixture 提供设备与配置，二者不能误合并。完整测试导航见 [tests/README.md](../tests/README.md) 与 [unit/README.md](../tests/unit/README.md)。

## 2.2 执行环境与推荐命令

在仓库根目录运行；每条按其环境要求单独执行。

```bash
# 轻量单测
python -m pip install -e '.[test]'
python -m pytest -q tests/unit

# CPU Ray 控制面；需要 Ray
python -m pytest -q tests/integration/ray

# 真实 VERL 接线；需要对应 VERL、Ray 与 vLLM 环境
python -m pytest -q tests/integration/verl

# CUDA 真实设备原语
VERL_MULTITASK_GPU_MODEL_PATH=/path/to/model \
  python -m pytest -q -s -m gpu_integration tests/integration/cuda/test_native_sleep_gpu.py

# Ascend NPU 真实设备原语
VERL_MULTITASK_NPU_MODEL_PATH=/path/to/model \
  python -m pytest -q -s -m npu_integration tests/integration/npu/test_native_sleep_npu.py
```

不要把 CPU 单测、pytest skip、配置组合成功或 mock `RELEASED` 等同于真实 GPU/NPU 资源回收通过。

## 2.3 完整双任务 E2E 命令与入口

先按当前机器修改 `examples/e2e/native_args.txt` 中模型、训练与验证数据路径。示例：

```bash
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . --ray-address auto --start-local-ray \
  --native-args examples/e2e/native_args.txt \
  --scenarios "control_plane exactly_once recovery lifecycle force"
```

如使用已存在的持久 Ray 集群，按真实地址使用 `--ray-address` 并移除 `--start-local-ray`。默认从真实 donor CE 与 named PG **自动生成 Lease**；无需静态 `lease.example.json`。若要求真实在途 FORCE 证明，另加 `--require-inflight-force`。

## 2.4 验收矩阵：必查功能与证据

| 项 | 必须检查 | 通过条件 |
| --- | --- | --- |
| 启动 / 接线 | 实际导入源码、两个 task_session、唯一 GS | 两个真实任务注册同一 GS 且互不覆盖 |
| 空闲上报 / Lease | donor CE、PG、bundle、node_id、物理 UUID、claim | 租约身份与真实放置一致，不借出未释放设备 |
| Control plane | submit/query、operation_id 幂等、Lease 推进 | 准入与终态一致；ACK 丢失可对账 |
| Exactly once | 相同样本重放、冲突提交、Queue 最终条目 | 不重复提交、不接受内容冲突 |
| Recovery | 超时、重试、部分失败、UNKNOWN/QUARANTINED | 无假成功、无重复设备变更、能查真实 Owner |
| Lifecycle | DONATE→ADD→REMOVE→RESTORE | 全程有 `RELEASED / SERVICE_COMMITTED`，真实卡身份与版本一致 |
| FORCE | BORROWED 限制、其他服务端、targeted abort、续推回执、终态 | 在途请求有闭合证明且资源真实释放；严格模式额外校验在途路径 |
| 参数同步 | 实际 Vpub、receiver/source manifest、版本与重建通信组 | 真实传输完成，未验证配置不得标记 WEIGHT_READY |
| 资源退出 | NPU/GPU 显存、运行进程、PG bundle、借还 owner | 无被遗留的幽灵 runtime；释放证据对应准确设备 |

## 2.5 结果判定与产物

`verify_two_verl_jobs.py` 返回码：`0=PASS`、`1=FAIL`、`2=BLOCKED`。总结果见 `logs/two_real_jobs/<run>/orchestration_summary.json`，场景详情见 `scenarios/<场景>/<运行ID>/result.json`；同时保留 donor/borrower 日志和明确的 `operation_id`。**仅所选场景全部 PASS，且需额外证据的条件全部满足，才能称该范围验收通过。**

建议每次验收归档：Git SHA、VERL SHA、Python/Ray/vLLM/设备驱动版本、机器/卡数、模型路径摘要、完整运行命令、Lease PG 和 UUID、操作日志、资源显存/进程证据、摘要及每个 scenario result。不得以不同运行轮次的 FORCE 回执拼接成同一次证明。

## 2.6 已观察到的验收结果及未关闭事项

**最新观测：2026-10-09 真实 Ascend NPU 双任务 E2E（运行目录 `logs/two_real_jobs/20261009-094238/`）**。本轮启动隔离本地 Ray，检测到 8 个 NPU 资源，使用 `multitask_hccl`；Donor 和 Borrower 分别以独立 `task_session` 注册到同一个 GroupScheduler。Auto Lease 基于真实 donor CE、named PG（`c500b2433a4fb2cd06fd94c0eb1602000000`）及设备身份 `NPU:b446be0859e5365c63d5e98ec8820137a1c1c1985377966e004112b6:1` 创建，并核验 PG bundle=0、`NPU=1.0`、`CPU=2.0`。

| 场景 | 08:43 历史轮次 | 09:42 最新轮次 |
| --- | --- | --- |
| `control_plane` | PASS | **PASS** |
| `exactly_once` | PASS | **PASS** |
| `recovery` | PASS | **PASS** |
| `lifecycle` | PASS | **PASS** |
| `force` | FAIL（第二次 ADD bootstrap 失败） | **PASS** |
| 总结果 | FAIL（退出码 1） | **PASS（退出码 0，5/5）** |

**验收结论**：最新轮次已完成五个已选择 E2E 场景，返回 `0`。这证明该轮正常生命周期与 FORCE 场景的默认验收通过；**日志未表明使用 `--require-inflight-force`，因此不能据此宣称真实在途 targeted abort + continuation 的严格验收通过**。此次也未提供其他 CUDA/NPU 组合的验收证据。

**仍需关闭的事项**：

- **严格 FORCE**：使用 `--require-inflight-force` 并检查同一 REMOVE operation 的中断、续推回执、attempt 终态与最终样本无重复。
- **进程收尾**：Python 在退出阶段提示 `ResourceWarning: subprocess 406621 is still running`；需要检查进程是否随后正常结束或属于清理遗漏，不以 `exit=0` 自动认定资源收尾无问题。
- **历史 ADD 失败**：`20261009-084341` 轮次的 `e2e-baa0d4fefb-add` 在 bootstrap 回滚后记录已验证 `RELEASED`，底层原始异常尚未确定。新轮次未复现不等于根因已修复；Trainer 已增加捕获 traceback 的诊断日志，复发时按同一 operation_id 排查。

本节基于用户提供的两次 E2E 控制台日志与历史 `force_cycle/result.json`，并非在本次文档更新中重新运行测试。

## 2.7 发布门禁

- 产品变更先跑 `tests/unit`，改动对应 Owner 的再跑相关 Ray/VERL 集成测试。
- 修改资源 lifecycle、checkpoint backend、FORCE 时必须在目标真实设备执行专项原语和双任务 E2E。
- 必须满足精确 `operation_id`、Lease 资源身份、参数版本、请求 owner 与设备释放证据，不允许靠放宽断言降低门禁。
- 评审合入时列出通过范围、失败项、未覆盖能力和运行链接；任何 `FAIL` / `BLOCKED` 均不可写成整套验收 PASS。
