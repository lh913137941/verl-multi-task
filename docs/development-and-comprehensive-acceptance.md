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

## 1.8 各模块字段、方法与复用点（按实际代码）

本节按照「继承/新增字段 → 主要方法与作用 → 原生复用点」逐一说明。字段类型在代码明确声明时写类型；其余保留实际成员名并说明用途，不臆造公共结构。以下划线开头的名称及 Manager/GS 内部账本均属实现细节，不能作为新的跨组件接口。所有运行能力仍以当前 fail-closed 合同和真实 E2E 证据为准。

**数据结构注意**：Lease claim 由 GS 经 spec 传至 Replica，关键字段 `claim_id / pg_id / bundle_index / node_id / gpu_uuid / cpu_request / gpu_fraction`；`build_borrowed_worker_plan()` 产生 `rank / actor_name / num_gpus / num_cpus / env_vars`，其中 `env_vars` 包含 `WORLD_SIZE / RANK / RAY_LOCAL_WORLD_SIZE / WG_PREFIX / WG_BACKEND`；Worker 实测提供 `node_id / pg_id / gpu_uuid / resource_name / accelerator_id`。当前 Borrowed 是单 claim、TP=1，不是旧草案的通用多卡 `node_layout` 结构。

### 1.8.1 `GroupScheduler` — `scheduler/group_scheduler.py`

**继承与职责**：Ray Actor（无 VERL 基类）。

#### 字段（按设计职责分组；均为 GS 内部实现，不是新增公共状态 DTO）

```python
# ① 任务注册和空泡报告
task_runners: dict[str, ActorHandle]     # task_session -> TaskRunner
idle_reports: dict[str, object]          # task_session -> {observed_at, candidates}

# ② 同一个 Lease 所有权账本及排他索引
leases: dict[str, Lease]                 # lease_id -> Lease（授权事实）
claim_id_owner: dict[str, str]           # claim_id -> lease_id（身份不可重用）
active_bundle_owner: dict[tuple[str, int], str]  # (pg_id, bundle_index) -> lease_id
active_gpu_owner: dict[str, str]         # gpu_uuid -> lease_id

# ③ 操作幂等、交接及补偿（Lease 账本的内部投影）
operation_commands: dict[str, OperationCommand]  # operation_id -> 命令意图
borrower_targets: dict[str, ReplicaKey]  # lease_id -> 本轮 borrower
release_evidence: dict[tuple[str, str], OperationEvidence]  # (lease_id, op_id) -> 证据
release_history: dict[str, list[str]]    # lease_id -> 已完成操作顺序（可审视冗余）
handoff_ready_leases: set[str]           # DONATE RELEASED 后的可交接阶段索引
```

**单一状态 Owner**：GS 仅写跨任务 Lease/资源归属；TaskRunner 的 `OperationJournal` 是任务操作状态 Owner，Manager/CE/LB/Rollouter 各自写 M/E/R/C。以上 11 个字段不是 11 套公共数据模型。`release_history` 与 `handoff_ready_leases` 可以从历史证据/命令推导部分信息，但目前还用于 ADD 准入、RESTORE 终结、补偿及幂等回放。删除前必须验证这些路径；**本次不删字段、不改执行顺序**。

**典型字段值**：`idle_reports[task_session] = {"observed_at": monotonic_time, "candidates": ({"replica_key": ReplicaKey, "kind": "NATIVE"}, ...)}`；`release_evidence[(lease_id, operation_id)]` 绑定精确操作及证据；`handoff_ready_leases` 为集合而非单独的 Lease 状态枚举。

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `__init__() -> None` | **新增 Actor 构造**；建立 GS 内部注册、Lease、所有权及操作对账索引；不创建独立流程服务。 |
| `runtime_kind() -> str` | **新增发现协议**；返回兼容性标识供 TaskRunner 识别。 |
| `attach_task(task_id, task_runner) -> None` | **新增注册入口**；验证真实 Ray ActorHandle，绑定 task_session；已有相同绑定可重放，冲突拒绝。 |
| `detach_task(task_id) -> None` | **新增生命周期入口**；删除 task_runners 和过期 idle_reports，不代表清除运行中的 Lease。 |
| `get_task_runners() -> dict[str, ActorHandle]` | **新增只读查询**；返回注册快照，避免外部修改内部映射。 |
| `submit_idle_report(report) -> dict` | **新增空泡上报**；核验 task_session、ReplicaKey 与候选去重，返回 `{"accepted": True, "candidate_count": N}`。 |
| `submit_operation(command) -> OperationRecord` | **新增操作路由**；校验 Lease、阶段与原 command 幂等；在 RPC **前**记录 operation intent、ADD borrower 或 RESTORE 预留，再转发 TaskRunner。 |
| `_target_matches_donor(target, lease) -> bool` | **内部辅助**；核实本轮 donor identity 与 Lease claim 一致。 |
| `open_lease(lease) -> Lease` | **GS 内部账本动作**；验证 Claim、PG bundle 和物理设备排他，登记一次 Lease；相同 Lease 可幂等返回快照。 |
| `advance_lease(lease_id, evidence) -> dict` | **GS 内部对账动作**；仅接受同一操作的合法 evidence，推进 DONATE/REMOVE、RESTORE 成功或 ADD/RESTORE 安全补偿；精确重放保持幂等。 |

**复用点**：见下文模块说明。 |
| `detach_task(task_id) -> None` | **本模块现有方法**；解除注册。返回值见签名；**复用点**：见下文模块说明。 |
| `get_task_runners() -> dict` | **本模块现有方法**；查询任务句柄。返回值见签名；**复用点**：见下文模块说明。 |
| `submit_idle_report(report)` | **本模块现有方法**；接收候选。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `submit_operation(command) -> OperationRecord` | **本模块现有方法**；路由命令。返回值见签名；**复用点**：见下文模块说明。 |
| `open_lease(lease) -> Lease` | **本模块现有方法**；登记 claims。返回值见签名；**复用点**：见下文模块说明。 |
| `advance_lease(lease_id, evidence) -> dict` | **本模块现有方法**；按真实证据推进租约。返回值见签名；**复用点**：见下文模块说明。 |

**复用点**：复用 Ray named/detached Actor、ActorHandle 与真实 PG 元数据；不代替 VERL ResourcePool。

### 1.8.2 `Discovery` — `scheduler/discovery.py`

**继承与职责**：函数入口（无实例字段）。

#### 字段

```python
# [新增字段] 无；GS 所有权只在 GroupScheduler 内
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `get_or_create_group_scheduler()` | **本模块现有方法**；发现或创建共享 GS。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 Ray named Actor 发现和 detached 生命周期。

### 1.8.3 `Contracts` — `orchestration/contracts.py`

**继承与职责**：标准 Enum / dataclass（非 VERL 子类）。

#### 字段

```python
ReplicaKey: task_session: str; replica_id: str; runtime_epoch: int = 0
OperationCommand: operation_id: str; kind: OperationKind; target: ReplicaKey; lease_id: str; force: bool | None = None
OperationRecord: operation_id: str; status: OperationStatus; result: str | None
OperationEvidence: operation_id: str; type: EvidenceType; timestamp: int; released_gpu_uuids: tuple[str, ...]
Lease: lease_id: str; claims: tuple[Mapping, ...]; expires_at: float
# 枚举：ReplicaKind, ReplicaState, AttemptState, OperationKind, OperationStatus, EvidenceType
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `__post_init__()` | **新增内部辅助或重写（见基类定义）**；类型/标识验证。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `OperationEvidence.now(...)` | **本模块现有方法**；构造带时间戳证据。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `Lease.claim_ids(), source_lease_ids(), bundle_keys(), gpu_uuids()` | **本模块现有方法**；提取 claims 身份。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 Python dataclass/Enum；仅描述已有合同，不新增类型。

### 1.8.4 `OperationJournal` — `orchestration/operation_journal.py`

**继承与职责**：独立操作账本。

#### 字段

```python
# [新增]
_records: dict       # operation_id -> OperationRecord
_commands: dict      # operation_id -> OperationCommand
_active_by_task: dict # 任务并发操作索引
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `begin(command) -> OperationRecord` | **本模块现有方法**；受理或同命令幂等返回。返回值见签名；**复用点**：见下文模块说明。 |
| `query(operation_id) -> OperationRecord \| None` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `command(operation_id) -> OperationCommand` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `mark_running(id) -> OperationRecord` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `reopen_unknown(id) -> OperationRecord` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `finish(...)` | **本模块现有方法**；写终态并校验身份。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 OperationCommand / Record，不持有 GPU/模型状态。

### 1.8.5 `ReplicaSyncGate` — `orchestration/replica_sync_gate.py`

**继承与职责**：asyncio 同步闸门。

#### 字段

```python
# [新增]
_lock: asyncio.Lock
_epoch: int
_owner: GateOwner | None
_blocked_reason: str | None
_blocked_operation_id: str | None
# GateLease: _gate, owner
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `acquire(...) -> GateLease` | **本模块现有方法**；独占同步权。返回值见签名；**复用点**：见下文模块说明。 |
| `GateLease.guard(call, *args, **kwargs)` | **本模块现有方法**；闸门内执行。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `GateLease.release() -> bool` | **本模块现有方法**；释放。返回值见签名；**复用点**：见下文模块说明。 |
| `block(owner, reason) -> None` | **本模块现有方法**；阻断。返回值见签名；**复用点**：见下文模块说明。 |
| `reconcile(...)` | **本模块现有方法**；对账恢复。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `health / owner / blocked_reason` | **本模块现有方法**；查询。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 asyncio.Lock 和原生 Trainer 权重更新节奏；不替换 VERL checkpoint 通信原语。

### 1.8.6 `RuntimeProfile / RayActor` — `integration/verl/runtime_profile.py；integration/verl/ray_actor.py`

**继承与职责**：轻量函数与校验器。

#### 字段

```python
# [新增持久字段] 无
# 配置项读取自 VERL/Hydra config
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `resolve_runtime_profile(config)` | **本模块现有方法**；选定受支持 profile。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `validate_runtime_profile(config) -> bool` | **本模块现有方法**；能力准入。返回值见签名；**复用点**：见下文模块说明。 |
| `unwrap_native_actor_class(actor_class) -> type` | **本模块现有方法**；解包 Ray remote 类。返回值见签名；**复用点**：见下文模块说明。 |

**复用点**：复用原生配置树、Ray Actor 类语义；不维护跨任务状态。

### 1.8.7 `TaskRunner` — `integration/verl/experimental_fully_async/task_runner.py`

**继承与职责**：MultiTaskFullyAsyncTaskRunner(FullyAsyncTaskRunner)。

#### 字段

```python
# [继承] components、VERL 原生 run() / 初始化链路
# [新增]
task_session: str
group_scheduler: ActorHandle
_control_ready: ...
_attached_to_gs: bool
_journal_lock: threading.Lock
_operation_journal: OperationJournal
_operation_threads: dict
_operation_leases: dict   # operation_id -> Lease 快照
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `run(config)` | **重写/扩展原生入口**；执行原生训练主流程并安装插件接线。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `submit_operation(...)` | **本模块现有方法**；接收命令。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `query_operation(operation_id) -> OperationRecord` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `native_placement_candidates() -> tuple[dict, ...]` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `_build_borrowed_spec(...): 派生单次 ADD spec` | **新增内部辅助或重写（见基类定义）**；参见代码具体参数与返回值。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `_execute_operation(id)` | **新增内部辅助或重写（见基类定义）**；编排 Trainer/Rollouter。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `_attach_task_with_reconciliation()` | **新增内部辅助或重写（见基类定义）**；共享 GS 注册对账。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 FullyAsyncTaskRunner 初始化、Ray RPC 与 VERL 训练主循环；新增任务级 operation 编排。

### 1.8.8 `Trainer` — `integration/verl/experimental_fully_async/trainer.py`

**继承与职责**：MultiTaskFullyAsyncTrainer(FullyAsyncTrainer)。

#### 字段

```python
# [继承] current_param_version, rollouter, actor_wg, config
# [新增]
task_session: str
_replica_sync_gate: ReplicaSyncGate
checkpoint_manager: MultiTaskCheckpointEngineManager # [扩展重赋值]
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `_setup_checkpoint_manager()` | **新增内部辅助或重写（见基类定义）**；建立 Native 参数成员。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `_fit_update_weights()` | **新增内部辅助或重写（见基类定义）**；受 G 闸门保护的原生权重推进。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `bootstrap_and_publish(operation) -> OperationEvidence` | **本模块现有方法**；ADD。返回值见签名；**复用点**：见下文模块说明。 |
| `remove_and_commit(operation) -> OperationEvidence` | **本模块现有方法**；REMOVE/DONATE。返回值见签名；**复用点**：见下文模块说明。 |
| `restore_and_publish(operation) -> OperationEvidence` | **本模块现有方法**；RESTORE。返回值见签名；**复用点**：见下文模块说明。 |
| `reconcile_exit(...)` | **本模块现有方法**；退出结果对账。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 FullyAsyncTrainer 的优化器、参数版本、CheckpointEngine 与 Rollouter；不重写 PPO。

### 1.8.9 `Rollouter / Client` — `integration/verl/experimental_fully_async/rollouter.py`

**继承与职责**：MultiTaskFullyAsyncRollouter(FullyAsyncRollouter)。

#### 字段

```python
# [继承] llm_server_manager, async_rollout_manager, reward_loop_manager 等
# [新增]
group_scheduler: ActorHandle
task_session: str
_pending_operation_targets: dict
_idle_report_signature, _idle_report_last_sent: ...
_idle_report_task: asyncio.Task | None
_force_handoff_timeout_s, _natural_drain_timeout_s: float
# Client 适配：_continuation_client_id, _continuation_enabled, _release_fences
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `native_placement_candidates() -> tuple[dict, ...]` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `collect_idle_candidates() -> tuple` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `submit_idle_report()` | **本模块现有方法**；上报空闲资源。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `prepare_replica(...)` | **本模块现有方法**；准备隐藏 Borrowed。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `prepare_exit(...)` | **本模块现有方法**；drain / FORCE 准入。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `get_pending_target(id) -> ReplicaKey` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `get_pending_replicas(id)` | **本模块现有方法**；读取待装参目标。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `commit_service_change(operation)` | **本模块现有方法**；发布或撤销路由。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `finalize_release(operation) -> OperationEvidence` | **本模块现有方法**；释放对账。返回值见签名；**复用点**：见下文模块说明。 |

**复用点**：复用 FullyAsyncRollouter、FullyAsyncLLMServerClient、RewardLoopManager 与原生 async rollout；增量封装 continuation 和 task-scoped RewardLoop。

### 1.8.10 `LLMServerManager` — `integration/verl/experimental_fully_async/llm_server_manager.py`

**继承与职责**：MultiTaskLLMServerManager(FullyAsyncLLMServerManager)。

#### 字段

```python
# [继承] rollout_replicas, server_addresses, server_handles, rollout_config
# [新增]
task_session: str
rollout_replica_class: ... # [重赋值] MultiTaskvLLMReplica
_load_balancer_cls: ...     # [重赋值] MultiTask LB
replica_state: dict[ReplicaKey, ReplicaState]
replica_kind: dict[ReplicaKey, ReplicaKind]
_runtime_inventory: dict[ReplicaKey, object]
borrowed_operations: dict[str, dict]
_native_release_evidence: dict
next_replica_rank: int
replica_operation_lock: asyncio.Lock
global_load_balancer: ActorHandle # [重赋值]
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `register_replica(...)->None` | **本模块现有方法**；登记 kind/state/runtime。返回值见签名；**复用点**：见下文模块说明。 |
| `transition_replica(key, state)->ReplicaState` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `replica_meta(key)->tuple[ReplicaKind, ReplicaState]` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `inspect_runtime(key)` | **本模块现有方法**；查询实际 runtime。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `validate_borrowed_spec(spec)->dict` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `create_borrowed_replica(spec)->dict` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `sleep(...), destroy(...)` | **重写/扩展原生入口**；生命周期。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `activate_service(key), deactivate_service(key)` | **本模块现有方法**；服务状态。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `query_release_evidence(...)` | **本模块现有方法**；查询 RELEASED。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 FullyAsyncLLMServerManager、Native Replica 初始化、Ray PG / Actor 与 vLLM 服务管理。

### 1.8.11 `Replica` — `rollout/replica.py`

**继承与职责**：MultiTaskvLLMReplica(vLLMReplica)。

#### 字段

```python
# [继承] replica_rank, config, model_config, world_size, nnodes, workers, servers, resource_pool, bundle_indices, rollout_mode, name_suffix, _server_address, _server_handle
# [新增]
replica_kind: ReplicaKind
placement_claims: tuple[dict, ...] | None
runtime_epoch: int
borrowed_worker_names: tuple[str, ...]
borrowed_server_names: tuple[str, ...]
borrowed_cleanup_verified: bool
server_class: ... # [重赋值] ray.remote(MultiTaskvLLMHttpServer)
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `get_ray_class_with_init_args()->RayClassWithInitArgs` | **重写/扩展原生入口**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `validate_placement(spec)->None` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `build_borrowed_worker_plan(spec)->dict` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `init_from_lease(...)` | **本模块现有方法**；独立 CE Worker/Server。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `worker_placements()->tuple[dict,...]` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `validate_worker_placement()->tuple[dict,...]` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `validate_server_runtime()->dict` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `cleanup_borrowed_runtime()->None` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `sleep(), wake_up(tags)` | **重写/扩展原生入口**；原生设备生命周期。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 vLLMReplica、RayWorkerGroup.from_detached 和指定已有 PG bundle；不复用 donor CE Worker，不创建 borrower PG。

### 1.8.12 `HTTP Server` — `rollout/http_server.py`

**继承与职责**：MultiTaskvLLMHttpServer(vLLMHttpServer)。

#### 字段

```python
# [继承/重赋值] engine, _server_port
# [新增]
_submission_paused: bool
_multitask_sleep_stage_value: str
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `runtime_health()->dict` | **本模块现有方法**；Engine/端口/Node 健康。返回值见签名；**复用点**：见下文模块说明。 |
| `shutdown_runtime()->dict` | **本模块现有方法**；真实引擎退出。返回值见签名；**复用点**：见下文模块说明。 |
| `sleep()->dict, wake_up(tags)->dict` | **重写/扩展原生入口**；原生 sleep/wake 包装。返回值见签名；**复用点**：见下文模块说明。 |
| `_wait_admission_barrier()` | **新增内部辅助或重写（见基类定义）**；安全准入。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `_validate_multitask_engine_capabilities(engine)` | **新增内部辅助或重写（见基类定义）**；原语准入。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 vLLMHttpServer、EngineCore 和 sleep/wake；不以 RPC 返回代替真实设备证明。

### 1.8.13 `LoadBalancer` — `rollout/load_balancer.py`

**继承与职责**：MultiTaskGlobalRequestLoadBalancer(GlobalRequestLoadBalancer)。

#### 字段

```python
# [继承] servers、服务选择/释放逻辑
# [新增]
routes: dict
active_request_server: dict
attempt_state: dict
draining_operations: dict
ready_operations: dict
continuation_handoffs: dict
_awaiting_service_restore: bool
_settled_retention: ...
_settled_count: int
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `acquire_server(request_id, **extra)` | **本模块现有方法**；请求准入。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `release_server(server_id, request_id)` | **本模块现有方法**；结清 attempt。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `query_attempt(request_id)` | **本模块现有方法**；查询 attempt 状态。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `confirm_continuation(request_id, client_id, ...)` | **本模块现有方法**；续推前缀回执。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `continuation_handoff_requests(operation_id)->tuple[str,...]` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `begin_drain(key, operation_id)` | **本模块现有方法**；停止新流量。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `commit_ready(...)` | **本模块现有方法**；服务发布。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `finish_remove(key)` | **本模块现有方法**；撤销服务。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `query_ready_operation(id)` | **本模块现有方法**；发布对账。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 GlobalRequestLoadBalancer 的请求分配、路由缓存及 Server 句柄；增加 request/attempt Owner 事实。

### 1.8.14 `MessageQueue` — `integration/verl/experimental_fully_async/message_queue.py`

**继承与职责**：MultiTaskMessageQueue(MessageQueue)。

#### 字段

```python
# [继承] 原生队列配置及消费机制
# [新增]
task_session: str
_completion_tmpdir: ...
_completion_db: ...
_next_completion_seq: int
_completion_lock: asyncio.Lock
# CompletionEvidence：完整样本提交回执
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `put_sample_once(sample)->CompletionEvidence` | **本模块现有方法**；幂等提交。返回值见签名；**复用点**：见下文模块说明。 |
| `put_sample(sample)->bool` | **重写/扩展原生入口**；保持原接口兼容。返回值见签名；**复用点**：见下文模块说明。 |
| `_lookup_completion(...)` | **新增内部辅助或重写（见基类定义）**；查重。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `_decode_identity(sample)->tuple[str,str]` | **新增内部辅助或重写（见基类定义）**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `shutdown()` | **重写/扩展原生入口**；关闭并清理辅助资源。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用原生 MessageQueue 与训练消费路径；增加完成样本唯一性存证和冲突拒绝。

### 1.8.15 `CheckpointEngineManager` — `checkpoint/checkpoint_engine_manager.py`

**继承与职责**：MultiTaskCheckpointEngineManager(CheckpointEngineManager)。

#### 字段

```python
# [继承] actor_wg, replicas, 原生同步配置
# [新增]
_effective_replica_map: dict
_pending_bootstrap_map: dict
_bootstrap_ready_map: dict
parameter_validation_enabled: bool
source_validation_enabled: bool
backend: str
# 属性：effective_replicas、pending_bootstrap
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `register_pending(...)` | **本模块现有方法**；登记待装参目标。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `bootstrap_target(...)` | **本模块现有方法**；同步目标 Vpub。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `commit_pending(...)` | **本模块现有方法**；pending -> effective。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `discard_pending(key)->None` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `add_effective(...)->None / remove_effective(key)->None` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `mark_all_loaded_version(version)->None` | **本模块现有方法**；参见代码具体参数与返回值。返回值见签名；**复用点**：见下文模块说明。 |
| `validate_parameter_sync(...)` | **本模块现有方法**；manifest/版本校验。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `_get_source_manifest()` | **新增内部辅助或重写（见基类定义）**；获取源端审计。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用 CheckpointEngineManager、RayWorkerGroup、VERL 原生发送/接收与通信组；只限定目标集合及证据条件。

### 1.8.16 `CheckpointEngineWorker` — `checkpoint/checkpoint_engine_worker.py`

**继承与职责**：MultiTaskCheckpointEngineWorker(CheckpointEngineWorker)。

#### 字段

```python
# [继承] 原生 checkpoint worker 与权重接收能力
# [新增]
_server_name_suffix: str
server_handle: ... # [适配赋值]
parameter_validation_enabled: bool
_last_parameter_manifest: dict | None
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `update_weights(global_steps)` | **本模块现有方法**；原生装参并更新 manifest。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `get_parameter_manifest()->dict` | **本模块现有方法**；接收侧审计。返回值见签名；**复用点**：见下文模块说明。 |
| `_MultiTaskServerAdapter._ensure_server_handle()->bool` | **新增内部辅助或重写（见基类定义）**；关联命名 Server。返回值见签名；**复用点**：见下文模块说明。 |

**复用点**：复用原生 CheckpointEngineWorker、ServerAdapter、vLLM 权重接收端。

### 1.8.17 `HCCLCheckpointEngine` — `checkpoint/hccl_checkpoint_engine.py`

**继承与职责**：MultiTaskHCCLCheckpointEngine(HCCLCheckpointEngine)。

#### 字段

```python
# [继承/复用] pyhccl, rank, world_size, send_buf, recv_buf
# [新增]
source_validation_enabled: bool
_source_manifest: dict | None
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `send_weights(weights, global_steps)` | **本模块现有方法**；HCCL 真实发送并更新源 manifest。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `get_source_manifest()->dict` | **本模块现有方法**；供目标/源对照。返回值见签名；**复用点**：见下文模块说明。 |
| `finalize()->None` | **本模块现有方法**；通信收尾。返回值见签名；**复用点**：见下文模块说明。 |
| `_tensor_sha256(tensor)->str` | **新增内部辅助或重写（见基类定义）**；摘要。返回值见签名；**复用点**：见下文模块说明。 |

**复用点**：复用原生 HCCLCheckpointEngine 通信器和 VERL 权重数据路径；不绕过完成校验。

### 1.8.18 `验收辅助` — `testing/npu_restore_sender.py；testing/startup_diagnostics.py`

**继承与职责**：辅助脚本/函数（非产品 Actor）。

#### 字段

```python
# [运行期] 模型、通信环境与诊断参数；不持有 GS/Lease Owner 状态
```

#### 方法与复用点

| **方法** | **分类与功能** |
| --- | --- |
| `NPU RESTORE sender` | **本模块现有方法**；测试版本变化与收发真实路径。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |
| `startup diagnostics` | **本模块现有方法**；打印/检查实际运行环境。返回值与异常以源码实现为准；**复用点**：见下文模块说明。 |

**复用点**：复用目标环境的 torch_npu、HCCL、Ray / VERL；不参与生产控制面。

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
