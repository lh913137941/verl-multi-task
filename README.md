# verl-multi-task

可选的 VERL experimental Fully Async 扩展包。继续使用原生训练入口，通过
`multitask.enabled=true` 选择 MultiTask 创建链；默认关闭。

当前实现合同：`docs/simplified-fusion-contract.md`（092203）。本轮实现目标是把
控制面和 owner 状态收敛到该合同，同时保留未验证 GPU 原语的显式失败边界。

## 安装与接入

在已经能运行 VERL 的训练环境中安装本仓：

```bash
uv pip install --python /path/to/training/python /path/to/verl-multi-task
# 开发时：
uv pip install --python /path/to/training/python -e /path/to/verl-multi-task
```

Driver、所有 Ray 节点及其运行环境都需要安装同一份扩展包，并使用兼容的
VERL/Ray/vLLM 环境。安装本包不会替换这些运行时依赖。

仍使用 VERL 原生入口：

```bash
python -m verl.experimental.fully_async_policy.fully_async_main
```

需要 VERL 入口具备 MultiTask 选择接线：配置规范化后、`run_ppo` 初始化 Ray 前，
按 `multitask.enabled` 延迟解析 `MultiTaskFullyAsyncTaskRunner`。本包不使用
`verl.plugins` 自动加载、不 monkey patch 原生类，也不提供另一份训练入口。

## 首版 profile

唯一支持的 profile 是 `experimental_fully_async_standalone`。当前首版边界：

- experimental Fully Async；
- pure STANDALONE；
- vLLM、non-PD；
- 单节点；
- 整卡借还；
- DP=1、PP=1；
- TP 必须能放入单节点并经过实际组合验证；
- `async_training.use_trainer_do_validate=false`；
- `async_training.use_dynamic_resource_scheduling=false`。

最小接入参数示例：

```text
multitask.enabled=true
actor_rollout_ref.hybrid_engine=false
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
actor_rollout_ref.rollout.calculate_log_probs=true
async_training.use_trainer_do_validate=false
async_training.use_dynamic_resource_scheduling=false
data.train_batch_size=0
data.gen_batch_size=1
```

`multitask.enabled=false` 时继续走原生 TaskRunner。显式启用后，缺包、缺依赖、
profile 不匹配或首版拓扑不满足都会报错，不静默回退到原生 MultiTask 路径。

## 创建链与状态所有者

```text
VERL main → resolve_runtime_profile → run_ppo
  → MultiTaskFullyAsyncTaskRunner
    ├─ MultiTaskFullyAsyncTrainer
    │  └─ MultiTaskCheckpointEngineManager
    └─ MultiTaskFullyAsyncRollouter
       └─ MultiTaskLLMServerManager
          ├─ MultiTaskGlobalRequestLoadBalancer
          └─ MultiTaskvLLMReplica
             ├─ MultiTaskCheckpointEngineWorker
             └─ MultiTaskvLLMHttpServer
```

092203 合同只保留四份 owner 真值：

- **M / Manager**：`replica_state[ReplicaKey]`、`replica_kind[ReplicaKey]`；
- **E / CE Manager**：`effective_replicas` 和参数侧事实；
- **R / LB**：原生 route/inflight + `active_request_server` + `attempt_state`；
- **C / Rollouter**：复用原生 `max_concurrent_samples`，生产窗口复用 `paused`。

Trainer 持有唯一同步门 G。不存在公共 `ReplicaRecord` 或 `AttemptRecord`。
生命周期状态只有 `CREATING / ACTIVE / DRAINING / DORMANT / RELEASED /
QUARANTINED`。

公共生命周期结构收敛为：

```text
OperationCommand(operation_id, kind, target, lease_id, force?)
OperationRecord(operation_id, status, result?)
OperationEvidence(operation_id, type, timestamp, released_gpu_uuids=())
Lease(lease_id, claims, expires_at)
```

`ReplicaKey` 只是共享身份。Lease 的 claim 至少携带 Ray `pg_id/bundle_index`
调度键和 `node_id/gpu_uuid` 物理校验键；首版只允许整 GPU claim。

## 当前已落代码的控制面能力

- TaskRunner 使用有限并发，让原生长时间 `run()` 执行期间仍能处理 GS 的
  `submit_operation/query_operation`；operation journal 用锁保护；
- 原生 STANDALONE replicas 初始化完成后登记为 `NATIVE/ACTIVE`，供 M 和空泡判断读取；
- LB 复用当前 VERL router API，并补逐 request `ADMITTED/TERMINATED/SETTLED`；
- 自然 release 进入 `SETTLED`；已验证 continuation 先进入 `TERMINATED`，随后由迟到 release 或退出提交收敛到 `SETTLED`；
- GS 维护最小 Lease 账本，claim_id 永久绑定原 lease；首版整卡下同一 PG/bundle 与 GPU UUID 在 RELEASED 前不能被第二个 lease 重复授权；
- RELEASED 必须与已登记的 `operation_id → lease_id` 对应，并完整覆盖 lease 的 GPU UUID 集合；
- native 参数同步继续复用原生实现，但进入同一个 Trainer G；
- Queue exactly-once 仍是原生样本之上的逻辑 key + digest 薄层；
- 空泡上报只选择“移除后仍能保持当前 committed capacity”的 ACTIVE surplus replica，不再把所有 ACTIVE replica 都当作可捐候选；
- borrowed placement 入口已在任何 Ray Actor 副作用前校验 borrower/source lease、claim、world_size、单节点 rank/local_rank、整卡约束和过期时间；Manager 同时按 borrower lease 建立幂等 `borrowed_operations` 记录并单调分配 `replica_rank`，相同 create 重试不会二次分配 rank、冲突重放会被拒绝；随后仍停在显式 `NotImplementedError`，直到 PG/bundle actor 创建经过真实环境验证；
- GS 句柄只保留在 TaskRunner/Rollouter 跨任务边界，不再下沉到 Manager/LB；Manager/LB 只维护本任务 M/R 与 runtime/request 事实；
- 对外 `OperationCommand` 仍只携带 `lease_id`；GS 在转发给 TaskRunner 时附带同一份已校验 Lease 快照，TaskRunner 只在内部生成 borrowed placement spec，再交给 Rollouter/Manager 校验。

## 明确未完成的 GPU/runtime 能力

以下能力仍必须在真实 VERL/vLLM/CUDA/NCCL 组合完成验证后实现；当前代码继续
显式抛出 `NotImplementedError`，不会用假 handle 或合成证据伪造成功：

- borrowed hidden runtime 创建与 lease-aware GPU 绑定；
- native DONATE 的真实 sleep/release；
- target-only 参数 bootstrap / replay；
- RESTORE 的真实 wake 与参数恢复；
- FORCE_VERIFIED 的 targeted abort + Client continuation；
- sleep/destroy 后基于真实进程/设备事实生成 RELEASED。

因此 `multitask.enabled=true` 目前表示“启用 092203 控制面与 native subclass
绑定”，不表示 GPU 借还闭环已经通过验收。

## 验证

本分支应分层验证，不把 mock/AST 当作 GPU 成功：

```bash
# 依赖轻量 unit
python -m pytest -q tests/unit

# 有 Ray 环境后
python -m pytest -q -m ray_integration tests/integration

# 指向兼容的真实 VERL checkout 后
MT_VERL_SOURCE_ROOT=/absolute/path/to/verl \
python -m pytest -q -m native tests/native_unit
```

本次 092203 迁移在会话环境中已实际执行依赖轻量的合同/operation journal/
exactly-once/G gate 核验；真实 Ray、完整仓库 unit、native VERL 和 GPU 测试仍需在
具备对应依赖的环境运行。当前 GitHub 分支没有可用的 Actions 运行结果，因此不在
README 中声明未执行的测试通过。

详细设计以当前 092203 设计文档和 `docs/simplified-fusion-contract.md` 为准。
