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
`verl.plugins` 自动加载、不 monkey patch 原生类，也不提供另一份训练入口。当前
v0.10.0.dev 基线固定到 VERL commit `f92febf50fe3db102273eaf59b1854f392ae761d`；
对应入口接线已提交到 `lh913137941/verl:chatgpt/092203-multitask-entry`
(commit `9f5b131195f848c9d78347179ed02f6fe48c42fc`)，并随本仓
`patches/verl-v0.10-fully-async-multitask-entry.patch` 一并交付。

## 首版 profile

唯一支持的 profile 是 `experimental_fully_async_standalone`。当前首版边界：

- experimental Fully Async；
- pure STANDALONE；
- vLLM、non-PD；
- 单节点；
- 整卡借还（物理独占由 GS 的 `pg_id/bundle_index + gpu_uuid` owner 保证；Ray actor 资源记账不是物理份额）；
- DP=1、PP=1；
- 当前已记录的真实 GPU 验收只覆盖 TP=1；TP>1 在补齐同卡借还/RESTORE 实测前由 runtime profile fail-closed；
- `async_training.use_trainer_do_validate=false`；
- `async_training.use_dynamic_resource_scheduling=false`。

最小接入参数示例：

```text
multitask.enabled=true
actor_rollout_ref.hybrid_engine=false
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true
actor_rollout_ref.rollout.calculate_log_probs=true
async_training.use_trainer_do_validate=false
async_training.use_dynamic_resource_scheduling=false
data.train_batch_size=0
data.gen_batch_size=1
```

`multitask.enabled=false` 或省略 `multitask.enabled` 时继续走原生 TaskRunner；单独设置 `multitask.runtime.profile` 不会隐式启用 MultiTask。显式启用后，缺包、缺依赖、profile 不匹配或首版拓扑不满足都会报错，不静默回退到原生 MultiTask 路径。

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
             ├─ VERL CheckpointEngineWorker
             └─ MultiTaskvLLMHttpServer
```

092203 合同只保留四份 owner 真值：

- **M / Manager**：`replica_state[ReplicaKey]`、`replica_kind[ReplicaKey]`；
- **E / CE Manager**：`effective_replicas` 和参数侧事实；
- **R / LB**：原生 route/inflight + `active_request_server` + `attempt_state`；SETTLED request 只保留有界 ACK-loss 查询窗口，超出窗口时连同旧 continuation proof 一并回收；
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
调度键和 `node_id/gpu_uuid` 物理校验键；首版只允许整 GPU claim，并要求 borrowed
`world_size` 严格等于 borrower 任务自己的 TP×DP×PP 拓扑，不能仅凭 claim 数改变模型并行拓扑。

## 当前已落代码的控制面能力

- TaskRunner 使用有限并发，让原生长时间 `run()` 执行期间仍能处理 GS 的
  `submit_operation/query_operation`；operation journal 用锁保护；
- 原生 STANDALONE replicas 初始化完成后登记为 `NATIVE/ACTIVE`，供 M 和空泡判断读取；borrowed ADD 一经 Manager 接受就用同一 `ReplicaKey` 登记为 `BORROWED/CREATING`，创建结果未能证明成功时收口到 `QUARANTINED`，不会跳过 M 直接发布服务；
- LB 复用当前 VERL router API，并补逐 request `ADMITTED/TERMINATED/SETTLED`；新增 `commit_ready()` 在 LB 单写者内原子登记 server + `ReplicaKey→server_id` route，并保存 operation-scoped `SERVICE_COMMITTED` receipt，ACK 丢失可用 `query_ready_operation()` 对账而不重复上架；
- 自然 release 进入 `SETTLED`；已验证 continuation 先进入 `TERMINATED`，随后由迟到 release 或退出提交收敛到 `SETTLED`；
- GS 维护最小 Lease 账本，claim_id 永久绑定原 lease；首版一个 Lease 只覆盖一个完整 donor replica，且一个 lease_id 对应一次 borrower 生命周期。DONATE 的 RELEASED 只把已预留 claims 推进到 borrower handoff-ready，只有 borrowed REMOVE 的 RELEASED 才把 PG/bundle 与 GPU UUID 归还可授权池；RESTORE 受理后临时重新占住这些 claims 防止 wake 期间被新 lease 抢占，`SERVICE_COMMITTED` 通过同一 `advance_lease()` 关闭该周期并释放临时 reservation，下一轮借卡必须新 lease_id；首版固定 `max_colocate_count=2`，claim 的 `gpu_fraction=0.5` 仅是 Ray 调度记账份额，不能解释成“半张物理 GPU 可同时借给另一任务”；
- RELEASED 必须与已登记的 `operation_id → lease_id` 对应，并完整覆盖 lease 的 GPU UUID 集合；runtime release 无法给出真实证据时 Manager 将目标从 DRAINING 收口到 QUARANTINED。TaskRunner 的同 operation UNKNOWN 重放会先查 Manager RELEASED 账本；若只是 E 已移除后的 R/C ACK 不确定，则在同一 G 下对账并解围栏，不重复释放物理资源；
- native 参数同步继续复用原生实现，但进入同一个 Trainer G；CE Manager 增加 owner-local pending-bootstrap 投影，borrowed runtime 在 `WEIGHT_READY` 前不会进入父类 `replicas` effective set/普通全成员同步；`bootstrap_target()` 已复用原生 CE `prepare/build_topology/init/update/finalize` 协议只同步 pending target，并要求目标 server 的 `global_steps` 精确等于当前发布版本后才生成幂等 `WEIGHT_READY` receipt，匹配 operation 的证据才能提升为 effective；
- Queue exactly-once 仍是原生样本之上的逻辑 key + digest 薄层；
- 空泡上报只选择“移除后仍能保持当前 committed capacity”的 ACTIVE surplus replica，不再把所有 ACTIVE replica 都当作可捐候选；GS 对已有 idle report 的 DONATE 只接受 10s 内且包含目标 NATIVE ReplicaKey 的事实；Lease donor 按 Manager/Trainer 的规范身份 `native-{replica_rank}` 精确匹配，不再接受 `r{rank}` 命名别名；
- borrowed placement 入口已在任何 Ray Actor 副作用前校验 borrower/source lease、claim、world_size、单节点 rank/local_rank、整卡约束和过期时间；Manager 同时按 borrower lease 建立幂等 `borrowed_operations` 记录并单调分配 `replica_rank`，相同 create 重试不会二次分配 rank、冲突重放会被拒绝；随后通过 Ray placement-group table 以 `pg_id` 核验 CREATED 状态、namespace、bundle/node 落点，并借 PG name 恢复后反查 handle id；Manager 内部 runtime backend 已接通 hidden borrowed runtime 创建和 verified borrowed destroy：创建成功只登记 `RUNTIME_READY/CREATING`，不会提前进入 E/R/C；destroy 只有在 server/worker 清理验证完成后才生成带完整 GPU UUID 的 `RELEASED`；`MultiTaskvLLMReplica` 进一步生成确定性的 borrower CE actor name、PG/bundle、Ray GPU/CPU 份额和 borrower rank/world 环境计划；底层 `_create_workers_from_claims()` 已按 claim clone `RayClassWithInitArgs`、建立 borrower 自己的 MASTER 通信根并用 `RayWorkerGroup.from_detached()` 包装新 handles，物理 placement 直接复用 VERL `CheckpointEngineWorker.__ray_call__` 执行只读 node/Ray accelerator/GPU UUID 探针；worker 创建后必须逐 rank 与 claim 核对 node/GPU UUID，失败时按确定性 actor name 执行 kill，并通过 Ray State API 确认不存在非 `DEAD` actor。`MultiTaskvLLMHttpServer` 现在还提供 vLLM `check_health()` 驱动的健康事实与单节点 graceful shutdown；`MultiTaskvLLMReplica.init_from_lease()` 能在底层完成 worker → 原生 `launch_servers()` → engine/server 健康校验，并在失败时按 server/worker 名称执行 shutdown/kill + Ray State `DEAD` 核验。该 hidden-create primitive 已由 Manager 调用，ADD 现已从 TaskRunner 正常受理，并通过 Trainer G 完成 target-only current-Vpub bootstrap、E 提交与 R/C/M 发布；无法证明创建/装参/发布结果时仍按 UNKNOWN/QUARANTINED 规则收口；
- GS 句柄只保留在 TaskRunner/Rollouter 跨任务边界，不再下沉到 Manager/LB；Manager/LB 只维护本任务 M/R 与 runtime/request 事实；
- 对外 `OperationCommand` 仍只携带 `lease_id`；GS 在转发给 TaskRunner 时附带同一份已校验 Lease 快照，TaskRunner 只在内部生成 borrowed placement spec，再交给 Rollouter/Manager 校验。

## 已接通但仍需真实 GPU/runtime 验收的能力

以下主流程代码已经接通并由 TaskRunner 受理，但仍必须在真实 VERL/vLLM/CUDA/NCCL 组合完成验收；运行时无法证明 placement、权重、请求交接或资源释放时会显式失败/隔离，不用假 handle 或合成证据伪造成功：

- ADD：hidden borrowed create → current-Vpub target bootstrap → E → R/C/M 已端到端接线；纯 placement 校验失败、runtime 创建前失败、或已证明 borrowed cleanup 的失败会以 FAILED+RELEASED 补偿收口，不再误记 UNKNOWN；真实 GPU/NCCL 验收仍需实际执行；
- DONATE：自然 drain → E/R/C 退出 → native level-2 sleep → GPU UUID `RELEASED` → GS handoff-ready 已端到端接线；自然 drain 由 `multitask.drain_timeout_s`（默认 300s）设上限并周期输出等待日志，超时进入隔离而不是无限轮询；真实同卡让渡闭环仍需实际执行；
- RESTORE 已在内部复用现有 `pending_bootstrap / bootstrap_target / commit_ready` 串起流程；weights-only wake 已移入 Trainer G 内的 CE bootstrap，随后完成当前 Vpub 全量装参、KV 恢复/版本确认并先提交 E，再 full wake、本地 C/M 就绪，最后由 R 对外发布；LB 回包丢失先按 `query_ready_operation()` 对账；E 提交前的 CE/bootstrap 失败可在 G 内 verified re-sleep 回 DORMANT；E 已 effective 后若 R 明确未发布，则先撤 C/M、再由 Trainer 在 G 内撤 E，最后真实 re-sleep 并以 RELEASED 补偿；R 仍未知则保持 runtime awake/ACTIVE 并 BLOCK G，同 operation 重放会重新查询 R，只有向前补齐或验证回滚成功才解围栏。opt-in GPU 验收还会先真实修改 sender output weights，避免只靠 `global_steps` 标签过关；当前环境尚未跑通该 GPU 闭环，但 TaskRunner 已受理 RESTORE，运行时按证据失败/隔离；
- FORCE_VERIFIED：borrowed-only targeted abort + continuation proof + partial-rollout retry 已接线，并有 opt-in 两 GPU 验收入口；当前会话未实际运行该 GPU 用例；
- native sleep 后还需用真实 GPU 实验确认显存让渡足以让 borrower 在同一物理 GPU 启动；控制面 receipt/mock 不作为验收。

首版 whole-GPU 借还还要求 VERL `_resolve_sleep_level()==2`；因此 MTP rollout / unmerged LoRA rollout 等会退化为 level-1 sleep 的配置在 runtime profile 阶段直接拒绝，不能生成 `RELEASED`。

paused production window 现在由 Rollouter 内部 reporter 自动向 GS 发送 idle metadata；相同候选集每 5s 刷新一次，避免 GS 长期持有陈旧 bubble 事实；报告失败只重试元数据，不中断生成。

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

# 真实 CUDA/vLLM；1 GPU 验 DONATE→同卡 borrower→REMOVE，>=2 GPU 继续验 current-Vpub RESTORE
VERL_MULTITASK_GPU_MODEL_PATH=/path/to/local/model \
python -m pytest -q -s -m gpu_integration tests/integration/test_native_sleep_gpu.py
```

当前会话环境无法拉取并执行当前分支的完整 pytest，也没有可用的 GitHub Actions
结果；历史轻量测试记录不能替代本轮修改后的验证。因此 README 只记录已落库的测试
入口，不声明当前 HEAD 的 unit/native/GPU 已通过。

详细设计以当前 092203 设计文档和 `docs/simplified-fusion-contract.md` 为准。
