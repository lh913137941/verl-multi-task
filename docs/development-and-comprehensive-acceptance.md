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

**2026-10-09 NPU E2E 已知问题**：`logs/two_real_jobs/20261009-084341/` 的 `force_cycle` 实际在第二次 ADD 失败，报 `ADD bootstrap failed; hidden runtime was verified RELEASED`，因此未进入 FORCE。日志显示 `checkpoint-finalize-complete` 与 `kv-resume-complete`，随后有 EngineCore shutdown 和 SIGTERM，但尚不足以确定底层根因。已在本分支 Trainer `bootstrap_and_publish` 异常捕获处增加包含 `operation_id`、`target`、`e_committed` 和 traceback 的日志（提交 `35ed25a`）；**仅增强诊断，尚未证明 ADD 根因已修复**。

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

**2026-10-09 真实 NPU 双任务执行**：

| 场景 | 观测结果 | 备注 |
| --- | --- | --- |
| `control_plane` | PASS | 同轮完成 |
| `exactly_once` | PASS | 同轮完成 |
| `recovery` | PASS | 同轮完成 |
| `lifecycle` | PASS | 同轮完成 |
| `force` | FAIL | `force_cycle` 的 ADD bootstrap 失败，未进入 FORCE |

该记录可证明所列通过场景在当次环境中完成，但不代表跨 CUDA/NPU、不同硬件组合均已验收。后续应在新诊断日志下复现失败，关联 `e2e-baa0d4fefb-add` 的原始 traceback，核实 CE target commit、服务发布、EngineCore 生命周期与二次借卡时序；修复后重新跑全量五场景及严格在途 FORCE。**目前不得声明 FORCE 真实 in-flight continuation 已通过。**

## 2.7 发布门禁

- 产品变更先跑 `tests/unit`，改动对应 Owner 的再跑相关 Ray/VERL 集成测试。
- 修改资源 lifecycle、checkpoint backend、FORCE 时必须在目标真实设备执行专项原语和双任务 E2E。
- 必须满足精确 `operation_id`、Lease 资源身份、参数版本、请求 owner 与设备释放证据，不允许靠放宽断言降低门禁。
- 评审合入时列出通过范围、失败项、未覆盖能力和运行链接；任何 `FAIL` / `BLOCKED` 均不可写成整套验收 PASS。
