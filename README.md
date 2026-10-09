# verl-multi-task

面向 VERL experimental Fully Async 的多任务共享调度扩展。

本仓不替换 VERL 训练入口，也不复制一套训练框架；它在原生 Fully Async +
STANDALONE + vLLM 链路上增加：

- 多任务共享资源生命周期：`ADD / DONATE / REMOVE / RESTORE`；
- borrowed replica 的动态创建、装参、发布和回收；
- FORCE REMOVE 的 targeted abort + continuation；
- Lease / operation evidence / request owner 的 fail-closed 控制面；
- 参数同步门、Exactly-once、异常恢复和 E2E 验收；
- CUDA 与 Ascend NPU 的设备级 acceptance 入口。

> 当前生产合同以
> [docs/simplified-fusion-contract.md](docs/simplified-fusion-contract.md)
> 为准；完整验收规则见
> [docs/e2e-acceptance.md](docs/e2e-acceptance.md)。

---

## 1. 当前支持范围

本仓目前对接 VERL 的 **experimental Fully Async + 独立 vLLM Rollout**，启用 MultiTask 后自动选择唯一支持的运行配置，普通使用者无需填写 Runtime Profile 名称。

| 项目 | 当前边界 |
| --- | --- |
| 设备 | CUDA GPU / Ascend NPU（不支持 CPU-only 资源借还） |
| 部署 | 独立 Rollout；单节点，TP / DP / PP = 1 / 1 / 1 |
| 资源共享 | 整张物理 GPU/NPU 的借还；不支持把 0.5 Ray 资源份额视为半张卡 |
| 生命周期 | ADD、DONATE、natural REMOVE、RESTORE |
| FORCE REMOVE | 仅 BORROWED，要求部分生成、其他可用 Server、真实续推证明 |
| 参数同步 | CUDA 使用原生 NCCL；Ascend NPU 使用 `multitask_hccl` |
| 暂不支持 | Hybrid、PD/disaggregation、未合并 LoRA 的借卡等未验证组合 |

**验收现状**：真实 NPU 双任务已完成共享 GS 注册、自动 Lease 物理身份校验和 lifecycle / force 验收；不代表所有场景及真实在途 FORCE 均已通过。以最新运行的 `orchestration_summary.json` 和 [GitHub Actions](https://github.com/lh913137941/verl-multi-task/actions) 为准。

所有关键操作采用 **fail-closed**：无法证明资源释放、参数版本或请求安全交接时，不会直接返回成功。

---

## 2. 主要流程：如何借卡、还卡

正常训练仍由 VERL Fully Async 驱动；MultiTask 只在共享调度操作发生时接管对应资源生命周期：

| 操作 | 执行流程 | 完成证据 |
| --- | --- | --- |
| **DONATE（出借）** | Native 停止接收新请求 → 等已有请求结清 → 退出服务/参数同步集合 → 休眠原 Runtime | `RELEASED` |
| **ADD（借入）** | 根据 Lease 在借出资源上创建隐藏 Borrowed Runtime → 同步当前模型权重 → 注册路由并对外服务 | `SERVICE_COMMITTED` |
| **REMOVE（归还）** | Borrowed 停止接新请求 → 等待请求结清 → 退出服务 → 销毁 Borrowed Runtime，归还卡 | `RELEASED` |
| **RESTORE（恢复）** | 原 Native Runtime 唤醒 → 同步最新权重、恢复 KV/服务 → 重新参与生成 | `SERVICE_COMMITTED` |

典型借还链路：**DONATE → ADD → REMOVE → RESTORE**。需要强制回收时，只有 Borrowed Replica 可走 **FORCE REMOVE**：旧请求必须完成 abort 和可验证 continuation，在另一可用 Server 续推，不能直接丢弃在途请求。

注意：DONATE 返回的 `RELEASED` 表示**物理卡使用权已释放**，但原 Native Replica 的状态仍为 `DORMANT`，等待 RESTORE；Borrowed 被销毁后才进入 `RELEASED` 状态。

---

## 3. 组件分工：哪些模块负责什么

| 组件 | 职责 |
| --- | --- |
| **GS / GroupScheduler** | 任务注册、Lease/资源归属账本、操作分发 |
| **TaskRunner** | 每个任务的控制入口；编排操作、查询进度和异常对账 |
| **Trainer / G** | 保留原生训练和更新权重流程；参数同步与生命周期操作通过同一同步闸门 |
| **E / CheckpointEngineManager** | 参与权重同步的副本、Bootstrap 和模型参数版本 |
| **Rollouter / C** | 生成流程、生产容量、请求 drain 与服务发布 |
| **M / LLMServerManager** | Native/Borrowed Replica 的创建、休眠、唤醒、销毁及状态 |
| **R / LoadBalancer** | 服务路由、请求准入、在途请求和续推回执 |

各模块只维护自己的事实，不使用一份重复的全局状态。操作采用 `OperationCommand` 分发，通过 `OperationRecord / OperationEvidence` 查询与证明结果；超时、ACK 丢失则按**同一 operation_id** 对账，不盲目重做资源变更。详细状态和接口见 [设计合同](docs/simplified-fusion-contract.md)。

---

## 4. 与 VERL 的接线关系

**入口不变、训练主循环不变。** VERL 原生 `fully_async_main` 在 `multitask.enabled=true` 时选择 MultiTask TaskRunner；否则使用原生 TaskRunner。MultiTask 通过少量 subclass 复用 VERL 的原生 Trainer、Rollouter、vLLM 和 Checkpoint Engine。

```text
VERL fully_async_main（原生启动入口）
  └─ MultiTaskFullyAsyncTaskRunner（每个任务一个）
       ├─ MultiTaskFullyAsyncTrainer（继承 FullyAsyncTrainer）
       │    └─ MultiTaskCheckpointEngineManager → CheckpointEngine Worker
       ├─ MultiTaskFullyAsyncRollouter（继承 FullyAsyncRollouter）
       │    └─ MultiTaskLLMServerManager
       │         ├─ MultiTaskGlobalRequestLoadBalancer
       │         └─ MultiTaskvLLMReplica → MultiTaskvLLMHttpServer
       └─ MultiTaskMessageQueue（完成样本恰好一次提交）

多个 TaskRunner ↔ 同一个 GroupScheduler（GS，独立 Ray Actor）
```

**正常训练数据流：** VERL 初始化 Trainer/独立 Rollout Server → 参数同步 → AgentLoop 经 LB 请求 vLLM 生成 → MessageQueue 提交完成样本 → Trainer 消费样本并更新权重；MultiTask 保留原生业务循环。

**跨任务控制流：** Rollouter 上报可借资源 → GS 管理 Lease 和分发操作 → 对应 TaskRunner 调用 Trainer/Rollouter 的生命周期动作 → Manager、Checkpoint Engine 和 LB 更新各自事实 → TaskRunner 返回 Evidence，GS 再推进 Lease。E2E 使用同一控制接口执行实际操作。

**入口接线方式：** 本仓不复制另一份 VERL 启动程序。第 6 节 E2E 启动器会检查当前 Python 实际导入的 VERL 入口，必要时备份并安装最小选择桥接；手动接入见第 5 节。

---

## 5. 安装与准备

先确保目标 VERL Fully Async 和 Ray/vLLM（Ascend 使用 vLLM-Ascend）本身能正常运行。本仓不会替代这些环境依赖。

- **只运行第 6 节 E2E：** 可以从源码目录直接运行，启动器自动把 `src/` 配置给本次进程和新建的 Ray Worker；不要求额外 `pip install -e .`。
- **开发或单测：** 在仓库根目录安装：

```bash
python -m pip install -e '.[test]'
```

- **手动通过原生 VERL 入口启动：** 设置 `multitask.enabled=true` 前，先检查接线：

```bash
python scripts/e2e/ensure_verl_multitask_bridge.py --check-only
```

当前仅有一种支持的 Runtime Profile，**无需手动填写 `experimental_fully_async_standalone`**。必要的运行约束由 [runtime_profile.py](src/multi_task_scheduler/integration/verl/runtime_profile.py) 检查，E2E 启动器会自动配置符合首版范围的参数。

入口补丁位于 [patches/verl-v0.10-fully-async-multitask-entry.patch](patches/verl-v0.10-fully-async-multitask-entry.patch)。多节点使用时须确保各节点都能访问相同源码和依赖。

---

## 6. 双任务 E2E：实际执行命令

你当前环境中的 `examples/e2e/native_args.txt` 已包含实际模型、训练集及验证集路径时，**直接运行下面这条命令即可**，不必另传 `--model-path`、`--train-files`、`--val-files`：

```bash
cd /workspace/n00873601/multi_rl_task_lh_clone/verl/multi_task_verl

python scripts/e2e/verify_two_verl_jobs.py \
  --repo . \
  --ray-address auto \
  --start-local-ray \
  --native-args /workspace/n00873601/multi_rl_task_lh_clone/verl/multi_task_verl/examples/e2e/native_args.txt \
  --scenarios "lifecycle force"
```

`--scenarios "lifecycle force"` 验证两条真实资源流程：
`DONATE → ADD → REMOVE → RESTORE`，以及
`DONATE → ADD → FORCE REMOVE → RESTORE`。

启动器会自动启动 donor/borrower 两个 VERL driver、将其注册到同一个 GS、依据真实 donor Placement Group 和设备身份生成测试 Lease，并运行所选场景。

**换机器或重新 clone 时先检查参数文件：** 仓库原始样例中的模型路径可能仍是 `/REPLACE_WITH_VERL_REPO_DIR/...` 占位符；必须先在 `native_args.txt` 中写好本机真实模型及数据路径。配置好后无需在命令行重复传路径。

常用变化：需要全部五项验收时改为 `--scenarios "control_plane exactly_once recovery lifecycle force"`；有**持久/多机 Ray** 时改用 `--ray-address <head地址>` 并删除 `--start-local-ray`。不希望启动器自动备份/桥接 VERL 入口时加 `--no-auto-bridge`。

---

## 7. CUDA 与 Ascend NPU

| 配置 | CUDA | Ascend NPU |
| --- | --- | --- |
| `trainer.device` | `cuda` | `npu`（当前 NPU 参数样例） |
| Ray 资源 | `GPU` | `NPU` |
| 权重同步 | 原生 `nccl` | `multitask_hccl`（底层 HCCL） |
| Native sleep | 支持条件下 level 2 | 平台安全的 level 1 |

使用第 6 节启动器时，权重同步后端会根据设备自动选择；不需要手工修改 CUDA/NPU backend。CUDA 需要自己的原生训练参数文件，不能直接照搬 `trainer.device=npu` 的 NPU 样例。

NPU 环境诊断（只检查版本、设备与模型文件结构，不代表真实模型已加载）：

```bash
VERL_MULTITASK_NPU_MODEL_PATH=/实际模型路径/Qwen3-0.6B \
  python scripts/e2e/diagnose_npu_runtime.py
```

如需验证真实模型加载、NPU sleep 或参数恢复，运行第 8 节硬件测试。

---

## 8. 测试：按运行环境选择

`tests/` 只有两类：[unit](tests/unit/)（CPU 快速回归）和 [integration](tests/integration/)（真实 Ray、VERL 或 GPU/NPU）。
在仓库根目录按需要运行：

```bash
# 日常必跑：无需原生 VERL 或加速卡
python -m pytest -q tests/unit

# CPU Ray 控制面（需要 Ray）
python -m pytest -q tests/integration/ray

# VERL 原生接线（需要真实 VERL、Ray 等 Python 依赖）
python -m pytest -q tests/integration/verl
```

CUDA/NPU 硬件测试需在对应实机运行；以下以已配置的本地模型路径为例：

```bash
# Ascend NPU：先单测真实模型加载，再按需测试 sleep / FORCE / RESTORE
export VERL_MULTITASK_NPU_MODEL_PATH=/实际模型路径/Qwen3-0.6B
python -m pytest -q -s -m npu_backend_smoke tests/integration/npu/test_native_sleep_npu.py
python -m pytest -q -s -m npu_integration tests/integration/npu/test_native_sleep_npu.py

# CUDA：真实 vLLM 和模型、设备验收
VERL_MULTITASK_GPU_MODEL_PATH=/实际模型路径 \
  python -m pytest -q -s -m gpu_integration tests/integration/cuda/test_native_sleep_gpu.py
```

CUDA 环境基线测试位于 `tests/integration/cuda/test_baseline_environment.py`，
运行前须设置 `MT_GPU_TEST_CONFIG`。单测及各集成层的依赖、命令与边界见
[tests/README.md](tests/README.md)；具体 Unit 文件职责见
[tests/unit/README.md](tests/unit/README.md)。

---

## 9. E2E 验收结果与排错

`verify_two_verl_jobs.py` 是唯一推荐的启动入口；内部由 `run_all.sh` 分发到五个 `validate_*` 验收脚本，用户无需逐个运行。

`--scenarios` 允许选择以下场景；只需更改第 6 节命令的最后一行：

| 场景 | 验证 |
| --- | --- |
| `control_plane` | GS/Lease、操作回执与恢复 |
| `exactly_once` | 完成样本去重、冲突拒绝 |
| `recovery` | UNKNOWN、异常和故障注入 |
| `lifecycle` | DONATE → ADD → REMOVE → RESTORE |
| `force` | DONATE → ADD → FORCE REMOVE → RESTORE |

日志在 `logs/two_real_jobs/<运行时间>/`，其中 `orchestration_summary.json` 汇总最终结果，具体场景错误在 `scenarios/<场景>/<运行ID>/result.json`。

```bash
RUN=$(ls -dt logs/two_real_jobs/*/ | head -n 1)
cat "${RUN}orchestration_summary.json"
find "${RUN}scenarios" -name result.json -print
grep -E 'FAIL|BLOCKED|Traceback|Error' "${RUN}e2e.log" | tail -n 40
```

`STATE=PASS` 且所选场景均为 PASS 才算该轮通过；退出码 `0=PASS`、`1=FAIL`、`2=BLOCKED`（环境或证明条件不足）。定位失败先查对应场景 `result.json`，再查 `donor.log` / `borrower.log`。

**FORCE 验证范围：** 默认 `force: PASS` 证明对应生命周期闭环，不一定命中真实在途请求；要严格要求真实在途 abort/continuation 证据，增加 `--require-inflight-force`（含 `force` 场景时使用）。

更详细的验收、异常恢复与手动 Lease 调试见 [E2E 验收文档](docs/e2e-acceptance.md)。

---

## 10. 实现原则

- **尽量不改 VERL 主流程**：通过原生入口选择和最小 subclass 接线；
- **Owner 单写者**：Manager、Checkpoint Engine、LB、Rollouter、GS 分别维护自己的真实状态；
- **关键操作要有证据**：物理资源释放、参数版本、请求交接不能靠配置或推测；
- **同一操作对账**：ACK 丢失或 UNKNOWN 使用相同 `operation_id` 幂等重放；
- **失败时安全封锁**：证据不完整时保持 DRAINING/BLOCKED/QUARANTINED，不能提前恢复流量。

更完整的时序与回滚路径见 [设计合同](docs/simplified-fusion-contract.md)。

---

## 11. 代码导航

| 代码位置 | 主要内容 |
| --- | --- |
| [`scheduler/`](src/multi_task_scheduler/scheduler/) | GroupScheduler、Lease、任务注册 |
| [`orchestration/`](src/multi_task_scheduler/orchestration/) | 操作、证据及同步闸门 |
| [`integration/verl/experimental_fully_async/`](src/multi_task_scheduler/integration/verl/experimental_fully_async/) | VERL TaskRunner / Trainer / Rollouter / Manager 接线 |
| [`rollout/`](src/multi_task_scheduler/rollout/) | vLLM Replica、路由、服务 |
| [`checkpoint/`](src/multi_task_scheduler/checkpoint/) | Checkpoint Engine、参数同步 |
| [`scripts/e2e/`](scripts/e2e/) | 双任务真实运行、分场景测试、诊断 |
| [`tests/`](tests/) | 单测、Ray、Native 与真实硬件测试 |

设计细节见 [simplified-fusion-contract.md](docs/simplified-fusion-contract.md)，验收规则见 [e2e-acceptance.md](docs/e2e-acceptance.md)。

---

## 12. 设计与融合来源

当前分支：[chatgpt/0928-merge-verl-expansion](https://github.com/lh913137941/verl-multi-task/tree/chatgpt/0928-merge-verl-expansion)。

本仓从 simplified-fusion 合同出发，吸收了 `verl_expansion` 中与现有 Lease、Owner 和 Evidence 机制兼容的实现；[历史参考](docs/verl-expansion-reference/) 仅用于对照。发生冲突时以当前代码和 [设计合同](docs/simplified-fusion-contract.md) 为准。
