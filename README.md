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

首版运行配置：`experimental_fully_async_standalone`，基于 VERL experimental Fully Async + 独立 vLLM Rollout。

| 能力 | 当前边界 |
| --- | --- |
| 设备 | CUDA GPU / Ascend NPU；不支持 CPU-only 资源借还 |
| 拓扑 | 单节点 Rollout；TP / DP / PP = 1 / 1 / 1 |
| 资源借还 | 整张物理 GPU/NPU，不能按 Ray 的 fractional resource 当作分卡借出 |
| 资源生命周期 | ADD、DONATE、natural REMOVE、RESTORE |
| FORCE REMOVE | 仅 BORROWED；需要 partial rollout、其他可用 Server 和可验证续推 |
| 参数同步 | CUDA 原生 NCCL；NPU 使用 `multitask_hccl` |
| 暂不支持 | PD/disaggregated rollout、未合并 LoRA 的借卡、需要更浅 sleep 的不兼容配置 |

**验证状态**：软件回归以 [GitHub Actions](https://github.com/lh913137941/verl-multi-task/actions) 为准，不在 README 固定易过期的通过数量。
真实 NPU 双任务已跑通共享 GS 注册、自动 Lease 物理身份检查及 lifecycle / force 场景；
**这不等于全套场景或有在途请求的 FORCE 续推都已验收通过**。
每轮真实环境的最终结论应查看第 9 节的 `orchestration_summary.json`。

能力或证据不足时保持 **fail-closed**，不会凭配置或超时推定资源已释放、样本已续推成功。

---

## 2. 生命周期

| 操作 | 主要流程 | 完成证据 |
| --- | --- | --- |
| **DONATE** | NATIVE/ACTIVE → DRAINING → 请求结清、退出服务 → 原 Runtime 睡眠并保留为 DORMANT | `RELEASED`（释放物理卡使用权） |
| **ADD** | Lease 已可借 → 创建隐藏 BORROWED Runtime → 装载当前权重 → 发布服务并 ACTIVE | `SERVICE_COMMITTED` |
| **REMOVE** | BORROWED/ACTIVE → DRAINING → 请求结清 → 销毁借用 Runtime、归还资源 | `RELEASED` |
| **RESTORE** | NATIVE/DORMANT → 恢复原 Runtime、同步当前权重 → 恢复服务并 ACTIVE | `SERVICE_COMMITTED` |

注意：**DONATE 的 `RELEASED` 是操作证据，不是 NATIVE Replica 的状态**；
Native Runtime 保留为 `DORMANT` 以便 RESTORE，同一借用 Runtime 在 REMOVE 后才进入 `RELEASED` 状态。

**FORCE REMOVE** 仍是 BORROWED-only：必须关闭旧服务准入、执行真实 abort，
由 Client 确认可交接前缀并在另一个 active Server 续推，旧 attempt 安全结清后才能放行。
不能因为 `partial_rollout=true` 就认定 FORCE 已验证。详见
[验收规则](docs/e2e-acceptance.md)。

---

## 3. Owner 真值

各组件只维护自己负责的事实，避免重复状态机：

| Owner | 负责什么 |
| --- | --- |
| **GS** / GroupScheduler | TaskRunner 注册、Lease / claim 归属、操作分发与账本 |
| **M** / LLMServerManager | Replica 类型、状态和真实 Runtime |
| **E** / CheckpointEngineManager | 参数版本、有效副本集合、Bootstrap |
| **R** / LoadBalancer | 服务路由、在途请求、attempt 终态和续推回执 |
| **C** / Rollouter | 生产容量和生成暂停/恢复 |
| **G** / Trainer gate | 参数同步与生命周期变更的互斥、异常封锁 |

Replica 状态只有 `CREATING / ACTIVE / DRAINING / DORMANT / RELEASED / QUARANTINED`；
请求 attempt 状态由 R 管理（`ADMITTED / TERMINATED / SETTLED`）。
跨组件以 `ReplicaKey`、`Lease`、`OperationCommand`、
`OperationRecord`、`OperationEvidence` 交互；UNKNOWN 必须查询 Owner 并按同一操作对账。
完整规则见 [simplified-fusion-contract.md](docs/simplified-fusion-contract.md)。

---

## 4. VERL 如何接入 MultiTask

继续使用 **VERL 原生 Fully Async 入口**，仅在启用 MultiTask profile 时选择扩展 TaskRunner：

```text
VERL fully_async_main
  └─ MultiTaskFullyAsyncTaskRunner
      ├─ MultiTaskFullyAsyncTrainer → CheckpointEngineManager / Worker
      └─ MultiTaskFullyAsyncRollouter
          └─ MultiTaskLLMServerManager
              ├─ GlobalRequestLoadBalancer
              └─ MultiTaskvLLMReplica → MultiTaskvLLMHttpServer
```

GS 通过 TaskRunner 转发跨任务操作；原生训练和生成主循环继续复用 VERL 实现。
包本身不提供另一套训练主程序。启用所需的 VERL **入口选择桥接**
由第 6 节启动器自动检查；如需自行运行 VERL 入口，参照第 5 节检查接线。

---

## 5. 安装与环境准备

先准备**已经能够运行 VERL Fully Async** 的 Python 环境，包括匹配的 Ray、vLLM
（Ascend 上为 vLLM-Ascend）及对应加速卡运行时。本仓 `pyproject.toml` 不会替你安装这些大型依赖。

```bash
# 在 verl-multi-task 仓库根目录：开发 / 测试环境
python -m pip install -e '.[test]'

# 只安装本仓 Python 包则改用：python -m pip install -e .
```

**只使用第 6 节双任务 E2E 启动器时**，可以直接从源码 checkout 运行：
启动器会配置自身及新建 Ray Worker 的 `PYTHONPATH`，不要求额外 editable 安装。
多节点运行时，各节点仍需能访问同一份 `multi_task_scheduler` 源码。

直接调用 `python -m verl.experimental.fully_async_policy.fully_async_main` 前，
先确认 VERL 已接入 MultiTask 入口：

```bash
python scripts/e2e/ensure_verl_multitask_bridge.py --check-only
```

入口补丁保存在
[`patches/verl-v0.10-fully-async-multitask-entry.patch`](patches/verl-v0.10-fully-async-multitask-entry.patch)；
常规 E2E 无需手动应用，自动检查与安全备份规则见第 6 节。

---

## 6. 快速开始：运行两个真实 VERL 任务

**推荐入口：`scripts/e2e/verify_two_verl_jobs.py`。** 不需要手动创建 Ray Placement Group、
填写 Lease 或分别启动 donor / borrower。先准备能正常运行的 VERL Fully Async 环境，以及
真实的模型目录、训练和验证数据。

在本仓库根目录执行（**替换三处 `/实际...` 路径**）：

```bash
# 已有 /tmp/native_args.txt 时保留原文件；样例默认针对 Ascend NPU。
test -f /tmp/native_args.txt || cp examples/e2e/native_args.txt /tmp/native_args.txt

python scripts/e2e/verify_two_verl_jobs.py \
  --repo . \
  --ray-address auto \
  --start-local-ray \
  --native-args /tmp/native_args.txt \
  --model-path /实际模型目录/Qwen3-0.6B \
  --train-files /实际数据目录/train.parquet \
  --val-files /实际数据目录/test.parquet \
  --scenarios "control_plane exactly_once recovery lifecycle force"
```

- `/tmp/native_args.txt` 每行一个 **Hydra `key=value`**（不是 shell 脚本）。
  样例见 [examples/e2e/native_args.txt](examples/e2e/native_args.txt)；如果文件里已填写真实模型和数据路径，
  可省略上面的 `--model-path`、`--train-files`、`--val-files`。
- `--start-local-ray`：优先连接现有 Ray；没有集群时启动临时单机 Ray，结束后关闭。
  使用持久或多机 Ray 时，指定 `--ray-address <head地址>`，**去掉** `--start-local-ray`。
- 启动器自动完成 MultiTask 入口检查、两个 VERL driver 启动、共享 GS 注册、
  从真实 donor CE / PG 自动发现物理 Lease，以及指定场景的验收。
  它会在必要时备份并修补当前 Python **实际导入**的 VERL 源码入口；
  禁止自动改动时加 `--no-auto-bridge`（要求入口已经接好）。
  源码 `src/` 会自动传给本次 driver 和新建 Ray worker。
- 初始默认每任务各申请 1 张 Trainer 和 1 张独立 Rollout 卡；需要调整时使用
  `--trainer-gpus` / `--rollout-gpus`，确保 Ray 有足够真实 GPU/NPU 资源。

**自己通过原生 VERL 入口运行时**，关键开关是
`multitask.enabled=true` 和
`multitask.runtime.profile=experimental_fully_async_standalone`。
其余运行条件（独立 Rollout、TP/DP/PP=1、参数同步后端等）必须满足
[`runtime_profile.py`](src/multi_task_scheduler/integration/verl/runtime_profile.py)；
上面的 E2E 启动器会统一补齐这些限定参数。未启用 MultiTask 时仍使用原生 VERL。

---

## 7. CUDA / Ascend NPU：只需关注的差异

| 项目 | CUDA | Ascend NPU |
| --- | --- | --- |
| `trainer.device` | `cuda` | `npu`（样例 `native_args.txt` 的默认值） |
| Ray 加速卡资源 | `GPU` | `NPU` |
| Checkpoint backend | 原生 `nccl` | `multitask_hccl` + HCCL |
| Native sleep | 支持条件下 level 2 | 当前 vLLM-Ascend 使用平台安全的 level 1 |

**使用第 6 节的启动器时，后端会按设备自动配置。** NPU 不要手工照抄 CUDA 的
`checkpoint_engine.backend=nccl`；实际需要 `multitask_hccl` 及对应
`custom_backend_module`，启动器已处理。

NPU 环境首次运行失败时，先做不启动训练的预检查：

```bash
VERL_MULTITASK_NPU_MODEL_PATH=/实际模型目录/Qwen3-0.6B \
  python scripts/e2e/diagnose_npu_runtime.py
```

这里检查 vLLM / vLLM-Ascend 版本配套、模型文件结构、`torch.npu` 可用性及
VERL 的 NPU 资源映射；**真正的模型加载和 Ray NPU 调度**仍需第 8 节 smoke
或第 6 节双任务 E2E 验证。若使用了 `torch_npu.contrib.transfer_to_npu`，不要仅用
`torch.cuda.is_available()` 判断设备类型。模型优先使用已验证的 BF16/FP16；
量化模型先单独验证 vLLM-Ascend 加载。设备级专项测试见第 8 节。

---

## 8. 测试：按需要选择

普通代码修改通常先跑 **Unit**；涉及 Ray Actor 时再跑 CPU Ray；
改动原生 VERL 适配时跑 Native：

```bash
python -m pytest -q tests/unit
python -m pytest -q -m ray_integration tests/integration
python -m pytest -q -m native tests/native_unit
```

有真实硬件、模型且需要验证设备级 sleep / 恢复时再跑（**二选一**）：

```bash
# Ascend NPU：先验证 backend，再验证完整 NPU 集成
export VERL_MULTITASK_NPU_MODEL_PATH=/实际模型目录/Qwen3-0.6B
python -m pytest -q -s -m npu_backend_smoke tests/integration/test_native_sleep_npu.py
python -m pytest -q -s -m npu_integration tests/integration/test_native_sleep_npu.py

# CUDA：在 CUDA 机器上运行
VERL_MULTITASK_GPU_MODEL_PATH=/实际模型目录 \
  python -m pytest -q -s -m gpu_integration tests/integration/test_native_sleep_gpu.py
```

这些专项测试与第 6 节的双任务 E2E 互为补充；不需要每次全部运行。

---

## 9. E2E 验收结果与排错

第 6 节的 `--scenarios` 可直接指定范围，无需换一套启动脚本：

| 场景 | 验证内容 |
| --- | --- |
| `control_plane` | GS / Lease / 注册与 ACK 丢失 |
| `exactly_once` | MessageQueue 样本去重与冲突拒绝 |
| `recovery` | 确定性异常与 UNKNOWN 对账 |
| `lifecycle` | DONATE → ADD → REMOVE → RESTORE |
| `force` | DONATE → ADD → FORCE REMOVE → RESTORE |

例如只复测两个失败场景，将第 6 节命令最后一行换成
`--scenarios "control_plane exactly_once"`；只跑生命周期则换成
`--scenarios "lifecycle force"`。

结果写入 `logs/two_real_jobs/<运行时间>/`。查看**最近一次**运行：

```bash
RUN=$(ls -dt logs/two_real_jobs/*/ | head -n 1)
cat "${RUN}orchestration_summary.json"
grep -E 'MULTITASK_E2E_RESULT|FAIL|BLOCKED' "${RUN}e2e.log" | tail -n 50
find "${RUN}scenarios" -name result.json -print
```

`STATE=PASS` 且所选场景全为 `PASS` 才算该轮通过。
退出码：`0=PASS`、`1=FAIL`、`2=BLOCKED`（环境或证据不足）。
场景失败时优先查看对应 `scenarios/<场景>/<运行ID>/result.json`，
再查 `donor.log` / `borrower.log`，不要只根据总表定位根因。

**严格在途 FORCE**：在含有 `force` 的 E2E 命令中额外添加
`--require-inflight-force`，要求与本次 FORCE 操作匹配、且实际中断/续推的正向证明；
默认 `force: PASS` **不等于**已经验证真实在途请求续推。

需要保留训练任务以便进一步排查时，可以加 `--keep-running`，但必须连接
**持久 Ray 集群**且不能同时使用 `--start-local-ray`。

详细的验收断言、异常恢复合同、手动 Lease 和 attach 模式见
[docs/e2e-acceptance.md](docs/e2e-acceptance.md)，避免把 README 变成内部设计手册。

---

## 10. 实现与安全原则

- **最小侵入**：尽量复用 VERL Fully Async、vLLM、Checkpoint Engine 的原生能力。
- **单写者与真实证据**：M/E/R/C/GS 各自维护真值；物理释放、权重版本、请求交接必须有对应 Owner 证明。
- **操作幂等**：ACK 丢失、UNKNOWN 通过同一 `operation_id` 重放并对账，不重复副作用。
- **安全优先**：证据不完整时保持 DRAINING / BLOCKED / QUARANTINED，不提前恢复准入。
- **样本恰好一次提交**：依赖 logical sample ID 与 payload digest 拒绝重复或冲突完成结果。

详细时序、回滚和异常恢复合同参阅
[设计约束](docs/simplified-fusion-contract.md) 和 [E2E 验收规则](docs/e2e-acceptance.md)。

---

## 11. 找代码与测试

| 位置 | 用途 |
| --- | --- |
| [`src/multi_task_scheduler/scheduler/`](src/multi_task_scheduler/scheduler/) | GroupScheduler、Lease 账本和注册 |
| [`src/multi_task_scheduler/orchestration/`](src/multi_task_scheduler/orchestration/) | 操作合同、证据、同步闸门 |
| [`src/multi_task_scheduler/rollout/`](src/multi_task_scheduler/rollout/) | Replica、负载均衡、vLLM 服务 |
| [`src/multi_task_scheduler/checkpoint/`](src/multi_task_scheduler/checkpoint/) | Checkpoint Engine、装参与参数同步 |
| [`src/multi_task_scheduler/integration/verl/`](src/multi_task_scheduler/integration/verl/) | VERL TaskRunner / Trainer / Rollouter 对接 |
| [`tests/`](tests/) | Unit、原生适配、Ray 和 GPU/NPU 集成测试 |
| [`scripts/e2e/`](scripts/e2e/) | 一键双任务 E2E、独立场景与诊断脚本 |

日常操作优先用第 6–9 节的命令，按问题再进入对应源码或
[专项验收文档](docs/e2e-acceptance.md)。

---

## 12. 设计与融合来源

当前维护分支：[`chatgpt/0928-merge-verl-expansion`](https://github.com/lh913137941/verl-multi-task/tree/chatgpt/0928-merge-verl-expansion)。

实现以 [simplified-fusion-contract.md](docs/simplified-fusion-contract.md) 和当前源码为准；
[`verl-expansion` 历史参考](docs/verl-expansion-reference/) 仅用于对照，不作为现行接口或状态机依据。
