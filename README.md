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

首版 profile：

```text
experimental_fully_async_standalone
```

当前边界：

| 项目 | 当前状态 |
| --- | --- |
| VERL execution model | experimental Fully Async |
| rollout deployment | STANDALONE |
| inference backend | vLLM / vLLM-Ascend |
| PD disaggregation | 不支持 |
| rollout topology | 单节点 |
| TP / DP / PP | 当前验证边界为 1 / 1 / 1 |
| resource lending | 整张物理 accelerator 借还 |
| CUDA | level-2 sleep；真实 GPU acceptance 单独运行 |
| Ascend NPU | VERL NPU platform + Ray `NPU`；vLLM-Ascend level-1 sleep |
| dynamic checkpoint membership | 复用 VERL Checkpoint Engine |
| FORCE REMOVE | borrowed-only；要求 partial rollout + continuation proof |

当前实现遵循 **fail-closed**：无法证明 placement、参数版本、请求交接或资源释放时，
不会合成成功证据，而是保留 DRAINING / BLOCKED / QUARANTINED 等可对账状态。

### 当前验证状态

截至当前分支最近一轮验证：

```text
unit:             336 passed
ray_integration:    6 passed
native_unit:        8 passed
```

CUDA / NPU 设备级测试需要在对应真实硬件、vLLM runtime 和本地模型环境中单独执行；
不能用 unit/mock 结果替代设备级验收。

---

## 2. 生命周期

核心资源流程：

```text
DONATE
NATIVE/ACTIVE
  -> DRAINING
  -> request drain
  -> E/R/C exit
  -> runtime sleep
  -> RELEASED
  -> Lease handoff-ready

ADD
Lease
  -> BORROWED/CREATING
  -> hidden runtime create
  -> current-Vpub bootstrap
  -> WEIGHT_READY
  -> E commit
  -> R/C/M publish
  -> ACTIVE

REMOVE
BORROWED/ACTIVE
  -> DRAINING
  -> request drain
  -> destroy runtime
  -> RELEASED
  -> physical slot returned

RESTORE
NATIVE/DORMANT
  -> reserve original slot
  -> wake/bootstrap current Vpub
  -> KV restore
  -> version check
  -> E commit
  -> final wake
  -> R/C/M publish
  -> ACTIVE
```

FORCE REMOVE 不绕过安全条件：

```text
BORROWED only
  + partial rollout capable
  + alternate active server
  + targeted abort
  + continuation proof
  + request state settled
  -> EXIT_READY
```

没有 continuation proof 的在途 request 不能直接被视为安全完成。

---

## 3. Owner 真值

实现只保留必要的 owner 状态，不引入重复的大一统状态对象：

| Owner | 真值 |
| --- | --- |
| M / LLMServerManager | `replica_state[ReplicaKey]`、`replica_kind[ReplicaKey]` |
| E / CheckpointEngineManager | effective replicas、pending bootstrap、参数版本事实 |
| R / LoadBalancer | route、inflight、request attempt state、continuation handoff |
| C / Rollouter | committed capacity / production window |
| GS | Lease 账本、TaskRunner 注册、operation 转发 |
| G / Trainer gate | 参数同步和生命周期变更串行边界 |

生命周期状态：

```text
CREATING
ACTIVE
DRAINING
DORMANT
RELEASED
QUARANTINED
```

公共控制结构保持精简：

```text
ReplicaKey
OperationCommand
OperationRecord
OperationEvidence
Lease
```

---

## 4. 创建链

MultiTask 只替换必要 subclass，VERL 主训练流程继续由原生入口驱动：

```text
VERL Fully Async main
  -> MultiTaskFullyAsyncTaskRunner
     -> MultiTaskFullyAsyncTrainer
        -> MultiTaskCheckpointEngineManager
     -> MultiTaskFullyAsyncRollouter
        -> MultiTaskLLMServerManager
           -> MultiTaskGlobalRequestLoadBalancer
           -> MultiTaskvLLMReplica
              -> MultiTaskCheckpointEngineWorker
              -> MultiTaskvLLMHttpServer
```

原生业务主循环尽量保持不变。例如 Rollouter 的 `fit()` 直接继承
VERL `FullyAsyncRollouter.fit`；MultiTask 逻辑通过已有初始化/生命周期 hook 接入。

本包本身不 monkey-patch VERL 原生类，也不提供另一份训练入口。

---

## 5. 安装

先准备一个能够正常运行目标 VERL 版本的环境，再安装本仓：

```bash
python -m pip install -e '.[test]'
```

或者只安装 runtime 包：

```bash
python -m pip install -e .
```

本仓使用 `src/` layout。

Driver 和所有 Ray worker 节点必须能够 import 同一版本的
`multi_task_scheduler`。测试环境中的 `tests/conftest.py` 会把 `src` 同步到
`PYTHONPATH`，保证本地 Ray worker 也能导入源码 checkout。

仍然使用 VERL 原生 Fully Async 入口：

```bash
python -m verl.experimental.fully_async_policy.fully_async_main
```

VERL 侧需要具备 MultiTask 选择接线；对应补丁保存在：

```text
patches/verl-v0.10-fully-async-multitask-entry.patch
```

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

## 10. 关键实现原则

- **最小侵入 VERL**：保留原生 Fully Async 主流程和业务方法；
- **单写者 owner**：M / E / R / C / GS 各自只写自己的真值；
- **证据驱动**：生命周期推进依赖 `OperationEvidence`，不靠推测；
- **Same-operation replay**：ACK loss / UNKNOWN 通过原 operation 对账，不重复副作用；
- **资源释放必须可证明**：placement、actor death、sleep、request settlement 都需真实 owner fact；
- **参数版本必须可证明**：RESTORE/ADD 目标在发布前确认 current Vpub；
- **Exactly-once**：以 logical sample key + payload digest 防止重复完成样本；
- **异常优先可恢复**：不能证明完成时保持 fenced/blocked/quarantined，而不是提前成功。

---

## 11. 目录导航

```text
src/multi_task_scheduler/
  scheduler/                         # GroupScheduler / discovery
  orchestration/                     # contracts / operation evidence
  rollout/                           # replica / LB / HTTP server
  checkpoint/                        # CE manager / CE worker
  integration/verl/                  # VERL Fully Async adapters

tests/
  unit/                              # dependency-light regression
  native_unit/                       # real VERL adapter contracts
  integration/
    test_group_scheduler.py          # CPU Ray integration
    test_native_sleep_gpu.py         # CUDA acceptance
    test_native_sleep_npu.py         # Ascend NPU acceptance

scripts/e2e/                          # lifecycle / recovery / exactly-once
docs/
  simplified-fusion-contract.md      # current control contract
  e2e-acceptance.md                  # acceptance semantics
  verl-expansion-reference/          # imported historical/reference material
```

---

## 12. 设计与融合来源

当前分支：

```text
chatgpt/0928-merge-verl-expansion
```

它在 simplified-fusion 合同上吸收了 `verl_expansion` 中与当前 owner/Lease/evidence
模型兼容的实现和验收资产，但没有直接恢复旧的重复状态结构或旧 wire contract。

历史参考保存在：

[docs/verl-expansion-reference/](docs/verl-expansion-reference/)

如果历史参考与当前代码或
[docs/simplified-fusion-contract.md](docs/simplified-fusion-contract.md)
冲突，以当前合同和源码为准。
