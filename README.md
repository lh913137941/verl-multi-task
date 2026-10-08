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

## 6. 启用 MultiTask

最小配置方向：

```text
multitask.enabled=true
multitask.runtime.profile=experimental_fully_async_standalone

actor_rollout_ref.hybrid_engine=false
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.data_parallel_size=1
actor_rollout_ref.rollout.pipeline_model_parallel_size=1
actor_rollout_ref.rollout.enable_sleep_mode=true
actor_rollout_ref.rollout.free_cache_engine=true
actor_rollout_ref.rollout.calculate_log_probs=true

actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true

async_training.use_trainer_do_validate=false
async_training.use_dynamic_resource_scheduling=false

data.train_batch_size=0
data.gen_batch_size=1
```

`trainer.device` 不需要为 MultiTask 手工写死。VERL 的 `auto_set_device()` 会在
profile 选择前把它归一化为当前平台的 `cuda` 或 `npu`；MultiTask profile
接受这两种 accelerator device，并继续拒绝 CPU-only 生命周期。

`multitask.enabled=false` 或未配置时继续使用原生 Fully Async TaskRunner。

显式启用后，如果 profile、拓扑、依赖或 runtime capability 不满足要求，会直接失败，
不会静默退回另一套 MultiTask 行为。

---

## 7. CUDA 与 Ascend NPU

### CUDA

CUDA 路径要求 whole-GPU DONATE 能进入 level-2 sleep。MTP rollout、未 merge 的 LoRA
等只能安全使用更浅 sleep level 的配置会 fail-closed。

真实 CUDA acceptance：

```bash
export VERL_MULTITASK_GPU_MODEL_PATH=/path/to/local/model

python -m pytest -q -s   -m gpu_integration   tests/integration/test_native_sleep_gpu.py
```

覆盖：

- native sleep -> 同物理 GPU borrower -> REMOVE；
- FORCE targeted abort + continuation；
- current-Vpub RESTORE。

### Ascend NPU

NPU 路径使用：

```text
VERL platform: huawei
device:        npu
Ray resource:  NPU
communication: HCCL
rollout:       vLLM-Ascend
```

VERL / vLLM-Ascend 当前把 NPU 的可用 sleep primitive 解析为 level 1；
MultiTask NPU 路径因此要求 platform-safe level-1 sleep，而不会伪装成 CUDA level 2。

为了保持首版 Lease/OperationEvidence 结构不膨胀，历史字段名
`gpu_uuid` 暂时同时承载物理 accelerator identity：

```text
CUDA: real GPU UUID
NPU:  NPU:<node_id>:<ray_accelerator_id>
```

### 关于 `transfer_to_npu`

Ascend 环境常启用：

```text
torch_npu.contrib.transfer_to_npu
```

它会全局 monkey-patch `torch.cuda.*`，因此：

**不要用 `torch.cuda.is_available()` / `torch.cuda.device_count()` 判断当前真实设备。**

NPU acceptance 只信：

```text
VERL_PLATFORM=huawei
verl.utils.device.get_device_name() == "npu"
verl.utils.device.get_resource_name() == "NPU"
torch.npu
Ray cluster resource "NPU"
```

NPU 测试会为 driver 和 Ray runtime 显式固定 `VERL_PLATFORM=huawei`；
CUDA acceptance 若检测到 `transfer_to_npu` 已加载则直接 skip，避免假 CUDA 绿灯。

真实 NPU acceptance 建议使用 plain BF16/FP16 模型；C8/ModelSlim/其它量化模型应先单独验证
vLLM-Ascend loader 兼容性，不要把量化 backend 失败混入生命周期验收。

先做静态 preflight，再验证 vanilla VERL + vLLM-Ascend 基线，最后跑 MultiTask lifecycle：

```bash
export VERL_MULTITASK_NPU_MODEL_PATH=/path/to/local/model

# /tmp 空间不足或使用率 >=95% 时，把 Ray session/object-spill 临时目录放到大盘。
export VERL_MULTITASK_RAY_TMPDIR=/path/to/large/local/filesystem/ray_tmp

# 0) 不启动 Ray/vLLM；检查版本配对、Git HEAD、NPU、模型量化判定和临时盘。
python scripts/e2e/diagnose_npu_runtime.py

# 1) 不经过 MultiTask subclass，分层验证 backend：
#    A. direct vLLM-Ascend default executor（单卡默认 uni）
#    B. direct vLLM-Ascend distributed_executor_backend=mp
#    C. 原生 VERL vLLMReplica（worker extension 是 VERL server contract 的一部分）
python -m pytest -q -s \
  -m npu_backend_smoke \
  tests/integration/test_native_sleep_npu.py

# 2) backend smoke 通过后再跑完整生命周期。
python -m pytest -q -s \
  -m npu_integration \
  tests/integration/test_native_sleep_npu.py
```

当前 VERL NPU 安装脚本使用同版本 lane 的 vLLM / vLLM-Ascend；例如当前脚本对应
`vLLM v0.23.0` + `vLLM-Ascend releases/v0.23.0`。不要把任意 vLLM source HEAD
与另一条 vLLM-Ascend release/main 混用。诊断脚本会从本地 VERL checkout 的
`scripts/install_vllm_mcore_npu.sh` 读取期望 pair 并与实际环境对照。

NPU acceptance 在启动 Ray 前会检查临时文件系统的剩余空间；空间不足时会
直接报告环境阻断，避免等到 vLLM EngineCore 初始化后才出现模糊的 WorkerProc 错误。

三条设备级验收：

```text
NPU DONATE:
native sleep -> same-slot borrowed runtime -> real generation -> REMOVE

NPU FORCE:
targeted abort -> continuation proof -> alternate replica completes request

NPU RESTORE:
sleep -> lend slot -> remove borrower -> mutate trainer Vpub
-> HCCL checkpoint transfer -> version check -> final wake -> generation
```

注意：VERL 的 Ascend Checkpoint Engine 实现仍通过现有
`checkpoint_engine.backend="nccl"` 配置入口选择，runtime 在 NPU 平台落到 HCCL；
不要仅根据配置字符串判断底层通信设备。

---

## 8. 分层测试

### Unit

```bash
python -m pytest -q tests/unit
```

### CPU Ray integration

```bash
python -m pytest -q   -m ray_integration   tests/integration
```

这层验证真实 Ray Actor / GroupScheduler 控制面，不需要 GPU/NPU。

### Native VERL adapter

```bash
python -m pytest -q   -m native   tests/native_unit
```

这层检查 MultiTask subclass 与真实 VERL 类的继承/方法合同。

### CUDA acceptance

```bash
export VERL_MULTITASK_GPU_MODEL_PATH=/path/to/model

python -m pytest -q -s   -m gpu_integration   tests/integration/test_native_sleep_gpu.py
```

### Ascend NPU acceptance

```bash
export VERL_MULTITASK_NPU_MODEL_PATH=/path/to/model
export VERL_MULTITASK_RAY_TMPDIR=/path/to/large/local/filesystem/ray_tmp

python -m pytest -q -s -m npu_integration tests/integration/test_native_sleep_npu.py
```

pytest marker 定义见 `pyproject.toml`。

---

## 9. E2E 与异常恢复

完整脚本位于：

```text
scripts/e2e/
```

快速验证控制面、恢复和 Exactly-once：

```bash
bash scripts/e2e/verify_cluster.sh   --quick   --lease /tmp/lease.json   --attach
```

完整集群生命周期：

```bash
bash scripts/e2e/verify_cluster.sh   --launcher /path/to/test_launcher.sh   --lease /tmp/lease.json
```

已有运行任务时：

```bash
python scripts/e2e/list_tasks.py

bash scripts/e2e/verify_cluster.sh   --attach   --lease /tmp/lease.json   --donor-session <donor-task-session>   --borrower-session <borrower-task-session>
```

Lease fixture：

```text
examples/e2e/lease.example.json
```

结果严格区分：

```text
0 = PASS
1 = FAIL
2 = BLOCKED
```

缺少硬件、拓扑或真实证据时返回 BLOCKED，而不是伪报 PASS。

完整说明：
[docs/e2e-acceptance.md](docs/e2e-acceptance.md)。


### 两个真实 Fully Async 任务的共享 GS 一键验收

先确保目标 VERL 已应用 `patches/verl-v0.10-fully-async-multitask-entry.patch`（或等价适配），
所有 Ray 节点都能导入本包，并准备可运行的原生 Fully Async 模型、数据、训练参数。
将这些 Hydra overrides 逐行保存为 `/tmp/native_args.txt`，不要只使用占位模型路径。

若脚本报 `VERL Fully Async entry has no MultiTask bridge`，先检查当前解释器
**实际导入**的入口文件（不是猜测相邻仓库，也不要直接删除前置检查）：

```bash
python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("verl.experimental.fully_async_policy.fully_async_main")
print("VERL entry:", spec.origin if spec else "<not installed>")
PY
```

在与该入口对应的、可写的 VERL **源码 checkout** 根目录执行补丁检查和安装，
路径按实际环境替换。修补已生效的入口时不要重复 `git apply`：

```bash
MT_ROOT=/path/to/verl-multi-task
VERL_ROOT=/path/to/verl
git -C "$VERL_ROOT" apply --check "$MT_ROOT/patches/verl-v0.10-fully-async-multitask-entry.patch"
git -C "$VERL_ROOT" apply "$MT_ROOT/patches/verl-v0.10-fully-async-multitask-entry.patch"
# 如果 Python 实际导入的是另一份已安装的 VERL，让当前环境使用修补后的源码：
python -m pip install -e "$VERL_ROOT" --no-deps
python - <<'PY'
import importlib
m = importlib.import_module("verl.experimental.fully_async_policy.fully_async_main")
print("VERL entry:", m.__file__)
assert callable(getattr(m, "_resolve_task_runner_class", None)), "MultiTask bridge not active"
PY
```

`git apply --check` 未通过时，先检查现有修改和 VERL 版本差异，
不要使用 `--reject` 或 `--3way` 强行套用到不兼容的源码。
补丁包含 YAML 配置的 `multitask` 键和 Python 入口选择逻辑，两部分缺一不可。

```bash
export RAY_ADDRESS=auto  # 或实际 Ray head 地址
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . \
  --ray-address "$RAY_ADDRESS" \
  --namespace multitask-jobs \
  --native-args /tmp/native_args.txt
```

脚本启动两个独立的原生 VERL Fully Async driver，强制启用 MultiTask 并使用相同的
Ray 地址和 job namespace；等待两个新的 task_session 附着到同一个 named detached GS，
然后通过 donor TaskRunner → Rollouter → CE Worker 只读采集真实 GPU/NPU 身份，
结合原生 Replica 所持有的 named Placement Group 验证 bundle、node、设备及 namespace。
验证通过后自动生成 `logs/two_real_jobs/<运行时间>/auto_lease.json` 并交给已有 E2E
驱动使用。默认选择 donor native rank 0；可用 `--donor-replica-rank` 明确指定。
若事实不完整、PG 无法验证或任务未就绪，直接 BLOCKED，不猜测 PG/卡号。
已有的 `--lease /tmp/lease.json --interactive-lease` 仍支持手工诊断。

这个自动 Lease 是 **E2E 编排产生的真实资源快照**，不是业务侧需要维护的配置项；
生产调度中仍由 GS 根据 idle report、容量、安全边界及资源归属选择 donor/borrower，
使用内部 `open_lease` 记账，不能仅凭自动发现便视为资源已获借出许可。
为避免两个真实任务在同一 Ray namespace 内产生相同 native rollout PG/CE/server 名，
MultiTask 对原生 Replica 的资源名称增加 task_session 作用域。

随后复用现有 `run_all.sh` 验证 control_plane、exactly_once、recovery、lifecycle 和 force。
可用 `--scenarios "lifecycle force"` 缩小范围；`--keep-running` 可保留两个 driver。
结果和训练日志保存在 `logs/two_real_jobs/<运行时间>/`；退出码为
`0=PASS / 1=FAIL / 2=BLOCKED`。真实 GPU/NPU 运行结果需在实际集群上判定。

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
