# verl-multi-task

基于 **VERL Fully Async + vLLM** 的多 RL 任务共享调度扩展。支持在任务之间借还 GPU/NPU，复用原生训练流程，提供 DONATE、ADD、REMOVE、RESTORE 和有条件的 FORCE REMOVE。

**第一次使用？直接按下面的「快速开始」运行。** 不需要理解 Lease、Runtime Profile 或内部状态机。

## 1. 快速开始：双任务 E2E

前提：已具备能正常运行的 VERL Fully Async、Ray、vLLM（Ascend NPU 使用 vLLM-Ascend）环境，以及真实模型和 Parquet 数据。以下示例针对 **Ascend NPU**，在仓库根目录执行：

```bash
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . \
  --start-local-ray \
  --native-args examples/e2e/native_args.txt \
  --model-path /实际模型路径/Qwen3-0.6B \
  --train-files /实际数据路径/train.parquet \
  --val-files /实际数据路径/test.parquet
```

将三处 `/实际...` 替换为本机路径。已有自己的训练参数文件时，将 `--native-args` 改为该文件路径；它是**每行一个 Hydra `key=value`** 的文本文件，不是 shell 脚本。

启动器会自动完成两个 VERL 任务的启动、共享 GroupScheduler 注册、从真实 Placement Group/NPU 身份生成测试 Lease，以及五项验收；不用手动创建 Ray 资源池或配置 MultiTask 内部 Profile。

- **已有持久 Ray 集群 / 多机**：使用 `--ray-address <Ray-head地址>`，移除 `--start-local-ray`。
- **CUDA**：使用适用于 CUDA 的训练参数（至少 `trainer.device=cuda`），不能直接沿用 NPU 样例；启动器按设备选择 NCCL/HCCL 后端。
- **VERL 入口桥接**：脚本默认会检查并在必要时**备份、修改当前 Python 实际导入的 VERL 源码入口**；不允许自动修改时加 `--no-auto-bridge`，但需事先完成接入。
- **资源需求**：默认每个任务申请 1 张 Trainer 卡、1 张独立 Rollout 卡。要调整使用 `--trainer-gpus` / `--rollout-gpus`。

## 2. 按需验收

默认运行全部五项。只想测试某一部分，在上一条命令末尾加 `--scenarios`：

| 场景 | 检查什么 |
| --- | --- |
| `control_plane` | 调度注册、Lease 和控制面异常 |
| `exactly_once` | 完成样本去重、冲突拒绝 |
| `recovery` | 异常恢复和操作对账（故障注入） |
| `lifecycle` | DONATE → ADD → REMOVE → RESTORE |
| `force` | DONATE → ADD → FORCE REMOVE → RESTORE |

例如，只复测控制面和样本提交，在第 1 节命令最后加上
`--scenarios "control_plane exactly_once"`。

如果要严格验证 **真实在途请求被中断并成功续推**，还需在包含 `force` 的命令中增加 `--require-inflight-force`。普通的 `force: PASS` 不代表一定命中了在途请求。

## 3. 结果与排错

完成后查看 `logs/two_real_jobs/<运行时间>/orchestration_summary.json`。**只有 `STATE=PASS` 且所选场景全部 PASS，才算本轮通过。**

| 退出码 | 含义 |
| --- | --- |
| `0` | PASS |
| `1` | FAIL |
| `2` | BLOCKED：环境或证据不足，不能视为通过 |

快速查看最近一次运行和各场景的失败原因：

```bash
RUN=$(ls -dt logs/two_real_jobs/*/ | head -n 1)
cat "${RUN}orchestration_summary.json"
find "${RUN}scenarios" -name result.json -print
grep -E 'FAIL|BLOCKED|Traceback|Error' "${RUN}e2e.log" | tail -n 30
```

定位时优先看失败场景的 `result.json`，再看 `donor.log`、`borrower.log`。如果是 NPU 环境或模型加载问题，可以先运行：

```bash
VERL_MULTITASK_NPU_MODEL_PATH=/实际模型路径/Qwen3-0.6B \
  python scripts/e2e/diagnose_npu_runtime.py
```

该脚本只检查环境、模型文件结构和设备信息，不代替真实模型加载或硬件 E2E。

## 4. 支持范围

| 支持 | 当前限制 |
| --- | --- |
| VERL experimental Fully Async + 独立 vLLM Rollout | 不支持 Hybrid、PD/disaggregation |
| CUDA GPU / Ascend NPU | 不支持 CPU-only 借卡 |
| 单节点 Rollout，TP/DP/PP = 1/1/1 | 暂不支持多节点 Rollout、多卡 TP |
| 整张物理加速卡的借还 | 不支持把 Ray 的 0.5 资源份额当半张物理卡借出 |
| ADD / DONATE / REMOVE / RESTORE | FORCE REMOVE 仅限可验证续推的 borrowed replica |

DONATE 会释放原生副本的**物理卡使用权**，但保留睡眠中的原 Runtime，以便 RESTORE；REMOVE 则销毁 borrowed Runtime。物理释放、参数版本或请求续推无法证明时，操作保持安全封锁，不会虚报成功。

当前真实 Ascend NPU 环境已验证双任务共享调度注册、自动 Lease 识别及部分生命周期场景；**全量 E2E 与严格在途 FORCE 以每轮实际结果为准**。软件测试状态见 [GitHub Actions](https://github.com/lh913137941/verl-multi-task/actions)。

## 5. 开发与手动接入

只运行第 1 节启动器时，可直接使用源码 checkout（启动器负责传递 `PYTHONPATH`）。若需开发或跑单测，可安装：

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/unit
```

按修改范围选择进一步的测试：

```bash
python -m pytest -q -m ray_integration tests/integration
python -m pytest -q -m native tests/native_unit
```

设备级测试在对应 GPU/NPU 实机运行，见 [验收说明](docs/e2e-acceptance.md)。

如果**不使用一键 E2E**，而是直接调用 VERL 原生 Fully Async 入口，先检查入口桥接：

```bash
python scripts/e2e/ensure_verl_multitask_bridge.py --check-only
```

手动接入时只需要显式启用 `multitask.enabled=true`，并满足当前运行条件；**无需手写 `experimental_fully_async_standalone`**，当前代码会自动选择唯一支持的运行配置。详细校验条件见 [runtime_profile.py](src/multi_task_scheduler/integration/verl/runtime_profile.py)。

## 6. 代码与设计文档

- [调度 / Lease](src/multi_task_scheduler/scheduler/) · [操作合同](src/multi_task_scheduler/orchestration/) · [Rollout / LB](src/multi_task_scheduler/rollout/) · [参数同步](src/multi_task_scheduler/checkpoint/) · [VERL 适配](src/multi_task_scheduler/integration/verl/)
- [E2E 脚本](scripts/e2e/) · [测试](tests/)
- [设计与安全约束](docs/simplified-fusion-contract.md) · [完整 E2E 验收规则](docs/e2e-acceptance.md) · [历史参考](docs/verl-expansion-reference/)

README 只保留使用流程。组件职责、状态机、操作幂等、异常恢复和完整验收证据，以源码及上述专项文档为准。
