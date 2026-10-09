# Unit 测试说明

本目录验证 **MultiTask 与 VERL 的接线、资源生命周期、请求/参数一致性以及失败恢复**。
测试优先使用轻量 Mock、AST 类/方法隔离和临时文件，使大多数回归可以在 CI 的
**CPU-only Python 环境**中运行；不需要启动两个真实 RL 任务。

> 使用方式：**改哪个组件，先跑对应文件，再跑全量 Unit**。
> Unit 能验证逻辑与 fail-closed 边界，但不能代替真实 Ray、VERL、vLLM 和 Ascend NPU/CUDA 的 E2E。

## 1. 如何运行

在仓库根目录执行：

```bash
# 首次准备测试依赖（原生 VERL、Ray、vLLM 不会由此自动安装齐全）
python -m pip install -e '.[test]'

# 全部单测；CI 也使用这个入口
python -m pytest -q tests/unit

# 只测所修改的组件（示例）
python -m pytest -q tests/unit/test_rollouter_wiring.py
python -m pytest -q tests/unit/test_checkpoint_wiring.py

# 精确重跑一个失败用例
python -m pytest -q tests/unit/test_taskrunner_wiring.py::test_taskrunner_replays_identical_lease_evidence_after_ack_loss
```

CI 配置：[unit.yml](../../.github/workflows/unit.yml)。
带 `@pytest.mark.parametrize` 的函数会展开为多个 pytest 用例，因此 CI 的通过数量
不等于源码里的 `test_*` 函数数，也不建议把固定数量作为验收条件。

## 2. 测试文件与覆盖范围

当前组织方式是 **15 个测试文件 + 1 个辅助文件**。

| 文件 | 核心测试内容及重点失败边界 |
| --- | --- |
| [`test_orchestration.py`](test_orchestration.py) | `OperationCommand/Record/Evidence`、Lease 物理归属、OperationJournal 幂等/UNKNOWN、ReplicaSyncGate 互斥与 fencing |
| [`test_scheduler_wiring.py`](test_scheduler_wiring.py) | GS 操作暂存与提交、DONATE donor 匹配、Lease 借还和恢复归属；TaskRunner ACK 不确定或报告过期时不错误放行 |
| [`test_taskrunner_wiring.py`](test_taskrunner_wiring.py) | ADD/DONATE/REMOVE/RESTORE/FORCE 命令编排、GS attach ACK、会话隔离和 Trainer PG 命名；重复命令、超时、UNKNOWN 对账 |
| [`test_trainer_wiring.py`](test_trainer_wiring.py) | Trainer 的参数同步闸门、ADD Bootstrap、DONATE/REMOVE、RESTORE 最新权重；服务发布失败、补偿及 QUARANTINED |
| [`test_rollouter_wiring.py`](test_rollouter_wiring.py) | 生成容量、闲置上报、ADD/REMOVE/RESTORE 发布与 drain、FORCE abort/continuation、RewardLoop 名称隔离；ACK 丢失和同操作重试 |
| [`test_replica_wiring.py`](test_replica_wiring.py) | Native/Borrowed Replica 身份及状态、物理 Claim/Placement Group、Worker 创建和清理；缺失真实 DEAD/释放证据时拒绝提交 |
| [`test_lb_wiring.py`](test_lb_wiring.py) | LoadBalancer 路由/准入、request attempt 状态、continuation 证明和历史 GC；零可用 Server 时仅对合法切换等待、超时不误成功 |
| [`test_server_wiring.py`](test_server_wiring.py) | vLLM HTTP Server 的健康、drain/shutdown、sleep/wake 阶段与准入屏障；EngineCore 输出处理、level 1/2 和失败回滚 |
| [`test_checkpoint_wiring.py`](test_checkpoint_wiring.py) | Checkpoint Engine 成员关系、Bootstrap 版本、参数 Manifest、Native RESTORE、任务隔离的 Server 名称；错误版本/不完整证明拒绝提交 |
| [`test_hccl_checkpoint_engine.py`](test_hccl_checkpoint_engine.py) | NPU `multitask_hccl` 兼容性：Communicator 销毁与重建、资源释放、Manifest 清理；使用 CPU 替身，不实际创建 HCCL 通信组 |
| [`test_message_queue_exactly_once.py`](test_message_queue_exactly_once.py) | logical sample ID + payload 的重复提交、冲突拒绝、磁盘账本、模糊 ACK 与原生丢弃语义 |
| [`test_runtime_profile.py`](test_runtime_profile.py) | 唯一支持运行配置的校验；CUDA/NPU backend、拓扑、sleep、LoRA/MTP 和非法组合 fail-closed |
| [`test_restore_vpub_mutation.py`](test_restore_vpub_mutation.py) | NPU RESTORE 的 FSDP1 live 参数写回、tied embedding 和导出 Vpub 校验；防止把陈旧参数视作已同步 |
| [`test_npu_memory_probe.py`](test_npu_memory_probe.py) | NPU 显存预检查、npu-smi 解析/回退和超时；不能因为探测失败推断设备可用 |
| [`test_e2e_scripts.py`](test_e2e_scripts.py) | 双任务 E2E 启动参数、VERL 入口桥接、Ray 自动连接、自动 Lease、脚本分发及严格 FORCE 回执校验；仅测试驱动逻辑，不启动真实双任务 |
| [`_wiring_support.py`](_wiring_support.py) | **非测试文件**：共享 `isolated()` AST 提取、Ray/VERL Mock、组件构造器和物理 Claim 工厂；不放场景断言 |

### 组件与主要流程对应

- **DONATE / REMOVE**：TaskRunner、Trainer、Rollouter、Replica、LB、Scheduler、Orchestration。
- **ADD / RESTORE / 参数同步**：TaskRunner、Trainer、Rollouter、Checkpoint、Server、Replica。
- **FORCE REMOVE / 在途续推**：Rollouter、LB、TaskRunner、E2E 脚本校验；不能仅由配置判定续推成功。
- **Exactly Once / 异常恢复**：MessageQueue、Orchestration、Scheduler、TaskRunner，以及对应的 E2E 故障注入。

## 3. 与 E2E 的关系

`scripts/e2e/validate_recovery_faults.sh` 会从本目录选取特定故障注入用例，
并运行 `test_orchestration.py` 与 `test_message_queue_exactly_once.py`。
**移动、重命名或删除这些测试时，必须同步更新该脚本中的 pytest 选择器**；
`test_e2e_scripts.py` 会检查九个精确选择器仍存在，且不会重复选择编排测试套件。

完整双任务生命周期/ FORCE 验收仍通过
[`verify_two_verl_jobs.py`](../../scripts/e2e/verify_two_verl_jobs.py) 执行，
例如在 `examples/e2e/native_args.txt` 已包含实际模型与数据路径的机器上：

```bash
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . \
  --ray-address auto \
  --start-local-ray \
  --native-args examples/e2e/native_args.txt \
  --scenarios "lifecycle force"
```

此命令需要真实运行时、模型、数据与 GPU/NPU 资源。默认 `force: PASS` 不等同于
验证了真实在途请求续推；如需该项证明，再加 `--require-inflight-force`。
详细的判据见 [E2E 验收说明](../../docs/e2e-acceptance.md)。

## 4. 维护约定

1. **一个组件优先维护一个测试文件**；专项设备后端、Exactly Once 等不同依赖/契约可以独立保留。
2. 用 `_wiring_support.py` 复用 AST 装载、Mock、Claim/Lease 工厂；每次构造返回新的可变对象，避免跨测试污染。
3. 保留关键流程的**成功、明确失败、未知结果/ACK 丢失、幂等重放及恢复**边界；不要因为断言对象相近就合并不同故障时序。
4. Mock/AST 单测主要证明本地合同，不应宣称已覆盖真实通信组、模型加载、Ray Actor 竞争或硬件行为。修改相应接线后还需运行 [native_unit](../native_unit/)、[integration](../integration/) 或真实 E2E。
