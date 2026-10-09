# MultiTask 关键流程 E2E / 异常恢复验收

这套验收脚本借鉴 `verl_expansion:main` 的做法：每个场景有独立日志、结构化结果、明确退出码，
且缺少真实环境或证据时返回 `BLOCKED`，不把“不具备验证条件”伪报为成功。

它没有直接复制 `D3_test.sh` / `D4_test.sh` / `D0_D4_*test.sh`，原因是这些脚本绑定
verl_expansion 的旧 D2/D3/D4 controller、rank/dict receipt、Ascend/HCCL fixture 和
`multitask.d3_bootstrap_test` / `multitask.d4_runtime_test` 测试钩子。当前分支的生产合同已经是
`ReplicaKey + Lease + OperationEvidence + ReplicaSyncGate + UNKNOWN reconciliation`；
照搬旧脚本会验证错误的入口和状态语义。

## 验收层级

| 脚本 | 证据层级 | 主要验证 |
| --- | --- | --- |
| `validate_lifecycle_cycle.sh` | 真实 VERL/Ray/GPU runtime | `DONATE -> ADD -> REMOVE -> RESTORE`；每步必须取得设计规定的终态 evidence，并验证 exact-command replay 幂等 |
| `validate_force_remove.sh` | 真实 VERL/Ray/vLLM runtime | `DONATE -> ADD -> FORCE REMOVE -> RESTORE`；要求 partial rollout 与另一个 active server。可选要求真实在途 continuation proof |
| `validate_control_plane.sh` | 真实 Ray Actor/RPC，CPU 即可 | GS fail-closed、过期 Lease、claim 冲突、stale idle report、RELEASED evidence 幂等、GS->TR ACK 丢失后 staging 保留与 same-op replay |
| `validate_exactly_once.sh` | 真实 Ray MessageQueue Actor | 同 logical sample + 同 payload 只提交一次；同 ID 不同 digest 拒绝；队列 overflow 保留 dropped_oldest evidence |
| `validate_recovery_faults.sh` | 确定性 fault-injection | advance_lease ACK loss、UNKNOWN replay、G BLOCKED reconcile、RESTORE rollback/quarantine、natural drain timeout same-op resume、FORCE partial handoff/abort ACK loss |
| `run_all.sh` | 汇总 | 串行运行以上场景，输出 PASS/FAIL/BLOCKED 与 `summary.json` |

`validate_recovery_faults.sh` 是故障注入验收，不冒充真实 GPU E2E。ACK 丢失、R 提交不确定、
abort reply 丢失等故障需要命中非常精确的 owner-commit 边界；在没有生产 fault hook 的前提下，
用确定性注入比随机 kill actor 更能证明状态机恢复语义。

## 退出码

- `0`：PASS，脚本要求的证据全部成立。
- `1`：FAIL，代码运行了，但得到错误结果或证据不完整。
- `2`：BLOCKED，环境、拓扑或依赖不足，不能据此声称能力已通过。

日志默认写到 `logs/multitask_e2e/`。

## 集群一行验收

实际集群上不需要逐个调用上面的脚本。推荐直接使用总入口。

启动一个测试训练并做完整验收：

```bash
bash scripts/e2e/verify_cluster.sh --launcher /path/to/multitask_test_launcher.sh --lease /tmp/lease.json
```

如果 donor / borrower 两个测试 job 已经在同一 Ray 集群里运行，先列出已注册 TaskRunner：

```bash
python scripts/e2e/list_tasks.py
```

然后指定两个不同的 task_session：

```bash
bash scripts/e2e/verify_cluster.sh --attach --lease /tmp/lease.json \
  --donor-session <donor-task-session> \
  --borrower-session <borrower-task-session>
```

只想先验证不依赖真实生命周期拓扑的控制面/异常恢复/Exactly-once：

```bash
bash scripts/e2e/verify_cluster.sh --quick --lease /tmp/lease.json --attach
```

完整模式默认运行：

```text
control_plane -> exactly_once -> recovery -> lifecycle -> force
```

最终只需要看命令退出码和脚本打印的 `summary.json` 路径：

- `0` = 全部要求通过；
- `1` = 至少一个能力验证失败；
- `2` = 环境/拓扑不足，存在 BLOCKED，不能宣称完整验收通过。

若希望阶段性执行时允许 BLOCKED，但仍保留结果记录，可加 `--allow-blocked`。

## 先跑不需要 GPU 的恢复合同

```bash
bash scripts/e2e/validate_recovery_faults.sh
bash scripts/e2e/validate_control_plane.sh
```

`validate_control_plane.sh` 需要 Ray，但不需要完整 VERL/vLLM/GPU。它创建临时 GS 与真实 Ray
TaskRunner substitute，专门验证 GS 账本和 RPC 不确定性；临时 Actor 在场景结束后销毁。

Exactly-once 需要能导入当前 VERL 和 Ray：

```bash
bash scripts/e2e/validate_exactly_once.sh
```

## 真实生命周期 E2E

真实生命周期需要一份物理 Lease fixture。复制：

```bash
cp examples/e2e/lease.example.json /tmp/lease.json
```

然后把 `pg_id`、`node_id`、`gpu_uuid`、`bundle_index` 和 donor rank 改成真实环境值。这里的 `pg_id` 是 Ray placement group 的十六进制 ID（`pg.id.hex()`），不是 placement group name。
`donor_task_id` 可以保留 `__TASK_SESSION__`，driver 会替换为本次附着的 TaskRunner session。
当前融合分支的 borrowed runtime 明确限制 `TP=DP=PP=1` 且 `len(claims)=1`，因此真实生命周期 fixture 只能选择**单 GPU donor replica**；不能拿多 GPU donor 的其中一张卡冒充完整 donor。后续若放开多 TP，需要先扩展生产实现和设计合同，再扩展本 fixture。

### 方式 A：脚本启动一个有限时长的测试训练

```bash
export MULTITASK_LAUNCH_SCRIPT=/path/to/your/multitask_test_launcher.sh
export MT_E2E_LEASE_FILE=/tmp/lease.json

bash scripts/e2e/validate_lifecycle_cycle.sh \
  <传给 launcher 的 Hydra overrides>
```

launcher 应启动足够长、但最终会正常退出的测试训练。driver 会等待 TaskRunner 注册到命名 GS 后，
通过正式的 `open_lease -> submit_operation -> query_operation` 链路执行生命周期。

### 方式 B：附着到已经运行的测试 job

```bash
export MT_E2E_ATTACH_ONLY=1
export RAY_ADDRESS=auto
export MT_E2E_LEASE_FILE=/tmp/lease.json

bash scripts/e2e/validate_lifecycle_cycle.sh
```

如果同一 GS 下同时挂了多个 TaskRunner，需要额外设置：

```bash
export MT_E2E_TASK_SESSION=<目标 task_session>
```

## FORCE REMOVE

```bash
export MT_E2E_LEASE_FILE=/tmp/lease.json
bash scripts/e2e/validate_force_remove.sh <launcher overrides>
```

该场景要求配置 `async_training.partial_rollout=true`，并且 FORCE 目标之外至少还有一个 active
rollout server。否则返回 BLOCKED，而不是把 preflight rejection 算成功。

默认 FORCE 场景证明真实 `abort_all_requests()` 路径能够完成并闭环。
如果要进一步要求本次测试**真的命中至少一个 ADMITTED request 并完成 continuation handoff**：

```bash
export MT_E2E_REQUIRE_INFLIGHT_FORCE=1
bash scripts/e2e/validate_force_remove.sh <持续产生 rollout request 的测试配置>
```

Rollouter 会输出一个结构化 receipt：

```text
MULTITASK_FORCE_HANDOFF {"operation_id": "...", "admitted_count": 1,
 "abort_ack_known": true, "aborted_count": 1, "confirmed_count": 1}
```

当 `MT_E2E_REQUIRE_INFLIGHT_FORCE=1` 时，校验器从本次 `result.json` 取得 FORCE REMOVE
的 `operation_id`，只检查同一操作的 receipt。每条 receipt 独立核验计数，不能拼接不同操作的日志。
已收到 abort ACK 时，必须有 `aborted_count > 0`，且
`aborted_count <= confirmed_count <= admitted_count`；如果 abort ACK 丢失，则必须
`aborted_count=null`，且本次全部 ADMITTED requests 都有 continuation proof。
没有命中本次操作，或没有真实在途续推时返回 BLOCKED；计数矛盾或证明不完整返回 FAIL。

## 可选参数清单审计

`MULTITASK_PARAMETER_VALIDATION=1` 开启 CE receiver 清单审计。每次传输开始即使旧清单失效；
只有非空权重流被完整消费且加载调用成功后，清单才标为 complete。取消、异常或提前返回都保留
本次未完成状态。Manager 检查每个目标都有 CE Worker，以及清单非空、参数名唯一、字段完整、
shape/numel 与汇总计数一致，再检查版本和各 receiver 的摘要。

receiver 摘要相同只证明接收清单一致。`MULTITASK_SOURCE_VALIDATION=1` 还要求 backend
提供真实 source manifest。Ascend NPU 的 Fully Async E2E 自动选用已有的
`multitask_hccl` 扩展（`custom_backend_module` 注入，`rebuild_group=true`），
并优先调用 vLLM-Ascend communicator 的 `close()` 进行同步销毁；
不支持时使用经过验证的底层 HCCL destroy API，未知接口则保持失败。
CUDA 仍使用原生 NCCL，未增加 NCCL 源端审计。
没有 source manifest 就不能声称源端到接收端校验通过。

## 综合验收

```bash
export MT_E2E_LEASE_FILE=/tmp/lease.json
export MULTITASK_LAUNCH_SCRIPT=/path/to/test_launcher.sh
export MT_E2E_REQUIRE_COMPLETE=1
bash scripts/e2e/run_all.sh <launcher overrides>
```

默认场景为：

```text
control_plane exactly_once recovery lifecycle force
```

可缩小范围：

```bash
MT_E2E_SCENARIOS="control_plane recovery" \
MT_E2E_REQUIRE_COMPLETE=0 \
bash scripts/e2e/run_all.sh
```

`run_all.sh` 会继续执行后续场景，并在最终 `summary.json` 中区分 PASS、FAIL 和 BLOCKED。

## 与设计文档关键合同的对应

- **ADD**：真实 hidden create、target-only weight bootstrap、E/R/C/M commit；终态必须是
  `SERVICE_COMMITTED`。exact replay 不得创建第二个 runtime。
- **DONATE / natural REMOVE**：必须先 DRAINING、结清 request，再 E/R/C 退出和物理释放；
  `RELEASED` 才允许 GS 推进 Lease。
- **RESTORE**：同一 native runtime 回来，参数恢复后再发布；终态必须是
  `SERVICE_COMMITTED`。确定性 fault suite 另验 re-sleep compensation 与 G BLOCKED。
- **FORCE_VERIFIED**：borrowed-only；必须 partial rollout + 其他 active server；压力模式要求
  abort request 与 continuation proof 数量闭合。
- **UNKNOWN / ACK loss**：GS 不把查询不到或空结果当“确定未受理”；same operation 重放进行对账。
- **G**：同步/生命周期冲突由同一 gate 串行；未知 side effect 进入 BLOCKED，只有原 operation
  的 owner-fact reconciliation 能解除。
- **Exactly-once**：`(task_session, logical_sample_id, payload_digest)` 保护完成样本；重复相同 payload
  幂等，不同 payload 必须拒绝。
