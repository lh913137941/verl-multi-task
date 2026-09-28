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

然后把 `pg_id`、`node_id`、`gpu_uuid`、`bundle_index` 和 donor rank 改成真实环境值。
`donor_task_id` 可以保留 `__TASK_SESSION__`，driver 会替换为本次附着的 TaskRunner session。
Lease 必须覆盖**完整 donor replica**，不能只填写其中一张卡来绕过设计约束。

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

当 `MT_E2E_REQUIRE_INFLIGHT_FORCE=1` 时，`admitted_count=0` 只能算 BLOCKED，不能证明 continuation。

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
