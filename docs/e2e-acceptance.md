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
| `validate_force_remove.sh` | 真实 VERL/Ray/vLLM runtime | `DONATE -> ADD -> FORCE REMOVE -> RESTORE`；严格在途 continuation proof 由主启动器的 `--require-inflight-force` 校验 |
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

## 推荐入口：两个真实 VERL 任务

仓库统一入口为 `scripts/e2e/verify_two_verl_jobs.py`。它通过原生 VERL Fully Async 入口
启动 donor/borrower，连接同一个 Ray / GroupScheduler，从真实 donor CE 与 Placement Group
发现物理资源，自动生成 Lease，然后由 `run_all.sh` 调用下方五个 `validate_*` 执行验收。

```bash
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . \
  --ray-address auto \
  --start-local-ray \
  --native-args examples/e2e/native_args.txt \
  --scenarios "lifecycle force"
```

`native_args.txt` 必须提前配置**真实的模型和数据路径**；仓库样例中的模型路径默认是占位符。
已有持久 Ray 集群时，指定实际 `--ray-address` 并移除 `--start-local-ray`。
启动器在必要时安全备份/修补当前实际导入的 VERL 入口，不使用另一套训练主循环。

`--scenarios` 可设为 `"control_plane exactly_once recovery lifecycle force"`，也可以只选择失败场景。
每次结果位于 `logs/two_real_jobs/<时间>/orchestration_summary.json`；
场景明细位于 `scenarios/<场景>/<运行ID>/result.json`。
退出码：0=PASS、1=FAIL、2=BLOCKED；没有真实完成证据绝不能算 PASS。

## 验收脚本为何保留

`run_all.sh` 的五条场景分别由以下脚本完成，**并非废弃入口**：

- `validate_control_plane.sh`：真实 Ray 控制面检查，调用 `control_plane_recovery.py`
- `validate_exactly_once.sh`：真实 Ray MessageQueue，调用 `exactly_once_driver.py`
- `validate_recovery_faults.sh`：通过明确列出的 pytest 故障注入用例验证恢复合同
- `validate_lifecycle_cycle.sh`：实际 DONATE → ADD → REMOVE → RESTORE，调用 `lifecycle_driver.py`
- `validate_force_remove.sh`：实际 DONATE → ADD → FORCE REMOVE → RESTORE，同样复用 `lifecycle_driver.py`

前 3 项可以独立运行，无需启动双任务，但控制面与队列检查仍需相应 Ray/VERL 环境：

```bash
bash scripts/e2e/validate_control_plane.sh
bash scripts/e2e/validate_exactly_once.sh
bash scripts/e2e/validate_recovery_faults.sh
```

生命周期验证只走**附着到当前真实双任务**的路径，由双任务启动器设置
`MT_E2E_ATTACH_ONLY=1`、`MT_E2E_LEASE_FILE`、
`MT_E2E_DONOR_SESSION`、`MT_E2E_BORROWER_SESSION`、`RAY_ADDRESS`。
不要拿示例 Lease 的虚拟 `pg_id` / `gpu_uuid` 当真实卡归属。

## 严格 FORCE 与回执

普通 `force: PASS` 只证明对应 FORCE 生命周期操作闭环；要进一步要求
**至少一次真实在途请求中断和 continuation handoff**，在主命令中增加
`--require-inflight-force`。严格校验由 `verify_two_verl_jobs.py` 完成，
从同一次运行的 `force_cycle/result.json` 取得精确 `operation_id`，
再匹配 `borrower.log` 中的 `MULTITASK_FORCE_HANDOFF` 回执。
不再有另一份 launcher-mode 的重复校验实现。

正常 abort ACK 要求 `aborted_count > 0`，且
`aborted_count <= confirmed_count <= admitted_count`；若 abort ACK 丢失，
要求全部边界 ADMITTED 请求均有 continuation proof。
无正向在途证明返回 BLOCKED；计数矛盾返回 FAIL。

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

统一从 `verify_two_verl_jobs.py` 选择场景，主脚本自动调用 `run_all.sh` 汇总。
`run_all.sh` 不负责启动训练、不创建物理 Lease；请勿将它当独立的集群启动入口。

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
