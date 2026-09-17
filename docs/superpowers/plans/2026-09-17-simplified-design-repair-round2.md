# 精简设计一致性修复（第二轮）

依据：用户提供的《多RL任务资源共享调度对接VERL_动态流程编排融合设计精简版(3)》。本轮以该文档的 §5、§6、§8-§15 为合同基线。

## 本轮修复

- [x] OperationStatus 对齐为 `ACCEPTED / RUNNING / SUCCEEDED / FAILED / UNKNOWN`；status、phase、outcome 分开保存。
- [x] Phase 对齐为 16 个阶段；`DONE` 不自动推导成功/失败，终态 status 必须显式记录。
- [x] OperationContext 改为 `protocol_version: int`、`gs_epoch: str`，GS 启动生成新的 epoch 字符串。
- [x] CompletionKey 改为 `(task_session, logical_sample_id)`，移除 turn/attempt 维度；首次提交保存并重放 CompletionEvidence。
- [x] Ledger / OperationJournal 的 operation replay 校验覆盖完整业务身份、lease、command_seq、target 与摘要；`remaining_budget_ms` 不参与业务摘要，同号重试不能新建另一场操作或重置原受理操作的总时限。
- [x] TaskRunner 严格执行“每任务一场生命周期操作”：同 operation_id 可幂等重放；不同 operation_id 必须等当前工单进入 DONE 后才能受理，并继续受严格递增 command_seq 保护。
- [x] Ledger 合并 OperationResult 前核验完整 OperationContext 与 ReplicaKey；迟到旧 session/lease/seq 结果不能靠更高 revision 覆盖。
- [x] PlacementSpec 首版收敛为单个 `NodePlacement` + `placement_digest`；删除旧多节点 `NodeBlock` wire shape，不保留兼容投影。
- [x] ReleaseKind 对齐为 `DONOR_SLEEP_RELEASED / BORROWER_RUNTIME_DESTROYED`，公共证明统一使用 `ReleaseEvidence`；删除旧 `ReleaseReceipt` alias。
- [x] `OperationKind` 为唯一公共操作枚举；删除旧 `OperationType` alias。
- [x] IdleCandidate 直接携带完整观察身份；删除 ID-only CandidateSet / CandidateRef 公共包装。
- [x] CE/LB owner 提交统一返回 typed `CommitReceipt`，不再保留 bool/int 兼容结果。
- [x] TaskRunner 控制入口统一为 `submit_operation/query_operation/probe_task`，删除 `begin_operation`；退出入口统一为 `prepare_exit`，删除 `begin_drain`。
- [x] GS 查询优先读取 TaskRunner 权威 operation journal，控制器不可达时只保守返回本地已知事实，不用健康状态猜执行结果。
- [x] FORCE_VERIFIED continuation 必须显式证明 `old_terminal.device_finished=True`；缺字段不能默认通过。
- [x] GS lease 授权闭环：释放边核验完整 `OperationResult + ServiceEvidence(REMOVE) + ReleaseEvidence`，要求逐 GPU 释放证明完整覆盖 lease placement，并把确认过的 release digest 绑定到后续 ADD/RESTORE authorization。
- [x] GS GPU 使用权账本与 lease handoff 同步：donor 释放后整组 GPU 才显式授权 borrower；borrower 真实释放后整组清回 native 隐式使用权；placement、native owner、lease id/epoch 任一不匹配都拒绝且不做部分更新。
- [x] 修正第一轮计划错误勾选，并更新相关 docstring 编号/语义。
- [x] CPU Ray GS 集成测试迁移到新 protocol/epoch/placement/status 合同（仍需真实执行环境验证）。

## 设计复核修正

主设计里 `OperationRecord.status` 与 `phase` 是两个独立字段，因此第二轮原排查稿中“由 phase 映射 status”的表述不能用于终态：`Phase.DONE` 可对应 `SUCCEEDED`、`FAILED` 或在对账后仍需保留的 `UNKNOWN`。实现改为显式终态 status。

用户上传的精简设计文档另发现两处纯文字瑕疵：

1. §6.7 “删除只重复包装这两个字段的 CompletionEvidence”应为删除 `CompletionRecord`。
2. 重复标题 `### 6.8### 6.8` 应为单个 `### 6.8`。

两处不改变业务语义；修正版作为本次交付附件提供。

## 验证边界

- 纯 Python 契约、journal、exactly-once、ledger、lease、scale transaction 为本轮优先回归层。
- CPU Ray 集成测试只验证 GS Actor/句柄/合同，不代表 GPU runtime 成功。
- borrowed create、真实 sleep/wake、target-only bootstrap、权重 replay、FORCE_VERIFIED 续推仍必须保持显式 `NotImplementedError`，直到 CUDA/NCCL/native backend 实测通过。
- 不通过 mock/AST 结果宣称设备能力已经交付。
