# 精简设计一致性修复（第二轮）

依据：用户提供的《多RL任务资源共享调度对接VERL_动态流程编排融合设计精简版(3)》。本轮以该文档的 §5、§6、§8-§15 为合同基线。

## 本轮修复

- [x] OperationStatus 对齐为 `ACCEPTED / RUNNING / SUCCEEDED / FAILED / UNKNOWN`，租约回执边同步要求 `SUCCEEDED`。
- [x] Phase 对齐为 16 个阶段；`status` 与 `phase` 独立保存，`DONE` 不自动推导成功/失败。
- [x] OperationContext 改为 `protocol_version: int`、`gs_epoch: str`，GS 启动生成新的 epoch 字符串。
- [x] CompletionKey 改为 `(task_session, logical_sample_id)`，移除 turn/attempt 维度；首次提交保存并重放 CompletionEvidence。
- [x] Ledger 的 operation replay 校验覆盖命令身份、lease、command_seq、target 与摘要；不同身份不再复用。
- [x] Ledger 合并 OperationResult 前核验完整 OperationContext；迟到旧 session/lease/seq 结果不能靠更高 revision 覆盖。
- [x] PlacementSpec 首版收敛为单个 `NodePlacement` + `placement_digest`，保留只读兼容投影，不再允许多节点 wire shape。
- [x] ReleaseKind 对齐为 `DONOR_SLEEP_RELEASED / BORROWER_RUNTIME_DESTROYED`，公共名改为 ReleaseEvidence；旧 ReleaseReceipt 仅保留 import alias。
- [x] OperationKind 为公共名称；旧 OperationType 仅保留 import alias。
- [x] 修正第一轮计划错误勾选，并更新相关 docstring 编号/语义。
- [x] CPU Ray GS 集成测试改为新 protocol/epoch/placement/status 合同。

## 设计复核修正

主设计里 `OperationRecord.status` 与 `phase` 是两个独立字段，因此第二轮原排查稿中“由 phase 映射 status”的表述不能用于终态：`Phase.DONE` 同时可对应 `SUCCEEDED`、`FAILED` 或 `UNKNOWN`。实现改为显式终态 status。

用户上传的精简设计文档另发现两处纯文字瑕疵：

1. §6.7 “删除只重复包装这两个字段的 CompletionEvidence”应为删除 `CompletionRecord`。
2. 重复标题 `### 6.8### 6.8` 应为单个 `### 6.8`。

两处不改变业务语义；修正版作为本次交付附件提供。

## 验证边界

- 纯 Python 契约、journal、exactly-once、ledger、lease、scale transaction 为本轮优先回归层。
- CPU Ray 集成测试只验证 GS Actor/句柄/合同，不代表 GPU runtime 成功。
- borrowed create、真实 sleep/wake、target-only bootstrap、权重 replay、FORCE_VERIFIED 续推仍必须保持显式 `NotImplementedError`，直到 CUDA/NCCL/native backend 实测通过。
- 不通过 mock/AST 结果宣称设备能力已经交付。
