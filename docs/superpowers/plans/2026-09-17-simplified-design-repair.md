# 精简设计一致性修复

依据：外层 `多RL任务资源共享调度对接VERL_动态流程编排融合设计精简版.md`。
用户已明确：严格按该设计修复，本轮不运行设备验收；现有可预约集群为 NPU，
不能作为 CUDA/NCCL 后端已验证的证据。

> 2026-09-17 复核更正：本文件原先把全部 8 项标为完成并不准确。
> 样本去重仍保留 `turn`，Ledger 幂等/结果身份校验也只完成了一部分。
> 第二轮修复以 `2026-09-17-simplified-design-repair-round2.md` 为准。

## 已确认的缺陷与修复状态

- [x] 空泡观测：缺失计数默认为 0；选择函数把当前窗口 epoch 当作观测 epoch。
  改为 UNKNOWN/None 拒绝候选，并检查真实观测代次与完整性。
- [x] GS 候选：无限 TTL、旧报告覆盖新报告、空报告不撤销、重复报告续期。
  改为有限相对 TTL、单调报告和整集合替换。
- [x] 幂等（第二轮补齐）：同 operation_id 的更高 lease_epoch 会覆盖原记录；缺失 digest 或
  改 command_seq 可绕过冲突；结果合并不检查完整身份。OperationJournal 第一轮已修，
  Ledger 的命令身份和结果身份校验在第二轮补齐。
- [x] 同步门：实际 Trainer 原生同步没有进入 G；异常释放 G 后继续调度。
  通过继承委托包住原生同步；异常/取消标为 BLOCKED，普通新操作拒绝。
- [x] 生命周期：从旧版 14 状态迁移到 8 状态，区分 native sleep 与 borrowed
  destroy，保留隔离和有证据的补偿边。
- [x] 样本去重（第二轮补齐）：CompletionKey 改为 `(task_session, logical_sample_id)`，
  不再包含 turn/attempt；`put_sample_once` 返回首次 CompletionEvidence，保留原生满丢返回语义。
- [x] 事务/绑定：禁止空 manifest/版本标签冒充快照、未完成清理时发布、LB 假
  drain/假 remove、CE overlay 冒充原生 E 提交、字符串 COMMITTED 冒充释放。
- [x] 补充接线测试、更新当前能力说明，并记录尚未实现的设备和原生 hook 边界。

## 验证边界

先运行每个缺陷的失败测试，再进行对应修复。使用 `.uv-cache/plugin-tests`
中的项目 Python；所有替身测试明确标注，设备测试不运行。
原生 VERL 类实现的变更受 AGENTS.md 限制；若样本接线确需新增原生工厂 hook，
先给出具体位置与最小方案。不会通过全局 monkey patch 或复制整段初始化规避限制。
