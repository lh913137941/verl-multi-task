# verl_expansion D2 reference

来源仓库：`ZiqiGuan892/verl_expansion`  
来源分支：`D2`  
来源提交：`ba920fd5f110bbf76811817a61f17411c7fd2dce`

此目录保存本次融合时有价值的 D2 资源能力、borrowed runtime 和架构材料，便于后续继续扩展。
这些文档是**参考材料，不是当前规范**。当前运行时规范仍是
`../simplified-fusion-contract.md` 以及当前分支源码。

没有直接进入生产路径的源仓能力主要包括：

- 多 source lease / 多 donor claim 聚合；
- 多 TP、多节点 borrowed runtime 与 fragmented/cross-PG D2 smoke；
- Ascend/NPU 专用启动脚本；
- 源仓旧的 lifecycle/reclaim receipt wire。

这些能力若后续启用，需要按当前 Lease、OperationEvidence、ReplicaSyncGate、UNKNOWN reconciliation
和异常恢复合同重新迁移，不能直接恢复旧接口或覆盖现有流程。
