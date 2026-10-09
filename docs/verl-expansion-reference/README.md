# verl_expansion main reference

来源仓库：`ZiqiGuan892/verl_expansion`  
来源分支：`main`  
来源提交：`761bd6ddc296f7170ade3489ba9fbc8c8c04db62`  
其 D2 基线：`ba920fd5f110bbf76811817a61f17411c7fd2dce`

本目录保存融合时有价值的 D2/D3/D4 资源能力、borrowed runtime 与综合验收设计。
这些文档是**参考材料，不是当前规范**。当前运行时规范仍是
`../simplified-fusion-contract.md` 以及当前分支源码。

已迁入当前生产/验收代码的 main 新增能力：
- CheckpointEngineWorker 可选 receiver parameter manifest / SHA256 审计；
- 独立 `multitask_hccl` backend 模块（未改变当前 CUDA/nccl profile 的默认选择）；
- 现行 E2E 采用 `scripts/e2e/verify_two_verl_jobs.py` 的五场景验收及独立诊断工具；历史 `D0_D4_E2E_RESULT` 判据已退出当前代码与单测，原始设计仍保留在本目录参考文档中；
- D3/D4/综合验收设计参考。

没有直接覆盖当前生产 controller 的 main 能力：
- 旧的 rank/dict receipt D3/D4 create chain；
- 多 source lease / 多 donor claim 聚合；
- 多 TP、多节点 borrowed runtime；
- Ascend/NPU 专用 lifecycle controller 与旧 reclaim wire。

这些能力若后续启用，必须按当前 ReplicaKey、Lease、OperationEvidence、
ReplicaSyncGate、UNKNOWN reconciliation 和异常恢复合同重新迁移，不能直接恢复旧接口。
