# 原生 VERL Fully Async 接入

本目录说明如何通过 VERL 原生 `fully_async_main` 启用 MultiTask。**不会提供第二套训练入口**；模型、数据、资源规模和训练超参数仍由原生 VERL 配置决定。

日常双任务共享调度 E2E 请使用 [examples/e2e](../e2e/) 和 [统一启动器](../../scripts/e2e/verify_two_verl_jobs.py)，无需手动组合下面的参数。

## 1. 启用条件

1. 先确认目标 VERL 的 Fully Async + 独立 vLLM Rollout 能正常运行；driver、Ray worker 和子进程均能导入同一版本的 `multi_task_scheduler`。
2. 使用已安装 MultiTask TaskRunner 选择桥接的 VERL checkout。仅安装本仓包不会自动改动上游 VERL。
3. 从仓库根目录检查当前 Python 实际导入的 VERL 入口：

   ```bash
   python scripts/e2e/ensure_verl_multitask_bridge.py --check-only
   ```

   检查失败表示桥接尚未就绪；可参照 [项目 README](../../README.md#5-安装与准备) 的安装说明处理，不要将其当作训练成功。
4. 继续以 `python -m verl.experimental.fully_async_policy.fully_async_main` 启动原训练命令。设置 `multitask.enabled=true` 启用，改为 `false` 则新任务使用原生路径。

启用后自动选择受支持的运行 profile，无需显式指定名称。当前范围为**单节点、整卡、独立 non-PD vLLM、DP=1、PP=1**；真实 CUDA 验收覆盖 TP=1，其他未经验证的组合保持 fail-closed。

## 2. 手动启动参数

以下为 **CUDA / NCCL 示例 overrides**，追加到已经可运行的原生训练命令（不是一份可直接执行的完整训练配置）：

```text
multitask.enabled=true
actor_rollout_ref.hybrid_engine=false
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.calculate_log_probs=true
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true
async_training.use_trainer_do_validate=false
async_training.use_dynamic_resource_scheduling=false
data.train_batch_size=0
data.gen_batch_size=1
```

NPU 环境必须使用匹配的设备与权重同步后端，不能照搬 CUDA 的 `nccl` 参数。E2E 启动器会按支持的设备设置相关参数。初始 Replica 数量仍由原生 rollout 资源配置决定，GS 不负责分配初始规模。

## 3. 如何验收

按证据层级依次检查：

1. **配置与接线**：用原生入口 `--cfg job` 检查参数组合；分别关闭、开启 `multitask.enabled`，确认实际 TaskRunner、Trainer、Rollouter、Manager、LB、Replica 与 Checkpoint Engine 类型符合预期。
2. **单任务运行**：确认原生训练循环不受影响、TaskRunner 可响应 `query_operation`，原生 Replica 注册为 `NATIVE/ACTIVE`，自然完成请求最终结清。
3. **双任务控制面**：启动两个独立训练任务，验证共享 detached GS、任务隔离与异常退出的互不影响。
4. **真实设备生命周期**：验证 DONATE 释放的物理 GPU UUID、同卡 Borrowed 创建/销毁、RESTORE 最新权重和 FORCE 的真实 targeted abort/continuation。仅有控制面 ACK、Mock 或 CPU Ray PASS 不能证明设备闭环。

CUDA 真实原语验收可在仓库根目录运行：

```bash
VERL_MULTITASK_GPU_MODEL_PATH=/path/to/local/model \
  python -m pytest -q -s -m gpu_integration \
  tests/integration/cuda/test_native_sleep_gpu.py
```

测试覆盖 native sleep/借还等能力；满足设备条件时还会检查 RESTORE 的权重重装路径。**这些原语测试不等价于完整 GS → TaskRunner 双任务 E2E 验收**。如需完整链路，使用 [双任务启动器](../../scripts/e2e/verify_two_verl_jobs.py) 并按 [E2E 验收标准](../../docs/e2e-acceptance.md) 检查日志、操作证据、参数版本与请求交接回执；真实 FORCE 在途续推需额外验证，不能从普通 `force: PASS` 推断。

记录运行命令、源码和依赖版本、实际导入路径、设备 UUID、显存变化及场景日志。测试环境与执行命令参见 [tests/README.md](../../tests/README.md)；设计和能力边界参见 [项目 README](../../README.md)。
