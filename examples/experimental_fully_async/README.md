# 原生入口配置与真实环境验收

当前接入使用 `multitask.enabled` 开关，并以 092203 simplified fusion contract
作为实现合同。

## 使用方法

1. 在已经可以运行原生 Fully Async 的训练环境中安装 `verl-multitask`；driver、
   Ray 节点和子进程必须能导入同一版本。
2. 使用已接入 MultiTask TaskRunner 选择逻辑的 VERL checkout；仅安装 wheel
   不会自动修改其他上游 checkout。
3. 继续从 `python -m verl.experimental.fully_async_policy.fully_async_main` 启动，
   将 [native_entry_overrides.txt](native_entry_overrides.txt) 中的参数追加到原有训练命令。
4. 设置 `multitask.enabled=false` 即可关闭，新启动任务保留原生路径。

`multitask.enabled=true` 默认选择唯一的
`experimental_fully_async_standalone` profile。当前首版只支持单节点、整卡、
DP=1、PP=1、non-PD vLLM；TP 必须能在单节点放下并经实际组合验证。

示例只提供接入条件，模型、数据、算法、训练步数以及 trainer/rollout 资源参数继续
使用原生配置。GS 不分配初始规模；初始 Replica 数量仍由原生 rollout 资源字段决定。

## 验收顺序

1. 在每个执行环境打印 `multi_task_scheduler.__file__`，核对安装位置与部署版本。
   先通过原生入口的 `--cfg job` 检查组合结果；这一步本身不证明 Actor 能运行。
2. 同一模型/数据/资源配置先关闭开关，再启用开关。核对根 TaskRunner、Trainer、
   Rollouter、Manager、Replica、LB、HTTP Server、CE Manager/Worker 的实际类型。
3. 检查 TaskRunner 在原生长时间 `run()` 期间仍能响应 `query_operation`，确认
   MultiTask Actor 的有限并发没有破坏原生训练循环。
4. 检查原生 STANDALONE replicas 初始化后都进入 Manager 的 M：
   `replica_kind=NATIVE`、`replica_state=ACTIVE`。
5. 检查 LB 的逐 request 状态：自然完成进入 `SETTLED`；FORCE 路径只有在真实
   continuation 证明成立后才允许 `TERMINATED`，迟到 release 不得覆盖该终态。
6. 启动两个隔离的训练 job，检查它们发现同一 detached GS；一个正常退出不影响另一个。
7. GPU 借还、target-only bootstrap、DONATE sleep、RESTORE wake、FORCE targeted abort
   和真实 RELEASED 逐卡核验必须单独做 GPU 验收。当前这些原语仍显式
   `NotImplementedError`，不可把控制面测试当成借还闭环完成。

记录完整命令、组合配置、两仓源码版本、环境依赖、实际导入路径、节点/GPU 数量、
Actor 类型和日志。若没有实际触发中断/续推，只记录该配置运行结果。

当前分支没有可用的 GitHub Actions 运行结果；README 不声明未执行的 native/GPU
测试通过。安装说明、状态 owner 与能力边界见 [README](../../README.md)。
