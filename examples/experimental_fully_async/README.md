# 原生入口配置与真实环境验收

更新日期：2026-09-16。当前接入使用 `multitask.enabled` 开关。
源码核对基线是相邻 VERL checkout 的 `f92febf50fe3db102273eaf59b1854f392ae761d`。

## 使用方法

1. 在已经可以运行原生 Fully Async 的训练环境中，安装 `verl-multitask`；
   driver、Ray 节点和子进程必须能导入同一版本。
2. 使用包含本轮 main/YAML 接线的 VERL；仅安装 wheel 不会修改其他上游 checkout。
3. 继续从 `python -m verl.experimental.fully_async_policy.fully_async_main` 启动，
   将 [native_entry_overrides.txt](native_entry_overrides.txt) 中的参数追加到原有训练命令。
4. 设置 `multitask.enabled=false` 即可关闭，新启动的任务保留原生路径。

`multitask.enabled=true` 默认选择唯一的
`experimental_fully_async_standalone` profile。
`multitask.runtime.profile` 可保持 null；显式设置时只接受该 profile。
显式 false 优先于保存的 profile。新版主配置默认 false，所以旧命令仅设置 profile
需要迁移为显式 enabled=true。

不支持的 profile、HYBRID/PD 等配置和启用时的导入失败均报错。
不使用伴生训练入口、复制的 Hydra primary、逐类 FQN 或 import-time monkey patch。

示例只提供接入条件，模型、数据、算法、训练步数以及 trainer/rollout 资源参数继续使用原生配置。
GS 不分配初始规模；Replica 数量仍由原生 rollout 资源字段决定。

## 验收顺序

1. 在每个执行环境打印 `multi_task_scheduler.__file__`，核对安装位置与部署版本。
   先通过原生入口的 `--cfg job` 检查组合结果；这一步本身不证明 Actor 能运行。
2. 同一模型/数据/资源配置先关闭开关，再启用开关。核对根 TaskRunner、Trainer、
   Rollouter、Manager、Replica、LB、HTTP Server、CE Manager/Worker 的实际类型。
3. 分别以 `async_training.partial_rollout=true` 和 `false` 检查生成、样本队列、
   训练更新、初始传权及至少一次后续传权。该参数保持原生客户端语义。
4. 启动两个隔离的训练 job，检查它们发现同一 GS；一个正常退出不影响另一个。
   单独记录初始化失败、训练失败和 GS 不可达时的行为。
5. GPU 借还、目标专属追参、强制回收和真实样本守恒必须另按精简设计第 19 节验收；
   当前运行时原语仍有显式未实现接口，不可把第 1–4 步当成借还闭环完成。

记录完整命令、组合配置、两仓源码版本、环境依赖、实际导入路径、节点/GPU 数量、
Actor 类型和日志。若没有实际触发中断/续推，只记录该配置运行结果。

本轮验证：182 项 unit（含真实 Hydra 组合、入口/构造链替身检查），以及 wheel 构建、
干净环境安装和关闭路径导入。未执行真实 Ray、原生 GPU 训练或借还测试。
安装说明与能力边界见 [README](../../README.md)。
