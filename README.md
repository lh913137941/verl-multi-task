# verl-multi-task

可选的 VERL experimental Fully Async 扩展包。继续使用原生训练入口，
通过 `multitask.enabled=true` 选择现有 MultiTask 创建链；默认关闭。

更新日期：2026-09-16。本次源码核对基线为相邻 `verl` 仓的
`f92febf50fe3db102273eaf59b1854f392ae761d`，不是旧文档中的 `a9ebd0bb`。
当前工作区包含尚未提交的编排实现，构建的 wheel 包含这些当前源码。

## 安装与接入

在已经能运行 VERL 的训练环境中安装本仓，无需把源码嵌套复制到 VERL 目录：

```bash
uv pip install --python /path/to/training/python /path/to/verl-multi-task
# 开发时可改用可编辑安装：
uv pip install --python /path/to/training/python -e /path/to/verl-multi-task
```

部署到其他节点可使用 wheel：

```bash
uv build --wheel
uv pip install --python /path/to/training/python dist/verl_multitask-0.1.0.dev0-py3-none-any.whl
```

driver、所有 Ray 节点及其运行环境都需要安装同一份扩展包，并使用兼容的
VERL/Ray/vLLM 环境。安装本包不会替换这些运行时依赖。
`import multi_task_scheduler` 和配置校验保持轻量，不初始化 Ray、发现 GS、
注册全局对象或修改原生类。

**还需要 VERL 入口接线。** 本工作区已在相邻 `verl` 仓的以下两处完成：

- `verl/experimental/fully_async_policy/fully_async_main.py`：配置规范化后、
  `run_ppo` 初始化 Ray 前，按开关延迟导入插件解析器。
- `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`：
  增加默认关闭的 `multitask` 配置。

仅在其他未接线的上游 checkout 安装 wheel，不会自动改变其训练入口。
本包不使用 `verl.plugins` 自动加载、`VERL_USE_EXTERNAL_MODULES` 或 monkey patch。
当前上游自动加载机制只导入模块，没有为 Fully Async 提供 TaskRunner 选择接口；
这里使用原生 `run_ppo(config, task_runner_class=...)` 的窄接入点。

## 开关与启动

在已经可运行的原生 Fully Async 命令后追加以下接入参数。
它们不是完整训练命令，仍需原有模型、数据、算法和资源参数：

```text
multitask.enabled=true
actor_rollout_ref.hybrid_engine=false
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
actor_rollout_ref.rollout.calculate_log_probs=true
async_training.use_trainer_do_validate=false
async_training.use_dynamic_resource_scheduling=false
data.train_batch_size=0
data.gen_batch_size=1
```

入口仍是：

```bash
python -m verl.experimental.fully_async_policy.fully_async_main
```

| 配置 | 行为 |
|---|---|
| `multitask.enabled=false`（默认） | 原生 TaskRunner；不导入插件、不创建 GS，即使保存了 profile |
| `multitask.enabled=true`，profile 缺失或 null | 选择唯一的 `experimental_fully_async_standalone` profile |
| `multitask.enabled=true`，profile 显式指定 | 只接受上述 profile；其他值报错 |
| 旧自定义配置完全没有 enabled 字段 | 保留显式 profile 的旧选择语义；缺失/null profile 关闭 |
| enabled 为字符串、数字或 null | 明确报错，不以 Python 真值转换开关 |

新版主 YAML 已有 `enabled=false`，所以只设置旧的 `multitask.runtime.profile`
不会启用插件；请改用 `multitask.enabled=true`。
关闭只需改为 `multitask.enabled=false`，不必卸载包。
显式启用后，缺包、缺依赖、配置错误会终止启动，不静默回退到原生训练。
原生配置不会被插件改写；节点/GPU 映射和 reward migration 仍先由 VERL 执行。

## 创建链与所有者

```text
VERL main → resolve_runtime_profile → run_ppo
  → MultiTaskFullyAsyncTaskRunner（Ray Actor，连接共享 GS）
    ├─ MultiTaskFullyAsyncTrainer（Ray Actor）
    │  └─ MultiTaskCheckpointEngineManager（本地对象）
    └─ MultiTaskFullyAsyncRollouter（Ray Actor）
       └─ MultiTaskLLMServerManager（本地对象）
          ├─ MultiTaskGlobalRequestLoadBalancer（Ray Actor）
          └─ MultiTaskvLLMReplica（本地对象）
             ├─ MultiTaskCheckpointEngineWorker（Ray Actor）
             └─ MultiTaskvLLMHttpServer（Ray Actor）
```

TaskRunner 创建 Trainer/Rollouter；Trainer 持有 CE Manager，
Rollouter 持有 LLM Manager。扩展复用原生训练循环、生成、客户端、MessageQueue 和参数同步。
本次开关不修改原生类实现，不引入另一份 Hydra primary 或逐类 FQN 配置。

GS 使用 detached 生命周期，默认 namespace 为 `verl-multi-task`，
name 为 `verl-multi-task-group-scheduler`。
TaskRunner 正常退出时只解绑自己。关闭新任务的插件不会销毁其他任务共用的 GS。
强制退出后的残留与跨崩溃恢复仍需独立验证。

## 能力边界

当前可选择的适配族是 experimental Fully Async + 纯 STANDALONE + vLLM 非 PD。
原生参数和布局限制仍有效，V1、HYBRID 和 PD 不受支持。

现有源码包含纯 Python 编排核心、GS 账本/租约及扩展绑定，但并未完成精简设计第 19 节
要求的 GPU 借还验收。真实 borrowed 创建、sleep/destroy、目标专属追参、
ADD/REMOVE/RESTORE 提交及强制中断续推等接口仍显式抛出 `NotImplementedError`。
原生同步与 G、实际发布快照、请求/样本证据等完整接线也不能由这些类的存在来证明。

因此 `enabled=true` 启用的是现有扩展运行路径，不代表已实现自动借卡或安全强制回收。
具体合同以外层《多 RL 任务资源共享调度对接 VERL：动态流程编排融合设计精简版》第 2、18、19 节为准。
关闭 capability 或通过 mock 测试不能替代该设计的 GPU 验收。

## 验证

本轮使用独立的 uv 环境、Python 3.12.14，保留了原先不可用的 `.venv`。

| 层次 | 本轮结果 | 证明范围 |
|---|---|---|
| unit | 182 项通过 | 编排单测、配置选择、原生入口 AST 执行、构造链替身、原生类源码保持不变 |
| Hydra | 包含在 unit 中，通过 | 实际 primary/defaults 与示例参数组合；搜索路径映射到源码目录，未启动 GPU 模块 |
| wheel | 构建、干净环境安装与导入通过 | 32 个源码模块完整打包；无缓存/环境文件；没有 Ray/VERL 时关闭解析仍可用 |
| CPU Ray / native_unit / GPU | 本轮未执行 | 未安装 Ray、PyTorch、vLLM；不声称真实 Actor、训练或借还闭环通过 |

复验单元测试（从本仓执行）：

```bash
uv venv --python 3.12 .venv-plugin
uv pip install --python .venv-plugin/bin/python -e '.[test]'
MT_VERL_SOURCE_ROOT=/absolute/path/to/verl \
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv-plugin/bin/python -m pytest -q -p no:cacheprovider tests/unit
```

Windows 使用环境的 `Scripts/python.exe`，并以 PowerShell 设置相同环境变量。
源代码对比测试要求指定的 VERL Git checkout 含有上述 `f92febf5` 基线对象；
源码归档可运行其余测试，但不具备该对比证据。
真实训练验收顺序见 [示例说明](examples/experimental_fully_async/README.md)。
