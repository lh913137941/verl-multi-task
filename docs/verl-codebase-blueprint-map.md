# 5.3 编排蓝图 → verl 代码仓：逻辑与模块详细梳理

> 本文回答一个问题：**5.3 编排蓝图（replica-sync 门 / ADD / REMOVE / NATIVE SYNC / RESTORE）落到 verl 代码仓里，会触达哪些模块、哪些类、哪些方法、原生逻辑是什么、蓝图要如何挂上去。**
>
> 本文是文档化梳理，**不改任何代码**。它是 5.3 蓝图的 verl 侧陪读材料，也是后续「编排核心（Slice A+B+C）」实现计划的论证依据。

---

## 0. 阅读前提与版本基线

### 0.1 三条路径约定

| 记号 | 物理路径 | 含义 |
| --- | --- | --- |
| `V/` | `verl/verl/` | verl 原生代码仓（被蓝图触达的一侧） |
| `M/` | `verl-multi-task/src/multi_task_scheduler/` | 伴生仓库（编排核心落地的一侧） |
| `R/` | `multi_task_verl/multi_task_scheduler/references/` | 调研文档仓（doc 17/26/27 在此） |

### 0.2 版本基线（重要，需显式声明）

- **本地 `V/` 当前 HEAD = `f92febf5`**（vanilla 上游，`grep multitask` 为空、`run_ppo` 固定用 `FullyAsyncTaskRunner`）。
- 文档/蓝图钉定的接线版本是 `a9ebd0bb`（含 `resolve_runtime_profile` 选择逻辑的 2 个文件改动）。
- **本文所有行号以本地 `f92febf5` 为准**。两版本在本文涉及的模块上结构基本一致，但 `checkpoint_engine/base.py` 里的 ServerAdapter 在本版已并入 `CheckpointEngineWorker` 的 `server_adapter: BaseRollout` 字段（见 §4.8），与旧版命名不同。

### 0.3 蓝图本身

- 蓝图文件：`C:\Users\星星\.claude\plans\binary-soaring-giraffe-agent-aac0fab8a5fdfe833.md`（5.3 实现蓝图，37 KB）。
- 蓝图依赖的原生时序结论：`R/17-verl-v0.9-dynamic-replica-scaling-timing-and-mutual-exclusion.md`。

---

## 1. 一张总表：蓝图编排原语 → verl 原生机制

5.3 蓝图在 M/ 侧要新增 5 个模块（`replica_sync_gate.py` / `replica_record.py` / `operation_journal.py` / `receipts.py` / `scale_transaction.py`），它们的**每一步最终都要调用 V/ 侧的原生方法**。下表是总映射：

| 蓝图原语（M/ 侧） | verl 原生对应（V/ 侧） | 原生是否已有该语义 |
| --- | --- | --- |
| `ReplicaSyncGate`（单 `G_standalone` 互斥门） | 无。原生靠 `FullyAsyncTaskRunner` 单 Actor 串行（`num_cpus=1, max_concurrency=1`）保证「无并发编排」，但没有跨 `CheckpointEngineManager` / LB / replica 的原子门 | ❌ 无，蓝图新增 |
| `OperationJournal`（操作状态机 + 重放） | 无。原生没有「增删操作」的持久/可重放记录，增删是即时副作用 | ❌ 无，蓝图新增 |
| `ReplicaRecord`（副本生命周期状态） | `RolloutReplica`（`replica.py:70`）+ `CheckpointEngineManager.replicas` 列表（`base.py:449/457`）。原生只有「在列表里 / 不在列表里」两态，无 MATERIALIZING→ROUTABLE→DRAINING 的显式状态机 | ⚠️ 部分：对象存在，状态机缺 |
| `ScaleTransaction.add`（materialize→bootstrap→CE→routable） | `RolloutReplica.init_standalone`（`replica.py:189`）→ `CheckpointEngineManager.add_replicas`（`base.py:449`）→ `GlobalRequestLoadBalancer.add_servers`（`router.py:227`） | ⚠️ 原生三步是「零散、无事务、无门」 |
| `ScaleTransaction.remove`（drain→CE remove→LB remove→sleep/destroy） | `GlobalRequestLoadBalancer.get_inflight_count`（`router.py:255`）→ `remove_servers`（`router.py:242`）→ `CheckpointEngineManager.remove_replicas`（`base.py:457`）→ `RolloutReplica.sleep`（`replica.py:269`） | ⚠️ 原生无 `begin_drain`，`remove_servers` 直接删 |
| `ScaleTransaction.bootstrap`（新副本灌权重） | `CheckpointEngineManager.update_weights`（`base.py:505`）——但它是「全量 abort + rebuild process group + NCCL」，不是单副本增量 bootstrap | ❌ 原生无单副本路径，蓝图需扩展 |
| `ScaleTransaction.native_sync`（周期同步） | `FullyAsyncTrainer._fit_update_weights`（`trainer.py:690`）→ `checkpoint_manager.update_weights`（`base.py:505`） | ✅ 原生有，蓝图只做「门包裹」 |
| `ScaleTransaction.restore`（donor/borrower 归还） | 无。蓝图复用 ADD 链路的 bootstrap + CE add + wake | ❌ 无，蓝图组合 |

**核心结论**：verl 原生「有资源对象、有即时增删副作用、有周期同步」，但**没有** ①互斥门、②操作状态机、③drain 等待、④单副本 bootstrap。5.3 蓝图的编排核心正是在 M/ 侧补齐这四件事，再把它们**薄薄地包在原生方法外面**。

---

## 2. 运行时五个 Actor 及其与蓝图的关系

verl Fully-Async 的运行时是 5 个 Ray Actor + 2 个「进程内对象」。蓝图的门必须明确「放哪一侧」，所以先理清 Actor 拓扑：

| Actor / 对象 | 位置 | 关键属性 | 对蓝图的意义 |
| --- | --- | --- | --- |
| `FullyAsyncTaskRunner` | `V/experimental/fully_async_policy/fully_async_main.py:36` | `@ray.remote(num_cpus=1)`，`run()`（`:46`）阻塞主循环 | 唯一的「编排串行点」；蓝图门若放这里，天然无并发 |
| `FullyAsyncTrainer` | `fully_async_trainer.py:54` | 持有 `checkpoint_manager`、`dynamic_resource_controller` | `CheckpointEngineManager` 的**宿主**；权重同步的触发者 |
| `FullyAsyncRollouter` | `fully_async_rollouter.py:330` | `@ray.remote(num_cpus=10, max_concurrency=100)`；持有 `llm_server_manager` | 高并发 Actor；副本/样本视图的**权威宿主** |
| `FullyAsyncLLMServerManager` | `fully_async_rollouter.py:54`（继承 `llm_server.py:346` 的 `LLMServerManager`） | 持有 `server_handles` / `server_addresses` / `rollout_replicas` | replica 对象的实际持有者（在 Rollouter 进程内） |
| `MessageQueue` / `MessageQueueClient` | `message_queue.py:27` / `:180` | `deque(maxlen=...)` | 样本队列，与副本增删间接相关（staleness） |
| `CheckpointEngineManager`（**非 Actor**） | `checkpoint_engine/base.py:380` | Trainer 进程内对象，`replicas` 列表 | 权重同步 + 副本集合变更的实际执行体 |
| `CheckpointEngineWorker`（**非 Actor，colocated**） | `checkpoint_engine/base.py:304` | 每个 rollout 进程内一个，持有 `server_adapter` | NCCL 收端，`update_weights`（`:354`）落地权重 |

**关键差异**（决定蓝图门放哪）：
- `FullyAsyncTaskRunner` 是 **串行 Actor**（`num_cpus=1`），任何「一次只做一件事」的编排天然满足。
- `FullyAsyncRollouter` 是 **高并发 Actor**（`max_concurrency=100`），但它的 `add_replicas`/`remove_replicas`（`:1199`/`:1205`）本质是薄封装，真正的资源动作在 `CheckpointEngineManager`（Trainer 进程）与 `GlobalRequestLoadBalancer`（独立 Actor，`router.py:133`）。
- 因此 5.3 的 `G_standalone` 门**必须落在 Trainer / TaskRunner 这一串行侧**（持有 `checkpoint_manager`），而不是 Rollouter 高并发侧。

---

## 3. 蓝图五大能力 → verl 具体挂点

### 3.1 `G_standalone` 互斥门（`ReplicaSyncGate`）

**原生现状**：没有任何跨对象的互斥门。原生「安全」只来自两点——
1. `FullyAsyncTaskRunner.run()`（`main.py:46`）是单 Actor 阻塞循环，编排调用串行；
2. `CheckpointEngineManager.update_weights`（`base.py:505`）内部自成一原子块。

**原生并发风险源**（蓝图门要防的就是它）：
- `update_weights` 内部顺序是 `abort_replicas → release_kv_cache_replicas → build_process_group → NCCL → resume_kv_cache_replicas`。其中 `build_process_group`（`base.py` 内）按 `self.replicas` 集合**重建通信拓扑**。
- 若此刻 `add_replicas`/`remove_replicas`（`base.py:449/457`）并发改了 `self.replicas`，会导致 **NCCL 拓扑与实际副本集合不一致**——这正是 doc 17 记录的时序互斥问题的根源。

**蓝图挂法**：`ReplicaSyncGate` 是 M/ 侧纯 `asyncio.Lock` + epoch 租约，**放在持有 `checkpoint_manager` 的进程内**。`ScaleTransaction` 的 `bootstrap_and_publish` / `remove_effective_replica` / `wrap_native_sync` / `restore_donor` 都在门内执行，门内的原生调用点分别是：
- `bootstrap_and_publish` → `CheckpointEngineManager.update_weights`（单副本语义，蓝图扩展）+ `GlobalRequestLoadBalancer.add_servers`（`router.py:227`）
- `remove_effective_replica` → `CheckpointEngineManager.remove_replicas`（`base.py:457`）+ `GlobalRequestLoadBalancer.remove_servers`（`router.py:242`）
- `wrap_native_sync` → `CheckpointEngineManager.update_weights`（`base.py:505`）

### 3.2 ADD（materialize → bootstrap → CE effective → routable）

原生链路（三步，零散无事务）：

| 步 | 蓝图状态 | 原生方法 | 行号 | 原生逻辑 |
| --- | --- | --- | --- | --- |
| 1 | MATERIALIZING | `RolloutReplica.init_standalone` | `replica.py:189` | 在 rollout GPU 上拉起 vLLM 服务进程 |
| 2 | CE_EFFECTIVE | `CheckpointEngineManager.add_replicas` | `base.py:449` | `self.replicas.extend(replicas)`，仅改列表 |
| 3 | ROUTABLE | `GlobalRequestLoadBalancer.add_servers` | `router.py:227` | 把 `server_id → ActorHandle` 加入路由表 |

**蓝图要补的两件事**：
1. **BOOTSTRAP**（步骤 2 与 3 之间）：新副本的权重必须灌到位才能接收请求。原生 `update_weights` 是**全量 abort + rebuild**，不能直接用于「只给新副本灌一次」。蓝图需在 `CheckpointEngineManager` 上扩展 per-replica bootstrap（或 `naive` backend 单点 sync）。
2. **wake**：`RolloutReplica.wake_up`（`replica.py:265`）→ `vLLMHttpServer.wake_up`（`vllm_async_server.py:833`）。⚠️ **standalone 下是 no-op**（`:852`），见 §6 风险 1。

### 3.3 REMOVE（drain → CE remove → LB remove → sleep/destroy）

原生链路：

| 步 | 蓝图状态 | 原生方法 | 行号 | 原生逻辑 |
| --- | --- | --- | --- | --- |
| 1 | DRAINING | `GlobalRequestLoadBalancer.get_inflight_count` | `router.py:255` | 返回该 server 当前 in-flight 数；**无 `begin_drain`** |
| 2 | CE_REMOVED | `CheckpointEngineManager.remove_replicas` | `base.py:457` | 从 `self.replicas` 列表过滤掉 |
| 3 | DESTROYED | `GlobalRequestLoadBalancer.remove_servers` | `router.py:242` | 从路由表直接删 |
| 4 | （睡眠） | `RolloutReplica.sleep` | `replica.py:269` | → `vLLMHttpServer.sleep`，standalone 下 no-op（`:863`） |

**蓝图要补的**：原生 `remove_servers` **立即删除**，不等待 in-flight 归零。蓝图的 `begin_drain` + `wait_inflight_zero` 必须在 M/ 侧 LB 封装里实现（轮询 `get_inflight_count`（`router.py:255`）或 `get_total_inflight`（`router.py:296`）直到 0，再调 `remove_servers`）。

### 3.4 NATIVE SYNC（周期权重同步）

唯一触发点：`FullyAsyncTrainer._fit_update_weights`（`trainer.py:690`）。

原生内部逻辑（`trainer.py:690-788`）：
1. `if self.local_trigger_step != 1: return None`（`:700`）——**只有 local_trigger_step==1 才真正同步**，否则是 no-op。
2. 若非 `only_hybrid`，调 `self.checkpoint_manager.update_weights(global_steps=...)`（`:733`）→ NCCL 广播到 standalone 副本。
3. 调 `self.rollouter.reset_staleness.remote()`（`:764`）重置 staleness。

**蓝图挂法**：`wrap_native_sync` 在门内包住整个 `checkpoint_manager.update_weights`（`base.py:505`）。**必须对齐 `local_trigger_step` 节律**——即门的 acquire/release 要和原生「step==1 才同步」的判据一致，否则门内可能包住一个 no-op。

### 3.5 RESTORE（donor / borrower 归还）

原生**无 RESTORE 语义**。蓝图把 RESTORE 定义为 ADD 链路的复用：
- **donor 归还**：`bootstrap → CE add_effective → wake → ROUTABLE`，但走 `sleep → mark_dormant`（**不 destroy**，保留进程/内存）。
- **borrower 归还**：`destroy_temporary`（真正销毁临时副本）。

挂点与 ADD 完全一致（§3.2），差别只在「销毁 or 休眠」的收尾分支。

---

## 4. 逐模块详解（V/ 侧，含精确行号）

> 行号以本地 `f92febf5` 为准。每个模块给「类/方法/行号/原生逻辑/蓝图如何挂」五要素。

### 4.1 `V/experimental/fully_async_policy/fully_async_main.py` — 任务驱动器

| 符号 | 行号 | 说明 |
| --- | --- | --- |
| `class FullyAsyncTaskRunner` | `:36` | `@ray.remote(num_cpus=1)`，串行编排 Actor |
| `run(config)` | `:46` | 阻塞主循环，调用 `_initialize_components → _create_* → _run_training_loop` |
| `_initialize_components(config)` | `:51` | 初始化 tokenizer/processor/worker group class |
| `_create_rollouter(config)` | `:117` | 创建 Rollouter Actor |
| `_create_trainer(config)` | `:138` | 创建 Trainer Actor |
| `_setup_hybrid_worker_group(config)` | `:159` | 混合 worker group |
| `_run_training_loop()` | `:186` | 驱动 trainer.fit / rollouter.fit 的主循环 |
| `run_ppo(...)` 调用 | `:238` | `run_ppo(config, task_runner_class=FullyAsyncTaskRunner)` |

**蓝图关系**：这是编排的「串行根」。`G_standalone` 门若在此 Actor 内创建，则全链无并发；若在 Trainer 内创建（因为 `checkpoint_manager` 在 Trainer），则需确认 Trainer 侧没有第二个并发入口。**推荐：门放 Trainer 进程**（与 `checkpoint_manager` 同进程，锁内可直接操作 `self.replicas`），因为门的保护对象是 `CheckpointEngineManager`。

### 4.2 `fully_async_trainer.py` — 训练器（权重同步 + CE 宿主）

| 符号 | 行号 | 原生逻辑 / 蓝图挂点 |
| --- | --- | --- |
| `class FullyAsyncTrainer(SeparateRayPPOTrainer)` | `:54` | 训练侧 Actor |
| `self.local_trigger_step = 1` | `:140` | 参数同步的节律计数器 |
| `_setup_checkpoint_manager()` | `:217` | **从 `rollouter.get_replicas.remote()` 拉副本列表**，`CheckpointEngineManager(config, actor_wg, replicas)` 实例化。→ 蓝图 ADD/REMOVE 都会改这个 `replicas` 集合 |
| `_setup_hybrid_checkpoint_manager()` | `:226` | 混合（trainer 侧）副本的 naive CE，**与 standalone 门无关**（dynamic scheduling 专属） |
| `_setup_dynamic_resource_controller()` | `:302` | 动态调度控制器，`only_hybrid` 由 standalone 副本数推导（`:319`） |
| `set_rollouter(rollouter)` | `:346` | 建立 trainer→rollouter 引用 |
| `init_workers()` | `:487` | 初始化 worker |
| `fit()` | `:498` | 训练主循环 |
| `fit_step()` | `:536` | 单步 |
| `_fit_generate()` | `:643` | 生成 |
| `_fit_update_local_step()` | `:676` | `local_trigger_step` 自增/归 1（`:684-688`） |
| `_fit_update_weights()` | `:690` | **唯一权重同步点**。step!=1 时 no-op（`:700`）；非 only_hybrid 时 NCCL 广播（`:733`）；末尾 `reset_staleness`（`:764`） |
| `_fit_validate()` | `:812` | 校验 |

**蓝图关系**：
- `ScaleTransaction.native_sync` = 门包 `checkpoint_manager.update_weights`，触发节律对齐 `_fit_update_weights` 的 `local_trigger_step==1`。
- `ScaleTransaction.add/remove` 的 CE 步骤 = 直接调 `self.checkpoint_manager.add_replicas/remove_replicas`（`base.py:449/457`）。

### 4.3 `fully_async_rollouter.py` — 采样子系统 + LLMServerManager 宿主

**两个类**（注意区分）：

| 符号 | 行号 | 说明 |
| --- | --- | --- |
| `class FullyAsyncLLMServerManager(LLMServerManager)` | `:54` | 副本管理器（在 Rollouter 进程内） |
| `_initialize_llm_servers(start_rank=0)` | `:78` | 两阶段初始化（hybrid + standalone），standalone 走 `init_standalone` |
| `add_replicas(resource_ids)` | `:144` | **hybrid 副本激活**（`hybrid_replicas`/`alive_replicas`/`alive_addresses`），调 `global_load_balancer.add_servers.remote`。→ **dynamic scheduling 专属，非 standalone 门** |
| `remove_replicas(resource_ids)` | `:206` | hybrid 副本下线 |
| `get_standalone_replicas()` | `:284` | 返回 standalone-only 副本 |
| `class FullyAsyncAgentLoopManager` | `:305` | agent loop |
| `class FullyAsyncRollouter(SeparateRayPPOTrainer)` | `:330` | `@ray.remote(num_cpus=10, max_concurrency=100)`，高并发 |
| `max_concurrent_samples = None` | `:427` | 并发样本容量 |
| `staleness_samples = 0` | `:433` | 陈旧样本计数 |
| `set_max_required_samples()` | `:491` | 计算 `max_concurrent_samples` |
| `get_replicas()` | `:526` | 返回副本列表（**Trainer 的 `_setup_checkpoint_manager` 依赖此 RPC**） |
| `reset_staleness()` | `:594` | 参数变更后重置 `staleness_samples`（`:605`） |
| `_init_async_rollout_manager()` | `:819` | 初始化异步 rollout manager |
| `_feed_samples()` / `_processor_worker()` | `:859` / `:902` | 样本流 |
| `fit()` | `:1076` | 采样主循环，`staleness >= max_required` 时等（`:1155`） |
| `add_replicas(resource_ids)` | `:1199` | **standalone 副本增加**（薄封装 + `_update_max_concurrent_samples`） |
| `remove_replicas(resource_ids)` | `:1205` | standalone 副本移除 |
| `rebalance_requests()` | `:1211` | 请求再均衡 |
| `_update_max_concurrent_samples()` | `:1274` | 按活跃副本数重算 `max_concurrent_samples` |
| `get_all_hybrid_replicas()` | `:1300` | 返回 hybrid 副本字典 |

**蓝图关系**：
- 蓝图的 **effective_replicas** 概念最终影响 `max_concurrent_samples`（`_update_max_concurrent_samples`：`:1274`）与 `reset_staleness`（`:594`）。
- `get_replicas()`（`:526`）是 Trainer 与 Rollouter 之间「副本对象经 Ray RPC 序列化传递」的唯一桥梁——蓝图 replica 状态必须**单一权威**，否则两份视图漂移（见 §6 风险 4）。

### 4.4 `workers/rollout/llm_server.py` — LLM server manager 基类

| 符号 | 行号 | 原生逻辑 |
| --- | --- | --- |
| `class LLMServerClient` | `:43` | 客户端（rollouter 侧调用入口） |
| `class LLMServerManager` | `:346` | 副本管理器基类 |
| `create(cls, *args)` | `:392` | 工厂：`_initialize_llm_servers` 后 `_init_global_load_balancer` |
| `_initialize_llm_servers(start_rank)` | `:399` | 创建 rollout 副本，生成 `server_handles`（`:460`）/`server_addresses`（`:461`） |
| `_init_global_load_balancer()` | `:481` | `get_router_handle(servers=dict(zip(addresses, handles)))` 建 LB Actor |
| `get_client(client_cls)` | `:491` | 返回客户端 |
| `get_server_addresses()` | `:511` | 地址列表 |
| `get_replicas()` | `:513` | 副本列表 |

**蓝图关系**：这是 `FullyAsyncLLMServerManager`（`rollouter.py:54`）的父类。蓝图的 LB 封装（`LoadBalancerOps`）最终指向这里建出的 `global_load_balancer` Actor（`router.py:133`）。

### 4.5 `workers/rollout/replica.py` — RolloutReplica 抽象

| 符号 | 行号 | 原生逻辑 / 蓝图挂点 |
| --- | --- | --- |
| `class RolloutReplica(ABC)` | `:70` | 副本抽象基类 |
| `init_hybrid(worker_group)` | `:131` | hybrid 初始化 |
| `init_hybrid_colocated(...)` | `:143` | hybrid 共置 |
| `init_colocated(resource_pool)` | `:160` | colocated 初始化 |
| `init_standalone()` | `:189` | **standalone 初始化**（蓝图 ADD 的 MATERIALIZING 步） |
| `get_ray_class_with_init_args()` | `:228` | 返回 `RayClassWithInitArgs`（replica 对象经 Ray 序列化的形态） |
| `launch_servers()` | `:242` | 拉起服务 |
| `server_address` / `server_handle`（property） | `:247` / `:252` | 地址 / 句柄 |
| `max_concurrency`（property） | `:257` | 并发容量 |
| `wake_up()` | `:265` | → 底层 `vLLMHttpServer.wake_up`（standalone no-op） |
| `sleep()` | `:269` | → 底层 `vLLMHttpServer.sleep`（standalone no-op） |
| `abort_all_requests()` | `:273` | 中止所有 in-flight 请求（**权重同步前置步**，`base.py:477` 会调用） |
| `resume_generation()` | `:277` | 中止后恢复 |
| `clear_kv_cache()` / `release_kv_cache()` / `resume_kv_cache()` | `:281` / `:285` / `:289` | kv-cache 管理（`update_weights` 的 `release/resume_kv_cache_replicas` 落点） |
| `class RolloutReplicaRegistry` | `:302` | 副本类注册表 |

**蓝图关系**：
- ADD 的 materialize = `init_standalone`（`:189`）。
- REMOVE 的 sleep = `sleep`（`:269`），**但 standalone no-op**（风险 1）。
- NATIVE SYNC 的 abort = `abort_all_requests`（`:273`，由 `base.py:477` 的 `abort_replicas` 批量调用）。

### 4.6 `workers/rollout/router.py` — GlobalRequestLoadBalancer

| 符号 | 行号 | 原生逻辑 / 蓝图挂点 |
| --- | --- | --- |
| `class RequestLoadBalancer(Protocol)` | `:32` | 路由协议（`acquire_server`/`add_servers`/`remove_servers`/`get_total_inflight` 等接口定义） |
| `class GlobalRequestLoadBalancer` | `:133` | 具体实现（独立 Ray Actor） |
| `acquire_server(request_id)` | `:170` | 分配请求到 server |
| `release_server(server_id, request_id)` | `:206` | 释放 |
| `add_servers(servers)` | `:227` | **加入路由表**（蓝图 ADD 的 ROUTABLE 步） |
| `remove_servers(server_ids)` | `:242` | **从路由表删除**（蓝图 REMOVE 的 LB 步；⚠️ 立即删，无 drain） |
| `get_inflight_count(server_id)` | `:255` | **单 server 当前 in-flight 数**（蓝图 DRAIN 等待的轮询对象） |
| `get_all_servers()` | `:259` | 全部 server |
| `clear_sticky_cache()` | `:263` | 清 sticky 缓存 |
| `get_status()` | `:287` | 状态快照 |
| `get_total_inflight()` | `:296` | **总 in-flight 数** |

**蓝图关系**：蓝图的 `LoadBalancerOps.begin_drain` / `wait_inflight_zero` / `commit_routable_idempotently` / `finish_remove` 全部指向这个 Actor 的方法。核心缺口：原生**无 `begin_drain`**，蓝图必须在 M/ 侧 LB 封装里加「停止新分配」开关，并用 `get_inflight_count`（`:255`）轮询归零。

### 4.7 `workers/rollout/vllm_rollout/vllm_async_server.py` — vLLM 推理服务

| 符号 | 行号 | 原生逻辑 |
| --- | --- | --- |
| `class vLLMHttpServer` | `:87` | vLLM HTTP 服务 Actor |
| `__init__` | `:94` | 初始化（`enable_sleep_mode` 判定在 `:297-298`） |
| `wake_up(tags)` | `:833` | `node_rank!=0` 直接 return；HYBRID→`engine.wake_up`+`reset_prefix_cache`；COLOCATED→`engine.wake_up`+`reset_prefix_cache`；**STANDALONE→`logger.info("skip wake_up in standalone mode")`**（`:852`） |
| `sleep()` | `:854` | `node_rank!=0 or not free_cache_engine` 直接 return；HYBRID→`_sleep_hybrid`；COLOCATED→`engine.sleep(level=1)`；**STANDALONE→`logger.info("skip sleep in standalone mode")`**（`:863`） |
| `wait_for_requests_to_drain()` | `:929` | 等待请求排空 |
| `abort_all_requests(reset_prefix_cache=True)` | `:932` | 中止所有请求 |

**蓝图关系（关键陷阱）**：standalone 副本的 `sleep`/`wake_up` **是 no-op**。这直接推翻了「蓝图 REMOVE 的 sleep 会释放内存」的假设——standalone 下真正释放/恢复内存的是 **NCCL 权重同步路径**（`base.py:505` 里的 `release_kv_cache_replicas`/`resume_kv_cache_replicas`），不是 `sleep`。蓝图的 `finish_remove_and_destroy` 里 `sleep + mark_dormant` 对 standalone 只是**逻辑标记**，不产生物理释放。

### 4.8 `checkpoint_engine/base.py` — CE worker + manager（权重同步核心）

**两个类**：

| 符号 | 行号 | 原生逻辑 / 蓝图挂点 |
| --- | --- | --- |
| `class CheckpointEngineWorker(Worker)` | `:304` | 每个 rollout 进程内 colocated 的收端 worker；`server_adapter: BaseRollout`（`:315`） |
| `update_weights(global_steps)` | `:354` | `checkpoint_engine.receive_weights` → `server_adapter.update_weights`（NCCL 收端落地权重） |
| `class CheckpointEngineManager` | `:380` | Trainer 进程内的协调器，持有 `replicas` 列表 |
| `__init__(config, actor_wg, replicas)` | `:398` | 持有 actor worker group + replica 列表 |
| `build_process_group(rollout)` | `:411` | **按 `self.replicas` 重建 NCCL 拓扑**（`prepare`→`build_topology`→`init_process_group`） |
| `add_replicas(replicas)` | `:449` | `self.replicas.extend(replicas)`（蓝图 ADD 的 CE 步；仅改列表，**下次 update_weights 才重建拓扑**） |
| `remove_replicas(replicas)` | `:457` | 从列表过滤（蓝图 REMOVE 的 CE 步） |
| `sleep_replicas()` | `:467` | `asyncio.gather(*[r.sleep() for r in self.replicas])` → standalone no-op |
| `wake_up_replicas()` | `:472` | `r.wake_up()` → standalone no-op |
| `abort_replicas()` | `:477` | `r.abort_all_requests()`（权重同步前置） |
| `resume_generation_replicas()` | `:482` | `r.resume_generation()` |
| `release_kv_cache_replicas()` / `resume_kv_cache_replicas()` | `:486` / `:491` | NCCL 同步前后释放/恢复 kv-cache（**standalone 真正内存管理的落点**） |
| `update_weights(global_steps)` | `:505` | **核心**：naive→直接 `actor_wg.update_weights`；否则 `abort_replicas → 建临时 RayWorkerGroup → release_kv_cache → build_process_group → NCCL → resume_kv_cache` |

**蓝图关系（最重要的一节）**：
1. `update_weights`（`:505`）是**全量、阻塞、重建拓扑**的原子块，蓝图 NATIVE SYNC 就是门包它。
2. `add_replicas`/`remove_replicas`（`:449`/`:457`）只改列表，**拓扑在下次 `update_weights` 才生效**——这就是「CE effective」与「实际 NCCL 拓扑」之间的时间窗，doc 17 的互斥问题即源于此。
3. 蓝图 **BOOTSTRAP 不能复用** `update_weights`（它是全量 abort），必须在 M/ 侧扩展 per-replica 增量灌权重（或 `naive` 单点 sync）。

### 4.9 `message_queue.py` — 样本队列

| 符号 | 行号 | 原生逻辑 |
| --- | --- | --- |
| `class MessageQueue` | `:27` | `deque(maxlen=max_queue_size)` 有界队列 |
| `put_sample(sample)` | `:55` | 入队（满则丢最旧） |
| `get_sample()` | `:85` | 出队 |
| `clear_queue()` | `:121` | 清空 |
| `class MessageQueueClient` | `:180` | 跨 Actor 客户端 |

**蓝图关系**：间接。副本增删影响 `max_concurrent_samples` → `staleness_samples`（`rollouter.py:594` 的 `len(active_tasks) + message_queue_client.get_queue_size()`）→ 队列消费节奏。蓝图不直接改队列，但 REMOVE 的 drain 期间队列可能在积压，需在时序上确认。

### 4.10 `trainer/main_ppo.py` — run_ppo 入口

| 符号 | 行号 | 说明 |
| --- | --- | --- |
| `run_ppo(config, task_runner_class)` | `:34` | 入口：`task_runner_class.remote()` 创建 TaskRunner |
| `class TaskRunnerV1` | `:103` | V1 版 TaskRunner |
| `TaskRunnerV1.run` | `:133` | `run_ppo(config, task_runner_class=TaskRunnerV1)`（`:184`） |
| 默认分支 | `:192` | `run_ppo(config, task_runner_class=TaskRunner)` |

**蓝图关系**：这是「接线点」。P1 已通过 `resolve_runtime_profile` 让 `fully_async_main.py:238` 的 `task_runner_class` 可选为 `MultiTaskFullyAsyncTaskRunner`。蓝图的 `G_standalone` 门若放 TaskRunner，则在这里的 `run()`（`fully_async_main.py:46`）里创建。

### 4.11 `single_controller/ray/base.py` — replica 对象序列化

| 符号 | 行号 | 说明 |
| --- | --- | --- |
| `_unwrap_ray_remote(cls)` | `:964` | 解开 `@ray.remote` 包装，还原原始类 |
| `create_colocated_worker_cls(class_dict)` | `:984` | 用 `_unwrap_ray_remote` 还原类（`:1009`/`:1020`） |

**蓝图关系**：`RolloutReplica` 对象经 Ray RPC 在 Trainer/Rollouter 之间传递时，最终以 `RayClassWithInitArgs` 序列化（`replica.py:228`）。蓝图在 M/ 侧 `ReplicaRecord` 里若持有 replica 引用，必须通过 `get_replicas()`（`rollouter.py:526`）取回的**反序列化副本**，而非进程本地对象——这是「单一权威」要求的来源。

---

## 5. 蓝图 M/ 侧 5 模块 ↔ verl 挂点对照表

| M/ 模块 | 关键方法 | 直接调用的 V/ 方法（行号） |
| --- | --- | --- |
| `replica_sync_gate.py` | `acquire` / `release` / `guard` | 无（纯 `asyncio.Lock` + epoch 租约，不触 V/） |
| `replica_record.py` | `transition_to` | 无（纯状态机，状态映射到 V/ 副本对象的生命周期，但本身不调用） |
| `operation_journal.py` | `begin` / `transition` | 无（纯状态机 + 重放，不触 V/） |
| `receipts.py` | （frozen dataclasses） | 无（数据载体，`BootstrapReceipt.weight_version` 对应 `update_weights(global_steps)` 的 `global_steps`） |
| `scale_transaction.py` | `prepare_replica` | `RolloutReplica.init_standalone`（`replica.py:189`） |
| | `bootstrap_and_publish` | CE per-replica bootstrap（蓝图扩展）+ `GlobalRequestLoadBalancer.add_servers`（`router.py:227`） |
| | `begin_drain` / `wait_inflight_zero` | `GlobalRequestLoadBalancer.get_inflight_count`（`router.py:255`） |
| | `remove_effective_replica` | `CheckpointEngineManager.remove_replicas`（`base.py:457`）+ `GlobalRequestLoadBalancer.remove_servers`（`router.py:242`） |
| | `finish_remove_and_destroy` | `RolloutReplica.sleep`（`replica.py:269`，standalone no-op） |
| | `wrap_native_sync` | `CheckpointEngineManager.update_weights`（`base.py:505`） |
| | `restore_donor` | CE bootstrap + `add_replicas`（`base.py:449`）+ `wake_up`（`replica.py:265`） |

> 说明：蓝图的 4 个 `*Ops` 协议（`RuntimeOps`/`CheckpointOps`/`LoadBalancerOps`/`CapacityOps`）是 M/ 侧的**测试替身边界**——真实实现里它们分别薄封装上表右列的 V/ 方法；单测里用 mock 替身替换，从而不 import verl / 不起 GPU / 不起 Ray。

---

## 6. 关键风险点 / 原生语义陷阱（逐条，含行号）

1. **standalone `sleep`/`wake_up` 是 no-op**（`vllm_async_server.py:852`/`:863`）
   → 蓝图 REMOVE 的「sleep 释放内存」在 standalone 下不成立。真正释放内存的是 NCCL 同步路径 `release_kv_cache_replicas`（`base.py:486`）。`finish_remove_and_destroy` 里 `sleep + mark_dormant` 只是逻辑标记，物理释放需另找路径（或明确接受「dormant 仍占内存」）。

2. **`update_weights` 是全量 abort + 重建拓扑，不是增量**（`base.py:505`）
   → 蓝图 BOOTSTRAP 不能复用 `update_weights`，否则新副本 bootstrap 会 abort 所有副本并重建全量拓扑。必须单开 per-replica 路径。

3. **LB 无 `begin_drain`，`remove_servers` 立即删**（`router.py:242`）
   → 蓝图 DRAINING 语义必须在 M/ 侧实现「停新请求 + 等 in-flight=0」（轮询 `get_inflight_count`：`router.py:255`），再删。原生无此门。

4. **Trainer 与 Rollouter 是两个 Actor，副本对象经 Ray RPC 序列化传递**（`get_replicas`：`rollouter.py:526` → `_setup_checkpoint_manager`：`trainer.py:217`）
   → 蓝图 replica 状态必须**单一权威**，否则 Trainer 侧 `checkpoint_manager.replicas` 与 Rollouter 侧 `llm_server_manager.rollout_replicas` 漂移。推荐：以 `CheckpointEngineManager.replicas` 为 CE 权威、以 LB Actor 为路由权威，`ReplicaRecord` 只做影子状态。

5. **门放哪一侧：`FullyAsyncRollouter` 是高并发 Actor（`num_cpus=10, max_concurrency=100`），`FullyAsyncTaskRunner` 是串行 Actor（`num_cpus=1`）**
   → `G_standalone` 必须放在**串行侧**（Trainer 或 TaskRunner，持有 `checkpoint_manager`），不能放 Rollouter。否则门内操作会被高并发打穿。

6. **`_fit_update_weights` 只在 `local_trigger_step==1` 真正同步**（`trainer.py:700`）
   → 蓝图 `wrap_native_sync` 的门 acquire/release 必须对齐该节律，否则门会包住 no-op；门内 `update_weights` 的 `global_steps` 也要与 `current_param_version` 一致。

7. **CE 拓扑在下次 `update_weights` 才生效**（`add_replicas` 只 `extend`，`base.py:449`；拓扑重建在 `build_process_group`，`base.py:411`，由 `update_weights` 触发）
   → 「CE effective」与「NCCL 拓扑已含新副本」之间有窗口。若新副本已 `add_servers`（routable）但拓扑未重建，会有请求打到「权重未同步」的副本上。这正是 doc 17 的互斥核心，也是 `G_standalone` 门要串行化 ADD 与 NATIVE SYNC 的直接原因。

---

## 7. 结论 / 对实现计划的影响

1. **verl 原生「有资源对象、有即时增删、有周期同步」，但缺「门 / 状态机 / drain / 单副本 bootstrap」四件** → 5.3 编排核心的四件套（`ReplicaSyncGate` / `OperationJournal` / `ReplicaRecord` / `ScaleTransaction`）落在 M/ 侧，只做**薄封装 + 状态机**，不重写 V/ 原生逻辑，符合「不改 verl 原类」约束。

2. **门的物理位置**：放持有 `CheckpointEngineManager` 的 Trainer 进程（串行侧），保护对象是 `checkpoint_manager.replicas` 集合 + LB Actor 路由表。

3. **两个必须 M/ 侧扩展的原生缺口**：①单副本 BOOTSTRAP（不能复用全量 `update_weights`）；②`begin_drain` + `wait_inflight_zero`（原生 `remove_servers` 立即删）。

4. **三个原生语义陷阱要在实现计划里显式测试**：standalone sleep/wake no-op、CE 拓扑延迟生效、`local_trigger_step` 节律门控。

---

*（本文为文档化梳理，未改动任何代码。行号以本地 `verl` HEAD `f92febf5` 为准；接线版本 `a9ebd0bb` 在本文涉及的模块上结构一致，但 `checkpoint_engine/base.py` 的 ServerAdapter 命名已并入 `CheckpointEngineWorker.server_adapter` 字段。）*
