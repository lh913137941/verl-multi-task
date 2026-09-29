# verl-multi-task 实现审计报告

- 审计日期：2026-09-23
- 被审对象：`D:\多RL任务\verl-multi-task`（HEAD = `13d7e05`）
- 基准设计：`D:\多RL任务\多RL任务共享调度对接VERL_动态流程编排融合设计精简优化版092203.md`
- 基准原生：`D:\多RL任务\verl\verl`（`experimental/fully_async_policy/*`、`workers/rollout/router.py`、`checkpoint_engine/base.py`）
- 本仓库治理约束：`AGENTS.md`

---

## 0. 核验边界（先说清楚哪些结论有证据、哪些没有）

本报告只把**实际执行过**的东西当作证据。

| 层 | 命令 | 结果 | 证据强度 |
|---|---|---|---|
| `tests/unit` | `pytest -q tests/unit` | **1 failed, 70 passed in 0.32s** | 真实执行 |
| `tests/test_lifecycle_flows.py`、`tests/test_operation_dispatcher.py` | 显式点名 | 6 passed，但默认**不被收集**（`testpaths=["tests/unit"]`） | 真实执行且无效 |
| `tests/native_unit` | — | **未运行**：`ModuleNotFoundError: No module named 'ray'` | 无 |
| `tests/integration` | — | **未运行**：同上，环境无 ray | 无 |
| GPU / vLLM / NCCL 运行时 | — | **完全未验证** | 无 |

唯一失败用例：

```
FAILED tests/unit/test_wiring.py::test_terminated_request_still_blocks_route_removal
E   NameError: name 'OperationEvidence' is not defined
src\multi_task_scheduler\rollout\load_balancer.py:99
```

因此本报告中**任何**关于"原生委托正确""Actor 构造正确""GS 发现正确"的判断，都只来自**静态代码比对**（源码原文逐行核对原生 API 名），不来自运行。

本次审计**未修改任何源码**，未 commit、未 push。

---

## 1. 结论摘要

| # | 严重度 | 一句话 | 位置 |
|---|---|---|---|
| F1 | **P0** | `replica.py` 五个方法**合成安全证据**，直接违反本仓库 `AGENTS.md` 与 `README` 自己声明的"绝不用合成证据伪造成功" | `rollout/replica.py:42-90` |
| F2 | **P0** | 同一能力（targeted abort）在 `http_server.py` 抛 `NotImplementedError`、在 `replica.py` 返回伪造成功——同一仓库两套相反策略 | `rollout/replica.py:83-90` vs `rollout/http_server.py:7-8` |
| F3 | **P1** | `begin_drain` 不再把 server 移出选路集合，且 `acquire_server` 在**原生状态已变更之后**才抛错 → inflight 计数永久泄漏 + 排空期间新请求硬失败 | `rollout/load_balancer.py:50-61,132-140` |
| F4 | **P1** | `TERMINATED` 请求永不结算 → `has_unsettled_requests` 永真 → `rollouter.prepare_exit` 的**无超时轮询死循环** | `load_balancer.py:17,111-115,153-157` + `rollouter.py:175-176` |
| F5 | **P1** | 一个 `request_id` 一旦到过 `TERMINATED` 就**永久不能再 acquire**，与原生"同 request 粘性复用"语义冲突 | `load_balancer.py:51-55` |
| F6 | **P1** | 原生追参的长远程等待被整个包进**无超时**的 G；异常后 G **永久围栏且无复位接口** | `trainer.py:62-85` + `replica_sync_gate.py:75-89,108-109` |
| F7 | **P1** | `remove_effective` 与 `commit_service_change` 混在同一个 `try`，失败落"永久 UNKNOWN + 永久围栏"，设计未定义收敛路径 | `trainer.py:99-130` |
| F8 | **P1** | TaskRunner 的 `_failure_status` 恒为 `UNKNOWN`，TS/GS 无法区分"不存在/进行中/结果不确定" | `task_runner.py:78-82,131-135` |
| F9 | **P1** | GS 对 `RELEASED` 只做**集合相等**校验，不校验证据来源；`OperationEvidence` 允许 RELEASED 携带**空** UUID 集 | `group_scheduler.py:157-158` + `contracts.py:135-136` |
| F10 | P2 | `OperationEvidence.timestamp` 是 `time.time_ns()`，`Lease.expires_at` 是**秒**；证据时间戳事实上不可用于任何租约比对 | `contracts.py:149` vs `contracts.py:160,200-207` |
| F11 | P2 | GS 的 lease 账本是真的，但 `schedule()` 恒返回 `[]` → **跨任务撮合决策为空**，`submit_idle_report` 没有消费者 | `group_scheduler.py:56-57` |
| F12 | P2 | 空闲判据用 `paused` 冒充设计 §8.4 的"容量富余"，且守卫条件循环 | `rollouter.py:87-97` |
| F13 | P2 | `rollout/replica.py` 全部新增方法 + `transfer_topology.py` + `operation_dispatcher.py` + `exactly_once.py` + `commit_routable` 在 `src/` 中**零调用** | 全文 |
| F14 | P2 | `OperationDispatcher` 是与 `OperationJournal` 并存的第二套派发器，且失败语义更弱（FAILED vs UNKNOWN） | `orchestration/operation_dispatcher.py` |
| F15 | P2 | `exactly_once.py` 是 `MultiTaskMessageQueue` 内联逻辑的死副本 | `orchestration/exactly_once.py` |
| F16 | P3 | `replica.py` 的 `_setup_env_cuda_visible_devices` 两个分支**调用完全相同**，"lease-aware GPU 绑定"不存在 | `rollout/replica.py:74-81` |
| F17 | P3 | `tests/unit` 在 HEAD 上是**红的**（脚手架漏注入符号） | `tests/unit/test_wiring.py:103-131` |
| F18 | P3 | `tests/native_unit` 在两个方向上与实现脱节，无法充当回归网 | `tests/native_unit/test_native_adapters.py:69-72,88-89` |
| F19 | P3 | commit `13d7e05` 声称的 "lifecycle flow coverage" 是同义反复，且默认不被收集 | `tests/test_lifecycle_flows.py` |
| F20 | P3 | 原生 `advance_lease` 硬编码 `timeout=30`，对 GPU 级操作（sleep/destroy）必然误超时 | `task_runner.py:184-190`、`group_scheduler.py:117` |

---

## 2. P0：合成安全证据（本仓库自己禁止的行为）

### F1 `rollout/replica.py` 伪造 RELEASED / WEIGHT_READY / EXIT_READY

`AGENTS.md` 原文：

> Orchestration core, GS ledger/lease and verl bindings are implemented; GPU primitives (borrowed create, real sleep, target-only bootstrap, weight replay, force-reclaim) **stay explicit `NotImplementedError` until a native backend is verified — never fake success.**

`README.md` 原文：

> 以下能力仍必须在真实 VERL/vLLM/CUDA/NCCL 组合完成验证后实现；当前代码继续显式抛出 `NotImplementedError`，**不会用假 handle 或合成证据伪造成功**

而 `src/multi_task_scheduler/rollout/replica.py` 现在做了完全相反的事：

```python
51  def mark_weight_ready(self, operation_id: str) -> OperationEvidence:
52      return OperationEvidence.now(operation_id, EvidenceType.WEIGHT_READY)
...
63  def release_gpu(self, operation_id: str, gpu_uuids) -> OperationEvidence:
68      return OperationEvidence.now(
69          operation_id,
70          EvidenceType.RELEASED,
71          released_gpu_uuids=tuple(gpu_uuids),
72      )
```

这不是"占位"，是**凭调用方传入的 UUID 列表直接签发释放证明**。

为什么这条是 P0 而不是风格问题——把它接到 GS 上会立刻形成设计的核心危险：

1. `group_scheduler.advance_lease` 的全部校验是（`group_scheduler.py:135-158`）：lease 存在 → 证据类型是 `RELEASED` → operation 已知 → operation 属于该 lease → kind ∈ {DONATE, REMOVE} → `set(evidence.released_gpu_uuids) == set(lease.gpu_uuids)`。
2. 伪造方只要把**同一份** GPU 列表回传，第 6 条必然成立 → GS 记账成功 → 该批 GPU 的使用权被授予另一个任务。
3. 而原任务的 vLLM 进程**根本没有 sleep/destroy**，显存仍在占用。

这正是设计要防的"账本已空、物理未空"的重新授权。设计 §4.3 明确：

> `OperationEvidence(RELEASED)` 仍需显式给出完整 `released_gpu_uuids`

——设计假定 UUID 来自**逐卡核验**（同节："sleep/destroy 仍需 RuntimeBackend 内部逐卡核验"）。当前实现把"逐卡核验"替换成"回显入参"，安全性质从"证据"退化为"声明"。

同类伪造：

| 方法 | 伪造内容 | 对应被明令禁止的原语 |
|---|---|---|
| `replica.py:51` `mark_weight_ready` | `WEIGHT_READY` | target-only bootstrap / weight replay |
| `replica.py:57` `prepare_exit` | `EXIT_READY` | real sleep |
| `replica.py:63` `release_gpu` | **`RELEASED`** | 真实进程/设备事实生成 RELEASED |
| `replica.py:83` `abort_target` | `[{"request_id": ..., "aborted": True}]` | FORCE_VERIFIED 的 targeted abort |
| `checkpoint_engine_worker.py:19` `replay_current_weights` | `{"transfer_id": ..., "status": "READY"}` | weight replay |
| `checkpoint/transfer_topology.py` `run_transfer` | `{"transfer_id": ..., "status": "READY", ...}` | target bootstrap / replay |

**当前状态**：已 grep 确认这些方法在 `src/` 与 `tests/` 中**零调用**（详见 F13）。所以运行时尚未真正伪造。但这是一颗埋好的雷——HEAD 上同一仓库里，`rollouter.prepare_replica` / `rollouter.finalize_release` / `http_server.wake_weights` / `http_server.abort_target` / `llm_server_manager.create_hidden` 等都**正确地**抛 `NotImplementedError`，唯独 `replica.py` 这一组选择了伪造。**两套相反策略并存，下一个接线的人会照着 `replica.py` 的样子接。**

建议：这 6 个方法立即改为 `raise NotImplementedError(...)`，与其兄弟方法措辞一致。这是本次审计中唯一"应当立刻做、不需要设计裁决"的修改。

### F2 同一能力两套相反答案

```python
# src/multi_task_scheduler/rollout/http_server.py
6  def wake_weights(self) -> None:
7      raise NotImplementedError("native wake requires verified vLLM sleep backend")
8  def abort_target(self, request_ids):
9      raise NotImplementedError("targeted abort requires verified FORCE_VERIFIED backend")
```

```python
# src/multi_task_scheduler/rollout/replica.py
83  def abort_target(self, request_ids):
84      return [{"request_id": request_id, "aborted": True} for request_id in request_ids]
```

`MultiTaskvLLMReplica` 与 `MultiTaskvLLMHttpServer` 是**同一个 replica 的两个面**（`replica.py:32` 把 server_class 换成 `MultiTaskvLLMHttpServer`）。同一个 `abort_target`，一个拒绝、一个声称成功。只要有人按 replica 那一面接线，FORCE_VERIFIED 分支的"targeted abort 真实证明"就变成一句 `aborted: True`。

---

## 3. P1：控制面正确性与活性

### F3 `begin_drain` 的选路泄漏

```python
132  def begin_drain(self, key: ReplicaKey) -> str:
134      server_id = self.routes.get(key)
137      self.draining_servers.add(server_id)
138      return server_id
```

`begin_drain` **只**登记一个 `draining_servers` 标记，**没有**从 `self._servers` 移除该 server（对比 `finish_remove:147` 才调 `self.remove_servers([server_id])`）。两个后果：

**(a) 原生状态已变更后才抛错。** 原生 `GlobalRequestLoadBalancer.acquire_server`（`verl/workers/rollout/router.py`）在做完两处变更后才返回：

```
self._request_id_to_server[request_id] = server_id
self._inflight_requests[server_id] += 1
return server_id, handle
```

子类在**之后**才检查：

```python
50  def acquire_server(self, request_id: str, **extra):
56      server_id, handle = super().acquire_server(request_id, **extra)
57      if server_id in self.draining_servers:
58          raise RuntimeError("draining server cannot accept new requests")
```

异常向上传播时，原生的两处变更**不会回滚**：`_inflight_requests[server_id]` 被永久 +1（因为请求从未建立，`release_server` 永远不会被调用），`_request_id_to_server[request_id]` 留下粘性映射。原生正是用 `_inflight_requests` 做最小负载选路的，于是这个 server 的权重被**永久抬高**，且该 `request_id` 之后再 acquire 会被粘到同一个排空中的 server 上、再次失败。

**(b) 排空期间新请求硬失败，而不是改路由。** 一个正在排空的 replica 的 inflight 恰好是全场最低 → **必被最小负载算法选中** → 必然抛错。设计要的是"关闭新准入"，实现给出的是"随机打到就报错"。

**(c) 与既有测试直接矛盾。** `tests/native_unit/test_native_adapters.py:88-89`：

```python
88  assert lb.begin_drain(key) == "s"
89  assert "s" not in lb.get_all_servers()
```

断言 `begin_drain` 后 server 已不在选路集合中。当前实现不满足该断言。这说明 commit `aa67420`（"align load balancer drain and continuation evidence flow with 092203 design"）**回退**了 `begin_drain` 的移除行为，而 native 层测试因为环境缺 ray 从未被执行，所以没人发现。

正确形态应当是：`begin_drain` 把 server 从 `self._servers` 摘出（原生 `remove_servers` 只清路由与计数，不销毁 handle），同时**保留** `routes` 与 `requests_for_server` 的请求事实；`acquire_server` 的守卫作为**纵深防御**保留，但置于 `super()` 之前（或改成不依赖抛异常的过滤）。

### F4 `TERMINATED` 永不结算 → 无超时死循环

四个事实叠加成一个死锁：

1. `confirm_continuation` 把状态推进到 `TERMINATED`（`load_balancer.py:98`）。
2. `_TERMINAL_ATTEMPT_STATES = {TERMINATED, SETTLED}`（`load_balancer.py:17`）——把 TERMINATED 与 SETTLED 并列为"终态"。
3. 但 `gc_settled_requests` 只在 `is AttemptState.SETTLED` 时清理（`load_balancer.py:153-157`），`release_server` 只在 `ADMITTED` 时推进到 `SETTLED`（`load_balancer.py:75-77`）——**没有任何路径能把 TERMINATED 变成 SETTLED 或把它清出 `attempt_state`**。
4. `has_unsettled_requests` 的定义是 `attempt_state.get(r) is not SETTLED`（`load_balancer.py:111-115`），且遍历的是 `requests_for_server(server_id)`（原生粘性映射）——TERMINATED 的请求映射同样不会被清。

于是 `has_unsettled_requests` 对该 server **永真**，`finish_remove` 永远抛 `ValueError("cannot remove route while requests remain unsettled")`（`load_balancer.py:145-147`）。

而 `rollouter.prepare_exit` 里是：

```python
175  while await lb.has_unsettled_requests.remote(server_id):
176      await asyncio.sleep(0.1)
```

**无超时、无重试上限、无逃生分支**。`prepare_exit` 直接 `await` 在这里，TaskRunner 的 `_execute_operation` 线程也被阻塞在这条链上。一次 Client continuation 就能让该 replica 的移除操作**永久挂起**。

设计 §8.3 确实把"requests terminated"列为排空必须等待的条件之一（`expires_at ≠ 释放证据` 一节也强调不能靠超时当释放证据）。设计给出的是**安全条件**，但没有给出"已终止请求如何退出 unsettled 集合"的**收敛路径**——因为设计假定 Client 接管续推后该请求会被重新认领并最终 release。当前实现把"安全条件"直接编码成"无限等待"，缺的正是那条收敛路径。

**这条需要设计裁决，不能只改代码**：要么引入"Client continuation 已核验 → 该请求的 R 责任已转移 → 允许从 `attempt_state` 毕业"的显式接口（`confirm_continuation` 返回的 `OperationEvidence` 本来就是给这个用的，但它现在只被丢弃），要么给排空一个显式的 fencing/放弃语义。当前 `test_terminated_request_still_blocks_route_removal` 把"TERMINATED 永久阻断"**固化为期望行为**，说明这是有意设计而非疏忽——更需要先定契约。

另外注意一个反向耦合：`confirm_continuation` 返回的 `OperationEvidence` 在**任何**调用点都没有被消费（唯一断言它的是过期测试 F18）。也就是说"Client 接管证明"这件事，产出了证据但没人验。

### F5 `request_id` 复用被永久封死

```python
51  state = self.attempt_state.get(request_id)
52  if state in {AttemptState.ADMITTED, AttemptState.TERMINATED}:
53      raise RuntimeError(
54          "first release allows at most one unsettled generation per request_id"
55      )
```

`ADMITTED` 的拒绝是对的（同一 request 未结算不得重复准入）。但 `TERMINATED` **不是"未结算"**——它是"Client 已接管、原 server 的这次生成已终止"。把它归入"未结算"，再加上 F4 的"永不结算"，结果是一个 `request_id` **一旦用过 continuation 就永远不能再被 acquire**。

而原生 `GlobalRequestLoadBalancer` 的整个设计前提就是 request_id 粘性复用（`_request_id_to_server` LRU 缓存 + `full_determinism` 时 `hash(request_id) % len(servers)`）。多轮对话/前缀续推复用同一 request_id 是常态。当前实现让这些请求在第一次 continuation 之后永久失败。

### F6 长远程等待被包进无超时的 G，失败即永久围栏

```python
62  async def _fit_update_weights(self):
65      gate = self.replica_sync_gate
66      lease = await gate.acquire(f"native-sync:{self.current_param_version}", GateKind.NATIVE_SYNC)
67      try:
68          result = await lease.guard(super()._fit_update_weights)
...
73      except BaseException as exc:
74          gate.block(lease.owner, f"Native synchronization outcome unknown: ...")
75          raise
```

原生 `super()._fit_update_weights` 最终落到 `CheckpointEngineManager.update_weights`（`verl/checkpoint_engine/base.py:504-557`），内部是一串**长远程等待**：`abort_replicas()` → `release_kv_cache_replicas()` → `build_process_group()` → `ray.get(actor_wg.update_weights + rollout.update_weights)` → `resume_kv_cache_replicas()` → `resume_generation_replicas()`。这条链在 GPU 规模上是分钟级。

而 `ReplicaSyncGate.acquire(timeout=None)`：

```python
108  async def acquire(self, timeout=None):
109      await self._lock.acquire()
```

**没有超时。** 整个任务的 ADD/REMOVE/DONATE 提交段都在等这把锁。设计 §4.3 说：

> G 不覆盖 Client continuation、runtime release 等等待阶段
> 不把长时间排空、接管或物理清理包进 G

但同时又说"原生同步与关键提交段共用 Trainer 的 G"。这两句需要一起读：设计的意图是原生追参**本身**要串行化（否则参数版本会穿插），但**不应**把排空/接管/释放这些*等待*包进来。当前实现的问题是第二半——`_fit_update_weights` 在持锁期间跨越了一整段可能挂起的远程调用，而排空路径（`remove_and_commit`）用同一把锁，于是"排空"事实上被"追参"阻塞。

更严重的是失败终局：

```python
# replica_sync_gate.py
75  def block(self, owner, reason):
...     # 之后所有 acquire -> _require_healthy -> raise GateFencedError
```

`gate.block()` 是**闩锁且无复位接口**（源码注释明确写了"deliberately no reset"）。一次瞬时网络抖动/一次 `ray.get` 超时，就把该任务**永久**打成 `GateFencedError`，此后所有 Add/Remove/Donate 全部失败且无法恢复。设计从未定义"G 的永久围栏"作为合法终局——设计里最接近的是"UNKNOWN 需人工/对账收敛"，但那是 Operation 级，不是 Gate 级。**这条需要设计裁决**：G 的围栏应当是可恢复的（fence epoch + 显式 reconcile 通过后重新开放），否则任何一次原生追参失败 = 任务不可用。

### F7 两段不同性质的操作被同一个 `try` 吞并

```python
114  await lease.guard(self.checkpoint_manager.remove_effective, target)
115  mutated = True
116  evidence = await lease.guard(self.rollouter.commit_service_change.remote, operation)
...
127  except BaseException as exc:
128      if mutated:
129          gate.block(lease.owner, f"Exit service commit outcome unknown: ...")
```

`remove_effective` 改的是 **E 视图**（CE 成员），`commit_service_change` 改的是 **R/C**（LB 路由、M 状态、并发容量）。`mutated` 只是一个 bool，把两者的失败合并成一件事。

后果被 `tests/unit/test_wiring.py::test_release_failure_after_service_commit_keeps_operation_unknown_and_fenced`（214-249）**固化为期望行为**：E 已改、R/C 未提交 → `OperationRecord` 落 `UNKNOWN`，且 G 永久围栏（F6 的不可复位闩锁）→ 任务停摆。

但这里的事实是**可区分的**：
- 若 `commit_service_change` 抛在 `finish_remove` 之前（`rollouter.py:205` 之前）→ LB 未动、M 已是 `DRAINING`、E 已移除 → 可补偿（把 E 加回，或重放提交）。
- 若抛在 `finish_remove` 之后、`deactivate_service` 之前 → LB 路由已删 → 只能前进不能回退。

`rollouter.commit_service_change` 里已经写了正确的补偿骨架（`rollouter.py:208-211`：异常时把 DRAINING → QUARANTINED），说明作者知道要区分阶段；但 Trainer 层把这个信息压平成了一个 bool。设计 §`证据链而非布尔` 正是反对这种压平。

同样地，`operation_journal` 的 `_RESOLVED = {SUCCEEDED, FAILED}`（`operation_journal.py:17`）意味着 `UNKNOWN` **不是** resolved —— `finish` 不会把它从 `active` 里弹出（87-88），于是 `active_operation(task)` 永远返回这个 UNKNOWN 的操作，`begin` 永远抛 "another lifecycle operation is active"（39-47）。**一个 UNKNOWN 就永久占住该任务的唯一操作槽位**，与 F6 的围栏叠加。

### F8 失败语义过粗，对账无依据

```python
# task_runner.py
78  def _failure_status(self, ...):
82      return OperationStatus.UNKNOWN
...
131  def query_operation(self, operation_id):
135      return OperationRecord(operation_id, OperationStatus.UNKNOWN, ...)  # + "not found"
```

设计的"先查事实，不猜结果"要求查询能区分事实。当前 TS 对外只有 `UNKNOWN` 一个答案，覆盖了三种完全不同的现实：操作 ID 不存在 / 操作仍在执行 / 操作结果不确定需要补偿。GS 的 `submit_operation` 依赖 `OperationRecord` 做对账（`group_scheduler.py:117-120`），拿到 `UNKNOWN` 无从判断该重发、该等待、还是该人工介入。设计 §7 的"证据链"在这里断掉了。

### F9 `RELEASED` 校验只做集合相等，允许空集

```python
# group_scheduler.py
157  if set(evidence.released_gpu_uuids) != set(lease.gpu_uuids):
158      raise ValueError("RELEASED evidence must exactly cover lease GPU claims")
```

```python
# contracts.py
135  if self.type is not EvidenceType.RELEASED and self.released_gpu_uuids:
136      raise ValueError("released_gpu_uuids are valid only for RELEASED evidence")
```

`OperationEvidence.__post_init__` 只拒绝"非 RELEASED 携带 UUID"，却**允许 RELEASED 携带空元组**（默认值就是 `()`，见 `contracts.py:120`）。源码注释称"空集由 GS 覆盖"——但 GS 的"覆盖"是**一致性**校验（两边相等），不是**非空**校验。当 `lease.gpu_uuids == ()` 时（例如纯 CPU 复用、或 claim 全部是 `cpu_request` 型），`set(()) == set(())` 成立，一份**什么都没释放**的证据可以通过校验并完成 lease 交接。

设计原文："`OperationEvidence(RELEASED)` 仍需显式给出完整 `released_gpu_uuids`"——"显式给出完整"应当意味着非空且与 lease claims 一一对应。建议在 `contracts.py` 收紧：RELEASED 必须非空；若确实存在零 GPU 的 lease，应在 `Lease` 层显式建模而不是让空证据无限通过。

---

## 4. P2：接线缺口与死代码

### F10 时间戳单位不一致

```python
# contracts.py
149  timestamp=time.time_ns(),        # 纳秒
...
160  expires_at: float = 0.0          # 秒（注释：0 means no fixed expiry）
203  # 校验只做 float 有限性 + >= 0
```

`OperationEvidence` 携带纳秒时间戳，`Lease` 携带秒级过期时刻。两者相差 10⁹ 倍，任何"该证据是否产生于租约有效期内"的比对都会得到无意义的结果。GS 目前不做这个比对（`advance_lease` 只做集合校验），所以**当前没有实际 bug，但证据里带了一个不可用的字段**——它迟早会被某个对账逻辑读到。要么统一单位，要么删掉这个字段。

### F11 跨任务撮合决策是空的

```python
# group_scheduler.py
56  def schedule(self):
57      return []
```

`AGENTS.md` 说 "GS discovery and entity creation are real, not empty"。核对下来：`discovery.py` 的实体发现、`open_lease`/`advance_lease`/`submit_operation` 的 lease 账本是**真的**（有完整校验、幂等重放检查、release_history）。但**决策**不是：

- `schedule()` 恒返回空列表。
- `open_lease` 在 `src/` 中**只被测试调用**（`group_scheduler.py:122`）。
- `submit_idle_report` 有入口（被 `rollouter.submit_idle_report` 调用），但没有任何消费者把它接到 `schedule()` 上。

设计 §6 的核心价值主张——把任务 A 的空闲 replica 跨任务派给任务 B——在 HEAD 上不存在。当前 GS 是一个**账本 + 校验器**，不是一个调度器。这不是缺陷（显式返回 `[]` 比假调度好），但 `README` 的控制面能力清单需要如实反映这一点。

### F12 空闲判据与设计 §8.4 不符

设计 §8.4 定义的**空泡判据 #2 = 容量富余：移除不破坏产能承诺**，数据源 = `committed_capacity`（C，内部）。

核对 `committed_capacity` 的数据源是对的：设计指出它复用 `max_concurrent_samples = len(get_replicas()) × concurrent_samples_per_replica`，而原生 `_update_max_concurrent_samples`（`fully_async_rollouter.py:1274-1294`）正是这么算的，且被 `remove_replicas` 调用后会重算。

但判据用错了：

```python
# rollouter.py
87  def collect_idle_candidates(self):
88      if self.production_window_open or self.committed_capacity <= 0:
89          return ()
...
93      return tuple(
94          (key, manager.replica_kind[key])
95          for key, state in manager.replica_state.items()
96          if state is ReplicaState.ACTIVE
97      )
```

即"只要 `paused` 就把**所有** ACTIVE replica 报成空闲候选"。问题：

- `paused` 的原生语义是**生成暂停**（队列满 / `_compute_rollout_resource_utilization` 触发），见 `fully_async_rollouter.py:907-912,1136-1148`。它与"移除这个 replica 之后还能不能满足 `max_required_samples`"**没有关系**。
- `committed_capacity <= 0` 这个守卫是**循环的**：容量为 0 时 rollouter 本就处于 paused。
- 结果是会把"必须保留才能满足产能承诺"的 replica 报成可移除候选。

缓解因素是最终决策在 GS（而 GS 的 `schedule()` 现在返回 `[]`，F11）——也就是说这条**目前不造成损害，因为下游不存在**。但一旦 F11 被补上，脏上报会直接变成错误决策。`tests/unit/test_wiring.py::test_rollouter_idle_detection_does_not_read_lb`（327-341）把这个判据固化成了期望行为。

### F13 大面积零调用代码

已用 grep 在 `src/` 与 `tests/` 全量确认零调用：

| 符号 | 位置 |
|---|---|
| `MultiTaskvLLMReplica.bind_lease` / `prepare_create` / `mark_weight_ready` / `release_gpu` / `abort_target` | `rollout/replica.py:42-90` |
| `MultiTaskCheckpointEngineWorker.replay_current_weights` / `replay_status` | `checkpoint/checkpoint_engine_worker.py:19+` |
| 整个 `transfer_topology.py` | `checkpoint/transfer_topology.py` |
| 整个 `operation_dispatcher.py` | `orchestration/operation_dispatcher.py` |
| 整个 `exactly_once.py` | `orchestration/exactly_once.py` |
| `MultiTaskGlobalRequestLoadBalancer.commit_routable` | `rollout/load_balancer.py:122` |
| `OperationDispatcher.evidence_for_service_commit` | `orchestration/operation_dispatcher.py:59` |

`AGENTS.md` 有一句直接相关：

> Remove unnecessary earlier P1 feature protocols, dummy services and tests. The user explicitly requests deletion without backups.

这 7 组代码正是"dummy services"的形态：有接口、有测试（部分）、无消费者。它们对读者的成本是真实的——`replica.py` 只有 90 行，其中 49 行是死代码，且这 49 行全部违反 F1。

### F14 两套并存的派发器

`OperationDispatcher`（63 行）与 `OperationJournal`（98 行）做的是同一件事（"同一 operation_id 只执行一次副作用、记录终态"），但语义不同：

| | `OperationJournal` | `OperationDispatcher` |
|---|---|---|
| 异常时的状态 | `UNKNOWN` | `FAILED` |
| 是否被产品代码使用 | 是（`task_runner.py`） | **否** |
| 唯一消费者 | — | `tests/test_operation_dispatcher.py` |

`UNKNOWN` 与 `FAILED` 的区别正是 F7/F8 的全部要害（"结果不确定" vs "确定失败"）。两套机制并存，且更弱的那套带着"exactly once"的测试名，会误导后来者。

`AGENTS.md`：

> Migrate the orchestration protocol as one unit. Do not add compatibility aliases or legacy wire forms for replaced types/interfaces.

这正是被禁止的形态。建议删除 `operation_dispatcher.py` 及其测试。

### F15 `exactly_once.py` 是死副本

`ExactlyOnceCompletionQueue` / `CompletedSample` / `CompletionEvidence`（118 行）复刻的是 `MultiTaskMessageQueue` 已经内联实现的能力（`message_queue.py` 的 `put_sample_once` + `_decode_identity`）。后者才是真正接在原生 `MessageQueueClient` 上的那一份（`task_runner._replace_message_queue`）。前者零调用。

### F16 `_setup_env_cuda_visible_devices` 是空壳

```python
74  def _setup_env_cuda_visible_devices(self, *args, **kwargs):
75      if self.replica_kind is ReplicaKind.BORROWED:
76          if not self.placement_claims:
77              raise ValueError("borrowed replica requires placement claims")
78          return super()._setup_env_cuda_visible_devices(*args, **kwargs)
81      return super()._setup_env_cuda_visible_devices(*args, **kwargs)
```

第 80 行与第 81 行**完全相同**。这个 `if` 只做了一次非空断言，对 `CUDA_VISIBLE_DEVICES` 的行为没有任何影响。

这意味着："borrowed hidden runtime 创建与 lease-aware GPU 绑定"（README 明列的实现项）**不存在**。一个 BORROWED replica 会走原生路径，拿到属于本 replica 的**全部**设备，而不是 lease 圈定的那几块 GPU。`placement_claims`（来自 `bind_lease`，`replica.py:43`）除了被断言非空之外**没有任何用途**。

当前不可达（BORROWED 需要 `rollouter.prepare_replica`，而它正确地抛 `NotImplementedError`），所以不构成现实风险。但它是"看起来实现了、实际没实现"的典型，且**静默**——如果哪天 `prepare_replica` 被接上，这一段不会报错，只会把错误的设备集交给 vLLM。

---

## 5. P2/P3：测试层

### F17 `tests/unit` 在 HEAD 上是红的

```
1 failed, 70 passed in 0.32s
FAILED tests/unit/test_wiring.py::test_terminated_request_still_blocks_route_removal
E   NameError: name 'OperationEvidence' is not defined
src\multi_task_scheduler\rollout\load_balancer.py:99
```

根因在**测试脚手架**，不在产品代码。`test_wiring.py` 用 AST 把类从原生基类上摘下来再 `exec` 到一个手工构造的 `env` 字典里：

- `taskrunner_class()`（81-100）正确注入了 `OperationEvidence`、`EvidenceType`。
- `load_balancer_class()`（103-131）**漏注入了这两个符号**。

`confirm_continuation` 在新版本里改为返回 `OperationEvidence.now(request_id, EvidenceType.EXIT_READY)`（`load_balancer.py:99-102`），于是 `isolated()` 出来的类一调用就 `NameError`。产品代码本身是对的。

严重度不高（一行 `env` 补全即可修），但它有两个附带影响：
1. `README` 的"验证"章节暗示 `pytest -q tests/unit` 可复现绿灯，与事实不符。
2. 脚本化的红/绿门禁在 HEAD 上是红的，任何人都无法用它判断自己有没有引入回归。

修法：在 `load_balancer_class()` 的 env 里加入 `OperationEvidence=OperationEvidence, EvidenceType=EvidenceType`，与 `taskrunner_class()` 对齐。

### F18 `tests/native_unit` 在两个方向上过期

这是**本该**捕获 F3 的那层测试，但它不可执行（环境无 ray），且内容是旧的：

```python
# tests/native_unit/test_native_adapters.py
69  assert (
70      lb.confirm_continuation("r", "client-1", "prefix-1")
71      is AttemptState.TERMINATED
72  )
```

实现已改为返回 `OperationEvidence`（`load_balancer.py:86,99`）—— 该断言现在必然失败。

```python
88  assert lb.begin_drain(key) == "s"
89  assert "s" not in lb.get_all_servers()
```

与当前 `begin_drain`（不移除 server，F3）矛盾 —— 该断言现在必然失败。

两个断言分别对应"该改测试"（返回类型变了）与"该改实现"（行为回退了）两种不同情况。**native 层测试目前不能充当任何回归网**，而它恰恰是唯一能验证原生委托/子类构造/Actor 接线的那一层。

本轮审计因环境缺 ray **未能运行**该层，因此关于原生接线的结论全部只是源码比对，不作为运行时证据。

### F19 `test_lifecycle_flows.py` 是同义反复

```python
4  def test_add_flow_evidence_chain():
5      chain = [EvidenceType.WEIGHT_READY, EvidenceType.SERVICE_COMMITTED]
6      assert chain[0] is EvidenceType.WEIGHT_READY
7      assert chain[-1] is EvidenceType.SERVICE_COMMITTED
...
21     transfer_ready = True
22     service_committed = EvidenceType.SERVICE_COMMITTED
23     assert transfer_ready
```

这四个用例断言的是**它自己刚刚构造的局部变量**，不触碰任何产品代码，因此不可能失败、也不可能发现回归。

两个问题叠加：
1. **零行为覆盖** —— commit `13d7e05` "Add lifecycle flow coverage for add remove donate restore" 声称的"流程覆盖"并不存在。
2. **默认不被收集** —— `pyproject.toml` 设了 `testpaths=["tests/unit"]`，所以 `pytest`（裸跑）也不会收集这个文件。它只在显式点名时执行。

`AGENTS.md`：

> Tests target native entry selection, actual GS discovery, subclass construction/wiring and native delegation. **Label mocked coverage explicitly.**
> Report test results by layer. Do not disguise a skipped or mocked integration test as a successful runtime check.

同义的占位测试比没有测试更糟：它让"已覆盖 add/remove/donate/restore"这句话出现在提交信息里。

### F20 硬编码 30s 超时

```python
# task_runner.py:184-190
ray.get(self.group_scheduler.advance_lease.remote(...), timeout=30)
# group_scheduler.py:117
ray.get(task_runner.submit_operation.remote(command), timeout=30)
```

这些链条的末端是 GPU 级操作（sleep/destroy/权重追参）。固定 30s 在真实规模下必然误超时；而 `ray.get(..., timeout=)` 超时后**不会取消**远端的 Actor 方法（Ray 语义），于是"超时报错"与"远端仍在执行并最终成功"会同时发生 —— 又制造一个 UNKNOWN。超时预算应当来自 lease/operation 的显式预算，而不是字面量。

---

## 6. 已核验通过（OK）

以下项经过逐行比对，**未发现问题**，记录以便区分"已查"与"未查"：

1. **`message_queue.py` 与原生属性名完全对齐**。`_lock` / `queue`（`deque(maxlen=...)`）/ `max_queue_size` / `dropped_samples` / `total_produced` / `_consumer_condition` 与原生 `verl/.../message_queue.py` 的 `MessageQueue` 逐一吻合；`put_sample_once` 的 drop-oldest 行为与返回值 `original_return_value` 语义与原生一致。这是"继承原生 Actor、只加去重"的正确做法。
2. **`contracts.py` 的 `Lease` claim 校验完整**（`contracts.py:162-207`）：非空 claims、每个 claim 必须有 `pg_id`/`node_id`/`gpu_uuid` 非空字符串、`bundle_index` 非负整数、`gpu_fraction` 必须为 1.0（"first release requires whole-GPU claims"）、`cpu_request` 非负、`gpu_uuid` 与 `(pg_id, bundle_index)` 均不得重复、`expires_at` 有限且 ≥ 0。这一层比大多数实现严谨。
3. **`llm_server_manager.py` 的状态迁移表** `_ALLOWED` 与设计 §M 的六态一致，非法迁移被拒绝；且 `MultiTaskLLMServerManager` 确实 `issubclass` 原生 `FullyAsyncLLMServerManager`。
4. **`replica_sync_gate.py` 的 `guard` 二次校验**（166-167）：获取锁之后重新检查健康状态，覆盖了"等待期间被 `block()`"的竞态。`GateFencedError` 语义明确。
5. **`unwrap_native_actor_class`** 通过 `__ray_actor_class__` 取回原生类，保证子类保留原生业务方法（`tests/native_unit:38-47` 的意图正确，虽然当前跑不了）。
6. **集成所依赖的原生 API 名全部存在，无命名漂移**（已逐个 grep 原生源码确认）：
   - `FullyAsyncTaskRunner`：`components` 字典、`_initialize_components`、`_create_rollouter`、`_create_trainer`、`_setup_hybrid_worker_group`、`_run_training_loop`（`fully_async_main.py:36-220`）
   - `FullyAsyncRollouter`：`get_max_queue_size`（:530）、`set_message_queue_client`（:486）、`get_replicas`（:526）、`get_active_server_count`（:280）、`_update_max_concurrent_samples`（:1274）、`set_max_required_samples`（:491）
   - `FullyAsyncTrainer`：`set_message_queue_client`（:298）
   - `CheckpointEngineManager.update_weights`（`checkpoint_engine/base.py:504-557`）的 6 步序列与 `trainer._fit_update_weights` 的包法一致
   - `GlobalRequestLoadBalancer`：`acquire_server` / `release_server` / `remove_servers` / `get_total_inflight` / `clear_sticky_cache` / `RequestLoadBalancer` Protocol / `require_acquire_fields` / `require_release_fields`（`workers/rollout/router.py`，434 行全文核对）
7. **`ABORT` 之外的所有"未实现"点都正确地抛 `NotImplementedError`**，且措辞指明了缺什么后端：`rollouter.prepare_replica:125`、`rollouter.finalize_release:221`、`rollouter.prepare_exit` 的 FORCE 分支:156、`trainer.bootstrap_and_publish`、`trainer.restore_and_publish`、`llm_server_manager` 的 `create_hidden`/`sleep`/`wake_weights`/`destroy`/`activate_service`、`http_server.wake_weights`/`abort_target`、`checkpoint_engine_manager.bootstrap_target`。**唯一的例外是 F1 那一组。**

---

## 7. 建议处理顺序

### 立即做（不需要设计裁决，纯纠错）

1. **F1 / F2**：`rollout/replica.py` 的 `bind_lease`、`prepare_create`、`mark_weight_ready`、`prepare_exit`、`release_gpu`、`abort_target` 全部改为 `raise NotImplementedError(...)`；`checkpoint_engine_worker.replay_current_weights`/`replay_status`、`transfer_topology.run_transfer` 同上。这是本仓库 `AGENTS.md` 的硬性要求，且是唯一有真实安全后果的项（伪造 `RELEASED` 可被 GS 全额接受）。
2. **F17**：`tests/unit/test_wiring.py::load_balancer_class()` 的 env 补 `OperationEvidence` 与 `EvidenceType` —— 让门禁回到绿。
3. **F9**：`contracts.py` 收紧 `OperationEvidence`：`RELEASED` 必须携带非空 `released_gpu_uuids`。
4. **F13 / F14 / F15**：删除零调用代码（`operation_dispatcher.py`、`exactly_once.py`、`transfer_topology.py`、`commit_routable`、`evidence_for_service_commit`）及其对应的死测试。`AGENTS.md` 已授权删除且不要备份。
5. **F19**：删除 `tests/test_lifecycle_flows.py`（同义反复），或改写成真正驱动 `TaskRunner._execute_operation` 的流程测试。

### 需要设计裁决后再改

6. **F4**（TERMINATED 的收敛路径）：设计 §8.3 只给了安全条件，没给收敛路径。在裁决之前，任何"让 TERMINATED 能被清理"的改动都可能削弱"`expires_at ≠ 释放证据`"这条约束。建议在设计中显式回答：**Client continuation 核验通过后，原 server 的 R 责任是否转移？转移后该 request 从哪个集合毕业？**
7. **F6**（G 的失败终局）：`ReplicaSyncGate.block()` 目前是不可复位的永久闩锁。设计需要定义"G 围栏后的恢复协议"（fence epoch + 显式 reconcile），或明确接受"一次原生追参失败 = 任务废弃"。
8. **F7**（E 与 R/C 的失败分离）：需要定义 UNKNOWN 的**收敛路径**（查询 → 补偿 → 重放 → 人工），而不是永久 UNKNOWN 占住唯一操作槽位。
9. **F12**（空闲判据）：按设计 §8.4 改为真正的容量富余判定（用 `committed_capacity` 与 `max_required_samples` 比较，而不是 `paused`）。
10. **F11**（GS 决策）：明确 `schedule()` 是"本期不实现"还是"待实现"，并让 `README` 如实反映。

### 实现层

11. **F3**：`begin_drain` 恢复"移出选路集合但保留请求事实"；`acquire_server` 的 draining 守卫移到 `super()` **之前**（或改为不抛异常的过滤），消除 inflight 泄漏。同步修 `tests/native_unit:88-89`。
12. **F5**：区分"未结算"（`ADMITTED`）与"已终止"（`TERMINATED`），让后者可被重新 acquire。
13. **F8**：`OperationStatus` 的查询语义分层（不存在 / 进行中 / 结果不确定）。
14. **F10**：统一 `timestamp` 与 `expires_at` 的单位。
15. **F16**：`_setup_env_cuda_visible_devices` 要么实现真正的 lease-aware 设备集，要么直接抛 `NotImplementedError`；不要保留两个相同分支。
16. **F18**：修复 `tests/native_unit` 的过期断言，并在有 ray 的环境里把它纳入门禁——这是唯一能验证原生接线的一层。
17. **F20**：把硬编码 `timeout=30` 换成来自 operation/lease 的显式预算。

---

## 附录：本次审计的方法与局限

**方法**：以设计文档 `精简优化版092203.md` 的 M/E/R/C 四真值、G 单闸门、证据链、`expires_at ≠ 释放证据` 等约束为基准，逐条对照 `src/multi_task_scheduler/**` 的实现；同时把集成层依赖的每一个原生符号、属性名、方法签名与 `D:\多RL任务\verl` 的源码原文核对；对零调用符号做全仓 grep。

**局限（必须如实声明）**：

- `tests/native_unit` 与 `tests/integration` **未运行**（环境无 ray）。所有涉及"原生委托是否正确""Actor 是否真的构造成功""GS 发现是否真的工作"的结论，**只是源码文本比对，不是运行时验证**。
- 无 GPU / vLLM / NCCL 环境，**任何**物理原语（create/sleep/destroy/权重追参/GPU 释放）均未验证。本报告中所有关于这些原语的判断，都是关于"代码是否诚实地拒绝"的判断，不是关于"物理行为是否正确"的判断。
- F4/F6/F7/F12/F11 涉及**设计意图的解释**。设计文档在这些点上给出了约束但未给出收敛路径，本报告的定性基于对 §4.3、§7、§8.3、§8.4 的解读；如与原始意图不符，以设计裁决为准。
- 本次审计**未修改任何源码**，**未 commit、未 push**。
