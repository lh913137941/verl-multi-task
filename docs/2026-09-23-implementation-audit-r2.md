# verl-multi-task 实现审计报告（第二轮）

- 审计日期：2026-09-23
- 被审对象：`D:\多RL任务\verl-multi-task`
- 分支：`chatgpt/092203-simplified-fusion`（**本报告标题里的 "092203" 由此而来**）
- HEAD：`19135ad`
- 上一轮审计修订：`13d7e05` → 见 `docs/2026-09-23-implementation-audit.md`（下称 **r1**，已被本报告取代）
- 基准设计：`D:\多RL任务\多RL任务共享调度对接VERL_动态流程编排融合设计精简优化版092203.md`
- 基准原生：`D:\多RL任务\verl\verl`
- 治理约束：本仓库 `AGENTS.md`

> **命名澄清**：`verl-multi-task-092203` 这个目录名不存在于磁盘上（全盘 `find` 确认）。`092203` 是**分支名** `chatgpt/092203-simplified-fusion` 的后缀。被审对象唯一对应 `D:\多RL任务\verl-multi-task` 的该分支。

---

## 0. 本轮的增量

r1 报告的 3 个 P0/P1 项已被三个提交处理：

| 提交 | 声称修复 | 实际结果 |
|---|---|---|
| `bf36aba` | remove synthetic lifecycle evidence from replica backend | ✅ **F1/F2 已修** |
| `42712aa` | require explicit release evidence payload | ✅ **F9 已修**，但 ❌ **引入回归 R1** |
| `19135ad` | correct drain admission and request settlement semantics | ⚠️ **F4 已修，F3a 已修**，但 ❌ **引入 R2/R6**，F3b/F5 未修 |

改动范围：**仅 3 个源文件，59 insertions / 137 deletions；`tests/` 一个字节都没动**（`git diff --stat 13d7e05..HEAD -- tests/` 为空）。这一点决定了下面相当一部分结论：产品代码往前走了，回归网没跟上。

---

## 1. 核验边界

| 层 | 命令 | 结果 | 证据强度 |
|---|---|---|---|
| `tests/unit` | `pytest -q tests/unit` | **2 failed, 69 passed in 0.40s** | 真实执行 |
| `tests/native_unit` | — | **未运行**：`ModuleNotFoundError: No module named 'ray'` | 无 |
| `tests/integration` | — | **未运行**：同上 | 无 |
| GPU / vLLM / NCCL | — | **完全未验证** | 无 |

失败的两个用例：

```
FAILED tests/unit/test_contracts.py::test_first_release_lease_is_whole_gpu_and_has_unique_bundle_and_uuid
FAILED tests/unit/test_wiring.py::test_terminated_request_still_blocks_route_removal
```

r1 时是 `1 failed, 70 passed`。**本轮红得更多**——新增的那个失败是产品代码回归导致的（见 R1），不是测试问题。

本报告未修改任何源码，未 commit、未 push。

---

## 2. 已修复项（逐条核验）

### F1 / F2 ✅ 已修复 —— 合成安全证据已清除

`src/multi_task_scheduler/rollout/replica.py` 现在：

```python
43  def prepare_create(self) -> dict:
44      raise NotImplementedError("replica creation requires verified native runtime backend")
48  def mark_weight_ready(self, operation_id: str):
49      raise NotImplementedError("weight readiness requires verified replay/bootstrap backend")
53  def prepare_exit(self, operation_id: str):
54      raise NotImplementedError("exit readiness requires verified vLLM sleep/drain backend")
58  def release_gpu(self, operation_id: str, gpu_uuids):
59      raise NotImplementedError("GPU release evidence requires verified runtime destroy backend")
71  def abort_target(self, request_ids):
72      raise NotImplementedError("targeted abort requires verified FORCE_VERIFIED backend")
```

`import` 也相应清理（移除了 `EvidenceType`/`OperationEvidence`）。`replica.abort_target` 与 `http_server.abort_target` 现在**语义一致**——同一能力不再有两套相反答案。这是 r1 中唯一有真实安全后果的项（伪造 `RELEASED` 可被 GS 全额接受），已消除。

### F9 ✅ 已修复 —— 且逻辑闭合

```python
# contracts.py
122  if self.type is EvidenceType.RELEASED:
123      if not self.released_gpu_uuids:
124          raise ValueError("RELEASED evidence requires explicit GPU uuids")
125  elif self.released_gpu_uuids:
126      raise ValueError("released_gpu_uuids are valid only for RELEASED evidence")
```

核对是否真的闭合：GS `advance_lease` 的唯一证据校验是 `set(evidence.released_gpu_uuids) != set(lease.gpu_uuids)`（`group_scheduler.py:157-158`）。`Lease.__post_init__` 要求至少一条 claim，且每条 claim 的 `gpu_uuid` 必须是非空字符串 → `lease.gpu_uuids` **恒非空**。因此空证据**永远不可能**通过集合相等校验。**这条修复是完整的，不是表面修补。**

### F4 ✅ 已修复（活性部分）—— 排空死循环消除

```python
78  def requests_for_server(self, server_id: str) -> tuple[str, ...]:
79      return tuple(r for r, s in self.active_request_server.items() if s == server_id)
81  def has_unsettled_requests(self, server_id: str) -> bool:
82      return any(self.attempt_state.get(r) is AttemptState.ADMITTED
83                 for r in self.requests_for_server(server_id))
```

两处改动都是对的，且第二处比表面看起来更重要：

1. `has_unsettled_requests` 现在**只认 `ADMITTED`**，TERMINATED 不再计入 → `rollouter.prepare_exit:175` 的 `while await lb.has_unsettled_requests.remote(server_id)` 无超时轮询不再可能永久自旋，`finish_remove` 可以成功。r1 的 F4 死锁消除。
2. `requests_for_server` 改为读**子类自己的** `active_request_server`，不再读原生 `_request_id_to_server`。这是一个**结构性改进**：原生粘性缓存是 LRU 且有 maxsize，会自然驱逐，把它当"请求事实"的来源本身就是错的。解耦之后，`remove_servers()` 不会再破坏排空所需的请求事实（原生 `remove_servers` 只 pop `_inflight_requests`/`_servers`，见 `router.py:240-244`）。

### F3a ✅ 已修复 —— inflight 计数泄漏已补偿

```python
45  server_id, handle = super().acquire_server(request_id, **extra)
46  if self._is_draining_server(server_id):
47      super().release_server(server_id, request_id=request_id)
48      raise RuntimeError("draining server cannot accept new requests")
```

r1 指出的"原生状态已变更之后才抛错"现在有了显式回滚，`_inflight_requests` 不再永久 +1。**但回滚不完整，见 R2。**

---

## 3. 本轮新引入的问题（回归）

### R1 🔴 P1 —— 删掉了 Lease 的 PG bundle 唯一性校验，仓库自己的测试立刻变红

commit `42712aa` 的 diff 中：

```python
-        bundle_keys = [
-            (claim["pg_id"], claim["bundle_index"])
-            for claim in claims
-        ]
-        if len(set(bundle_keys)) != len(bundle_keys):
-            raise ValueError("lease claims must not repeat a PG bundle")
```

这不是可选项，是 r1 §6.2 明确列为"**已核验通过**"的安全校验之一（我当时写的是："`gpu_uuid` 与 `(pg_id, bundle_index)` 均不得重复"）。

**实锤证据**——仓库自带的测试直接抓到了这个回归：

```
_____ test_first_release_lease_is_whole_gpu_and_has_unique_bundle_and_uuid _____

    with pytest.raises(ValueError, match="repeat a PG bundle"):
E   Failed: DID NOT RAISE <class 'ValueError'>

tests\unit\test_contracts.py:98: Failed
```

**为什么这条重要**：`gpu_uuid` 去重与 bundle 去重**不互相蕴含**。下面这份畸形 lease 现在能通过全部校验：

```python
Lease("l1", (
    {"pg_id": "pg1", "node_id": "n0", "bundle_index": 0, "gpu_uuid": "gpu-a"},
    {"pg_id": "pg1", "node_id": "n0", "bundle_index": 0, "gpu_uuid": "gpu-b"},
), 0)
```

`gpu_uuid` 不重复 ✓，bundle `(pg1, 0)` **重复**——而物理上一个 PG bundle 就是一块卡，这两条 claim 声称从同一个 bundle 里拿到了两块不同的卡，无法落地。原来这条校验在**契约层**拦住它；现在它一路通过 `Lease`、通过 `open_lease`、通过 GS 记账，直到真实分配阶段才炸——而那时故障现场已经远离了错误来源。设计原则"先查事实，不猜结果"要的正是把这类不可能的事实挡在契约层。

修法：恢复该校验（`gpu_uuid` 去重之后、`expires_at` 校验之前）。

### R2 🔴 P1 —— 粘性缓存回滚不完整，导致某个 `request_id` 永久不可服务

`acquire_server` 的回滚调的是原生 `release_server`，而原生实现是：

```python
# verl/workers/rollout/router.py:208
def release_server(self, server_id: str, request_id: str | None = None) -> None:
    """...
    ``request_id`` is accepted for signature parity with content-aware
    balancers ...; this balancer tracks request counts only
    and ignores it.
    """
    if server_id not in self._inflight_requests:
        return
    if self._inflight_requests[server_id] > 0:
        self._inflight_requests[server_id] -= 1
```

**它只减计数，不删 `_request_id_to_server[request_id]`**（docstring 明写 "ignores it"）。而 `acquire_server` 在抛错前已经执行了 `self._request_id_to_server[request_id] = server_id`（`router.py:203`）。

于是这条链是**确定性的、可重复的、无出口的**：

1. 第一次 `acquire_server(rid)` → 选中排空中的 server S → 写入粘性映射 `rid → S` → 补偿减计数 → 抛错。
2. 第二次 `acquire_server(rid)` → `router.py:180` 粘性命中 → `if server_id in self._inflight_requests`（S 仍在，因为 `begin_drain` 没移除它）→ 命中 → `_inflight_requests[S] += 1` → 返回 S → 补偿 → **再次抛错**。
3. 无限重复。

**这个 `request_id` 永远无法被服务，且永远不会被改路由到健康副本。** 因为粘性缓存是 LRU（maxsize 有界），症状是"偶发、可复现、换一个 request_id 就好"，属于最难定位的那类故障。

修法：补偿时显式清掉粘性条目（`self._request_id_to_server.pop(request_id, None)`）。更好的做法见 R3。

### R3 🔴 P1 —— F3b 未修：排空副本仍参与选路，且原生本来就能优雅改路由

`begin_drain` 仍然只加标记、不移除：

```python
100  def begin_drain(self, key: ReplicaKey):
103      server_id = self.routes.get(key)
106      self.draining_servers.add(server_id)
107      return server_id
```

server 仍在 `_servers` 与 `_inflight_requests` 里，于是原生**最小负载选路**对**全新 request_id** 也会选中它：

```python
# verl/workers/rollout/router.py:199
min_count = min(self._inflight_requests.values())
candidates = [sid for sid, count in self._inflight_requests.items() if count == min_count]
server_id = random.choice(candidates)
```

一个正在排空的副本 inflight 恰好最低（**它不再接新活**）→ 极大概率落入 `candidates` → 被选中 → 抛错。**新请求不是"优雅改路由到其它副本"，而是随机硬失败。** 这就是 r1 的 F3b，未修。

而原生**本就有**优雅改路由的机制，只差一次移除：

```python
# verl/workers/rollout/router.py:183
if server_id in self._inflight_requests:
    self._inflight_requests[server_id] += 1
    return server_id, self._servers[server_id]
# Server was removed, clear stale cache entry and re-select
del self._request_id_to_server[request_id]
```

只要 server 不在 `_inflight_requests` 里，原生就会清掉陈旧粘性条目并**重新选一个健康副本**。所以正确的修复恰恰是**在 `begin_drain` 里调 `self.remove_servers([server_id])`**：

- `remove_servers` 只 pop `_inflight_requests` 与 `_servers`（`router.py:240-244`），**不碰** `_request_id_to_server` → 请求事实不丢；
- 而 F4 的修复已经把 `requests_for_server` 改成读子类自己的 `active_request_server`，**排空跟踪已完全不再依赖原生缓存** → 移除它不再有任何副作用；
- 这也正是 `tests/native_unit/test_native_adapters.py:88-89` 断言的行为。

换句话说：`19135ad` 把"移除"从 `begin_drain` 拿掉是对的**前提**已经不存在了——F4 的解耦恰恰解除了当初移除它的唯一顾虑。这两处改动应当一起做。

### R4 🟡 P2 —— `confirm_continuation` 用 `request_id` 冒充 `operation_id`

```python
76  return OperationEvidence.now(request_id, EvidenceType.EXIT_READY)
```

`OperationEvidence.operation_id` 这个字段在别处一律是**生命周期操作 ID**：`rollouter.prepare_exit` 传 `operation_id`（`rollouter.py:178`），`trainer.remove_and_commit` 校验 `evidence.operation_id != operation.operation_id`（`trainer.py:120`），GS `advance_lease` 用它反查 `self.operation_commands.get(evidence.operation_id)`（`group_scheduler.py:146`）。

这里塞进去的是 **request 标识**，属于另一个命名空间。两个 ID 空间共用一个字段名，且没有任何类型层面的区分（都是 `str`）。

当前无实害——因为这份 `EXIT_READY` 证据在所有调用点都被丢弃（唯一曾消费它的是 `tests/native_unit:69-72`，而那个断言已经过期）。但它是"等待接线"的错误契约：一旦有人把它喂给任何按 `operation_id` 索引的逻辑，就会查不到（`"RELEASED evidence references an unknown operation"` 那类错误）。

要么给续推证明一个独立的证据类型/字段，要么在这里传真正的 `operation_id`。

### R5 🟡 P2 —— 无界泄漏换了位置

`release_server`：

```python
53  def release_server(self, server_id, request_id=None):
54      state = self.attempt_state.get(request_id) if request_id else None
55      if request_id:
56          owner = self.active_request_server.get(request_id)
57          if owner is not None and owner != server_id:
58              raise ValueError("request release belongs to another server")
59          if state is AttemptState.SETTLED:
60              return
61      super().release_server(server_id, request_id=request_id)
62      if request_id and state is AttemptState.ADMITTED:
63          self.attempt_state[request_id] = AttemptState.SETTLED
64          self.active_request_server.pop(request_id, None)
```

只有 `ADMITTED` 才会推进到 `SETTLED` 并 pop `active_request_server`。`TERMINATED` 走完 `super().release_server` 后**两个 map 都不清理**；而 `gc_settled_requests` 只清 `SETTLED`（`:122`）：

```python
120  def gc_settled_requests(self, request_ids):
121      for request_id in request_ids:
122          if self.attempt_state.get(request_id) is AttemptState.SETTLED:
123              self.attempt_state.pop(request_id, None)
124              self.active_request_server.pop(request_id, None)
```

**没有任何路径能把 `TERMINATED` 清出去。** 每个发生过 continuation 的请求都永久留在 `attempt_state` 与 `active_request_server` 里，`requests_for_server` 会永远返回这些陈旧 ID。

影响：目前**无正确性后果**（陈旧 TERMINATED 条目在 `has_unsettled_requests` 里返回 False）。但这是**随 continuation 次数单调增长的无界内存增长**，且让 `requests_for_server` 这个"事实查询"接口返回的事实越来越脏。r1 的 F4 从"死锁"降级成了"泄漏"——是改善，但没清干净。

### R6 🟡 P2 —— 同一状态，两条方法给出相反语义

同一个提交里：

```python
81  def has_unsettled_requests(self, server_id: str) -> bool:
82      return any(self.attempt_state.get(r) is AttemptState.ADMITTED ...)   # TERMINATED 不算 unsettled
```

```python
43  if state in {AttemptState.ADMITTED, AttemptState.TERMINATED}:
44      raise RuntimeError("request already owns an unsettled generation")  # 把 TERMINATED 当成 unsettled
```

`19135ad` 的整个立意就是"TERMINATED 不是未结算"（否则 F4 修不掉），但 `acquire_server` 的文案还停留在旧语义。**r1 的 F5 未修复**（TERMINATED 后同一 `request_id` 仍永久不可再 acquire），而且现在这个未修的行为挂着一句与姊妹方法矛盾的错误信息。

如果 F5 是**有意的**（"一个 request_id 只允许一次 continuation"），文案应改为类似 "request already reached TERMINATED"；如果不是有意的，就该放行。**这条需要设计裁决**：原生 `_request_id_to_server` 的整个存在意义就是 request_id 粘性复用，把 `TERMINATED` 永久钉死与它冲突；但"续推后原 request 不得复活"也可能是设计要的安全条件。设计文档需要明确。

### R7 🟡 P3 —— 契约文件的文档倒退

`42712aa` 顺带删除了四个公共数据结构的 docstring，并丢掉了 `expires_at` 的关键语义说明：

```python
-        """GS ledger entry: authorized claims and expiry, without a public state machine."""
...
-            raise ValueError("expires_at must be >= 0 (0 means no fixed expiry)")
+            raise ValueError("expires_at must be >= 0")
```

`contracts.py` 是**设计契约**文件，不是实现文件——它的 docstring 是设计文档在代码里的落点。而且删掉的这句"0 means no fixed expiry"正在被产品代码依赖：

```python
# group_scheduler.py:104
expired = bool(lease.expires_at and time.time() >= lease.expires_at)
```

`0` 是哨兵值（表示无固定过期）。删掉说明后，`0` 是合法值这件事在契约层变得不可读。

同一提交还把大量多行校验压成单行（`Lease.__post_init__` 从 44 行压到 29 行），`OperationEvidence.now` 的签名从 8 行压成 1 行（约 120 字符）。功能等价，但可读性净下降，而这些正是最需要逐行审读的安全校验。建议恢复 docstring 与 `expires_at` 的语义注释，格式化的压缩不必回滚但不宜再继续。

---

## 4. 仍未处理（r1 的其余发现，本次 3 个提交未触及）

| r1 编号 | 内容 | 状态 |
|---|---|---|
| **F5** | `TERMINATED` 后同一 `request_id` 永久不可再 acquire | **未修**，且文案自相矛盾（见 R6） |
| **F6** | 原生追参的长远程等待被包进**无超时**的 G；`gate.block()` 是**无复位接口**的永久闩锁 | **未修** |
| **F7** | `remove_effective`（E）与 `commit_service_change`（R/C）失败被压平成一个 bool；UNKNOWN 永久占住该任务唯一操作槽 | **未修** |
| **F8** | `_failure_status` 恒为 `UNKNOWN`，查询无法区分"不存在/进行中/结果不确定" | **未修** |
| **F10** | `OperationEvidence.timestamp = time.time_ns()`（纳秒）vs `Lease.expires_at`（秒）。新增反证：`group_scheduler.py:104` 用 `time.time()` 与其比较，确认秒是约定单位 | **未修** |
| **F11** | `group_scheduler.schedule()` 恒返回 `[]`；`open_lease` 只被测试调用；`submit_idle_report` 无消费者 → **跨任务撮合决策为空** | **未修** |
| **F12** | 空闲判据用 `paused` 冒充设计 §8.4 的"容量富余"，且守卫条件循环 | **未修** |
| **F13** | 死代码：`transfer_topology.py`、`operation_dispatcher.py`、`exactly_once.py` 全部仍在；`bind_lease`/`commit_routable`/`replay_current_weights`/`run_transfer`/`evidence_for_service_commit` 仍**零调用**（本轮 grep 复查确认） | **未修** |
| **F14** | `OperationDispatcher` 与 `OperationJournal` 两套派发器并存，失败语义更弱（FAILED vs UNKNOWN）；`AGENTS.md` 明禁 | **未修** |
| **F15** | **本条更正**：`exactly_once.py` **不是**死代码。`CompletionEvidence` 被 `message_queue.py:13` 使用、`DuplicateCompletionError` 被 `message_queue.py:56` 抛出且被 `tests/native_unit:112` 断言。仅 `ExactlyOnceCompletionQueue`/`CompletedSample` 在 `src/` 无消费者，但 `tests/unit/test_exactly_once.py` 的 7 个用例是**真实有效**的 drop-oldest / digest-conflict 语义测试，且是这套语义在无 ray 环境下唯一可测的表达。**决定保留，不删除。** r1/r2 前文把它列为待删死副本是错的 | 已复核更正 |
| **F16** | `_setup_env_cuda_visible_devices` 仍是纯校验空壳（去掉了重复的 `return`，现在更明显地看出 `if` 不影响行为）；`bind_lease` 设的 `placement_claims` 除被断言非空外**无任何用途** → "lease-aware GPU 绑定"不存在 | **未修** |
| **F17** | `tests/unit/test_wiring.py::load_balancer_class()` 仍漏注入 `OperationEvidence`/`EvidenceType` | **未修**（仍是红的） |
| **F18** | `tests/native_unit/test_native_adapters.py:69-72` 与 `:88-89` 两处断言仍与实现相反 | **未修** |
| **F19** | `tests/test_lifecycle_flows.py` 仍是同义反复，且默认不被收集 | **未修** |
| **F20** | `task_runner.py:184-190`、`group_scheduler.py:117` 仍硬编码 `timeout=30` | **未修** |

---

## 5. 已核验通过（本轮复查仍然成立）

1. **`message_queue.py` 与原生属性名完全对齐**（`_lock` / `queue`=deque(maxlen) / `max_queue_size` / `dropped_samples` / `total_produced` / `_consumer_condition`），drop-oldest 与 `original_return_value` 返回语义对齐原生。
2. **原生 API 名零漂移**（逐个 grep 原生源码确认）：`FullyAsyncTaskRunner.components`/`_initialize_components`/`_create_rollouter`/`_create_trainer`/`_run_training_loop`（`fully_async_main.py:36-220`）；`FullyAsyncRollouter.get_max_queue_size:530`/`set_message_queue_client:486`/`get_replicas:526`/`get_active_server_count:280`/`_update_max_concurrent_samples:1274`/`set_max_required_samples:491`；`FullyAsyncTrainer.set_message_queue_client:298`；`CheckpointEngineManager.update_weights`（`checkpoint_engine/base.py:504-557`）六步序列；`GlobalRequestLoadBalancer` 全部接口。
3. **`committed_capacity` 的数据源与设计 §8.4 一致**（`len(get_replicas()) × concurrent_samples_per_replica`，`fully_async_rollouter.py:1274-1294`）——问题只在**判据**错（F12），数据源是对的。
4. **`llm_server_manager.py` 的 `_ALLOWED` 状态迁移表**与设计 §M 六态一致，非法迁移被拒绝。
5. **`replica_sync_gate.guard` 的二次健康校验**（166-167）覆盖了"等待期间被 block"的竞态。
6. **除 F16 那处空壳外，所有未实现点都正确地抛 `NotImplementedError` 并指明缺什么后端**（`rollouter.prepare_replica:125`、`rollouter.finalize_release:221`、FORCE 分支 `:156`、`trainer.bootstrap_and_publish`/`restore_and_publish`、`llm_server_manager` 的 `create_hidden`/`sleep`/`wake_weights`/`destroy`/`activate_service`、`http_server.wake_weights`/`abort_target`、`checkpoint_engine_manager.bootstrap_target`、以及本轮修好的 `replica.py` 五个）。**r1 的 P0 已无遗留。**

---

## 6. 建议处理顺序

### 立刻做（纯纠错，无需设计裁决）

1. **R1**：恢复 `contracts.py` 的 PG bundle 唯一性校验。这是仓库自带测试已经抓到红的回归，且是契约层的安全校验。
2. **R2 + R3**：**一起做**。在 `begin_drain` 里恢复 `self.remove_servers([server_id])`；R2 的粘性污染由此一并消失（原生 `router.py:183-187` 会自动清陈旧条目并改路由到健康副本），无需单独 pop。同步修正 `tests/native_unit/test_native_adapters.py:88-89` 为期望行为。
3. **F17**：`tests/unit/test_wiring.py::load_balancer_class()` 的 exec env 补 `OperationEvidence=OperationEvidence, EvidenceType=EvidenceType`——与 `taskrunner_class()` 对齐。让门禁回到绿。
4. **R7**：恢复 `contracts.py` 四个公共结构的 docstring，以及 `expires_at` 的 `(0 means no fixed expiry)` 语义注释。
5. **F13/F14/F15**：删除零调用代码（`operation_dispatcher.py`、`exactly_once.py`、`transfer_topology.py`、`commit_routable`、`evidence_for_service_commit`、`bind_lease`）及其死测试。`AGENTS.md` 已授权删除且不要备份。
6. **F19**：删除或重写 `tests/test_lifecycle_flows.py`。

### 需要设计裁决后再改

7. **R6 / F5**：`TERMINATED` 之后同一 `request_id` 能否再次 acquire？设计需要在"原生粘性复用语义"与"续推后原 request 不得复活"之间明确取舍，并统一 `acquire_server` 与 `has_unsettled_requests` 的措辞。
8. **R4**：Client 续推证明应当用什么标识（`operation_id` 还是独立的 request 空间）？这决定 `OperationEvidence` 的字段语义。
9. **F4 剩余 / R5**：`TERMINATED` 的毕业条件。设计 §8.3 给了安全条件（`expires_at ≠ 释放证据`）但没给收敛路径，R5 的泄漏只是把它从"死锁"变成"泄漏"，没解决本质。建议在设计里显式回答：**Client continuation 核验通过后，原 server 的 R 责任是否转移？转移后该 request 从哪个集合毕业？**
10. **F6**：G 的失败终局。`ReplicaSyncGate.block()` 目前是不可复位的永久闩锁，一次原生追参失败 = 任务永久不可用。需要定义恢复协议（fence epoch + 显式 reconcile）或明确接受该终局。
11. **F7**：UNKNOWN 的收敛路径（查询 → 补偿 → 重放 → 人工），而不是永久 UNKNOWN 占住唯一操作槽位。
12. **F12**：按设计 §8.4 改为真正的容量富余判定（`committed_capacity` 与 `max_required_samples` 比较，而不是 `paused`）。
13. **F11**：明确 `schedule()` 是"本期不实现"还是"待实现"，并让 `README` 如实反映 GS 当前是账本而非调度器。

### 实现层

14. **F8**：`OperationStatus` 查询语义分层（不存在 / 进行中 / 结果不确定）。
15. **F10**：统一 `timestamp` 与 `expires_at` 的单位。
16. **F16**：`_setup_env_cuda_visible_devices` 要么实现真正的 lease-aware 设备集，要么直接抛 `NotImplementedError`；不要保留一个不影响行为的 `if`。
17. **F18**：修复 `tests/native_unit` 的过期断言，并在有 ray 的环境里纳入门禁——这是唯一能验证原生接线的一层，而 F3/R3 这类问题正是它该抓的。
18. **F20**：把硬编码 `timeout=30` 换成来自 operation/lease 的显式预算。

---

## 7. 本轮执行记录

按 §6 的**"立刻做"**与**"实现层"**两档执行；**"需要设计裁决"一档（第 7–13 项）一律未动**——它们要求先定契约，擅自实现会削弱设计里已有的安全约束。

### 已执行

| 项 | 动作 | 文件 |
|---|---|---|
| **R1** | 恢复 `bundle_keys`（PG bundle 唯一性）校验 | `orchestration/contracts.py` |
| **R7** | 恢复 `ReplicaKey`/`OperationRecord`/`OperationEvidence`/`Lease` 四个 docstring；恢复 `expires_at` 的 `(0 means no fixed expiry)` 语义说明；恢复被压行的校验格式 | `orchestration/contracts.py` |
| **R2+R3** | `begin_drain` 恢复 `remove_servers([server_id])`——排空即移出选路池。R2 的粘性污染由原生 `router.py:183-187` 自动清除陈旧条目并改路由到健康副本，无需单独 pop | `rollout/load_balancer.py` |
| **F13** | 删除零调用符号：`commit_routable`、`bind_lease`；删除整个死模块 `operation_dispatcher.py`（含 `evidence_for_service_commit`）、`transfer_topology.py` | 多个 |
| **F14/F19** | 删除 `tests/test_operation_dispatcher.py`（唯一消费者是死模块）、`tests/test_lifecycle_flows.py`（同义反复） | `tests/` |
| **F16** | `_setup_env_cuda_visible_devices` 的 BORROWED 分支改为显式 `NotImplementedError`，不再是无行为的 `if` | `rollout/replica.py` |
| **F1 尾项** | `replay_current_weights`/`replay_status` 改为显式 `NotImplementedError`（按 AGENTS.md「weight replay 保持显式 NotImplementedError，绝不伪造成功」**保留钩子**而非删方法，与非 GPU 原语的处理方式对称）；移除只存伪造记录的 `_replay_records` | `checkpoint/checkpoint_engine_worker.py` |
| **F17** | `load_balancer_class()` 的 exec env 补 `OperationEvidence`/`EvidenceType`，与 `taskrunner_class()` 对齐；移除已不存在的 `_TERMINAL_ATTEMPT_STATES` | `tests/unit/test_wiring.py` |
| **F18** | `confirm_continuation` 断言改为校验返回的 `OperationEvidence.type`；`finish_remove` 的 match 串改为 `"requests remain admitted"` | `tests/native_unit/test_native_adapters.py` |
| — | 新增 `test_drained_server_leaves_the_routing_pool_for_new_requests`，直接验证 R3 的修复意图 | `tests/unit/test_wiring.py` |

### 一处需要显式说明的测试改写

原 `test_wiring.py::test_terminated_request_still_blocks_route_removal` 断言的是 `19135ad` **之前**的语义（TERMINATED 计入 unsettled、从而阻断路由移除）。这与该提交对 `has_unsettled_requests` 的修改、以及设计 §8.3「排空等待 requests **terminated**」直接冲突——TERMINATED 就是"已终止"，不该继续阻断。

改写为 `test_admitted_request_blocks_removal_until_verified_continuation`，覆盖两条事实：**ADMITTED 阻断**、**已验证 continuation 之后的 TERMINATED 不阻断**。请注意：这**不是**第 7 项（R6/F5）那个待裁决问题——R6/F5 问的是"TERMINATED 之后同一 `request_id` 能否再次 acquire"，本次**未动**。

### 未执行（等设计裁决）

R6/F5、R4、F4 剩余/R5、F6、F7、F8、F10、F11、F12、F20 —— 见 §6 第 7–13 项与 §4 表。

### 测试证据

- `tests/unit`：**72 passed, 0 failed**（改动前为 `2 failed, 69 passed`）。数字自洽：69 通过 + 修复 2 个失败 + 新增 1 个用例 = 72。删除的两个根级测试文件本就不在 `testpaths=["tests/unit"]` 内，不影响计数。
- `tests/native_unit`：**仍未运行**（环境无 ray）。F18 的两处断言修正**未经运行验证**，仅保证与实现文本一致。
- `tests/integration`、GPU / vLLM / NCCL：**未验证**。

---

## 附录：方法与局限

**方法**：逐条复核 r1 的 20 项发现在 HEAD `19135ad` 上的状态；对新提交做全文 diff；把集成层依赖的每个原生符号、属性名、方法体与 `D:\多RL任务\verl` 源码原文核对（本轮重点核对了 `router.py` 的 `acquire_server`/`release_server`/`remove_servers`/`get_all_servers` 的**方法体**，R2/R3 的结论直接来自这三段实现）；对零调用符号做全仓 grep 复查。

**局限（必须如实声明）**：

- `tests/native_unit` 与 `tests/integration` **未运行**（环境无 ray）。所有涉及原生委托/Actor 构造/GS 发现的结论**只是源码文本比对，不是运行时验证**。
- 无 GPU / vLLM / NCCL，**任何**物理原语（create/sleep/destroy/权重追参/GPU 释放）均未验证。本报告关于这些的判断都是"代码是否诚实拒绝"的判断，不是"物理行为是否正确"的判断。
- **R2/R3 的故障链是从原生源码逐行推演出来的，未在真实 Ray 集群上复现**（环境无 ray）。推演依据是 `router.py:180-187`（粘性命中路径不检查 draining 标记）、`router.py:203`（抛错前已写粘性映射）、`router.py:208-220`（release 不删粘性条目）、`router.py:199-201`（最小负载选路含排空副本）四段实现。**建议在补上 ray 环境后优先用 `tests/native_unit` 验证这三条。**
- R6、F5、F6、F7、F11、F12 涉及**设计意图的解释**。设计文档在这些点上给出约束但未给出收敛路径，本报告的定性基于对 §4.3、§7、§8.3、§8.4 的解读；如与原始意图不符，以设计裁决为准。
- 未修改任何源码，未 commit、未 push。
