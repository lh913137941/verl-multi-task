# 测试目录与运行指南

按 **依赖的真实运行环境** 分层，而不是把所有测试混合执行。在仓库根目录运行以下命令。

```text
tests/
├── unit/                 # CPU 快速回归：Mock、AST 隔离，无需 Ray/VERL/NPU
│   └── README.md         # 各组件单元测试覆盖详情
├── integration/
│   ├── ray/              # 真实 CPU Ray Actor / GroupScheduler
│   ├── verl/             # 原生 VERL + Ray 类与适配器接线（原 native_unit）
│   ├── cuda/             # NVIDIA GPU + vLLM，含环境基线与生命周期
│   └── npu/              # Ascend NPU + vLLM-Ascend，含 smoke 与生命周期
└── conftest.py           # 共用源码路径、Ray Worker PYTHONPATH 设置
```

## 先选需要哪一层

| 修改内容 | 优先运行 | 要求 |
| --- | --- | --- |
| 合同、调度、生命周期及 Mock 接线 | `python -m pytest -q tests/unit` | `pip install -e '.[test]'`，不要求原生 VERL/Ray |
| GS 与 Lease 的真实 Ray RPC | `python -m pytest -q tests/integration/ray` | 安装 Ray，CPU 可运行 |
| 对接 VERL 原生类、Actor、Client | `python -m pytest -q tests/integration/verl` | 与目标环境一致的 VERL、Ray、vLLM 等依赖 |
| CUDA/vLLM 资源回收和参数恢复 | `tests/integration/cuda/`（见下方命令） | NVIDIA GPU、CUDA/vLLM、真实模型 |
| NPU/vLLM-Ascend 资源回收和参数恢复 | `tests/integration/npu/`（见下方命令） | Ascend NPU、torch_npu、vLLM-Ascend、真实模型 |

**默认 CI 只执行 `tests/unit`。** 在 pytest 的命令行加 `-m` 并不能阻止不匹配的测试模块被导入，因此有真实 VERL/CUDA/NPU 依赖的测试被分开存放；不要用
`pytest tests/integration` 代替上述按环境选择的命令。

### CUDA 实机

```bash
# 模型已存在于本机；验证原生 sleep、FORCE、RESTORE
VERL_MULTITASK_GPU_MODEL_PATH=/实际模型目录 \
  python -m pytest -q -s -m gpu_integration tests/integration/cuda/test_native_sleep_gpu.py

# 可选：只检查运行前的 CUDA 环境基线
# JSON 配置见 examples/experimental_fully_async/gpu_test_config.example.json
MT_GPU_TEST_CONFIG=/实际基线配置.json \
  python -m pytest -q tests/integration/cuda/test_baseline_environment.py
```

### Ascend NPU 实机

```bash
export VERL_MULTITASK_NPU_MODEL_PATH=/实际模型目录

# 最小真实模型加载与生成检查
python -m pytest -q -s -m npu_backend_smoke tests/integration/npu/test_native_sleep_npu.py

# 全部 NPU 验收场景：sleep / 借还 / FORCE / RESTORE
python -m pytest -q -s -m npu_integration tests/integration/npu/test_native_sleep_npu.py
```

`pytest.skip` 不是 PASS；检查 pytest 总结和真实输出，不能将缺少设备、
模型或环境依赖误报为已验证。不同设备上的依赖组合不保证能在同一 Python 环境中同时导入。

## 与 E2E 的关系

单元测试通过 Mock/AST 对合同与故障分支进行快速回归，真实 Ray、VERL 与硬件集成
检查本机运行时能力。**完整双 RL 任务共享调度闭环**仍以
[`verify_two_verl_jobs.py`](../scripts/e2e/verify_two_verl_jobs.py) 为入口：

```bash
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . --ray-address auto --start-local-ray \
  --native-args examples/e2e/native_args.txt \
  --scenarios "lifecycle force"
```

前提是 `native_args.txt` 已配置真实模型和数据；`force: PASS` 不代表严格在途续推证明，
需要时追加 `--require-inflight-force`。测试证据及失败处理见
[综合 E2E 验收规则](../docs/e2e-acceptance.md)，各 Unit 文件职责见
[unit/README.md](unit/README.md)。

## 维护约定

- 新增 **无需原生环境** 的边界/异常测试放入 `unit/`，复用 `_wiring_support.py`。
- 需要真实 Ray/VERL/设备依赖的测试放入对应 `integration/{ray,verl,cuda,npu}/`。
- 移动测试时同步更新 README、`.github/workflows/unit.yml`、`scripts/e2e` 和直接读取测试源码的校验脚本。历史审计文档中的旧路径仅表示当时的快照。
- 不删除重要的成功、失败、UNKNOWN/ACK 丢失、幂等恢复断言来降低测试数量。
