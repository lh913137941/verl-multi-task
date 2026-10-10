# 依赖与测试环境

本仓是接入 VERL experimental Fully Async 的轻量扩展，不重新锁定 PyTorch、vLLM、CUDA/NPU 等原生训练依赖。

| 环境 | 安装方式 | 用途 |
| --- | --- | --- |
| CPU unit | `python -m pip install -e '.[test]'` | `tests/unit`（无需 Ray 或 GPU） |
| CPU Ray integration | `python -m pip install -r requirements/ut.txt` | `tests/integration/ray`，基于 CPU 的真实 Ray |
| 原生 VERL integration | 使用目标 VERL/vLLM 环境并安装本仓 | `tests/integration/verl` |
| CUDA / Ascend NPU | 使用对应真实 VERL/vLLM/torch 设备环境 | `tests/integration/{cuda,npu}` |

`ut.txt` 只为 CPU Ray 集成提供明确的测试依赖，版本与 `pyproject.toml` 的 `test` extra 对齐。不要将其误认为 CUDA/NPU、原生 VERL 或生产环境依赖的 lockfile。

原先的 `native-ut.txt` 没有任何可安装依赖（只有注释），因此移除；原生环境的前置要求以本说明和 [测试指南](../tests/README.md) 为准。不同硬件栈不得随意合并为通用 `requirements.txt`。
