# 双任务 E2E 示例

- `native_args.txt`：唯一保留的 Hydra 原生训练参数样例。使用前填写真实的模型、训练与验证数据路径。

E2E 默认从真实 donor 资源位置自动构造 Lease，无需提供静态 Lease JSON。可选 `--lease` 仅用于使用已验证的人工诊断输入。

从仓库根目录运行：

```bash
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . --ray-address auto --start-local-ray \
  --native-args examples/e2e/native_args.txt \
  --scenarios "lifecycle force"
```

参见 [示例总览](../README.md)、[测试分层](../../tests/README.md) 和 [验收规范](../../docs/e2e-acceptance.md)。运行结果必须区分 Mock/CPU 和真实设备证据；仅运行成功不证明严格在途 continuation。
