# 双任务 E2E 示例

- `native_args.txt`：唯一保留的 Hydra 原生训练参数样例。使用前填写真实的模型、训练与验证数据路径。
- `lease.example.json`：Lease 结构示例；只有具备真实可验证的资源 placement 时才能用于完整生命周期验收。

从仓库根目录运行：

```bash
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . --ray-address auto --start-local-ray \
  --native-args examples/e2e/native_args.txt \
  --scenarios "lifecycle force"
```

参见 [示例总览](../README.md)、[测试分层](../../tests/README.md) 和 [验收规范](../../docs/e2e-acceptance.md)。运行结果必须区分 Mock/CPU 和真实设备证据；仅运行成功不证明严格在途 continuation。
