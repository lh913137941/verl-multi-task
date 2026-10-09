# 示例入口

按照使用目的选择示例；示例文件不能替代真实环境或硬件验收。

| 位置 | 用途 | 入口 |
| --- | --- | --- |
| [e2e/](e2e/) | 双任务共享调度 E2E（包括租约样例） | [native_args.txt](e2e/native_args.txt)、[lease.example.json](e2e/lease.example.json) |
| [experimental_fully_async/](experimental_fully_async/) | 使用原生 VERL Fully Async 入口与 MultiTask 开关 | [使用与验收说明](experimental_fully_async/README.md) |

## 双任务 E2E

`e2e/native_args.txt` 是**唯一**维护的 native_args 配置样例；它是 Hydra `key=value` overrides 文件，不是 shell 脚本。运行前替换其中模型与数据绝对路径（默认路径只代表样例环境），然后从仓库根目录执行：

```bash
python scripts/e2e/verify_two_verl_jobs.py \
  --repo . --ray-address auto --start-local-ray \
  --native-args examples/e2e/native_args.txt \
  --scenarios "lifecycle force"
```

完整运行前置条件、强制回收验收边界与结果解释见 [项目说明](../README.md) 和 [E2E 验收规范](../docs/e2e-acceptance.md)。

## 原生 Fully Async

在已有可运行的 VERL 环境里，通过 `experimental_fully_async/native_entry_overrides.txt` 追加 MultiTask 专有参数；训练模型/数据/资源参数继续使用自己的原生命令。详见 [原生入口说明](experimental_fully_async/README.md)。

请勿将原生接入的 overrides 与双任务 E2E 的完整 native_args 混为一个配置文件。
