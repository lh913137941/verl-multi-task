# Optional VERL plugin implementation plan

**Goal:** Install `verl-multitask` independently and select its existing runtime from
VERL's native experimental Fully Async entry with `multitask.enabled=true`.

**Architecture:** Keep the nine existing extension classes and their creation chain.
Add only an optional, lazy import at the native entry and disabled Hydra defaults.
The plugin owns configuration validation; neither installation nor package import
registers globals, initializes Ray, or changes native classes.

**Spec:** `../../../多RL任务资源共享调度对接VERL_动态流程编排融合设计精简版.md`,
sections 2, 18 and 19. This delivery covers optional integration, not the complete
GPU sharing acceptance contract in section 19.

## Constraints

- Work in the user-named checkouts, preserving all existing uncommitted changes.
- Do not commit or push. Do not change native class implementations.
- Default disabled; explicit false takes precedence over a stored profile.
- Explicit true selects the sole supported profile when profile is null/missing.
- For old custom configurations without `enabled`, retain explicit profile selection.
- Invalid selection and missing enabled dependencies fail before Ray startup.
- Keep original Hydra primary and training entry; no companion launcher, class-FQN
  configuration, automatic plugin registry, monkey patch, or fake GPU completion.
- Existing explicit failures for unverified GPU primitives remain explicit.

## Tasks

- [ ] Add failing tests for switch precedence, default selection, invalid booleans,
  import failures, and selection after native config normalization. Update the source
  comparison baseline to this checkout's actual `f92febf50fe3db102273eaf59b1854f392ae761d`.
- [ ] Extend `integration/verl/runtime_profile.py` without heavyweight imports;
  add narrow lazy selection to VERL `fully_async_main.main` and `multitask` defaults
  to the original `fully_async_ppo_trainer.yaml`.
- [ ] Verify actual Hydra composition against the source configuration and package
  installation/import outside the source directory. Keep Ray/GPU tests separately
  labeled; do not infer native runtime correctness from substitutes.
- [ ] Update package metadata, README and examples with install/enable/disable
  instructions, precise current boundaries, and test results. Correct stale claims
  about a different VERL commit and completed GPU integration.
- [ ] Run unit suite, check both diffs and whitespace, and report unverified layers.

## Test commands

Use a uv-managed project environment, with `MT_VERL_SOURCE_ROOT` pointing explicitly
to `D:\多RL任务\verl` and `PYTHONPATH` to this package's `src` for source tests:

```text
python -m pytest -q -p no:cacheprovider tests/unit
uv build --wheel --out-dir dist
uv pip install --python <clean-env-python> --no-deps <wheel>
```

The existing `.venv` points to an unavailable Python 3.10 executable; use a separate
environment under ignored `.uv-cache` without deleting or replacing it.
