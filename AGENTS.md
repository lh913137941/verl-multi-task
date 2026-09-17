# verl-multi-task development rules

This independent repository contains the `multi_task_scheduler` Python package.
It is not the documentation directory of the outer repository.

- Keep the existing Git history/configuration. Do not commit or push without a user request.
- Current authorized work: implement the **current simplified fusion contract** in `docs/simplified-fusion-contract.md`. `docs/verl-5.3-dynamic-orchestration-design.md` is historical background only and must not be used to restore superseded signatures or evidence shapes.
- Migrate the orchestration protocol as one unit. Do not add compatibility aliases or legacy wire forms for replaced types/interfaces.
- Replace the ABC-only scaffold with actual native subclasses/composition and preserve native training behavior.
- Necessary native entry/profile wiring is allowed; do not modify original verl class implementations or restore Impl/SPI patches.
- Remove unnecessary earlier P1 feature protocols, dummy services and tests. The user explicitly requests deletion without backups.
- Do not remove unrelated documents, Git metadata, environments or historical P0 backups under this cleanup authorization.
- Keep package imports dependency-light and free of Ray initialization, actor discovery, global registration and monkey patches.
- An unimplemented runtime adapter must fail explicitly, never fall back to a fake/native class and pretend integration succeeded.
- Orchestration core, GS ledger/lease and verl bindings are implemented; GPU primitives (borrowed create, real sleep, target-only bootstrap, weight replay, force-reclaim) stay explicit NotImplementedError until a native backend is verified — never fake success.
- GS discovery and entity creation are real, not empty. Retain only simple state/handles and interfaces needed by this startup chain.
- Reports to GroupScheduler contain metadata only. Donor runtime handles must never enter borrower placement contracts.
- The manager owns local replica runtime references. Checkpoint Engine and load balancing hold separate projections.
- Preserve the single `experimental_fully_async_standalone` profile and the native verl training entry.
- Do not add a companion training entry, mirrored Hydra primary, class-FQN configuration, or old Impl/SPI dependency.
- Use `apply_patch` for source/document edits when working in a checkout. Do not overwrite unrelated user changes.
- Use a project virtual environment and `uv` for environment management; install only the dependencies needed by selected tests.
- Tests target native entry selection, actual GS discovery, subclass construction/wiring and native delegation. Label mocked coverage explicitly.
- Selected real dependency tests may use a source copy inside verl. Never claim GPU/native-runtime success from AST or mocks.
- Source deployment must preserve unknown files, record provenance and never copy Git, environments or model data. Do not build a copy framework for this stage.
- Document each component's owner, creation point, inherited behavior and the boundary of unimplemented features.
- Report test results by layer. Do not disguise a skipped or mocked integration test as a successful runtime check.
