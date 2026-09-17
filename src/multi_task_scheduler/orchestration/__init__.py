"""Pure-Python orchestration primitives for standalone replica scaling.

This package must never import Ray, verl, torch, or vLLM. It holds the
task-internal concurrency gate, lifecycle/operation state machines, the
cross-component data contracts, and the transaction ordering for the
DONATE -> ADD -> REMOVE -> RESTORE loop. Runtime adapters that talk to
native verl/GPU are bound elsewhere (``integration``/``checkpoint``/
``rollout``/``scheduler``) and raise explicitly when unverified.
"""
