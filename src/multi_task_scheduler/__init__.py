"""Source-first MultiTask passthrough integration for experimental Fully Async.

Importing this package never imports verl/Ray or starts distributed services.
"""

__version__ = "0.1.0.dev0"
# Orchestration core + GS ledger/lease + verl bindings implemented; the GPU
# primitives remain explicit NotImplementedError until a native backend is verified.
IMPLEMENTATION_STAGE = "ORCHESTRATION_CORE_AND_BINDINGS"
