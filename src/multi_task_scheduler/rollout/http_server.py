"""Native vLLM server extension with no new generation behavior.

The native ``sleep`` (vLLM sleep mode) and generation methods stay inherited.
``wake_weights`` and ``wake_kv_and_validate`` are the donor wake path (section
5.5) and stay explicit failures until the native backend is verified; they must
restore weights before KV and never resume generation before both.
"""

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class MultiTaskvLLMHttpServer(vLLMHttpServer):
    """Replica wraps this class in Ray; every native method remains inherited."""

    def wake_weights(self, replica_id: str) -> str:
        """Restore weights only; generation stays disabled."""
        raise NotImplementedError(
            "weight wake requires verified native vLLM sleep backend"
        )

    def wake_kv_and_validate(self, receipt) -> str:
        """Restore KV, clear stale prefix/MM cache, validate engine; not routed."""
        raise NotImplementedError(
            "KV wake + validation requires verified native vLLM sleep backend"
        )

    def abort_target(self, replica_id: str) -> object:
        """Abort a borrowed target's in-flight requests (force reclaim)."""
        raise NotImplementedError(
            "target abort requires verified native backend"
        )
