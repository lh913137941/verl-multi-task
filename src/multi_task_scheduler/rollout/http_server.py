"""Native vLLM HTTP server extension; generation behavior remains inherited."""
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

class MultiTaskvLLMHttpServer(vLLMHttpServer):
    def wake_weights(self) -> None:
        raise NotImplementedError("native wake requires verified vLLM sleep backend")
    def abort_target(self, request_ids):
        raise NotImplementedError("targeted abort requires verified FORCE_VERIFIED backend")
