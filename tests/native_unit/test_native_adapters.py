import inspect
import pytest
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager, FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer
from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import MultiTaskLLMServerManager
from multi_task_scheduler.integration.verl.experimental_fully_async.rollouter import MultiTaskFullyAsyncRollouter
from multi_task_scheduler.integration.verl.experimental_fully_async.task_runner import MultiTaskFullyAsyncTaskRunner
from multi_task_scheduler.integration.verl.experimental_fully_async.trainer import MultiTaskFullyAsyncTrainer
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import AttemptState
from multi_task_scheduler.rollout.load_balancer import MultiTaskGlobalRequestLoadBalancer
pytestmark=pytest.mark.native

def test_actor_subclasses_keep_native_business_methods():
    for ext,native,method in [(MultiTaskFullyAsyncTaskRunner,FullyAsyncTaskRunner,"_run_training_loop"),(MultiTaskFullyAsyncTrainer,FullyAsyncTrainer,"fit"),(MultiTaskFullyAsyncRollouter,FullyAsyncRollouter,"fit")]:
        ec=unwrap_native_actor_class(ext); nc=unwrap_native_actor_class(native); assert issubclass(ec,nc); assert getattr(ec,method) is getattr(nc,method)

def test_manager_and_lb_extend_native_classes():
    assert issubclass(MultiTaskLLMServerManager,FullyAsyncLLMServerManager); assert issubclass(MultiTaskGlobalRequestLoadBalancer,GlobalRequestLoadBalancer)

def test_lb_adds_only_request_exit_metadata():
    server=object(); lb=MultiTaskGlobalRequestLoadBalancer({"s":server},full_determinism=True)
    assert lb.require_release_fields()==["request_id"]; assert lb.acquire_server("r")==("s",server); assert lb.query_attempt("r") is AttemptState.ADMITTED
    lb.release_server("s",request_id="r"); assert lb.query_attempt("r") is AttemptState.SETTLED
