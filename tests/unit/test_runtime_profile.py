import copy
import pytest
from multi_task_scheduler.integration.verl.runtime_profile import PROFILE_ID, ProfileConfigurationError, validate_runtime_profile

def config():
    return {"multitask":{"runtime":{"profile":PROFILE_ID}},"actor_rollout_ref":{"hybrid_engine":False,"rollout":{"mode":"async","name":"vllm","calculate_log_probs":True,"nnodes":1,"n_gpus_per_node":8,"tensor_model_parallel_size":4,"data_parallel_size":1,"pipeline_model_parallel_size":1,"disaggregation":{"enabled":False},"checkpoint_engine":{"backend":"nccl"}}},"rollout":{"nnodes":1,"n_gpus_per_node":8},"async_training":{"use_trainer_do_validate":False,"use_dynamic_resource_scheduling":False},"data":{"train_batch_size":0,"gen_batch_size":1}}

def test_valid_profile_is_not_mutated():
    c=config(); before=copy.deepcopy(c); assert validate_runtime_profile(c); assert c==before

@pytest.mark.parametrize("path,value",[("nnodes",2),("data_parallel_size",2),("pipeline_model_parallel_size",2)])
def test_first_release_scope_rejects_cross_node_dp_pp(path,value):
    c=config(); c["actor_rollout_ref"]["rollout"][path]=value
    if path=="nnodes": c["rollout"]["nnodes"]=value
    with pytest.raises(ProfileConfigurationError): validate_runtime_profile(c)

def test_tp_must_fit_single_node():
    c=config(); c["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"]=9
    with pytest.raises(ProfileConfigurationError): validate_runtime_profile(c)
