import os
from pathlib import Path
import pytest
from multi_task_scheduler.integration.verl.runtime_profile import validate_runtime_profile

def test_current_profile_shape_without_native_import():
    c={"multitask":{"enabled":True,"runtime":{"profile":"experimental_fully_async_standalone"}},"actor_rollout_ref":{"hybrid_engine":False,"rollout":{"mode":"async","name":"vllm","calculate_log_probs":True,"nnodes":1,"n_gpus_per_node":8,"tensor_model_parallel_size":4,"data_parallel_size":1,"pipeline_model_parallel_size":1,"disaggregation":{"enabled":False},"checkpoint_engine":{"backend":"nccl"}}},"rollout":{"nnodes":1,"n_gpus_per_node":8},"async_training":{"use_trainer_do_validate":False,"use_dynamic_resource_scheduling":False},"data":{"train_batch_size":0,"gen_batch_size":1}}
    assert validate_runtime_profile(c)

def test_native_source_root_is_optional_for_control_plane_unit_suite():
    root=os.environ.get("MT_VERL_SOURCE_ROOT")
    if root: assert Path(root).is_absolute()
