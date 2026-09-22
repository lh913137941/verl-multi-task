import ast
import threading
from pathlib import Path
from multi_task_scheduler.orchestration.contracts import OperationCommand, OperationRecord, OperationStatus, ReplicaKey, ReplicaKind, ReplicaState
from multi_task_scheduler.orchestration.operation_journal import OperationJournal
SOURCE=Path(__file__).resolve().parents[2]/"src/multi_task_scheduler"
INTEGRATION="integration/verl/experimental_fully_async"

def isolated(relative,name,parent,**scope):
    path=SOURCE/relative; tree=ast.parse(path.read_text()); node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name); node.bases=[ast.Name(id="Parent",ctx=ast.Load())]; node.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[]); env={"Parent":parent,**scope}; exec(compile(ast.fix_missing_locations(module),str(path),"exec"),env); return env[name]

def test_taskrunner_minimal_journal_surface():
    class Parent:
        def __init__(self): self.components={}
    cls=isolated(f"{INTEGRATION}/task_runner.py","MultiTaskFullyAsyncTaskRunner",Parent,OperationJournal=OperationJournal,OperationCommand=OperationCommand,OperationRecord=OperationRecord,OperationStatus=OperationStatus,threading=threading)
    runner=cls(); runner.task_session="task-a"; cmd=OperationCommand("op","ADD",ReplicaKey("task-a","r0"),"l1")
    assert runner.submit_operation(cmd).status is OperationStatus.ACCEPTED
    assert runner.query_operation("missing").status is OperationStatus.UNKNOWN

def test_manager_owns_state_and_kind_maps_without_replica_record():
    class Parent:
        def __init__(self,*args): pass
    allowed={ReplicaState.CREATING:{ReplicaState.ACTIVE,ReplicaState.RELEASED,ReplicaState.QUARANTINED},ReplicaState.ACTIVE:{ReplicaState.DRAINING},ReplicaState.DRAINING:{ReplicaState.ACTIVE,ReplicaState.DORMANT,ReplicaState.RELEASED,ReplicaState.QUARANTINED},ReplicaState.DORMANT:{ReplicaState.ACTIVE,ReplicaState.QUARANTINED},ReplicaState.RELEASED:set(),ReplicaState.QUARANTINED:set()}
    cls=isolated(f"{INTEGRATION}/llm_server_manager.py","MultiTaskLLMServerManager",Parent,MultiTaskvLLMReplica=object(),MultiTaskGlobalRequestLoadBalancer=object(),ReplicaKey=ReplicaKey,ReplicaKind=ReplicaKind,ReplicaState=ReplicaState,_ALLOWED=allowed)
    m=cls(object()); key=ReplicaKey("task-a","r0"); m.register_replica(key,ReplicaKind.BORROWED); m.transition_replica(key,ReplicaState.ACTIVE); m.transition_replica(key,ReplicaState.DRAINING); m.transition_replica(key,ReplicaState.RELEASED)
    assert m.replica_meta(key)==(ReplicaKind.BORROWED,ReplicaState.RELEASED); assert not hasattr(m,"_lifecycle")

def test_rollouter_idle_detection_does_not_read_lb():
    class Parent:
        def __init__(self,*args,**kwargs): self.paused=True; self.max_concurrent_samples=8
    cls=isolated(f"{INTEGRATION}/rollouter.py","MultiTaskFullyAsyncRollouter",Parent,ReplicaKey=ReplicaKey,ReplicaKind=ReplicaKind,ReplicaState=ReplicaState,OperationRecord=OperationRecord)
    r=cls(object(),object()); key=ReplicaKey("task-a","r0"); r.llm_server_manager=type("M",(),{"replica_state":{key:ReplicaState.ACTIVE},"replica_kind":{key:ReplicaKind.NATIVE}})()
    assert r.collect_idle_candidates()==((key,ReplicaKind.NATIVE),); r.paused=False; assert r.collect_idle_candidates()==()
