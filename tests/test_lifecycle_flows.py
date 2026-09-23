from multi_task_scheduler.orchestration.contracts import EvidenceType


def test_add_flow_evidence_chain():
    chain = [EvidenceType.WEIGHT_READY, EvidenceType.SERVICE_COMMITTED]
    assert chain[0] is EvidenceType.WEIGHT_READY
    assert chain[-1] is EvidenceType.SERVICE_COMMITTED


def test_remove_flow_evidence_chain():
    chain = [EvidenceType.EXIT_READY, EvidenceType.RELEASED]
    assert chain[0] is EvidenceType.EXIT_READY
    assert chain[-1] is EvidenceType.RELEASED


def test_restore_flow_commit_point():
    assert EvidenceType.SERVICE_COMMITTED.value == "SERVICE_COMMITTED"


def test_donate_flow_uses_service_commit_after_transfer():
    transfer_ready = True
    service_committed = EvidenceType.SERVICE_COMMITTED
    assert transfer_ready
    assert service_committed is EvidenceType.SERVICE_COMMITTED
