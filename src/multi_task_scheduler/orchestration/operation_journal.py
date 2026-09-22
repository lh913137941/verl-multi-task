"""Small TaskRunner-owned lifecycle journal for the current protocol."""

from __future__ import annotations

from .contracts import OperationCommand, OperationRecord, OperationStatus


class OperationIdentityError(RuntimeError):
    """Raised for conflicting replay or overlapping lifecycle operations."""


_TERMINAL = {
    OperationStatus.SUCCEEDED,
    OperationStatus.FAILED,
    OperationStatus.UNKNOWN,
}


class OperationJournal:
    """Idempotency plus one unfinished lifecycle operation per task session."""

    def __init__(self) -> None:
        self._records: dict[str, OperationRecord] = {}
        self._commands: dict[str, OperationCommand] = {}
        self._active_by_task: dict[str, str] = {}

    def begin(self, command: OperationCommand) -> OperationRecord:
        if not isinstance(command, OperationCommand):
            raise TypeError("begin requires OperationCommand")
        existing = self._records.get(command.operation_id)
        if existing is not None:
            if self._commands[command.operation_id] != command:
                raise OperationIdentityError(
                    f"conflicting replay for operation {command.operation_id!r}"
                )
            return existing

        task_session = command.target.task_session
        active_id = self._active_by_task.get(task_session)
        if active_id is not None:
            active = self._records[active_id]
            if active.status not in _TERMINAL:
                raise OperationIdentityError(
                    f"another lifecycle operation is active for task {task_session!r}: {active_id!r}"
                )
            self._active_by_task.pop(task_session, None)

        record = OperationRecord(operation_id=command.operation_id)
        self._commands[command.operation_id] = command
        self._records[command.operation_id] = record
        self._active_by_task[task_session] = command.operation_id
        return record

    def query(self, operation_id: str) -> OperationRecord | None:
        return self._records.get(operation_id)

    def command(self, operation_id: str) -> OperationCommand:
        return self._commands[operation_id]

    def mark_running(self, operation_id: str, result: str | None = None) -> OperationRecord:
        record = self._records[operation_id]
        if record.status in _TERMINAL:
            raise OperationIdentityError("terminal operation cannot return to RUNNING")
        record.status = OperationStatus.RUNNING
        if result is not None:
            record.result = result
        return record

    def finish(
        self,
        operation_id: str,
        status: OperationStatus,
        result: str | None = None,
    ) -> OperationRecord:
        status = OperationStatus(status)
        if status not in _TERMINAL:
            raise ValueError("finish requires SUCCEEDED, FAILED, or UNKNOWN")
        record = self._records[operation_id]
        if record.status in _TERMINAL:
            if record.status is not status or record.result != result:
                raise OperationIdentityError("conflicting terminal replay")
            return record
        record.status = status
        record.result = result
        command = self._commands[operation_id]
        if self._active_by_task.get(command.target.task_session) == operation_id:
            self._active_by_task.pop(command.target.task_session, None)
        return record

    def active_operation(self, task_session: str) -> str | None:
        operation_id = self._active_by_task.get(task_session)
        if operation_id is None:
            return None
        if self._records[operation_id].status in _TERMINAL:
            self._active_by_task.pop(task_session, None)
            return None
        return operation_id
