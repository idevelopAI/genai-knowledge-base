import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


TERMINAL_STATUSES = {"completed", "failed"}


class JobCapacityError(RuntimeError):
    pass


@dataclass
class IndexJob:
    job_id: str
    operation: str
    status: str
    request_id: str
    created_at: str
    updated_at: str
    filename: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class IndexJobStore:
    def __init__(self, max_jobs: int = 100):
        self.max_jobs = max_jobs
        self._jobs: dict[str, IndexJob] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _prune_terminal_jobs(self) -> None:
        terminal_jobs = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in TERMINAL_STATUSES
            ),
            key=lambda job: job.updated_at,
        )
        while len(self._jobs) >= self.max_jobs and terminal_jobs:
            oldest = terminal_jobs.pop(0)
            self._jobs.pop(oldest.job_id, None)

    def create(
        self,
        *,
        operation: str,
        request_id: str,
        filename: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._prune_terminal_jobs()
            if len(self._jobs) >= self.max_jobs:
                raise JobCapacityError("Index job queue is at capacity.")

            timestamp = self._timestamp()
            job = IndexJob(
                job_id=str(uuid.uuid4()),
                operation=operation,
                status="queued",
                request_id=request_id,
                filename=filename,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._jobs[job.job_id] = job
            return asdict(job)

    def update(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = status
            job.result = result
            job.error = error
            if filename is not None:
                job.filename = filename
            job.updated_at = self._timestamp()
            return asdict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job is not None else None

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()
