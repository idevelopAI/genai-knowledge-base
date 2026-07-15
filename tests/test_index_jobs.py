import pytest

from index_jobs import IndexJobStore, JobCapacityError


def test_job_lifecycle_returns_isolated_snapshots():
    store = IndexJobStore(max_jobs=2)
    created = store.create(operation="reindex", request_id="request-1")
    created["status"] = "modified-outside-store"

    queued = store.get(created["job_id"])
    assert queued["status"] == "queued"

    store.update(created["job_id"], status="running")
    result = {"status": "reindexed", "chunks_created": 3}
    completed = store.update(
        created["job_id"],
        status="completed",
        result=result,
    )

    assert completed["status"] == "completed"
    assert completed["result"] == result


def test_terminal_jobs_are_pruned_before_rejecting_new_work():
    store = IndexJobStore(max_jobs=2)
    first = store.create(operation="reindex", request_id="request-1")
    store.update(first["job_id"], status="completed", result={})
    second = store.create(operation="reindex", request_id="request-2")

    third = store.create(operation="upload", request_id="request-3")

    assert store.get(first["job_id"]) is None
    assert store.get(second["job_id"])["status"] == "queued"
    assert store.get(third["job_id"])["status"] == "queued"


def test_active_job_capacity_is_enforced():
    store = IndexJobStore(max_jobs=1)
    store.create(operation="reindex", request_id="request-1")

    with pytest.raises(JobCapacityError, match="capacity"):
        store.create(operation="upload", request_id="request-2")
