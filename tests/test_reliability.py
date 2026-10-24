import pytest

from reliability import CircuitBreaker, CircuitOpenError, TTLCache


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_ttl_cache_expires_entries_and_evicts_least_recently_used():
    clock = Clock()
    cache = TTLCache[str](max_entries=2, ttl_seconds=10, clock=clock)
    cache.set("first", "one")
    cache.set("second", "two")

    assert cache.get("first") == "one"

    cache.set("third", "three")

    assert cache.get("second") is None
    assert cache.get("first") == "one"
    clock.advance(10)
    assert cache.get("first") is None


def test_disabled_ttl_cache_never_stores_values():
    cache = TTLCache[str](max_entries=0, ttl_seconds=10)

    cache.set("key", "value")

    assert cache.enabled is False
    assert cache.get("key") is None


def test_circuit_breaker_opens_and_allows_one_half_open_probe():
    clock = Clock()
    circuit = CircuitBreaker(
        failure_threshold=2,
        recovery_seconds=30,
        clock=clock,
    )

    circuit.before_call()
    circuit.record_failure()
    circuit.before_call()
    circuit.record_failure()

    with pytest.raises(CircuitOpenError) as exc_info:
        circuit.before_call()
    assert exc_info.value.retry_after == 30
    assert circuit.snapshot().state == "open"

    clock.advance(30)
    circuit.before_call()
    assert circuit.snapshot().state == "half_open"

    with pytest.raises(CircuitOpenError):
        circuit.before_call()

    circuit.record_success()
    assert circuit.snapshot().state == "closed"


def test_failed_half_open_probe_reopens_circuit():
    clock = Clock()
    circuit = CircuitBreaker(
        failure_threshold=1,
        recovery_seconds=20,
        clock=clock,
    )
    circuit.record_failure()
    clock.advance(20)
    circuit.before_call()

    circuit.record_failure()

    assert circuit.snapshot().state == "open"
    assert circuit.snapshot().retry_after == 20


def test_non_transient_probe_error_allows_another_recovery_attempt():
    clock = Clock()
    circuit = CircuitBreaker(failure_threshold=1, recovery_seconds=20, clock=clock)
    circuit.record_failure()
    clock.advance(20)
    circuit.before_call()

    circuit.release_probe()
    circuit.before_call()
    circuit.record_success()

    assert circuit.snapshot().state == "closed"
