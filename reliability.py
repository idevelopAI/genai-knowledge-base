import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class CircuitSnapshot:
    state: str
    consecutive_failures: int
    retry_after: int


class CircuitOpenError(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__("Provider circuit is open.")
        self.retry_after = retry_after


class TTLCache(Generic[ValueT]):
    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max(0, max_entries)
        self.ttl_seconds = max(0, ttl_seconds)
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, ValueT]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_seconds > 0

    def get(self, key: str) -> ValueT | None:
        if not self.enabled:
            return None

        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def set(self, key: str, value: ValueT) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._entries[key] = (self._clock() + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int,
        recovery_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(1, recovery_seconds)
        self._clock = clock
        self._consecutive_failures = 0
        self._open_until: float | None = None
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def before_call(self) -> None:
        now = self._clock()
        with self._lock:
            if self._open_until is None:
                return
            if now < self._open_until:
                raise CircuitOpenError(self._retry_after(now))
            if self._probe_in_flight:
                raise CircuitOpenError(1)
            self._probe_in_flight = True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._open_until = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        now = self._clock()
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._probe_in_flight
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._open_until = now + self.recovery_seconds
            self._probe_in_flight = False

    def release_probe(self) -> None:
        """Allow another recovery attempt after a non-transient call error."""
        with self._lock:
            self._probe_in_flight = False

    def snapshot(self) -> CircuitSnapshot:
        now = self._clock()
        with self._lock:
            if self._open_until is None:
                state = "closed"
            elif now < self._open_until:
                state = "open"
            else:
                state = "half_open"
            return CircuitSnapshot(
                state=state,
                consecutive_failures=self._consecutive_failures,
                retry_after=self._retry_after(now),
            )

    def reset(self) -> None:
        self.record_success()

    def _retry_after(self, now: float) -> int:
        if self._open_until is None:
            return 0
        return max(0, math.ceil(self._open_until - now))
