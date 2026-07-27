from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable


@dataclass(frozen=True)
class ActionLease:
    lease_id: str
    holder: str
    contract_id: str
    issued_at_ns: int
    expires_at_ns: int

    def is_expired(self, now_ns: int) -> bool:
        return now_ns >= self.expires_at_ns


class ActionLeaseManager:
    def __init__(self, clock: Callable[[], int] | None = None) -> None:
        self._clock = clock or time.time_ns
        self._active: ActionLease | None = None
        self._counter = 0

    def acquire(self, holder: str, contract_id: str, duration_ms: int) -> ActionLease:
        now = int(self._clock())
        self.expire_if_needed(now)
        if self._active is not None:
            raise RuntimeError("actuator lease already held")
        self._counter += 1
        lease = ActionLease(
            lease_id=f"lease-{self._counter}",
            holder=holder,
            contract_id=contract_id,
            issued_at_ns=now,
            expires_at_ns=now + duration_ms * 1_000_000,
        )
        self._active = lease
        return lease

    def release(self, lease_id: str) -> None:
        if self._active is None:
            return
        if self._active.lease_id != lease_id:
            raise ValueError("cannot release another holder's lease")
        self._active = None

    def expire_if_needed(self, now_ns: int | None = None) -> bool:
        now = int(self._clock()) if now_ns is None else now_ns
        if self._active is not None and self._active.is_expired(now):
            self._active = None
            return True
        return False

    def active(self) -> ActionLease | None:
        return self._active
