"""Token-bucket rate limiter with SQLite persistence."""

import asyncio
import random
from datetime import datetime, timezone

import sqlalchemy as sa

from prospect.db import get_engine, rate_limits, init_db


class RateLimiter:
    def __init__(self, action_type: str, max_per_day: int, delay_min: float = 2.0, delay_max: float = 6.0):
        self.action_type = action_type
        self.max_per_day = max_per_day
        self.delay_min = delay_min
        self.delay_max = delay_max
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = init_db()
        return self._engine

    def _today_start(self) -> datetime:
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_count(self) -> int:
        """Get today's action count."""
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(rate_limits.c.count, rate_limits.c.window_start).where(
                    rate_limits.c.action_type == self.action_type
                )
            ).first()

            if row is None:
                return 0

            window_start = row.window_start
            if isinstance(window_start, str):
                window_start = datetime.fromisoformat(window_start)
            if window_start.tzinfo is None:
                window_start = window_start.replace(tzinfo=timezone.utc)

            if window_start < self._today_start():
                # Reset for new day
                conn.execute(
                    sa.update(rate_limits)
                    .where(rate_limits.c.action_type == self.action_type)
                    .values(count=0, window_start=self._today_start())
                )
                conn.commit()
                return 0

            return row.count

    def remaining(self) -> int:
        return max(0, self.max_per_day - self.get_count())

    def can_acquire(self) -> bool:
        return self.get_count() < self.max_per_day

    def increment(self):
        """Record one action."""
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(rate_limits.c.id).where(
                    rate_limits.c.action_type == self.action_type
                )
            ).first()

            if row is None:
                conn.execute(
                    sa.insert(rate_limits).values(
                        action_type=self.action_type,
                        count=1,
                        window_start=self._today_start(),
                    )
                )
            else:
                conn.execute(
                    sa.update(rate_limits)
                    .where(rate_limits.c.action_type == self.action_type)
                    .values(count=rate_limits.c.count + 1)
                )
            conn.commit()

    async def wait_and_acquire(self):
        """Wait a human-like delay, then increment the counter. Raises if limit reached."""
        if not self.can_acquire():
            raise RuntimeError(
                f"Daily limit reached for {self.action_type}: "
                f"{self.get_count()}/{self.max_per_day}. Try again tomorrow."
            )

        # Human-like delay with gaussian jitter
        base = random.uniform(self.delay_min, self.delay_max)
        jitter = random.gauss(0, 0.3)
        delay = max(self.delay_min, base + jitter)
        await asyncio.sleep(delay)

        self.increment()
