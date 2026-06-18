"""Small dependency-free progress display."""

from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, label: str, total: int | None, *, enabled: bool = True):
        self.label = label
        self.total = total if total is None or total >= 0 else 0
        self.enabled = enabled and sys.stderr.isatty()
        self.current = 0
        self.started = time.monotonic()
        self.last_render = 0.0

    def __enter__(self) -> "ProgressBar":
        self.render(force=True)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finish()

    def update(self, amount: int = 1) -> None:
        self.current += amount
        self.render()

    def finish(self) -> None:
        if not self.enabled:
            return
        if self.total is not None and self.current < self.total:
            self.current = self.total
        self.render(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_render < 0.1:
            return
        self.last_render = now
        elapsed = max(now - self.started, 1e-9)
        rate = self.current / elapsed
        if self.total is None:
            body = f"{self.current} done | {rate:.1f}/s"
        elif self.total == 0:
            body = f"{bar(1.0)} 0/0 | 100%"
        else:
            fraction = min(1.0, self.current / self.total)
            body = f"{bar(fraction)} {self.current}/{self.total} | {fraction * 100:5.1f}% | {rate:.1f}/s"
        sys.stderr.write(f"\r{self.label}: {body}")
        sys.stderr.flush()


def bar(fraction: float, width: int = 28) -> str:
    filled = int(round(width * max(0.0, min(1.0, fraction))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"
