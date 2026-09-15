"""Serving concerns for a shared deployment.

Two problems, both arising from several people using one GPU.

*Serialization.* Ollama generates one response at a time. Without a queue,
concurrent requests pile up inside Ollama with no feedback -- the third person
waits a minute with a spinner and no idea why. RequestQueue puts an explicit
queue in front, so each person can be told where they are in line.

*Politeness.* The card is shared with other people's jobs. Ollama keeps a model
resident in VRAM for five minutes after the last request by default, which is
invisible to the user and very visible to whoever wants the card at 2am. A
short keep-alive releases it promptly, at the cost of a few seconds' reload on
the next question.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass


class QueueTimeout(RuntimeError):
    """Waited too long for a turn."""


class RequestQueue:
    """First-come-first-served access to a single resource.

    A plain lock would serialise correctly but could not answer "how many
    people are ahead of me", and Python locks make no fairness guarantee, so a
    waiter can in principle be skipped repeatedly. Holding an explicit deque of
    tokens gives both ordering and position.

    Abandoning the queue -- by timing out, or by the user closing the tab --
    removes that token, so the people behind advance rather than waiting on a
    turn that will never be taken.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._queue: deque = deque()

    def depth(self) -> int:
        """How many requests are queued, including the one being served."""
        with self._cv:
            return len(self._queue)

    @contextmanager
    def slot(self, on_wait=None, poll: float = 0.4, timeout: float = 900):
        """Block until it is our turn, then yield.

        `on_wait(ahead)` is called roughly every `poll` seconds while waiting,
        with the number of requests still in front.
        """
        token = object()
        with self._cv:
            self._queue.append(token)

        deadline = time.monotonic() + timeout
        try:
            while True:
                with self._cv:
                    ahead = self._queue.index(token)
                if ahead == 0:
                    break
                if time.monotonic() > deadline:
                    raise QueueTimeout(
                        f"still {ahead} ahead after {timeout:.0f}s")
                if on_wait is not None:
                    on_wait(ahead)
                time.sleep(poll)
            yield
        finally:
            with self._cv:
                try:
                    self._queue.remove(token)
                except ValueError:
                    pass
                self._cv.notify_all()


# --------------------------------------------------------------------------
# hardware and backend checks
# --------------------------------------------------------------------------

@dataclass
class GPU:
    index: int
    name: str
    total_mb: int
    used_mb: int
    util_pct: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb

    def __str__(self) -> str:
        return (f"GPU {self.index} ({self.name}): {self.free_mb/1024:.1f} GB "
                f"free of {self.total_mb/1024:.1f}, {self.util_pct}% busy")


def gpus(timeout: float = 5) -> list[GPU] | None:
    """Query nvidia-smi. None if it is unavailable (no driver, no GPU)."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None

    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            out.append(GPU(int(parts[0]), parts[1], int(parts[2]),
                           int(parts[3]), int(parts[4])))
        except ValueError:
            continue
    return out


def ollama_up(host: str = "http://localhost:11434", timeout: float = 3) -> bool:
    try:
        import requests
        return requests.get(f"{host.rstrip('/')}/api/tags",
                            timeout=timeout).ok
    except Exception:                                   # noqa: BLE001
        return False


def ollama_loaded(host: str = "http://localhost:11434",
                  timeout: float = 3) -> list[str]:
    """Models currently resident in memory."""
    try:
        import requests
        r = requests.get(f"{host.rstrip('/')}/api/ps", timeout=timeout)
        r.raise_for_status()
        return [m.get("name", "?") for m in r.json().get("models", [])]
    except Exception:                                   # noqa: BLE001
        return []


def local_backend_ready(host: str = "http://localhost:11434",
                        need_mb: int = 6000) -> tuple[bool, str]:
    """Can we serve a local request right now, and if not, why not?

    Returns (ready, message). The message is shown to the user, so it says
    what is wrong and what they can do, not just that something failed.
    """
    if not ollama_up(host):
        return False, ("The local model service is not responding. It may be "
                       "restarting; try again shortly, or switch to the "
                       "hosted model in the sidebar.")

    cards = gpus()
    if cards is None:
        # No nvidia-smi does not mean no service -- Ollama falls back to CPU,
        # which is slow but works.
        return True, "Running without a GPU; answers will be slower than usual."

    if ollama_loaded(host):
        return True, "Model is loaded and ready."

    roomiest = max(cards, key=lambda g: g.free_mb)
    if roomiest.free_mb < need_mb:
        detail = "; ".join(str(g) for g in cards)
        return False, (
            f"The GPU is busy with other work, so the model cannot load right "
            f"now ({detail}). Switch to the hosted model in the sidebar, or "
            f"try again later.")

    return True, f"Ready. {roomiest}"