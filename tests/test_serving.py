"""Tests for the serving layer: request queueing and hardware checks."""

from __future__ import annotations

import threading
import time

import pytest

from xenonrag import serving
from xenonrag.serving import (GPU, QueueTimeout, RequestQueue,
                              local_backend_ready)


# --------------------------------------------------------------------------
# queue
# --------------------------------------------------------------------------

def test_single_request_does_not_wait():
    q = RequestQueue()
    seen = []
    with q.slot(on_wait=seen.append, poll=0.01):
        pass
    assert seen == []


def test_requests_are_served_in_order():
    q = RequestQueue()
    order = []

    def worker(name):
        with q.slot(poll=0.01):
            order.append(name)
            time.sleep(0.05)

    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, args=(f"u{i}",))
        threads.append(t)
        t.start()
        time.sleep(0.02)       # stagger so arrival order is unambiguous
    for t in threads:
        t.join()

    assert order == ["u0", "u1", "u2", "u3"]


def test_only_one_request_runs_at_a_time():
    """The whole point: Ollama generates one response at a time."""
    q = RequestQueue()
    concurrent = []
    live = 0
    lock = threading.Lock()

    def worker():
        nonlocal live
        with q.slot(poll=0.01):
            with lock:
                live += 1
                concurrent.append(live)
            time.sleep(0.04)
            with lock:
                live -= 1

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max(concurrent) == 1


def test_waiter_is_told_how_many_are_ahead():
    q = RequestQueue()
    seen = []

    def hog():
        with q.slot(poll=0.01):
            time.sleep(0.15)

    a = threading.Thread(target=hog)
    a.start()
    time.sleep(0.03)

    with q.slot(on_wait=seen.append, poll=0.01):
        pass
    a.join()

    assert seen and all(n == 1 for n in seen)


def test_abandoning_the_queue_unblocks_the_people_behind():
    """A timed-out waiter must not leave a gap nobody can step past."""
    q = RequestQueue()
    events = []

    def hog():
        with q.slot(poll=0.01):
            time.sleep(0.4)
            events.append("hog")

    def quitter():
        try:
            with q.slot(poll=0.01, timeout=0.1):
                events.append("quitter ran")
        except QueueTimeout:
            events.append("quitter gave up")

    def after():
        with q.slot(poll=0.01, timeout=5):
            events.append("after")

    a = threading.Thread(target=hog)
    a.start()
    time.sleep(0.03)
    b = threading.Thread(target=quitter)
    b.start()
    time.sleep(0.02)
    c = threading.Thread(target=after)
    c.start()
    for t in (a, b, c):
        t.join()

    assert "quitter gave up" in events
    assert "after" in events


def test_timeout_raises():
    q = RequestQueue()

    def hog():
        with q.slot(poll=0.01):
            time.sleep(0.3)

    t = threading.Thread(target=hog)
    t.start()
    time.sleep(0.03)
    with pytest.raises(QueueTimeout):
        with q.slot(poll=0.01, timeout=0.08):
            pass
    t.join()


def test_an_exception_inside_the_slot_still_releases_it():
    q = RequestQueue()
    with pytest.raises(ValueError):
        with q.slot(poll=0.01):
            raise ValueError("boom")
    assert q.depth() == 0
    with q.slot(poll=0.01, timeout=0.5):
        pass


def test_depth_reports_queued_requests():
    q = RequestQueue()
    assert q.depth() == 0

    started = threading.Event()

    def hog():
        with q.slot(poll=0.01):
            started.set()
            time.sleep(0.15)

    t = threading.Thread(target=hog)
    t.start()
    started.wait(1)
    assert q.depth() == 1
    t.join()
    assert q.depth() == 0


# --------------------------------------------------------------------------
# hardware checks
# --------------------------------------------------------------------------

def test_gpu_free_memory():
    g = GPU(index=1, name="NVIDIA TITAN V", total_mb=12288, used_mb=746,
            util_pct=0)
    assert g.free_mb == 11542
    assert "11.3 GB free" in str(g)


def test_not_ready_when_the_service_is_down(monkeypatch):
    monkeypatch.setattr(serving, "ollama_up", lambda *a, **k: False)
    ready, why = local_backend_ready()
    assert not ready
    assert "not responding" in why
    # The message must tell the user what to do, not just that it broke.
    assert "hosted model" in why


def test_ready_without_a_gpu_but_warns_about_speed(monkeypatch):
    monkeypatch.setattr(serving, "ollama_up", lambda *a, **k: True)
    monkeypatch.setattr(serving, "gpus", lambda *a, **k: None)
    ready, why = local_backend_ready()
    assert ready
    assert "slower" in why


def test_not_ready_when_the_card_is_full(monkeypatch):
    monkeypatch.setattr(serving, "ollama_up", lambda *a, **k: True)
    monkeypatch.setattr(serving, "ollama_loaded", lambda *a, **k: [])
    monkeypatch.setattr(serving, "gpus", lambda *a, **k: [
        GPU(0, "TITAN V", 12288, 11904, 99),
        GPU(1, "TITAN V", 12288, 11000, 80),
    ])
    ready, why = local_backend_ready(need_mb=6000)
    assert not ready
    assert "busy with other work" in why


def test_ready_when_one_card_has_room(monkeypatch):
    """Only the roomiest card matters -- a full GPU 0 is irrelevant if GPU 1
    is free."""
    monkeypatch.setattr(serving, "ollama_up", lambda *a, **k: True)
    monkeypatch.setattr(serving, "ollama_loaded", lambda *a, **k: [])
    monkeypatch.setattr(serving, "gpus", lambda *a, **k: [
        GPU(0, "TITAN V", 12288, 11904, 99),
        GPU(1, "TITAN V", 12288, 746, 0),
    ])
    ready, why = local_backend_ready(need_mb=6000)
    assert ready
    assert "GPU 1" in why


def test_already_loaded_model_is_ready_regardless_of_free_memory(monkeypatch):
    """The memory it needs is already allocated to it."""
    monkeypatch.setattr(serving, "ollama_up", lambda *a, **k: True)
    monkeypatch.setattr(serving, "ollama_loaded", lambda *a, **k: ["qwen3:8b"])
    monkeypatch.setattr(serving, "gpus", lambda *a, **k: [
        GPU(1, "TITAN V", 12288, 11900, 95)])
    ready, _ = local_backend_ready(need_mb=6000)
    assert ready


def test_gpu_query_survives_missing_nvidia_smi(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(serving.subprocess, "run", boom)
    assert serving.gpus() is None