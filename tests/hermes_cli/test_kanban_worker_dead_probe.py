"""Transient identity-read failures must never book a crash of a live worker (fleet R5).

The crash sweep (``_reclaim_dead_workers``) declared live workers dead whenever the spawn
fingerprint could not be re-read: ``_pid_recycled`` failed CLOSED, so one process-table / psutil
hiccup on a loaded dispatch host released the claim and counted a failure (42 of 43 'pid not
alive' reclaims were exactly that; heartbeats were 2-70 s fresh and are never consulted here).

The fix splits the question into verdicts: only a PROVEN recycle (MISMATCH) or a GONE pid may
book a crash. An UNREADABLE verdict is an unknown, never evidence: the card is held for as long
as the unreadability lasts — no confirmation window and no amount of elapsed time converts it
into a death — and it is never signalled (missing identity is refusal, not permission, #99558).
The sighting ledger observes how long THIS pid/fingerprint incarnation has been unreadable; it
never authorises a crash, and stale memory never vouches for an incarnation it did not see.
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kbd._DEAD_PROBE_MEMORY.clear()
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()
        kbd._DEAD_PROBE_MEMORY.clear()


def _running_with_pid(conn, *, pid, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
            (started_at, old, old, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid),
        )
    return tid


def _real_fingerprint(pid: int) -> str:
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time

    return f"{current_instantiation_epoch()}|{get_process_start_time(pid)}"


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=15)
    return proc.pid


def _live_child() -> subprocess.Popen:
    """A real live process that is not this test (a stand-in worker incarnation)."""
    return subprocess.Popen([sys.executable, "-c", "import threading; threading.Event().wait()"])


def _ledger_entry(conn, task_id):
    path = kbd._dead_probe_ledger_path(None)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(task_id)


def _rewrite_ledger(conn, task_id, entry) -> None:
    path = kbd._dead_probe_ledger_path(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({task_id: entry}), encoding="utf-8")


# --- verdict semantics -------------------------------------------------------


def test_unreadable_fingerprint_reads_as_alive(board, monkeypatch):
    """Our own live PID with a fingerprint whose birth CANNOT be re-read is alive-but-unverified:
    the claim is extended, not reclaimed, and nothing is signalled."""
    conn = board
    killed: list = []
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)

    assert kbd._worker_alive(os.getpid(), "9e-fake|123456789") is True
    assert kbd._pid_recycled(os.getpid(), "9e-fake|123456789") is False
    assert kbd._worker_is_signal_eligible(os.getpid(), "9e-fake|123456789") is False
    assert kbd.detect_crashed_workers(conn) == []
    assert kb.get_task(conn, tid).status == "running"
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 0
    assert killed == []


def test_start_time_read_failure_reads_as_alive(board, monkeypatch):
    """Legacy integer fingerprint: a read failure (None) is UNREADABLE, not 'recycled'."""
    conn = board
    tid = _running_with_pid(conn, pid=os.getpid(), started_at=1)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: None)

    state, probe = kbd.probe_worker_identity(os.getpid(), 1)
    assert state == kbd.WORKER_IDENTITY_UNREADABLE
    assert probe["read_failure"] == "start_time_unreadable"
    assert kbd._worker_alive(os.getpid(), 1) is True
    assert kb.get_task(conn, tid).status == "running"


def test_proven_mismatch_still_reclaims_immediately(board):
    """A live PID with a DIFFERENT verifiable fingerprint is a proven recycle: single probe,
    immediate reclaim, never signalled (behaviour preserved from the fingerprint contract)."""
    conn = board
    killed: list = []
    stranger = _real_fingerprint(os.getpid()) + "-foreign"
    tid = _running_with_pid(conn, pid=os.getpid(), started_at=stranger, max_runtime=1)

    assert kbd.detect_crashed_workers(conn) == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == []
    assert killed == []


def test_gone_pid_reclaims_immediately(board):
    """A pid that is verifiably NOT in the process table books on the first probe (control)."""
    conn = board
    tid = _running_with_pid(conn, pid=_dead_pid(), started_at=_real_fingerprint(os.getpid()))
    assert kbd.detect_crashed_workers(conn) == [tid]
    assert kb.get_task(conn, tid).status == "ready"


# --- an unknown verdict is never death evidence ---------------------------------


def test_persistent_unreadable_sightings_are_never_booked(board, monkeypatch):
    """Two UNREADABLE verdicts are two unknowns, not a death: no amount of elapsed time converts
    an unreadable identity into crash evidence — only this tick's GONE/MISMATCH may book."""
    conn = board
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")

    assert kbd.detect_crashed_workers(conn) == []           # tick 1: hold
    assert kb.get_task(conn, tid).status == "running"
    entry = _ledger_entry(conn, tid)
    assert entry is not None
    entry["first_seen"] = time.time() - 3601                # an hour of unreadability
    _rewrite_ledger(conn, tid, entry)

    assert kbd.detect_crashed_workers(conn) == []           # still unknown → still held
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid is not None


def test_read_recovery_drops_the_sighting(board, monkeypatch):
    """When the birth read recovers and matches, the worker is simply alive: no crash was ever
    booked and the stale sighting is dropped."""
    conn = board
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    real = _real_fingerprint(os.getpid())
    tid = _running_with_pid(conn, pid=os.getpid(), started_at=real)

    assert kbd.detect_crashed_workers(conn) == []
    assert _ledger_entry(conn, tid) is not None

    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: _real_fingerprint(pid))
    assert kbd.detect_crashed_workers(conn) == []           # alive again: nothing to book
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid is not None
    assert _ledger_entry(conn, tid) is None                 # sighting dropped on recovery


def test_lost_disk_ledger_memory_never_vouches_for_a_new_incarnation(board, monkeypatch):
    """A disk ledger lost to cleanup/corruption must not let in-process memory of a PREVIOUS
    incarnation answer for the new worker's first unreadable sighting: the note restarts."""
    conn = board
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    child = _live_child()
    try:
        tid = _running_with_pid(conn, pid=child.pid, started_at="9e-fake|777")
        # The only record of THIS incarnation was the disk row, and it is gone; the surviving
        # in-process memory belongs to an OLDER incarnation of the same task (61 s stale, foreign
        # pid/fingerprint).
        kbd._DEAD_PROBE_MEMORY[tid] = (time.time() - 3601, 4242, "stale")
        now = time.time()

        assert kbd.detect_crashed_workers(conn) == []       # hold — never booked on sighting one
        task = kb.get_task(conn, tid)
        assert task.status == "running" and task.worker_pid is not None
        entry = _ledger_entry(conn, tid)
        assert entry is not None
        assert entry["pid"] == child.pid and entry["fingerprint"] == "9e-fake|777"
        assert entry["first_seen"] == pytest.approx(now, abs=10)  # restarted, not vouched
    finally:
        child.kill()
        child.wait(timeout=15)


def test_disk_row_loss_same_incarnation_memory_keeps_the_age(board, monkeypatch):
    """When the disk row is lost but in-process memory saw the SAME incarnation, the note keeps
    its age: first_seen is a property of the incarnation, not of the last write."""
    conn = board
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")

    assert kbd.detect_crashed_workers(conn) == []
    kbd._dead_probe_ledger_path(None).unlink(missing_ok=True)
    aged = time.time() - 30
    kbd._DEAD_PROBE_MEMORY[tid] = (aged, os.getpid(), "9e-fake|123456789")

    assert kbd.detect_crashed_workers(conn) == []
    entry = _ledger_entry(conn, tid)
    assert entry is not None
    assert entry["first_seen"] == pytest.approx(aged, abs=5)


def test_stale_sighting_for_a_previous_incarnation_is_renoted(board, monkeypatch):
    """A sighting recorded against an older pid/fingerprint must not answer for the current
    worker: its first sighting restarts the note against THIS incarnation."""
    conn = board
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")
    old = {"first_seen": time.time() - 9999, "pid": 4242, "fingerprint": "old"}
    _rewrite_ledger(conn, tid, old)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)

    assert kbd.detect_crashed_workers(conn) == []           # re-noted against THIS incarnation
    entry = _ledger_entry(conn, tid)
    assert entry["pid"] == os.getpid() and entry["fingerprint"] == "9e-fake|123456789"


def test_same_incarnation_sighting_preserves_first_seen(board, monkeypatch):
    """A sub-minute tick must not reset the sighting's age: first_seen is preserved while the
    pid/fingerprint keep matching (a high-frequency ticker could otherwise defer forever)."""
    conn = board
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")

    assert kbd.detect_crashed_workers(conn) == []
    entry = _ledger_entry(conn, tid)
    aged = time.time() - 30
    entry["first_seen"] = aged
    _rewrite_ledger(conn, tid, entry)

    assert kbd.detect_crashed_workers(conn) == []
    entry = _ledger_entry(conn, tid)
    assert entry["first_seen"] == pytest.approx(aged, abs=5)


def test_darwin_ps_failure_keeps_the_kill_zero_answer(board, monkeypatch):
    """A darwin ``ps`` stat probe that fails (non-zero rc) keeps the kill(0) positive answer,
    like the timeout branch: ps failing to READ a process is not evidence that it exited."""
    from gateway import status

    def broken_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(subprocess, "run", broken_run)
    assert kbd._pid_alive(os.getpid()) is True


# --- signal safety on the unreadable path -------------------------------------


def test_termination_refuses_unreadable_identity(board, monkeypatch):
    """The fail-open reclaim must NOT turn into a licence to signal by number: an unreadable live
    worker is reported as surviving (signal_refused), never signalled."""
    killed: list = []
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    info = kbd._terminate_reclaimed_worker(
        os.getpid(), f"{kb._host_prefix()}:x",
        signal_fn=lambda pid, sig: killed.append((pid, sig)),
        started_at="9e-fake|123456789",
    )
    assert killed == []
    assert info["termination_attempted"] is False
    assert info.get("signal_refused") is True
    assert info.get("identity_unreadable") is True
    assert info["terminated"] is False


def test_timeout_path_never_signals_an_unreadable_worker(board, monkeypatch):
    """max-runtime enforcement over a live-but-unreadable worker: no SIGTERM, no SIGKILL, the card
    stays held until the identity reads again or the pid exits."""
    conn = board
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    killed: list = []
    tid = _running_with_pid(
        conn, pid=os.getpid(), started_at="9e-fake|123456789", max_runtime=1,
    )
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == []
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"


def test_max_runtime_still_signals_a_gone_worker(board):
    """A max-runtime overrun whose worker is verifiably GONE keeps the old contract's harmless
    SIGTERM attempt (the raised ProcessLookupError is suppressed) and is booked: a dead pid is
    not an identity hazard. Control for tests/hermes_cli/test_kanban_core_functionality.py."""
    conn = board
    killed: list = []
    gone = _dead_pid()
    tid = _running_with_pid(conn, pid=gone, started_at="9e-fake|123456789", max_runtime=1)

    assert kbd.enforce_max_runtime(
        conn, signal_fn=lambda pid, sig: killed.append((pid, sig)),
    ) == [tid]
    assert killed == [(gone, signal.SIGTERM)]
    assert kb.get_task(conn, tid).status == "ready"
