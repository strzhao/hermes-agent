"""Regression coverage for #147339: an ``internal_notification`` user row
must not blind ``_find_inflight_user_task``'s completion rescan.

A finished task ``A`` (real user ask -> assistant final answer) followed by a
``display_kind=internal_notification`` user row ``B`` used to be judged
in-flight forever: the completion rescan ``break``s on its first non-assistant
row, so it never reaches A's final answer and compaction replays the finished
work in the ``STILL IN PROGRESS`` handoff section -- exactly what the method's
docstring forbids ("replaying the ask would hand the model finished work as a
fresh instruction").

The fix skips user rows that are operational traffic (``display_kind`` set and
not a /steer row, i.e. the same filter ``_is_actionable_user_turn`` applies at
the anchor loop) while keeping the existing semantics for every other row
kind: a real human follow-up still ends the scan (the newest task remains
in-flight), and a trailing tool result still counts as mid-task.
"""

from typing import Any, Dict, List
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _make_compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        return ContextCompressor(
            model="test",
            quiet_mode=True,
            protect_first_n=2,
            protect_last_n=2,
        )


def _user(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def _final_answer(text: str = "Task A is complete: the digest was written to the inbox.") -> Dict[str, Any]:
    return {"role": "assistant", "content": text}


def _internal_notification_row(text: str = "[internal] background job notification: cron heartbeat recorded") -> Dict[str, Any]:
    return {"role": "user", "content": text, "display_kind": "internal_notification"}


def test_internal_notification_after_final_answer_does_not_replay_finished_task():
    """#147339: A -> final answer -> internal_notification B must yield None.

    Fails on main: the rescan breaks on B (first non-assistant row) and never
    reaches A's final answer, so the finished task is returned for replay.
    """
    compressor = _make_compressor()
    task_a = _user("Task A: brief the inbox and write a digest.")
    messages: List[Dict[str, Any]] = [
        task_a,
        _final_answer(),
        _internal_notification_row(),
    ]
    assert compressor._find_inflight_user_task(messages) is None


def test_transcript_without_notification_row_still_yields_none():
    """Control arm (pre-existing correct behavior): A -> final answer is
    finished with or without the notification row present."""
    compressor = _make_compressor()
    messages: List[Dict[str, Any]] = [
        _user("Task A: brief the inbox and write a digest."),
        _final_answer(),
    ]
    assert compressor._find_inflight_user_task(messages) is None


def test_real_user_followup_after_final_answer_keeps_task_inflight():
    """Second invariant: a REAL human follow-up after the final answer ends the
    completion scan; the newest real user turn stays in-flight. This pins the
    fix's narrowness -- only non-actionable user rows are skipped."""
    compressor = _make_compressor()
    task_a = _user("Task A: brief the inbox and write a digest.")
    messages: List[Dict[str, Any]] = [
        task_a,
        _final_answer(),
        _user("Also please summarize the archive folder."),
    ]
    inflight = compressor._find_inflight_user_task(messages)
    assert inflight is not None
    assert inflight is messages[2]


def test_trailing_tool_result_after_notification_row_still_midtask():
    """A trailing tool result after the notification row means the run was
    interrupted mid-task: the ask is still owed an answer. The notification
    skip must not swallow this mid-task signal."""
    compressor = _make_compressor()
    messages: List[Dict[str, Any]] = [
        _user("Task A: brief the inbox and write a digest."),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "tool output"},
        _internal_notification_row(),
        {"role": "tool", "tool_call_id": "c1", "content": "more output"},
    ]
    inflight = compressor._find_inflight_user_task(messages)
    assert inflight is not None
    assert inflight is messages[0]
