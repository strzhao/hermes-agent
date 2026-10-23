"""Sent-but-not-saved hook context: ``pre_llm_call`` results with ``ephemeral: true``.

Issue #143739: every ``pre_llm_call`` context injection used to be persisted —
``_collect_pre_llm_call_context`` folded all hook results into one string, the
prologue stamps it into the ``api_content`` sidecar, the flush writes it to the
user row, and every later turn replays those bytes. Plugins (memory/RAG/guard-
rail class) need the opposite of one shape: context the model sees THIS turn
but that leaves no trace — not in the row, not in replay, not in review/compac-
tion views (which read clean content and are automatically clean).

Contract under test (additive; everything else byte-identical):
- hook returns ``str`` or ``{"context": str}``            -> durable (today's behaviour)
- hook returns ``{"context": str, "ephemeral": True}``    -> sent-but-not-saved:
  on the wire for every API call of this turn, trailing the durable stream,
  never in the sidecar => never persisted, never replayed.
- unknown shapes fail open into the durable path (turn completes, nothing new
  is rejected).

Invariants (one row per test, referenced in the delivery FORGE.md):
  (a) ephemeral: row ``content`` and ``api_content`` carry no sentinel;
  (b) ephemeral: replay through the real chain carries no sentinel;
  (c) ephemeral: every iteration's wire bytes carry the sentinel;
  (d) durable shapes (incl. bare str): wire, sidecar and replay unchanged.

The DB assertions open the session sqlite file READ-ONLY (``mode=ro``): the
same protocol production verification must use against a live state.db.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.turn_context import _collect_pre_llm_call_context, build_api_messages

# Unique sentinels: never substrings of each other, never produced by fixtures.
EPH = "EPHEMERAL-ONLY-SENTINEL-2f8a1c"
DUR = "DURABLE-CTX-SENTINEL-9b4d7e"
DUR_FALSE = "DURABLE-CTX-SENTINEL-5e7f31"
RAW = "RAW-STRING-CTX-SENTINEL-1a2b3c"


# ---------------------------------------------------------------------------
# Unit: _collect_pre_llm_call_context buckets hook results (durable, ephemeral)
# ---------------------------------------------------------------------------

class _HookAgent:
    """Bare namespace covering what _collect reads off the agent."""

    session_id = "sess-collect"
    model = "test-model"
    platform = "cli"


def _collect(agent=None, hook_results=()):
    with patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=list(hook_results),
    ):
        return _collect_pre_llm_call_context(
            agent or _HookAgent(), effective_task_id="t", turn_id="turn-1",
            original_user_message="hello", messages=[], conversation_history=None,
        )


def _buckets(result):
    """Split the collector's return into ``(durable, ephemeral)``.

    Tolerates the pre-fix shape (a single str = everything durable) so the
    red phase fails at the ephemeral assertion — the gap itself — instead of
    at tuple unpacking."""
    if isinstance(result, tuple):
        durable, eph = result
    else:
        durable, eph = result, ""
    return durable or "", eph or ""


class TestCollectBuckets:
    def test_ephemeral_hook_lands_only_in_ephemeral_bucket(self):
        durable, eph = _buckets(_collect(hook_results=[{"context": EPH, "ephemeral": True}]))
        assert EPH in eph  # RED pre-fix: the ephemeral bucket does not exist
        assert EPH not in durable

    def test_mixed_hooks_separate_and_keep_order(self):
        # Ephemeral hook registered FIRST: durable stream must still come out
        # as one consecutive block, ephemeral trailing as its own stream.
        durable, eph = _buckets(_collect(hook_results=[
            {"context": EPH, "ephemeral": True},
            {"context": DUR},
        ]))
        assert durable == DUR
        assert eph == EPH

    def test_bare_string_is_durable(self):
        durable, eph = _buckets(_collect(hook_results=[RAW]))
        assert RAW in durable
        assert eph == ""


# ---------------------------------------------------------------------------
# Unit: build_api_messages appends ephemeral to the CURRENT turn's wire copy
# ---------------------------------------------------------------------------

class _SendAgent:
    api_mode = "chat_completions"
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = 10_000.0

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _wire_user(history, **kwargs):
    kwargs.setdefault("ext_prefetch_cache", "")
    kwargs.setdefault("plugin_user_context", "")
    kwargs.setdefault("moa_config", None)
    kwargs.setdefault("active_system_prompt", "")
    request, _ = build_api_messages(
        _SendAgent(), history, current_turn_user_idx=len(history) - 1, **kwargs
    )
    return request[-1]["content"]


class TestBuildApiMessagesEphemeral:
    def test_string_content_durable_then_ephemeral_trailing(self):
        content = _wire_user(
            [{"role": "user", "content": "hello"}],
            plugin_user_context=DUR, ephemeral_user_context=EPH,
        )
        # RED pre-fix: the kwarg does not exist yet (the new API is the gap).
        assert content == "hello\n\n" + DUR + "\n\n" + EPH

    def test_ephemeral_alone_still_rides_the_wire(self):
        content = _wire_user(
            [{"role": "user", "content": "hello"}], ephemeral_user_context=EPH,
        )
        assert content == "hello\n\n" + EPH

    def test_list_content_appends_wire_only_text_part(self):
        live = [{"type": "text", "text": "what is this"}]
        content = _wire_user(
            [{"role": "user", "content": live}],
            ephemeral_user_context=EPH,
        )
        assert content == [*live, {"type": "text", "text": EPH}]
        assert live == [{"type": "text", "text": "what is this"}]  # live list untouched

    def test_empty_ephemeral_is_byte_identical_to_omitting_it(self):
        history = [{"role": "user", "content": "hello"}]
        plain = _wire_user(history, plugin_user_context=DUR)
        empty = _wire_user(history, plugin_user_context=DUR, ephemeral_user_context="")
        assert plain == empty == "hello\n\n" + DUR


# ---------------------------------------------------------------------------
# E2E: real AIAgent + in-process mock provider + real SessionDB
# ---------------------------------------------------------------------------

class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        if type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            resp = _text_resp("DONE")
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        tcs = msg.get("tool_calls")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [{"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
        if content:
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        if tcs:
            for ti, tc in enumerate(tcs):
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": ti, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]},
                    "finish_reason": None}]})
        chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tcs else "stop"}]})
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a, **kw):
        pass


def _tc_resp(name: str, args: str = "{}") -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": name, "arguments": args}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


class _WireEnv:
    """Mock provider + isolated HERMES_HOME + real SessionDB + mutable hook
    results. ``hooks`` is the list returned for every ``pre_llm_call``; tests
    rewrite it between turns."""

    def __init__(self, tmp_path):
        _MockHandler.captured_requests = []
        _MockHandler.response_queue = []
        self.handler = _MockHandler
        self.srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

        self.test_home = tempfile.mkdtemp(prefix="hermes_ephemeral_")
        os.makedirs(os.path.join(self.test_home, ".hermes"))
        self.prev_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = os.path.join(self.test_home, ".hermes")

        from hermes_state import SessionDB

        self.db_path = Path(self.test_home) / "state.db"
        self.db = SessionDB(db_path=self.db_path)
        self.sid = "sess-eph"
        self.hooks: list = []
        self._patcher = patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda hook_name, **kw: list(self.hooks),
        )
        self._patcher.start()

    def make_agent(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{self.port}/v1",
            provider="openai-compat", model="test-model",
            max_iterations=10, enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, platform="cli",
            session_db=self.db, session_id=self.sid,
        )
        agent.valid_tool_names = {"read_file"}
        return agent

    def chat_requests(self):
        # The model context-length probe also hits the mock; keep chat payloads.
        return [r for r in self.handler.captured_requests if "messages" in r]

    def user_rows_readonly(self):
        """The user rows exactly as persisted, through a READ-ONLY connection."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            return conn.execute(
                "SELECT content, api_content FROM messages "
                "WHERE session_id = ? AND role = 'user' ORDER BY id",
                (self.sid,),
            ).fetchall()
        finally:
            conn.close()

    def close(self):
        self._patcher.stop()
        self.srv.shutdown()
        self.db.close()
        shutil.rmtree(self.test_home, ignore_errors=True)
        if self.prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self.prev_home


@pytest.fixture()
def wire(tmp_path):
    env = _WireEnv(tmp_path)
    try:
        yield env
    finally:
        env.close()


def _run_two_iterations(env):
    """One turn that makes two API calls (tool round, then final text).

    The agent is created BEFORE queueing: its init-time context-length probe
    hits the mock and would otherwise drain the first queued response."""
    agent = env.make_agent()
    env.handler.response_queue.append(_tc_resp("read_file", '{"file_path": "/nonexistent-path"}'))
    env.handler.response_queue.append(_text_resp("DONE"))
    env.handler.captured_requests = []
    result = agent.run_conversation("hello please", conversation_history=[], task_id="t")
    return agent, result


def _sent_user_contents(env):
    return [
        m["content"] for r in env.chat_requests()
        for m in r.get("messages", []) if m.get("role") == "user"
    ]


class TestEphemeralEndToEnd:
    def test_sent_every_iteration_trailing_durable(self, wire):
        """Invariant (c) + wire order: the ephemeral hook registered FIRST still
        trails the durable stream on the wire, and rides EVERY iteration."""
        wire.hooks = [{"context": EPH, "ephemeral": True}, {"context": DUR}]
        _agent, result = _run_two_iterations(wire)

        sent = _sent_user_contents(wire)
        assert len(sent) == 2
        for content in sent:  # sent this turn, every iteration
            assert EPH in content
            durable_end, eph_start = content.find(DUR), content.find(EPH)
            assert -1 < durable_end < eph_start  # ephemeral trails durable
        assert sent[0] == sent[1]  # identical bytes across builds
        assert result["final_response"] == "DONE"

    def test_row_carries_neither_content_nor_sidecar_sentinel(self, wire):
        """Invariant (a): the persisted user row is clean in BOTH columns,
        while the durable stream still lands in the sidecar."""
        wire.hooks = [{"context": EPH, "ephemeral": True}, {"context": DUR}]
        _run_two_iterations(wire)

        rows = wire.user_rows_readonly()
        assert rows, "the turn's user row must exist"
        content, api_content = rows[-1]
        assert EPH not in (content or "")  # RED pre-fix: clean content holds
        assert EPH not in (api_content or "")  # RED pre-fix: sidecar leaked it
        assert DUR in (api_content or "")  # durable stream unchanged
        assert content == "hello please"  # clean transcript untouched

    def test_replay_from_persisted_rows_is_clean(self, wire):
        """Invariant (b): turn N+1 (history reloaded from the store, hooks now
        empty) replays the durable bytes and none of the ephemeral ones."""
        wire.hooks = [{"context": EPH, "ephemeral": True}, {"context": DUR}]
        _run_two_iterations(wire)

        history = wire.db.get_messages_as_conversation(wire.sid)
        wire.hooks = []
        agent2 = wire.make_agent()  # probe fires here, on an empty queue
        wire.handler.response_queue.append(_text_resp("second answer"))
        wire.handler.captured_requests = []
        agent2.run_conversation("second question", conversation_history=history, task_id="t2")

        replayed = _sent_user_contents(wire)[0]
        assert EPH not in replayed  # RED pre-fix: the sidecar replayed it
        assert DUR in replayed  # durable replay unchanged


class TestDurableGuardsEndToEnd:
    """Byte-level regression guards for today's behaviour (base-green)."""

    def test_plain_dict_context_persisted_and_replayed(self, wire):
        wire.hooks = [{"context": DUR}]
        _run_two_iterations(wire)

        rows = wire.user_rows_readonly()
        assert DUR in (rows[-1][1] or "")  # sidecar carries it
        history = wire.db.get_messages_as_conversation(wire.sid)
        wire.hooks = []
        agent2 = wire.make_agent()  # probe fires here, on an empty queue
        wire.handler.response_queue.append(_text_resp("again"))
        wire.handler.captured_requests = []
        agent2.run_conversation("second question", conversation_history=history, task_id="t2")
        assert DUR in _sent_user_contents(wire)[0]  # replay carries it

    def test_explicit_false_matches_default(self, wire):
        wire.hooks = [{"context": DUR_FALSE, "ephemeral": False}]
        _run_two_iterations(wire)
        assert DUR_FALSE in _sent_user_contents(wire)[0]  # still sent
        assert DUR_FALSE in (wire.user_rows_readonly()[-1][1] or "")  # still persisted

    def test_bare_string_context_persisted_and_replayed(self, wire):
        wire.hooks = [RAW]
        _run_two_iterations(wire)
        assert RAW in _sent_user_contents(wire)[0]
        assert RAW in (wire.user_rows_readonly()[-1][1] or "")
        history = wire.db.get_messages_as_conversation(wire.sid)
        wire.hooks = []
        agent2 = wire.make_agent()  # probe fires here, on an empty queue
        wire.handler.response_queue.append(_text_resp("again"))
        wire.handler.captured_requests = []
        agent2.run_conversation("second question", conversation_history=history, task_id="t2")
        assert RAW in _sent_user_contents(wire)[0]


class TestAutoCleanViews:
    def test_summary_chain_surfaces_no_ephemeral(self, wire):
        """The compaction/forced-summary request construction substitutes the
        persisted sidecar into visible text — with an ephemeral turn's rows it
        must surface none of it (and all of the durable stream)."""
        from agent.chat_completion_helpers import _iteration_summary_api_messages

        wire.hooks = [{"context": EPH, "ephemeral": True}, {"context": DUR}]
        _run_two_iterations(wire)

        agent = wire.make_agent()
        agent._cached_system_prompt = "SYS"
        rows = wire.db.get_messages_as_conversation(wire.sid)
        visible = json.dumps(_iteration_summary_api_messages(agent, rows))
        assert EPH not in visible  # RED pre-fix: substitute surfaced the leak
        assert DUR in visible  # durable stream still visible to the summary

    def test_memory_review_snapshot_is_clean(self, wire):
        """The end-of-turn review reads the live transcript snapshot; the
        ephemeral text must not be visible to it."""
        wire.hooks = [{"context": EPH, "ephemeral": True}]
        captured = {}

        def _capture(self, **kwargs):
            captured.update(kwargs)

        agent = wire.make_agent()
        # Fire the review on this turn: counter tripwire only needs these three.
        agent._memory_nudge_interval = 1
        agent.valid_tool_names = {"memory", "read_file"}
        agent._memory_store = object()

        from agent.message_content import flatten_message_text

        with patch("run_agent.AIAgent._spawn_background_review", _capture):
            wire.handler.response_queue.append(_text_resp("DONE"))
            agent.run_conversation("hello please", conversation_history=[], task_id="t")

        assert captured.get("messages_snapshot"), "review must fire for this test to bind"
        snapshot_text = "\n".join(
            flatten_message_text(m.get("content")) for m in captured["messages_snapshot"]
        )
        assert EPH not in snapshot_text


class TestFailOpen:
    def test_unrecognized_hook_shapes_change_nothing(self, wire):
        """Non-dict / non-str / context-less results are ignored exactly as
        before: the turn completes, the row exists, nothing new is rejected."""
        wire.hooks = [{"unknown_key": "x"}, 42, {"context": ""}]
        _agent, result = _run_two_iterations(wire)

        assert result["final_response"] == "DONE"
        assert result.get("failed") is not True
        content, api_content = wire.user_rows_readonly()[-1]
        assert content == "hello please"
        assert api_content is None  # nothing injected, nothing stamped
        assert _sent_user_contents(wire)[0] == "hello please"
