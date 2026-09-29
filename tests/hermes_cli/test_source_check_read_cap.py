"""Oversized compare bodies must not cost the banner its behind count (#127940).

``_request_with`` caps response bodies at 2 MiB; a checkout far enough behind
gets a truncated compare JSON, the parse fails into ``None``, and
``_behind_count`` degrades to the count-less sentinel. These tests pin the
fallback: the count is answerable locally whenever the checkout's objects are
present.
"""

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import source_check


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=root, text=True).strip()


def _checkout(root: Path, head: str) -> source_check._Checkout:
    return source_check._Checkout(
        root=root, git="git", embedded=None, head=head, current_branch="main",
        origin="https://github.com/example/fork.git", repository="example/fork", dirty=False)


def _truncated_two_mib_body(ahead_by: int) -> str:
    """A compare body as ``_request_with`` delivers it: cut at exactly 2 MiB."""
    body = json.dumps({"ahead_by": ahead_by, "commits": [], "padding": "p" * (3 * 1024 * 1024)})
    return body[: 2 * 1024 * 1024]


@pytest.fixture
def behind_checkout(tmp_path):
    """A checkout sitting 7 commits behind a local ``ahead`` branch."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "commit", "--allow-empty", "-m", "base")
    base = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-b", "ahead")
    for i in range(7):
        _git(root, "commit", "--allow-empty", "-m", f"up {i}")
    target = _git(root, "rev-parse", "ahead")
    _git(root, "checkout", "main")
    return root, base, target


def test_truncated_compare_body_still_yields_the_local_count(behind_checkout, monkeypatch):
    root, head, target = behind_checkout
    body = _truncated_two_mib_body(ahead_by=7)
    # The truncation point must actually break the JSON — that is the defect.
    with pytest.raises(json.JSONDecodeError):
        json.loads(body)
    monkeypatch.setattr(
        source_check, "_request", lambda url, accept="application/vnd.github+json": body)
    behind, commits = source_check._behind_count(_checkout(root, head), target)
    expected = int(_git(root, "rev-list", "--count", f"{head}..{target}"))
    assert behind == expected
    assert commits == []


def test_well_formed_payload_still_counts_through_the_api(behind_checkout, monkeypatch):
    root, head, target = behind_checkout
    payload = {"ahead_by": 13, "commits": [{"sha": "f" * 40, "commit": {
        "message": "remote tip", "author": {"name": "Fixture"},
        "committer": {"date": "2026-01-01T00:00:00Z"}}}]}
    monkeypatch.setattr(
        source_check, "_request",
        lambda url, accept="application/vnd.github+json": json.dumps(payload))
    # 13 on purpose: a stray fallback to the local count would answer 7 and fail.
    behind, commits = source_check._behind_count(_checkout(root, head), target)
    assert behind == 13
    assert [c["sha"] for c in commits] == ["f" * 40]


def test_missing_objects_keep_the_countless_sentinel(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "commit", "--allow-empty", "-m", "base")
    head = _git(root, "rev-parse", "HEAD")
    target = "f" * 40  # A well-formed SHA this checkout has never seen.
    body = _truncated_two_mib_body(ahead_by=5)
    monkeypatch.setattr(
        source_check, "_request", lambda url, accept="application/vnd.github+json": body)
    assert source_check._behind_count(_checkout(root, head), target) == (
        source_check.UPDATE_AVAILABLE_NO_COUNT, [])
