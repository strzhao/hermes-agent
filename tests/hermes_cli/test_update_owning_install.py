"""``hermes update`` from a PM environment's workspace goes to the checkout that owns it (#122627)."""
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.update_owning_install import owning_install_root, retarget_to_owning_install
from pm.environments import install_key


def _checkout(root: Path, name: str = "hermes-agent") -> Path:
    checkout = root / name
    (checkout / "hermes_cli").mkdir(parents=True)
    (checkout / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    return checkout


def _generation(home: Path, state_name: str) -> tuple[Path, Path]:
    """``(workspace, venv)`` of a PM generation whose state dir is named after its checkout."""
    generation = home / "installs" / state_name / "environments" / "0123abcd"
    (generation / "venv").mkdir(parents=True)
    workspace = generation / "workspace"
    (workspace / "hermes_cli").mkdir(parents=True)
    (workspace / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    return workspace, generation / "venv"


def _run_from(monkeypatch, venv: Path) -> None:
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(venv.parent.parent.parent / "python"))
    monkeypatch.delenv("PYTHONPATH", raising=False)


def test_pm_generation_venv_is_owned_by_the_checkout_its_install_key_names(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    workspace, venv = _generation(tmp_path, install_key(checkout))
    (tmp_path / "installs" / install_key(checkout) / "inputs").mkdir()
    record = tmp_path / "installs" / install_key(checkout) / "inputs" / ".project-root"
    record.write_text(str(checkout), encoding="utf-8")
    _run_from(monkeypatch, venv)

    assert owning_install_root(workspace) == checkout.resolve()

    record.write_text(str(tmp_path / "elsewhere"), encoding="utf-8")  # not the path the key hashes
    assert owning_install_root(workspace) is None


def test_workspace_snapshot_without_an_install_record_is_owned_by_the_checkout_its_key_names(
    tmp_path, monkeypatch
):
    """The reporter's layout: editable workspace snapshot, no ``inputs/.project-root`` at all."""
    checkout = _checkout(tmp_path)
    workspace, venv = _generation(tmp_path, install_key(checkout))
    assert not (tmp_path / "installs" / install_key(checkout) / "inputs").exists()
    _run_from(monkeypatch, venv)

    assert owning_install_root(workspace) == checkout.resolve()


def test_the_recordless_branch_never_adopts_a_checkout_another_install_key_names(tmp_path, monkeypatch):
    """A state dir named for some other checkout must not be sent to ``<root>/hermes-agent``."""
    _checkout(tmp_path)
    workspace, venv = _generation(tmp_path, "0000000000000000")
    _run_from(monkeypatch, venv)

    assert owning_install_root(workspace) is None


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _checkout_one_commit_behind(home: Path) -> Path:
    """A real clone whose ``origin/main`` is one commit ahead, i.e. ``--check`` has something to report."""
    origin = home / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True, capture_output=True)
    seed = _checkout(home, "seed")
    _git(seed, "init", "-q", "-b", "main")
    _git(seed, "config", "user.email", "t@example.com")
    _git(seed, "config", "user.name", "t")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "c0")
    _git(seed, "push", "-q", str(origin), "HEAD:refs/heads/main")
    subprocess.run(
        ["git", "clone", "-q", f"file://{origin}", str(home / "hermes-agent")],
        check=True, capture_output=True, text=True,
    )
    checkout = home / "hermes-agent"
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "t")
    _git(checkout, "commit", "-q", "--allow-empty", "-m", "ahead")
    _git(checkout, "push", "-q", "origin", "HEAD:main")
    _git(checkout, "reset", "-q", "--hard", "HEAD~1")
    assert _git(checkout, "rev-list", "--count", "HEAD..origin/main") == "1"
    return checkout


def test_check_reports_the_checkout_behind_from_a_workspace_that_has_no_git(tmp_path, monkeypatch, capsys):
    """The leg the issue reports: ``--check`` from the workspace snapshot answers for the checkout."""
    home = tmp_path / "home"
    checkout = _checkout_one_commit_behind(home)
    workspace, venv = _generation(home, install_key(checkout))
    assert not (workspace / ".git").exists()
    _run_from(monkeypatch, venv)

    from hermes_cli import update_cmd

    resolved = owning_install_root(workspace)
    assert resolved == checkout.resolve()
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", resolved)
    monkeypatch.setattr("hermes_cli.update_contract.evaluate_update_admission", lambda root: None)
    monkeypatch.setattr("hermes_cli.source_check._github_compare_behind", lambda *a, **k: None)

    update_cmd._cmd_update_check("main")

    out = capsys.readouterr().out
    assert "Not a git repository" not in out
    assert "Update available" in out


def test_retarget_replays_the_command_on_the_resolved_checkout(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    workspace, venv = _generation(tmp_path, install_key(checkout))
    _run_from(monkeypatch, venv)
    calls = []

    def _call(argv, cwd=None, env=None):
        calls.append((argv, cwd, env))
        return 0

    monkeypatch.setattr("hermes_cli.update_owning_install.subprocess.call", _call)
    monkeypatch.setattr(sys, "argv", ["hermes", "update", "--check"])

    with pytest.raises(SystemExit) as exited:
        retarget_to_owning_install(workspace)

    assert exited.value.code == 0
    argv, cwd, env = calls[0]
    assert cwd == checkout.resolve()
    assert env["PYTHONPATH"] == str(checkout.resolve())
    assert argv[-3:] == ["hermes_cli.main", "update", "--check"]
