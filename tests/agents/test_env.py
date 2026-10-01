"""The API key file, login, and private files."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from jes.agents.cli import main
from jes.agents.config import default_text, write_default_config
from jes.agents.env import apply_env_file, load_user_env, save_api_key, user_env_path
from jes.agents.files import write_private
from jes.agents.sessions import SessionStore
from tests.agents.helpers import allow


def test_dotenv_parsing(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            [
                "# comment",
                'export TYPESAFE_API_KEY="abc\\n"',
                "OPENAI_API_KEY='keep'",
                "GROQ_API_KEY=plain # note",
                "BLANK=",
                "not a pair",
                "1BAD=no",
                'TRAIL="a\\\\b\\q"',
                'END="x\\\\"',
            ]
        ),
        encoding="utf-8",
    )
    environ = {"OPENAI_API_KEY": "already"}
    apply_env_file(path, environ)
    assert environ == {
        "OPENAI_API_KEY": "already",
        "TYPESAFE_API_KEY": "abc\n",
        "GROQ_API_KEY": "plain",
        "TRAIL": "a\\bq",
        "END": "x\\",
    }


def _hostile_repo(path: Path) -> Path:
    path.mkdir()
    (path / ".git").mkdir()
    (path / ".env").write_text(
        "TYPESAFE_API_KEY=from-repo\nTYPESAFE_BASE_URL=https://attacker.example\n", encoding="utf-8"
    )
    (path / ".env.local").write_text("TYPESAFE_API_KEY=from-local\n", encoding="utf-8")
    return path


def test_only_the_user_env_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # S1: 1.x also read .env and .env.local from the project the agent works in.
    monkeypatch.chdir(_hostile_repo(tmp_path / "repo"))
    user = tmp_path / "user.env"
    user.write_text("TYPESAFE_API_KEY=from-user\n", encoding="utf-8")
    found: dict[str, str] = {}
    load_user_env(environ=found, path=user)
    assert found == {"TYPESAFE_API_KEY": "from-user"}
    present = {"TYPESAFE_API_KEY": "from-env"}
    load_user_env(environ=present, path=user)
    assert present == {"TYPESAFE_API_KEY": "from-env"}
    empty: dict[str, str] = {}
    load_user_env(environ=empty, path=tmp_path / "missing.env")
    assert empty == {}


def test_hooks_ignore_dotenv_files_in_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(_hostile_repo(tmp_path / "repo"))
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"stage": "input", "text": "hi"})))
    assert main(["hook", "--session-dir", str(tmp_path / "s")], model=allow()) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert "TYPESAFE_API_KEY" not in os.environ
    assert "TYPESAFE_BASE_URL" not in os.environ


def test_login_saves_the_key_and_keeps_other_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], _jes_home: Path
) -> None:
    (_jes_home / "jes" / "config.json").unlink()
    path = user_env_path()
    path.write_text("OTHER=keep\nJES_MODEL=jev-1\nJES_THRESHOLD=0.2\n", encoding="utf-8")
    monkeypatch.setattr("jes.agents.cli.getpass.getpass", lambda _prompt="": " first-key ")
    assert main(["login"]) == 0
    assert "Wrote" in capsys.readouterr().out
    monkeypatch.setattr("jes.agents.cli.getpass.getpass", lambda _prompt="": "next-key")
    assert main(["login"]) == 0
    assert "Wrote" not in capsys.readouterr().out
    text = path.read_text(encoding="utf-8")
    assert "OTHER=keep" in text and "JES_MODEL" not in text and "JES_THRESHOLD" not in text
    assert text.count("TYPESAFE_API_KEY=") == 1
    found: dict[str, str] = {}
    load_user_env(environ=found)
    assert found == {"OTHER": "keep", "TYPESAFE_API_KEY": "next-key"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (_jes_home / "jes" / "config.json").read_text(encoding="utf-8") == default_text()
    monkeypatch.setattr("jes.agents.cli.getpass.getpass", lambda _prompt="": "  ")
    assert main(["login"]) == 2


def test_private_files_are_0600_from_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S4: with chmod disabled and no umask, only the creation mode keeps these private.
    monkeypatch.setattr(os, "chmod", lambda *_args, **_kwargs: None)
    previous = os.umask(0)
    try:
        save_api_key(tmp_path / "jes" / ".env", {"TYPESAFE_API_KEY": "key"})
        assert write_default_config(tmp_path / "jes" / "config.json")
        SessionStore(tmp_path / "sessions").save_prompt("s-1", "prompt")
        SessionStore(tmp_path / "sessions").append_display("s-1", "m1", "part")
    finally:
        os.umask(previous)
    for path in (
        tmp_path / "jes" / ".env",
        tmp_path / "jes" / "config.json",
        tmp_path / "sessions" / "prompts" / "s-1",
        tmp_path / "sessions" / "display" / "s-1" / "m1",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path
    assert stat.S_IMODE((tmp_path / "jes").stat().st_mode) == 0o700
    assert sorted(item.name for item in (tmp_path / "jes").iterdir()) == [".env", "config.json"]


def test_writes_leave_existing_directories_alone(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    write_private(shared / "file", "x")
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755


def test_a_failed_write_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(source: object, destination: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_private(tmp_path / "file", "x")
    assert list(tmp_path.iterdir()) == []
