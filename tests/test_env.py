"""Project .env loading."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from jes._env import apply_env_file, load_config, load_project_env, write_config
from jes.cli import main
from jes.config import default_document, write_default_config
from jes.hook import SessionStore
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend


def _jes_root(path: Path) -> Path:
    path.mkdir()
    (path / "pyproject.toml").write_text('[project]\nname = "jes"\n', encoding="utf-8")
    return path


def test_apply_env_file_parses_quotes_and_keeps_existing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
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
    apply_env_file(env_file, environ)
    assert environ["TYPESAFE_API_KEY"] == "abc\n"
    assert environ["OPENAI_API_KEY"] == "already"
    assert environ["GROQ_API_KEY"] == "plain"
    assert environ["TRAIL"] == "a\\bq"
    assert environ["END"] == "x\\"
    assert "BLANK" not in environ
    assert "1BAD" not in environ


def test_load_project_env_uses_nearest_jes_root(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    root = _jes_root(tmp_path / "repo")
    nested = root / "evals"
    nested.mkdir()
    (root / ".env").write_text("TYPESAFE_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    load_project_env()
    assert os.environ.get("TYPESAFE_API_KEY") == "from-file"


def test_load_project_env_ignores_other_projects_and_missing_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "pyproject.toml").write_text('name = "other"\n', encoding="utf-8")
    (other / ".env").write_text("TYPESAFE_API_KEY=nope\n", encoding="utf-8")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    load_project_env(other)
    assert os.environ.get("TYPESAFE_API_KEY") is None

    root = _jes_root(tmp_path / "repo")
    load_project_env(root)
    assert os.environ.get("TYPESAFE_API_KEY") is None


def _hostile_repo(path: Path) -> Path:
    """A project whose dotenv files try to redirect jes to another server."""

    path.mkdir()
    (path / ".git").mkdir()
    (path / ".env").write_text(
        "TYPESAFE_API_KEY=from-repo\nTYPESAFE_BASE_URL=https://attacker.example\n",
        encoding="utf-8",
    )
    (path / ".env.local").write_text("TYPESAFE_API_KEY=from-local\n", encoding="utf-8")
    return path


def test_load_config_reads_only_the_user_env(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(_hostile_repo(tmp_path / "repo"))
    user = tmp_path / "user.env"
    user.write_text("TYPESAFE_API_KEY=from-user\nEXTRA=from-user\n", encoding="utf-8")

    found: dict[str, str] = {}
    load_config(environ=found, user_env=user)
    assert found == {"TYPESAFE_API_KEY": "from-user", "EXTRA": "from-user"}

    present = {"TYPESAFE_API_KEY": "from-env"}
    load_config(environ=present, user_env=user)
    assert present == {"TYPESAFE_API_KEY": "from-env", "EXTRA": "from-user"}

    missing: dict[str, str] = {}
    load_config(environ=missing, user_env=tmp_path / "absent.env")
    assert missing == {}


def test_hook_ignores_dotenv_files_in_the_project(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    monkeypatch.chdir(_hostile_repo(tmp_path / "repo"))
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL"):
        # setenv then delenv, so the test restores "unset" even if the hook sets it.
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"stage": "input", "text": "hi"})))
    model = FakeBackend(default_answer=YesNoAnswer(0.0, "probability"), max_units=1_000_000)

    assert main(["hook", "--session-dir", str(tmp_path / "sessions")], model=model) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert "TYPESAFE_API_KEY" not in os.environ
    assert "TYPESAFE_BASE_URL" not in os.environ


def test_private_files_are_0600_from_creation(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # With chmod disabled and no umask, only the creation mode can keep these files private.
    monkeypatch.setattr(os, "chmod", lambda *_args, **_kwargs: None)
    previous = os.umask(0)
    try:
        write_config(tmp_path / "jes" / ".env", {"TYPESAFE_API_KEY": "key"})
        assert write_default_config(tmp_path / "jes" / "config.json")
        SessionStore(tmp_path / "sessions").put("sess-1", "prompt")
    finally:
        os.umask(previous)
    for path in (
        tmp_path / "jes" / ".env",
        tmp_path / "jes" / "config.json",
        tmp_path / "sessions" / "prompts" / "sess-1",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path
    for directory in (tmp_path / "jes", tmp_path / "sessions", tmp_path / "sessions" / "prompts"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, directory
    assert sorted(item.name for item in (tmp_path / "jes").iterdir()) == [".env", "config.json"]


def test_writes_leave_existing_directories_alone(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    write_config(shared / ".env", {"TYPESAFE_API_KEY": "key"})
    assert write_default_config(shared / "config.json")
    assert not write_default_config(shared / "config.json")
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755


def test_login_writes_the_user_env(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("jes.cli.getpass.getpass", lambda _prompt="": "secret-key")
    (tmp_path / "jes").mkdir()
    (tmp_path / "jes" / ".env").write_text(
        "OTHER=keep\nJES_MODEL=jev-1\nJES_THRESHOLD=0.2\nJES_HAZARDS=S1\nJES_ALLOWED_TOOLS=Read\n",
        encoding="utf-8",
    )
    assert main(["login"]) == 0
    created = tmp_path / "jes" / "config.json"
    assert json.loads(created.read_text(encoding="utf-8")) == default_document()
    path = tmp_path / "jes" / ".env"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    monkeypatch.setattr("jes.cli.getpass.getpass", lambda _prompt="": "next-key")
    assert main(["login"]) == 0
    text = path.read_text(encoding="utf-8")
    assert "OTHER=keep" in text
    assert "JES_MODEL" not in text
    assert "JES_THRESHOLD" not in text
    assert "JES_HAZARDS" not in text
    assert "JES_ALLOWED_TOOLS" not in text
    assert text.count("TYPESAFE_API_KEY=") == 1
    found: dict[str, str] = {}
    load_config(environ=found, user_env=path)
    assert found["TYPESAFE_API_KEY"] == "next-key"
    assert found["OTHER"] == "keep"
    config = tmp_path / "jes" / "config.json"
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    changed = config.read_text(encoding="utf-8").replace("0.5", "0.2", 1)
    config.write_text(changed, encoding="utf-8")
    monkeypatch.setattr("jes.cli.getpass.getpass", lambda _prompt="": "later-key")
    assert main(["login"]) == 0
    assert config.read_text(encoding="utf-8") == changed

    monkeypatch.setattr("jes.cli.getpass.getpass", lambda _prompt="": "  ")
    assert main(["login"]) == 2
