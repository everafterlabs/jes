"""Project .env loading."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from pytest import MonkeyPatch

from jes._env import apply_env_file, load_config, load_project_env
from jes.cli import main
from jes.config import default_document


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


def test_config_lookup_order(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".env").write_text(
        "TYPESAFE_API_KEY=from-dotenv\nEXTRA=from-dotenv\n",
        encoding="utf-8",
    )
    (root / ".env.local").write_text(
        "TYPESAFE_API_KEY=from-local\nREGION=from-local\n",
        encoding="utf-8",
    )
    user = tmp_path / "user.env"
    user.write_text(
        "TYPESAFE_API_KEY=from-user\nEXTRA=from-user\nREGION=from-user\n",
        encoding="utf-8",
    )

    local: dict[str, str] = {}
    load_config(nested, environ=local, user_env=user)
    assert local["TYPESAFE_API_KEY"] == "from-local"
    assert local["REGION"] == "from-local"
    assert local["EXTRA"] == "from-dotenv"

    present = {"TYPESAFE_API_KEY": "from-env", "EXTRA": "from-env"}
    load_config(root, environ=present, user_env=user)
    assert present["TYPESAFE_API_KEY"] == "from-env"
    assert present["EXTRA"] == "from-env"
    assert present["REGION"] == "from-local"

    (root / ".env.local").unlink()
    dotenv: dict[str, str] = {}
    load_config(root, environ=dotenv, user_env=user)
    assert dotenv["TYPESAFE_API_KEY"] == "from-dotenv"
    assert dotenv["EXTRA"] == "from-dotenv"
    assert dotenv["REGION"] == "from-user"

    (root / ".env").unlink()
    fallback: dict[str, str] = {}
    load_config(root, environ=fallback, user_env=user)
    assert fallback["TYPESAFE_API_KEY"] == "from-user"


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
    load_config(tmp_path / "empty", environ=found, user_env=path)
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

