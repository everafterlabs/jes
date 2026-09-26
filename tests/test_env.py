"""Project .env loading."""

from __future__ import annotations

import os
from pathlib import Path

from pytest import MonkeyPatch

from jes._env import apply_env_file, load_project_env
from jes.backends import SystemOne


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


def test_hosted_prefers_existing_env_then_file_then_argument(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    root = _jes_root(tmp_path / "repo")
    (root / ".env").write_text("TYPESAFE_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    from_env = SystemOne.hosted()
    assert from_env._api_key == "from-env"
    from_env.close()

    monkeypatch.delenv("TYPESAFE_API_KEY")
    from_file = SystemOne.hosted()
    assert from_file._api_key == "from-file"
    from_file.close()

    explicit = SystemOne.hosted(api_key="explicit")
    assert explicit._api_key == "explicit"
    explicit.close()
