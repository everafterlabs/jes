"""Guard config.json."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import pytest

from jes.cli import main
from jes.config import (
    config_path,
    default_document,
    load_policies,
    policies_from_config,
    write_default_config,
)
from jes.payload import ConfigError
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend


def _guards(document: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = document["guards"]
    assert isinstance(raw, dict)
    return raw


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sessions: Path,
    payload: object,
    model: FakeBackend,
) -> tuple[int, object, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    code = main(["hook", "--session-dir", str(sessions)], model=model)
    captured = capsys.readouterr()
    parsed: object = json.loads(captured.out) if captured.out.strip() else None
    return code, parsed, captured.err


def test_default_enables_the_three_guards() -> None:
    assert [item.name for item in policies_from_config(default_document())] == [
        "injection",
        "indirect_injection",
        "hazards",
    ]


def test_disabled_guard_is_omitted() -> None:
    document = default_document()
    _guards(document)["injection"]["enabled"] = False
    assert [item.name for item in policies_from_config(document)] == [
        "indirect_injection",
        "hazards",
    ]


def test_hazard_categories_and_allowed_tools_reach_the_factories() -> None:
    document = default_document()
    guards = _guards(document)
    guards["hazards"]["categories"] = ["S14"]
    guards["allowed_tools"] = {"enabled": True, "names": ["Read"]}
    policies = list(policies_from_config(document))
    hazards_policy = next(item for item in policies if item.name == "hazards")
    tools = next(item for item in policies if item.name == "allowed_tools")
    assert set(hazards_policy.questions(None)) == {"S14", "any"}
    assert tools.names == frozenset({"Read"})


def test_each_enabled_guard_is_built() -> None:
    document = default_document()
    guards = _guards(document)
    guards["toxicity"]["enabled"] = True
    guards["toxicity"]["labels"] = ["insult"]
    guards["topics"]["enabled"] = True
    guards["topics"]["deny"] = ["weapons"]
    guards["invisible_text"]["enabled"] = True
    guards["invisible_text"]["mode"] = "all"
    guards["invisible_text"]["block"] = True
    guards["allowed_tools"] = {"enabled": True, "names": ["Read"]}
    guards["canary"]["enabled"] = True
    guards["canary"]["token"] = "canary-token"
    guards["regex"]["enabled"] = True
    guards["regex"]["patterns"] = ["secret"]
    guards["regex"]["action"] = "redact"
    guards["regex"]["match"] = "fullmatch"
    guards["regex"]["require"] = True
    guards["regex"]["fold"] = True
    guards["substrings"]["enabled"] = True
    guards["substrings"]["terms"] = ["password"]
    guards["substrings"]["action"] = "redact"
    guards["substrings"]["whole_words"] = True
    guards["token_limit"]["enabled"] = True
    guards["token_limit"]["mode"] = "truncate"
    guards["pii"]["enabled"] = True
    guards["pii"]["entities"] = ["CREDIT_CARD"]
    guards["pii"]["input_mode"] = "mask"
    guards["pii"]["untrusted_mode"] = "redact"
    guards["pii"]["output_mode"] = "block"
    guards["pii"]["restore"] = False
    guards["secrets"]["enabled"] = True
    guards["secrets"]["redact"] = "partial"
    names = [item.name for item in policies_from_config(document)]
    assert "toxicity" in names
    assert "topics" in names
    assert "invisible_text" in names
    assert "canary" in names
    assert "regex" in names
    assert "substrings" in names
    assert "token_limit" in names
    assert "pii" in names
    assert "secrets" in names

    guards["secrets"]["redact"] = "hmac"
    guards["secrets"]["key"] = base64.b64encode(b"k" * 32).decode()
    assert any(item.name == "secrets" for item in policies_from_config(document))


def test_config_file_must_be_an_object(tmp_path: Path) -> None:
    broken = tmp_path / "config.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"invalid config\.json"):
        load_policies(broken)
    broken.write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be an object"):
        load_policies(broken)
    with pytest.raises(ConfigError, match="unknown config field"):
        policies_from_config({"guards": {}, "model": "jev-1"})
    with pytest.raises(ConfigError, match="must contain guards"):
        policies_from_config({"guards": []})


def test_invalid_config_is_rejected() -> None:
    document = default_document()
    guards = _guards(document)
    guards["injection"]["threshold"] = "nope"
    with pytest.raises(ConfigError, match="threshold"):
        policies_from_config(document)

    document = default_document()
    _guards(document)["judge"] = {"enabled": True}
    with pytest.raises(ConfigError, match="unknown guard"):
        policies_from_config(document)

    document = default_document()
    _guards(document)["topics"] = {"enabled": True, "threshold": 0.5, "deny": []}
    with pytest.raises(ConfigError, match="deny"):
        policies_from_config(document)


def test_missing_config_tells_the_user_to_login(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    code, body, err = _run(
        monkeypatch,
        capsys,
        tmp_path / "sessions",
        {"stage": "input", "text": "hi"},
        FakeBackend(default_answer=YesNoAnswer(0.0, "probability"), max_units=1_000_000),
    )
    assert code == 2
    assert body is None
    assert "jes login" in err


def test_injection_threshold_changes_the_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    write_default_config()
    path = config_path()
    model = FakeBackend(
        {"injection.violation": YesNoAnswer(0.6, "probability")},
        default_answer=YesNoAnswer(0.0, "probability"),
        max_units=1_000_000,
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["guards"]["injection"]["threshold"] = 0.4
    path.write_text(json.dumps(document), encoding="utf-8")
    code, body, _err = _run(
        monkeypatch,
        capsys,
        tmp_path / "sessions",
        {"stage": "input", "text": "hello"},
        model,
    )
    assert code == 0
    assert isinstance(body, dict)
    assert body["ok"] is False

    document["guards"]["injection"]["threshold"] = 0.9
    path.write_text(json.dumps(document), encoding="utf-8")
    code, body, _err = _run(
        monkeypatch,
        capsys,
        tmp_path / "sessions",
        {"stage": "input", "text": "hello"},
        model,
    )
    assert code == 0
    assert isinstance(body, dict)
    assert body["ok"] is True
