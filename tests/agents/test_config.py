"""config.json: parsing, every guard, and errors that name the problem."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from jes.agents.config import (
    DEFAULT_MODEL,
    config_path,
    default_config,
    default_text,
    load_config,
    parse_config,
    write_default_config,
)
from jes.errors import ConfigError
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend
from tests.agents.helpers import argv, run


def _document() -> dict[str, Any]:
    return copy.deepcopy(json.loads(default_text()))


def _names(document: dict[str, Any]) -> list[str]:
    return [policy.name for policy in parse_config(document).policies()]


def test_the_default_config_enables_three_guards() -> None:
    config = default_config()
    assert config.model == DEFAULT_MODEL
    assert [policy.name for policy in config.policies()] == [
        "injection",
        "indirect_injection",
        "hazards",
    ]


def test_model_is_optional_and_checked() -> None:
    document = _document()
    document["model"] = "jev-1.13.0"
    assert parse_config(document).model == "jev-1.13.0"
    for bad in ("", "  ", 5):
        document["model"] = bad
        with pytest.raises(ConfigError, match="model must be"):
            parse_config(document)


def test_disabled_and_enabled_guards() -> None:
    document = _document()
    document["guards"]["injection"]["enabled"] = False
    document["guards"]["tool_safety"] = {"enabled": True, "threshold": 0.4}
    assert _names(document) == ["indirect_injection", "hazards", "tool_safety"]
    document["guards"]["tool_safety"] = {"enabled": True}
    with pytest.raises(ConfigError, match="tool_safety requires threshold"):
        parse_config(document).policies()


def test_fields_reach_the_factories() -> None:
    document = _document()
    guards = document["guards"]
    guards["hazards"]["categories"] = ["S14"]
    guards["allowed_tools"] = {"enabled": True, "names": ["Read"]}
    policies = {policy.name: policy for policy in parse_config(document).policies()}
    assert list(policies["hazards"].questions) == ["S14"]
    assert policies["allowed_tools"].names == frozenset({"Read"})


def test_every_guard_builds() -> None:
    document = _document()
    guards = document["guards"]
    guards["toxicity"].update(enabled=True, labels=["insult"])
    guards["topics"].update(enabled=True, deny=["weapons"])
    guards["invisible_text"].update(enabled=True, mode="all", block=True)
    guards["allowed_tools"] = {"enabled": True, "names": ["Read"]}
    guards["tool_safety"]["enabled"] = True
    guards["canary"].update(enabled=True, token="canary-token")
    guards["regex"].update(
        enabled=True,
        patterns=["secret"],
        action="redact",
        match="fullmatch",
        require=True,
        fold=True,
        timeout_ms=20,
    )
    guards["substrings"].update(enabled=True, terms=["password"], action="redact", whole_words=True)
    guards["token_limit"].update(enabled=True, mode="truncate")
    guards["pii"].update(
        enabled=True,
        entities=["CREDIT_CARD"],
        input_mode="mask",
        untrusted_mode="redact",
        output_mode="block",
        tool_call_mode="flag",
        restore=False,
    )
    guards["secrets"].update(enabled=True, redact="partial")
    assert _names(document) == list(guards)
    guards["secrets"].update(redact="hmac", key=base64.b64encode(b"k" * 32).decode())
    assert "secrets" in _names(document)


def test_minimal_guards_use_factory_defaults() -> None:
    document = {
        "guards": {
            "invisible_text": {"enabled": True},
            "regex": {"enabled": True, "patterns": ["x"]},
            "substrings": {"enabled": True, "terms": ["x"]},
            "token_limit": {"enabled": True, "limit": 10},
            "pii": {"enabled": True, "entities": ["EMAIL_ADDRESS"]},
            "secrets": {"enabled": True},
        }
    }
    assert _names(document) == list(document["guards"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(extra=1), "unknown config field extra"),
        (lambda d: d.update(guards=[]), "must contain guards"),
        (lambda d: d["guards"].update(judge={"enabled": True}), "unknown guard judge"),
        (lambda d: d["guards"].update(injection=[]), "injection must be an object"),
        (lambda d: d["guards"]["injection"].update(color="red"), "unknown injection field color"),
        (lambda d: d["guards"]["injection"].pop("enabled"), "injection requires enabled"),
        (lambda d: d["guards"]["injection"].update(enabled="yes"), "injection requires enabled"),
        (
            lambda d: d["guards"]["injection"].update(threshold="high"),
            "invalid injection threshold",
        ),
        (lambda d: d["guards"]["injection"].update(threshold=True), "invalid injection threshold"),
        (lambda d: d["guards"]["injection"].update(threshold=1.5), "invalid injection threshold"),
        (
            lambda d: d["guards"]["injection"].update(threshold=float("nan")),
            "invalid injection threshold",
        ),
        (
            lambda d: d["guards"]["invisible_text"].update(block="no"),
            "invalid invisible_text block",
        ),
        (
            lambda d: d["guards"]["invisible_text"].update(mode="some"),
            "invalid invisible_text mode",
        ),
        (lambda d: d["guards"]["hazards"].update(categories="S1"), "invalid hazards categories"),
        (lambda d: d["guards"]["hazards"].update(categories=[1]), "invalid hazards categories"),
        (lambda d: d["guards"]["canary"].update(token=5), "invalid canary token"),
        (lambda d: d["guards"]["token_limit"].update(limit=0), "invalid token_limit limit"),
        (lambda d: d["guards"]["token_limit"].update(limit=True), "invalid token_limit limit"),
        (lambda d: d["guards"]["secrets"].update(key="not base64!"), "invalid secrets key"),
        (lambda d: d["guards"]["secrets"].update(key=""), "invalid secrets key"),
        (
            lambda d: d["guards"]["secrets"].update(key=base64.b64encode(b"short").decode()),
            "at least 32 bytes",
        ),
    ],
)
def test_parse_errors(change: Any, message: str) -> None:
    document = _document()
    change(document)
    with pytest.raises(ConfigError, match=message):
        parse_config(document)
    with pytest.raises(ConfigError, match="must be an object"):
        parse_config([])


@pytest.mark.parametrize(
    ("guard", "body", "message"),
    [
        ("topics", {"enabled": True, "threshold": 0.5, "deny": []}, "topics: topics needs"),
        ("allowed_tools", {"enabled": True, "names": []}, "allowed_tools: "),
        ("canary", {"enabled": True, "token": ""}, "canary: "),
        ("hazards", {"enabled": True, "threshold": 0.5, "categories": ["S99"]}, "hazards: unknown"),
        ("regex", {"enabled": True}, "regex requires patterns"),
        ("secrets", {"enabled": True, "redact": "hmac"}, "requires key"),
        (
            "secrets",
            {"enabled": True, "key": base64.b64encode(b"k" * 32).decode()},
            "only used when",
        ),
    ],
)
def test_build_errors(guard: str, body: dict[str, object], message: str) -> None:
    document = _document()
    document["guards"][guard] = body
    with pytest.raises(ConfigError, match=message):
        parse_config(document).policies()


def test_config_files(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    with pytest.raises(ConfigError, match="jes login"):
        load_config(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="not JSON"):
        load_config(path)
    assert write_default_config(tmp_path / "fresh.json")
    assert not write_default_config(tmp_path / "fresh.json")
    assert load_config(tmp_path / "fresh.json").model == DEFAULT_MODEL
    assert config_path().name == "config.json"


def test_the_threshold_in_the_file_changes_the_decision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    model = FakeBackend({"injection.violation": YesNoAnswer(0.6)}, default=YesNoAnswer(0.0))
    document = json.loads(config_path().read_text(encoding="utf-8"))
    for threshold, ok in ((0.4, False), (0.9, True)):
        document["guards"]["injection"]["threshold"] = threshold
        config_path().write_text(json.dumps(document), encoding="utf-8")
        code, body, _err = run(
            monkeypatch,
            capsys,
            argv("hook", tmp_path / "s"),
            {"stage": "input", "text": "hi"},
            model,
        )
        assert code == 0
        assert isinstance(body, dict) and body["ok"] is ok
