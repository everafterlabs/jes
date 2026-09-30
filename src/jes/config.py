"""Agent guard config at ``~/.config/jes/config.json``."""

from __future__ import annotations

import base64
import json
import math
import os
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict, cast

from jes.errors import PolicyError
from jes.payload import ConfigError
from jes.policies import (
    Policy,
    allowed_tools,
    canary,
    hazards,
    indirect_injection,
    injection,
    invisible_text,
    pii,
    regex,
    secrets,
    substrings,
    token_limit,
    topics,
    toxicity,
)

_Action = Literal["block", "redact"]
_Match = Literal["search", "fullmatch"]
_Invisible = Literal["targeted", "all"]
_TokenMode = Literal["block", "truncate"]
_Redact = Literal["all", "partial", "hmac"]
_PiiInput = Literal["redact", "mask", "block"]
_PiiUntrusted = Literal["mask", "redact", "block"]
_PiiOutput = Literal["flag", "redact", "block"]


class InjectionGuard(TypedDict):
    enabled: bool
    threshold: NotRequired[float]


class IndirectInjectionGuard(TypedDict):
    enabled: bool
    threshold: NotRequired[float]


class HazardsGuard(TypedDict):
    enabled: bool
    threshold: NotRequired[float]
    categories: NotRequired[list[str]]


class ToxicityGuard(TypedDict):
    enabled: bool
    threshold: NotRequired[float]
    labels: NotRequired[list[str]]


class TopicsGuard(TypedDict):
    enabled: bool
    threshold: NotRequired[float]
    deny: NotRequired[list[str]]


class InvisibleTextGuard(TypedDict):
    enabled: bool
    mode: NotRequired[_Invisible]
    block: NotRequired[bool]


class AllowedToolsGuard(TypedDict):
    enabled: bool
    names: NotRequired[list[str]]


class CanaryGuard(TypedDict):
    enabled: bool
    token: NotRequired[str]


class RegexGuard(TypedDict):
    enabled: bool
    patterns: NotRequired[list[str]]
    action: NotRequired[_Action]
    match: NotRequired[_Match]
    require: NotRequired[bool]
    fold: NotRequired[bool]
    timeout_ms: NotRequired[int]


class SubstringsGuard(TypedDict):
    enabled: bool
    terms: NotRequired[list[str]]
    action: NotRequired[_Action]
    whole_words: NotRequired[bool]
    fold: NotRequired[bool]


class TokenLimitGuard(TypedDict):
    enabled: bool
    limit: NotRequired[int]
    encoding: NotRequired[str]
    mode: NotRequired[_TokenMode]


class PiiGuard(TypedDict):
    enabled: bool
    entities: NotRequired[list[str]]
    input_mode: NotRequired[_PiiInput]
    untrusted_mode: NotRequired[_PiiUntrusted]
    output_mode: NotRequired[_PiiOutput]
    restore: NotRequired[bool]


class SecretsGuard(TypedDict):
    enabled: bool
    redact: NotRequired[_Redact]
    key: NotRequired[str]


class Guards(TypedDict, total=False):
    injection: InjectionGuard
    indirect_injection: IndirectInjectionGuard
    hazards: HazardsGuard
    toxicity: ToxicityGuard
    topics: TopicsGuard
    invisible_text: InvisibleTextGuard
    allowed_tools: AllowedToolsGuard
    canary: CanaryGuard
    regex: RegexGuard
    substrings: SubstringsGuard
    token_limit: TokenLimitGuard
    pii: PiiGuard
    secrets: SecretsGuard


class GuardConfig(TypedDict):
    """The object stored in ``config.json``."""

    guards: Guards


class _TypedDictClass(Protocol):
    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]


def config_path() -> Path:
    """``~/.config/jes/config.json``, honoring ``XDG_CONFIG_HOME``."""

    raw = os.environ.get("XDG_CONFIG_HOME", "").strip()
    home = Path(raw) if raw else Path.home() / ".config"
    return home / "jes" / "config.json"


def _default_text() -> str:
    return files("jes").joinpath("data", "config.json").read_text(encoding="utf-8")


def default_document() -> GuardConfig:
    """The config ``jes login`` writes, checked against ``GuardConfig``."""

    return parse_config(json.loads(_default_text()))


def write_default_config(path: Path | None = None) -> bool:
    """Write the default config when it is absent. Return whether it was written."""

    target = config_path() if path is None else path
    if target.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    target.write_text(_default_text(), encoding="utf-8")
    os.chmod(target, 0o600)
    return True


def load_policies(path: Path | None = None) -> Sequence[Policy]:
    """Read the config file and build the enabled policies."""

    target = config_path() if path is None else path
    if not target.is_file():
        raise ConfigError("missing config.json; run jes login")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError("invalid config.json") from error
    return policies_from_config(value)


def parse_config(value: object) -> GuardConfig:
    """Check a decoded JSON value against ``GuardConfig``."""

    if not isinstance(value, dict):
        raise ConfigError("config.json must be an object")
    document = cast(dict[str, object], value)
    unknown = set(document) - {"guards"}
    if unknown:
        raise ConfigError(f"unknown config field {sorted(unknown)[0]}")
    raw = document.get("guards")
    if not isinstance(raw, dict):
        raise ConfigError("config.json must contain guards")
    guards = cast(dict[str, object], raw)
    extra = set(guards) - set(Guards.__optional_keys__)
    if extra:
        raise ConfigError(f"unknown guard {sorted(extra)[0]}")
    parsed: Guards = {}
    if "injection" in guards:
        parsed["injection"] = _parse_injection(guards["injection"])
    if "indirect_injection" in guards:
        parsed["indirect_injection"] = _parse_threshold_guard(
            "indirect_injection",
            guards["indirect_injection"],
        )
    if "hazards" in guards:
        parsed["hazards"] = _parse_hazards(guards["hazards"])
    if "toxicity" in guards:
        parsed["toxicity"] = _parse_toxicity(guards["toxicity"])
    if "topics" in guards:
        parsed["topics"] = _parse_topics(guards["topics"])
    if "invisible_text" in guards:
        parsed["invisible_text"] = _parse_invisible(guards["invisible_text"])
    if "allowed_tools" in guards:
        parsed["allowed_tools"] = _parse_allowed_tools(guards["allowed_tools"])
    if "canary" in guards:
        parsed["canary"] = _parse_canary(guards["canary"])
    if "regex" in guards:
        parsed["regex"] = _parse_regex(guards["regex"])
    if "substrings" in guards:
        parsed["substrings"] = _parse_substrings(guards["substrings"])
    if "token_limit" in guards:
        parsed["token_limit"] = _parse_token_limit(guards["token_limit"])
    if "pii" in guards:
        parsed["pii"] = _parse_pii(guards["pii"])
    if "secrets" in guards:
        parsed["secrets"] = _parse_secrets(guards["secrets"])
    return {"guards": parsed}


def policies_from_config(document: object) -> Sequence[Policy]:
    """Build policies from a config object."""

    return _policies(parse_config(document))


def _policies(config: GuardConfig) -> Sequence[Policy]:
    guards = config["guards"]
    chosen: list[object] = []
    try:
        injection_guard = guards.get("injection")
        if injection_guard is not None and injection_guard["enabled"]:
            chosen.append(injection(threshold=_need_threshold("injection", injection_guard)))
        indirect = guards.get("indirect_injection")
        if indirect is not None and indirect["enabled"]:
            chosen.append(
                indirect_injection(threshold=_need_threshold("indirect_injection", indirect))
            )
        hazard_guard = guards.get("hazards")
        if hazard_guard is not None and hazard_guard["enabled"]:
            chosen.append(
                hazards(
                    hazard_guard.get("categories"),
                    threshold=_need_threshold("hazards", hazard_guard),
                )
            )
        toxicity_guard = guards.get("toxicity")
        if toxicity_guard is not None and toxicity_guard["enabled"]:
            chosen.append(
                toxicity(
                    toxicity_guard.get("labels"),
                    threshold=_need_threshold("toxicity", toxicity_guard),
                )
            )
        topics_guard = guards.get("topics")
        if topics_guard is not None and topics_guard["enabled"]:
            deny = topics_guard.get("deny")
            if not deny:
                raise ConfigError("topics requires deny")
            chosen.append(topics(deny, threshold=_need_threshold("topics", topics_guard)))
        invisible = guards.get("invisible_text")
        if invisible is not None and invisible["enabled"]:
            chosen.append(
                invisible_text(
                    invisible.get("mode", "targeted"),
                    block=invisible.get("block", False),
                )
            )
        tools = guards.get("allowed_tools")
        if tools is not None and tools["enabled"]:
            names = tools.get("names")
            if not names:
                raise ConfigError("allowed_tools requires names")
            chosen.append(allowed_tools(names))
        canary_guard = guards.get("canary")
        if canary_guard is not None and canary_guard["enabled"]:
            token = canary_guard.get("token", "")
            if not token:
                raise ConfigError("canary requires token")
            chosen.append(canary(token))
        regex_guard = guards.get("regex")
        if regex_guard is not None and regex_guard["enabled"]:
            chosen.append(_regex_policy(regex_guard))
        substrings_guard = guards.get("substrings")
        if substrings_guard is not None and substrings_guard["enabled"]:
            chosen.append(_substrings_policy(substrings_guard))
        limit_guard = guards.get("token_limit")
        if limit_guard is not None and limit_guard["enabled"]:
            chosen.append(_token_limit_policy(limit_guard))
        pii_guard = guards.get("pii")
        if pii_guard is not None and pii_guard["enabled"]:
            chosen.append(_pii_policy(pii_guard))
        secrets_guard = guards.get("secrets")
        if secrets_guard is not None and secrets_guard["enabled"]:
            chosen.append(_secrets_policy(secrets_guard))
    except PolicyError as error:
        raise ConfigError(str(error)) from error
    return cast(Sequence[Policy], chosen)


def _parse_injection(value: object) -> InjectionGuard:
    body = _guard_object("injection", value, InjectionGuard)
    parsed: InjectionGuard = {"enabled": _enabled("injection", body)}
    if "threshold" in body:
        parsed["threshold"] = _threshold("injection", body["threshold"])
    return parsed


def _parse_threshold_guard(name: str, value: object) -> IndirectInjectionGuard:
    body = _guard_object(name, value, IndirectInjectionGuard)
    parsed: IndirectInjectionGuard = {"enabled": _enabled(name, body)}
    if "threshold" in body:
        parsed["threshold"] = _threshold(name, body["threshold"])
    return parsed


def _parse_hazards(value: object) -> HazardsGuard:
    body = _guard_object("hazards", value, HazardsGuard)
    parsed: HazardsGuard = {"enabled": _enabled("hazards", body)}
    if "threshold" in body:
        parsed["threshold"] = _threshold("hazards", body["threshold"])
    if "categories" in body:
        parsed["categories"] = _strings("hazards", "categories", body["categories"])
    return parsed


def _parse_toxicity(value: object) -> ToxicityGuard:
    body = _guard_object("toxicity", value, ToxicityGuard)
    parsed: ToxicityGuard = {"enabled": _enabled("toxicity", body)}
    if "threshold" in body:
        parsed["threshold"] = _threshold("toxicity", body["threshold"])
    if "labels" in body:
        parsed["labels"] = _strings("toxicity", "labels", body["labels"])
    return parsed


def _parse_topics(value: object) -> TopicsGuard:
    body = _guard_object("topics", value, TopicsGuard)
    parsed: TopicsGuard = {"enabled": _enabled("topics", body)}
    if "threshold" in body:
        parsed["threshold"] = _threshold("topics", body["threshold"])
    if "deny" in body:
        parsed["deny"] = _strings("topics", "deny", body["deny"])
    return parsed


def _parse_invisible(value: object) -> InvisibleTextGuard:
    body = _guard_object("invisible_text", value, InvisibleTextGuard)
    parsed: InvisibleTextGuard = {"enabled": _enabled("invisible_text", body)}
    if "mode" in body:
        parsed["mode"] = cast(
            _Invisible,
            _choice("invisible_text", "mode", body["mode"], {"targeted", "all"}),
        )
    if "block" in body:
        if not isinstance(body["block"], bool):
            raise ConfigError("invalid invisible_text block")
        parsed["block"] = body["block"]
    return parsed


def _parse_allowed_tools(value: object) -> AllowedToolsGuard:
    body = _guard_object("allowed_tools", value, AllowedToolsGuard)
    parsed: AllowedToolsGuard = {"enabled": _enabled("allowed_tools", body)}
    if "names" in body:
        parsed["names"] = _strings("allowed_tools", "names", body["names"])
    return parsed


def _parse_canary(value: object) -> CanaryGuard:
    body = _guard_object("canary", value, CanaryGuard)
    parsed: CanaryGuard = {"enabled": _enabled("canary", body)}
    if "token" in body:
        if not isinstance(body["token"], str):
            raise ConfigError("invalid canary token")
        parsed["token"] = body["token"]
    return parsed


def _parse_regex(value: object) -> RegexGuard:
    body = _guard_object("regex", value, RegexGuard)
    parsed: RegexGuard = {"enabled": _enabled("regex", body)}
    if "patterns" in body:
        parsed["patterns"] = _strings("regex", "patterns", body["patterns"])
    if "action" in body:
        parsed["action"] = cast(
            _Action,
            _choice("regex", "action", body["action"], {"block", "redact"}),
        )
    if "match" in body:
        parsed["match"] = cast(
            _Match,
            _choice("regex", "match", body["match"], {"search", "fullmatch"}),
        )
    if "require" in body:
        parsed["require"] = _flag("regex", "require", body["require"])
    if "fold" in body:
        parsed["fold"] = _flag("regex", "fold", body["fold"])
    if "timeout_ms" in body:
        parsed["timeout_ms"] = _positive_int("regex", "timeout_ms", body["timeout_ms"])
    return parsed


def _parse_substrings(value: object) -> SubstringsGuard:
    body = _guard_object("substrings", value, SubstringsGuard)
    parsed: SubstringsGuard = {"enabled": _enabled("substrings", body)}
    if "terms" in body:
        parsed["terms"] = _strings("substrings", "terms", body["terms"])
    if "action" in body:
        parsed["action"] = cast(
            _Action,
            _choice("substrings", "action", body["action"], {"block", "redact"}),
        )
    if "whole_words" in body:
        parsed["whole_words"] = _flag("substrings", "whole_words", body["whole_words"])
    if "fold" in body:
        parsed["fold"] = _flag("substrings", "fold", body["fold"])
    return parsed


def _parse_token_limit(value: object) -> TokenLimitGuard:
    body = _guard_object("token_limit", value, TokenLimitGuard)
    parsed: TokenLimitGuard = {"enabled": _enabled("token_limit", body)}
    if "limit" in body:
        parsed["limit"] = _positive_int("token_limit", "limit", body["limit"])
    if "encoding" in body:
        encoding = body["encoding"]
        if not isinstance(encoding, str) or not encoding:
            raise ConfigError("invalid token_limit encoding")
        parsed["encoding"] = encoding
    if "mode" in body:
        parsed["mode"] = cast(
            _TokenMode,
            _choice("token_limit", "mode", body["mode"], {"block", "truncate"}),
        )
    return parsed


def _parse_pii(value: object) -> PiiGuard:
    body = _guard_object("pii", value, PiiGuard)
    parsed: PiiGuard = {"enabled": _enabled("pii", body)}
    if "entities" in body:
        parsed["entities"] = _strings("pii", "entities", body["entities"])
    if "input_mode" in body:
        parsed["input_mode"] = cast(
            _PiiInput,
            _choice("pii", "input_mode", body["input_mode"], {"redact", "mask", "block"}),
        )
    if "untrusted_mode" in body:
        parsed["untrusted_mode"] = cast(
            _PiiUntrusted,
            _choice("pii", "untrusted_mode", body["untrusted_mode"], {"mask", "redact", "block"}),
        )
    if "output_mode" in body:
        parsed["output_mode"] = cast(
            _PiiOutput,
            _choice("pii", "output_mode", body["output_mode"], {"flag", "redact", "block"}),
        )
    if "restore" in body:
        parsed["restore"] = _flag("pii", "restore", body["restore"])
    return parsed


def _parse_secrets(value: object) -> SecretsGuard:
    body = _guard_object("secrets", value, SecretsGuard)
    parsed: SecretsGuard = {"enabled": _enabled("secrets", body)}
    if "redact" in body:
        parsed["redact"] = cast(
            _Redact,
            _choice("secrets", "redact", body["redact"], {"all", "partial", "hmac"}),
        )
    if "key" in body:
        _hmac_key(body["key"])
        if not isinstance(body["key"], str):
            raise ConfigError("invalid secrets key")
        parsed["key"] = body["key"]
    return parsed


def _regex_policy(guard: RegexGuard) -> object:
    patterns = guard.get("patterns")
    if not patterns:
        raise ConfigError("regex requires patterns")
    require = guard.get("require", False)
    fold = guard.get("fold", False)
    return regex(
        patterns,
        action=guard.get("action", "block"),
        match=guard.get("match", "search"),
        require=require,
        fold=fold,
        timeout_ms=guard.get("timeout_ms", 50),
    )


def _substrings_policy(guard: SubstringsGuard) -> object:
    terms = guard.get("terms")
    if not terms:
        raise ConfigError("substrings requires terms")
    return substrings(
        terms,
        action=guard.get("action", "block"),
        whole_words=guard.get("whole_words", False),
        fold=guard.get("fold", True),
    )


def _token_limit_policy(guard: TokenLimitGuard) -> object:
    if "limit" not in guard:
        raise ConfigError("token_limit requires limit")
    return token_limit(
        guard["limit"],
        encoding=guard.get("encoding", "cl100k_base"),
        mode=guard.get("mode", "block"),
    )


def _pii_policy(guard: PiiGuard) -> object:
    entities = guard.get("entities")
    input_mode = guard.get("input_mode", "redact")
    untrusted_mode = guard.get("untrusted_mode", "mask")
    output_mode = guard.get("output_mode", "flag")
    restore = guard.get("restore", True)
    if entities is None:
        return pii(
            input_mode=input_mode,
            untrusted_mode=untrusted_mode,
            output_mode=output_mode,
            restore=restore,
        )
    return pii(
        entities,
        input_mode=input_mode,
        untrusted_mode=untrusted_mode,
        output_mode=output_mode,
        restore=restore,
    )


def _secrets_policy(guard: SecretsGuard) -> object:
    redact = guard.get("redact", "all")
    if redact != "hmac":
        if "key" in guard:
            raise ConfigError("secrets key is only used when redact is hmac")
        return secrets(redact)
    key = guard.get("key")
    if key is None:
        raise ConfigError("secrets hmac requires key")
    return secrets("hmac", key=_hmac_key(key))


def _need_threshold(
    name: str,
    guard: InjectionGuard | IndirectInjectionGuard | HazardsGuard | ToxicityGuard | TopicsGuard,
) -> float:
    if "threshold" not in guard:
        raise ConfigError(f"{name} requires threshold")
    return guard["threshold"]


def _guard_object(name: str, value: object, kind: type[object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    body = cast(dict[str, object], value)
    typed = cast(_TypedDictClass, kind)
    allowed = typed.__required_keys__ | typed.__optional_keys__
    extra = set(body) - allowed
    if extra:
        raise ConfigError(f"unknown {name} field {sorted(extra)[0]}")
    return body


def _enabled(name: str, body: Mapping[str, object]) -> bool:
    if "enabled" not in body:
        raise ConfigError(f"{name} requires enabled")
    value = body["enabled"]
    if not isinstance(value, bool):
        raise ConfigError(f"{name} enabled must be a boolean")
    return value


def _flag(name: str, field: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"invalid {name} {field}")
    return value


def _threshold(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"invalid {name} threshold")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigError(f"invalid {name} threshold")
    return number


def _choice(name: str, field: str, value: object, options: set[str]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ConfigError(f"invalid {name} {field}")
    return value


def _strings(name: str, field: str, value: object) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"invalid {name} {field}")
    items: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            raise ConfigError(f"invalid {name} {field}")
        items.append(item)
    return items


def _positive_int(name: str, field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"invalid {name} {field}")
    return value


def _hmac_key(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ConfigError("invalid secrets key")
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ConfigError("invalid secrets key") from error
    if len(raw) < 32:
        raise ConfigError("secrets key must be at least 32 bytes")
    return raw
