"""Development slices. Test files and unlicensed tasks are not loaded."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from evals.runner import Example

HOLDOUT_SALT = "jes-dev-v1"


@dataclass(frozen=True, slots=True)
class PromptSlice:
    labeled: list[Example]
    unlabeled: list[Example]
    held_out: int


def source_id(example_id: str) -> str:
    """Document id. BIPIA suffixes such as ``:clean`` stay on the same document."""

    if example_id.startswith("bipia:"):
        parts = example_id.split(":")
        if len(parts) >= 4:
            return ":".join(parts[:4])
    return example_id


def development_cohort(example_id: str) -> bool:
    """Keep four fifths for development. The other fifth is not sent to a model."""

    digest = hashlib.sha256(f"{HOLDOUT_SALT}:{example_id}".encode()).digest()
    return digest[0] % 5 != 0


def load_xstest_prompts(path: Path) -> tuple[PromptSlice, PromptSlice]:
    """Return injection and hazard slices from ``xstest_prompts.csv`` only."""

    if path.name != "xstest_prompts.csv":
        raise ValueError("only xstest_prompts.csv is cleared")
    injection_labeled: list[Example] = []
    injection_unlabeled: list[Example] = []
    hazard_labeled: list[Example] = []
    held_out = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            example_id = f"xstest:{row['id']}"
            if not development_cohort(example_id):
                held_out += 1
                continue
            text = row["prompt"]
            safe = row["label"] == "safe"
            if safe:
                injection_labeled.append(Example(example_id, example_id, text, False))
            else:
                injection_unlabeled.append(Example(example_id, example_id, text, False))
            hazard_labeled.append(Example(example_id, example_id, text, not safe))
    return (
        PromptSlice(injection_labeled, injection_unlabeled, held_out),
        PromptSlice(hazard_labeled, [], held_out),
    )


def load_bipia_train(context_path: Path, attack_path: Path, task: str) -> PromptSlice:
    """One end-inserted train attack per train context. Test files are refused."""

    if task not in {"table", "code"}:
        raise ValueError("only table and code are cleared")
    if context_path.name != "train.jsonl":
        raise ValueError("test contexts are reserved")
    if not attack_path.name.startswith(f"{'text' if task == 'table' else 'code'}_attack_train"):
        raise ValueError("test attacks are reserved")
    attacks = _attack_variants(attack_path)
    if not attacks:
        raise ValueError("attack file has no variants")
    labeled: list[Example] = []
    held_out = 0
    for index, row in enumerate(_jsonl(context_path)):
        example_id = f"bipia:{task}:train:{index}"
        if not development_cohort(example_id):
            held_out += 1
            continue
        context = _as_text(row["context"])
        name, attack = attacks[index % len(attacks)]
        poisoned = f"{context}\n{attack}"
        labeled.append(Example(f"{example_id}:clean", example_id, context, False, "untrusted"))
        labeled.append(
            Example(
                f"{example_id}:end:{name}",
                f"{example_id}:end:{name}",
                poisoned,
                True,
                "untrusted",
            )
        )
    return PromptSlice(labeled, [], held_out)


def locked_source_ids(root: Path) -> dict[str, dict[str, list[str]]]:
    """Ids only. Burned rows were sent; reserved rows were not."""

    xstest_burned, xstest_reserved = _hash_split(
        _xstest_ids(root / "xstest" / "xstest_prompts.csv")
    )
    table_burned, table_reserved = _hash_split(
        _bipia_ids(root / "bipia" / "benchmark" / "table" / "train.jsonl", "table", "train")
    )
    code_burned, code_reserved = _hash_split(
        _bipia_ids(root / "bipia" / "benchmark" / "code" / "train.jsonl", "code", "train")
    )
    return {
        "xstest-prompts": {"burned": xstest_burned, "reserved_holdout": xstest_reserved},
        "bipia-table": {
            "burned": table_burned,
            "reserved_holdout": table_reserved,
            "reserved_test": _bipia_ids(
                root / "bipia" / "benchmark" / "table" / "test.jsonl",
                "table",
                "test",
            ),
        },
        "bipia-code": {
            "burned": code_burned,
            "reserved_holdout": code_reserved,
            "reserved_test": _bipia_ids(
                root / "bipia" / "benchmark" / "code" / "test.jsonl",
                "code",
                "test",
            ),
        },
    }


def _hash_split(example_ids: list[str]) -> tuple[list[str], list[str]]:
    burned = [example_id for example_id in example_ids if development_cohort(example_id)]
    reserved = [example_id for example_id in example_ids if not development_cohort(example_id)]
    return burned, reserved


def _xstest_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [f"xstest:{row['id']}" for row in csv.DictReader(handle)]


def _bipia_ids(path: Path, task: str, split: str) -> list[str]:
    return [f"bipia:{task}:{split}:{index}" for index, _row in enumerate(_jsonl(path))]


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = _mapping(cast(object, json.loads(line)))
        if parsed is not None:
            rows.append(parsed)
    return rows


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = cast(list[object], value)
        return "\n".join(str(part) for part in parts)
    raise ValueError("context is not text")


def _attack_variants(path: Path) -> list[tuple[str, str]]:
    parsed = _mapping(cast(object, json.loads(path.read_text(encoding="utf-8"))))
    if parsed is None:
        raise ValueError("attack file is not an object")
    variants: list[tuple[str, str]] = []
    for name in sorted(parsed):
        body = parsed[name]
        if not isinstance(body, list):
            continue
        items = cast(list[object], body)
        if items and isinstance(items[0], str):
            variants.append((f"{name}-0", items[0]))
    return variants


def _mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    return {key: item for key, item in raw.items() if isinstance(key, str)}
