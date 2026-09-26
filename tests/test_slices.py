"""Development slice rules."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evals.slices import development_cohort, load_bipia_train, load_xstest_prompts


def test_holdout_is_stable_and_partial() -> None:
    first = [development_cohort(f"row-{index}") for index in range(50)]
    assert first == [development_cohort(f"row-{index}") for index in range(50)]
    assert 0 < first.count(False) < 50


def test_xstest_keeps_unsafe_out_of_injection_gold(tmp_path: Path) -> None:
    path = tmp_path / "xstest_prompts.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["id", "prompt", "type", "label", "focus", "note"])
        writer.writeheader()
        for index in range(30):
            writer.writerow(
                {
                    "id": str(index),
                    "prompt": f"prompt {index}",
                    "type": "safe_contexts" if index % 2 == 0 else "contrast_safe_contexts",
                    "label": "safe" if index % 2 == 0 else "unsafe",
                    "focus": "x",
                    "note": "",
                }
            )
    injection, hazards = load_xstest_prompts(path)
    assert all(not row.label for row in injection.labeled)
    assert all("unsafe" not in row.example_id for row in injection.labeled)
    assert any(row.label for row in hazards.labeled)
    assert all(row.stage == "input" for row in hazards.labeled)
    with pytest.raises(ValueError):
        load_xstest_prompts(tmp_path / "model_completions.csv")


def test_bipia_refuses_test_and_email(tmp_path: Path) -> None:
    contexts = tmp_path / "train.jsonl"
    attacks = tmp_path / "text_attack_train.json"
    contexts.write_text(
        "".join(
            json.dumps({"context": f"A short table {index}.", "question": "q", "ideal": "a"})
            + "\n"
            for index in range(15)
        ),
        encoding="utf-8",
    )
    attacks.write_text(json.dumps({"Task Automation": ["do the extra task"]}), encoding="utf-8")
    loaded = load_bipia_train(contexts, attacks, "table")
    assert loaded.labeled
    assert {row.label for row in loaded.labeled} <= {False, True}
    assert all(row.stage == "untrusted" for row in loaded.labeled)
    with pytest.raises(ValueError):
        load_bipia_train(tmp_path / "test.jsonl", attacks, "table")
    with pytest.raises(ValueError):
        load_bipia_train(contexts, attacks, "email")
