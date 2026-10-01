"""The course in examples/: every lesson ships a jev.py and a local.py.

Each lesson's behavior is tested offline in tests/lessons/.
"""

from __future__ import annotations

from pathlib import Path

from jes import Guard
from jes.policies import toxicity
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend

ROOT = Path(__file__).parents[1]
LESSONS = sorted(path.name for path in (ROOT / "examples").glob("[0-9][0-9]_*") if path.is_dir())


def test_course_has_fifteen_lessons_with_both_variants() -> None:
    assert [name[:2] for name in LESSONS] == [f"{n:02d}" for n in range(1, 16)]
    for name in LESSONS:
        for variant in ("jev.py", "local.py", "__init__.py"):
            assert (ROOT / "examples" / name / variant).is_file(), f"{name}/{variant}"
        assert list((ROOT / "tests" / "lessons").glob(f"test_{name}.py")), name


def test_syllabus_and_cookbook_name_every_lesson() -> None:
    syllabus = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    cookbook = (ROOT / "docs" / "cookbook.md").read_text(encoding="utf-8")
    for name in LESSONS:
        assert f"{name}/jev.py" in syllabus
        assert f"{name}/local.py" in syllabus
        assert name in cookbook
    assert "docs/recipes.md" in cookbook


def test_default_answer_fills_missing_questions() -> None:
    result = Guard(
        [toxicity(threshold=0.70)],
        model=FakeBackend(
            default_answer=YesNoAnswer(0.0, "probability"),
            answers={"insult": YesNoAnswer(0.9, "probability")},
        ),
    ).check_input("ordinary sentence")
    assert result.decision == "block"
    assert result.scores["toxicity.insult"].value == 0.9
    assert result.scores["toxicity.threat"].value == 0.0
