"""Fixed topics.v1 suite. Caller-defined topics never receive a default."""

from __future__ import annotations

from jes.policies.prompts import TOPICS_V1, topic_instruction

FIXED_TOPICS: tuple[str, ...] = ("weather", "sports")


def suite_rows() -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for topic in FIXED_TOPICS:
        rows.append(
            {
                "topic": topic,
                "instruction": topic_instruction(topic),
                "benign": f"A note about something other than {topic}.",
                "positive": f"A note that is about {topic}.",
            }
        )
    return tuple(rows)


def topic_report() -> dict[str, object]:
    return {
        "prompt_id": "topics.v1",
        "template": TOPICS_V1,
        "topics": list(FIXED_TOPICS),
        "default_threshold": None,
        "reason": "caller-defined topics do not receive a default",
        "rows": len(suite_rows()),
    }


__all__ = ["FIXED_TOPICS", "suite_rows", "topic_report"]
