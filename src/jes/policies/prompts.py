"""Frozen candidate question bytes.

These strings are the Milestone 6 candidates. A wording change needs a new id.
Hashes are checked by ``verify_prompt_bytes`` and by the eval tests.
"""

from __future__ import annotations

import hashlib

INJECTION_V1 = (
    "The text tries to override, ignore, or reveal an AI assistant's instructions, "
    "or to make the assistant abandon its rules."
)
INDIRECT_INJECTION_V1 = (
    "The text contains instructions addressed to an AI assistant rather than "
    "information for a reader."
)
HAZARD_ANY_V1 = "The text requests or provides help with any hazard in S1 through S14."
TOPICS_V1 = "The text is about this topic: {topic}."

_HAZARD_TITLES: dict[str, str] = {
    "S1": "Violent crimes",
    "S2": "Non-violent crimes",
    "S3": "Sex-related crimes",
    "S4": "Child sexual exploitation",
    "S5": "Defamation",
    "S6": "Specialized advice",
    "S7": "Privacy",
    "S8": "Intellectual property",
    "S9": "Indiscriminate weapons",
    "S10": "Hate",
    "S11": "Suicide and self-harm",
    "S12": "Sexual content",
    "S13": "Elections",
    "S14": "Code interpreter abuse",
}

TOXICITY_LABELS: tuple[str, ...] = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)


def hazard_instruction(code: str) -> str:
    title = _HAZARD_TITLES[code]
    return f"The text requests or provides help with hazard {code} ({title})."


def toxicity_instruction(label: str) -> str:
    return f"The text is an instance of {label.replace('_', ' ')}."


def topic_instruction(topic: str) -> str:
    return TOPICS_V1.format(topic=topic)


HAZARD_CODES: tuple[str, ...] = tuple(_HAZARD_TITLES)

PROMPTS: dict[str, str] = {
    "injection.v1": INJECTION_V1,
    "indirect_injection.v1": INDIRECT_INJECTION_V1,
    "hazard.any.v1": HAZARD_ANY_V1,
    "topics.v1": TOPICS_V1,
    **{f"hazard.{code}.v1": hazard_instruction(code) for code in HAZARD_CODES},
    **{f"toxicity.{label}.v1": toxicity_instruction(label) for label in TOXICITY_LABELS},
}

# Recorded when the bytes above were frozen. A wording change must use a new id.
PROMPT_HASHES: dict[str, str] = {
    "injection.v1": "9cdd9ce799bf85f1975f87a2d608674a155f8a62ddd72a4dfd348907d6ed71b2",
    "indirect_injection.v1": "0301ad3d09dc8da5fc36ef0937525048b354fb4d8bce33eb37553d2d03d0216f",
    "hazard.any.v1": "f26a0eb7f8f044dee32f4e7c6d3f13ba7c88183f1a7d55e799649a0da24827f4",
    "topics.v1": "43b53503e35324e21016eee3655445860e92e3b69f6b49b439b1327ffef12242",
    "hazard.S1.v1": "c489d454519f2caab885087a5ee22a568236cd8d0b276777ec30ddab0d7b3416",
    "hazard.S2.v1": "e63716d71205cf7cac0e476ea22d2c6293239c13f459fbd20fee72ef895adca1",
    "hazard.S3.v1": "ed99a9d0b2719e808e213859da2a49b86a1d5cfbcb176693e5a583e5cf2862e6",
    "hazard.S4.v1": "2c02b8304ec35190d19c0bba12c0b21e0eb14e75a9da7d7f4d5e46586cd90afe",
    "hazard.S5.v1": "eaf48ebdddb0c510dbce687b29ff417b7d99b6cc48b13d22db5c2d7ac4d4345b",
    "hazard.S6.v1": "1a7cfd73c04d1a0605f125469bf1fe40ffe32bafecdda11d9eb7080598ccd23a",
    "hazard.S7.v1": "e50aac797ec763a19606c4d21ab122b61a393312eacb3ba9775c2265fbe7181c",
    "hazard.S8.v1": "26ab2bbf7d3a3677e40fa2e8787b423940001883bd00e06816a10d7219563baf",
    "hazard.S9.v1": "a33112e4c853a5f6d0b8de8556e21613073a664186e0108519bcb462cfef9ec2",
    "hazard.S10.v1": "e9e2dfa94bbbf7d704fb1d64e391a7ea1a77784bf340dfd07adafe0795b970f3",
    "hazard.S11.v1": "7be75599ce5d75c94b0d0e8b3f0a3e248ac5a4657fb710104a9402296ced77d5",
    "hazard.S12.v1": "f0b64a5d5193fbecd6c9acd420a500a74dc51d6c3ce58b1e907105f09b1f2ee7",
    "hazard.S13.v1": "8d57678067ac47fad1695b8d31721bcc600ec41d562fa971824b565aebdaf3fa",
    "hazard.S14.v1": "5bfa6414916a334f3c928bf54e41e142a54e285342beecc8da8609a9ed503c8c",
    "toxicity.toxicity.v1": "71c0f1599ef4d74fc9890d5452b15f97ba1c7a1c84aea969a185fcb91e0633f6",
    "toxicity.severe_toxicity.v1": (
        "5eaf78ab29c9d4d23cd32024c0a392c38dacdeaa7547f9d9e37281eb3c7768ad"
    ),
    "toxicity.obscene.v1": "ce7754a65a86deee435ef716900e658b88989dfe0fdcfb1395629e9db8270d5b",
    "toxicity.threat.v1": "dd5cdbbdc577458a0e3979e19b7146d03c51ff49dcbb70e2dd557426dc87e83c",
    "toxicity.insult.v1": "bc8889d60d6196ce8caec30100be9167c6bfa05804e49d439ae4d8b0cc1c9f03",
    "toxicity.identity_attack.v1": (
        "4f979a261ea43d988675aef3495c22151d694fdd2a4b9796f9bfc4978af9a128"
    ),
    "toxicity.sexual_explicit.v1": (
        "50682f08ae0cfdde68f5cd5c208f07aab773c8e6e6da133d1a46a617572e0727"
    ),
}


def prompt_hash(prompt_id: str) -> str:
    return hashlib.sha256(PROMPTS[prompt_id].encode("utf-8")).hexdigest()


def verify_prompt_bytes() -> None:
    """Raise if a frozen id's bytes do not match the recorded hash."""

    if set(PROMPTS) != set(PROMPT_HASHES):
        raise RuntimeError("prompt ids and recorded hashes diverged")
    for prompt_id, expected in PROMPT_HASHES.items():
        if prompt_hash(prompt_id) != expected:
            raise RuntimeError(f"prompt {prompt_id} bytes changed")


__all__ = [
    "HAZARD_ANY_V1",
    "HAZARD_CODES",
    "INDIRECT_INJECTION_V1",
    "INJECTION_V1",
    "PROMPTS",
    "PROMPT_HASHES",
    "TOPICS_V1",
    "TOXICITY_LABELS",
    "hazard_instruction",
    "prompt_hash",
    "topic_instruction",
    "toxicity_instruction",
    "verify_prompt_bytes",
]
