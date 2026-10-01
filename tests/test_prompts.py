"""The built-in questions keep their exact bytes. A wording change needs a new id."""

from __future__ import annotations

import hashlib

from jes.policies.prompts import PROMPTS

# Recorded when each question was frozen.
FROZEN: dict[str, str] = {
    "injection.v1": "9cdd9ce799bf85f1975f87a2d608674a155f8a62ddd72a4dfd348907d6ed71b2",
    "indirect_injection.v1": "0301ad3d09dc8da5fc36ef0937525048b354fb4d8bce33eb37553d2d03d0216f",
    "tool_safety.v1": "3c804b577251f63268f173316656add619e05f8670ef4e498adef25503723720",
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


def test_frozen_prompt_bytes_match_their_recorded_hashes() -> None:
    assert set(PROMPTS) == set(FROZEN)
    for prompt_id, text in PROMPTS.items():
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == FROZEN[prompt_id], prompt_id
