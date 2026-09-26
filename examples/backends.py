"""Backend constructors. Importing this module does not call a model."""

from jes.backends import LiteLLMJudge, LlamaGuard4, PromptGuard2, SystemOne


def hosted_jev() -> SystemOne:
    # Reads TYPESAFE_API_KEY from the environment or a project .env file.
    return SystemOne.hosted()


def local_laya() -> SystemOne:
    return SystemOne.local(
        "http://127.0.0.1:8000",
        model="english",
        max_request_bytes=8_192,
    )


def litellm_local() -> LiteLLMJudge:
    # Requires jes[litellm]. logprobs and verbalized are different profiles.
    return LiteLLMJudge(
        "ollama/llama3.1:8b",
        mode="logprobs",
        provider="ollama",
        provider_profile="ollama.chat.v1",
        revision="pinned",
        max_request_bytes=8_192,
    )


def prompt_guard() -> PromptGuard2:
    # Requires jes[prompt-guard]. Answers injection only.
    return PromptGuard2.local(revision="pinned-checkpoint")


def llama_guard() -> LlamaGuard4:
    # Requires jes[llama-guard]. Answers hazard.any and hazard.S1 through S14.
    return LlamaGuard4.local(revision="pinned-checkpoint")
