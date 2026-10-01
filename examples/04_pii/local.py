"""Lesson 04 (local): Keep PII away from the model and give it back to the user.

pii() swaps an email address for a placeholder on the way in and restores it
on the way out. One Redactions store per conversation keeps placeholders
stable. pii is local, so only the chat model differs from local.py.

Run:   uv run --group examples python -m examples.04_pii.local
Needs: Ollama with qwen3:1.7b (ollama pull qwen3:1.7b)
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from examples._common import print_check
from jes import Guard, Redactions
from jes.policies import pii


def chat_model() -> BaseChatModel:
    # Thinking off. ChatOllama takes its timeout in client_kwargs.
    return init_chat_model(
        "ollama:qwen3:1.7b",
        base_url="http://localhost:11434",
        temperature=0,
        reasoning=False,
        client_kwargs={"timeout": 120},
    )


SYSTEM = (
    "Answer in one short sentence. Copy any [JES_v1_PII_...] placeholder "
    "exactly as written; it stands for the user's real value."
)
# Demo only. Load a real 32-byte key from your secret manager.
STORE_KEY = b"k" * 32


def main() -> None:
    guard = Guard([pii()])
    # The scope names the store; loading a saved store needs the same scope.
    store = Redactions(scope=b"conversation-1")

    incoming = guard.check_input(
        "Please confirm my email address: ada@example.com", redactions=store
    )
    reply = chat_model().invoke([("system", SYSTEM), ("user", incoming.onward)]).text
    outgoing = guard.check_output(reply, prompt=incoming, redactions=store)
    # Same store, so the same address gets the same placeholder.
    again = guard.check_input("ada@example.com again", redactions=store)

    # Persist the store encrypted between requests. associated_data binds the
    # blob to its context; loading it needs the same bytes.
    blob = store.dumps(STORE_KEY, associated_data=b"cookbook")

    print("== pii")
    print_check("input", incoming)
    print_check("output", outgoing)
    print_check("again", again)
    print(f"  reply: {outgoing.onward}")
    print(f"  saved store: {len(blob)} encrypted bytes")


if __name__ == "__main__":
    main()
