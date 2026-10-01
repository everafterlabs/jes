"""Lesson 04 (Jev): Keep PII away from the model and give it back to the user.

pii() swaps an email address for a placeholder before the model sees it, and
restores it in the reply. One Redactions store per conversation keeps the
placeholder stable. pii is local, so only the chat model differs between
jev.py and local.py. Prints what the model saw, a later message, and the reply.

Run:   uv run --group examples python -m examples.04_pii.jev
Needs: OPENAI_API_KEY
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from examples._common import Check, require_env, show
from jes import Guard, Redactions
from jes.policies import pii
from jes.types import InputResult, ScanResult

SYSTEM = (
    "Answer in one short sentence. Copy any [JES_v1_PII_...] placeholder "
    "exactly as written; it stands for the user's real value."
)
# Demo only; load a real 32-byte key from your secret manager, never commit one.
STORE_KEY = b"k" * 32


def chat_model() -> BaseChatModel:
    # OpenAI gpt-5.4-mini, the chat model that answers the user.
    return init_chat_model("openai:gpt-5.4-mini", timeout=60, max_retries=2)


def main() -> tuple[InputResult, ScanResult, InputResult, bytes]:
    # One store per conversation; scope names it, and loading a saved store needs it.
    store = Redactions(scope=b"conversation-1")
    # pii is local, so this guard needs no decision model.
    guard = Guard([pii()])

    incoming = guard.check_input(
        "Please confirm my email address: ada@example.com", redactions=store
    )
    # The model sees incoming.onward, never the address.
    reply = chat_model().invoke([("system", SYSTEM), ("user", incoming.onward)]).text
    # The user sees outgoing.onward, with the address restored.
    outgoing = guard.check_output(reply, prompt=incoming, redactions=store)

    # A later message: the same address gets the same placeholder from this store.
    again = guard.check_input("ada@example.com again", redactions=store)
    # Save the store encrypted between requests. associated_data ties the blob to
    # where it belongs (say, your app's name); loading needs the same bytes.
    blob = store.dumps(STORE_KEY, associated_data=b"cookbook")

    show("pii", [Check("input", incoming), Check("output", outgoing), Check("again", again)])
    print(f"  model saw: {incoming.onward}")
    print(f"  model saw later (same placeholder): {again.onward}")
    print(f"  user sees: {outgoing.onward}")
    print(f"  saved store: {len(blob)} encrypted bytes")
    return incoming, outgoing, again, blob


if __name__ == "__main__":
    require_env("OPENAI_API_KEY")
    main()
