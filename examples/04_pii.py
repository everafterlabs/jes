"""Lesson 04: Keep PII away from the model and give it back to the user.

pii() swaps an email address for a placeholder before the model sees it, and
restores it in the reply. One Redactions store per conversation keeps the
placeholder stable. Prints what the model saw, what a later message
became, and what the user sees.

Run:   uv run --group examples python -m examples.04_pii
Mock:  add --mock to run offline, no keys needed.
Needs: OPENAI_API_KEY
"""

from examples._backend import Check, chat_model, parse_mock, show
from examples._mocks import scripted_chat
from jes import Guard, Redactions
from jes.policies import pii
from jes.types import InputResult, ScanResult

SYSTEM = (
    "Answer in one short sentence. Copy any [JES_v1_PII_...] placeholder "
    "exactly as written; it stands for the user's real value."
)
# Demo only; load a real 32-byte key from your secret manager, never commit one.
STORE_KEY = b"k" * 32


def main(*, mock: bool = False) -> tuple[InputResult, ScanResult, InputResult, bytes]:
    # One store per conversation; scope names it, and loading a saved store needs it.
    store = Redactions(scope=b"conversation-1")
    # pii is local, so this guard needs no decision model.
    guard = Guard([pii()])

    incoming = guard.check_input("email me at ada@example.com", redactions=store)
    # The model sees incoming.onward, never the address. The mock repeats the
    # placeholder, as SYSTEM asks a real model to.
    llm = scripted_chat([f"Sure: {incoming.onward}"]) if mock else chat_model()
    reply = llm.invoke([("system", SYSTEM), ("user", incoming.onward)]).text
    # The user sees outgoing.onward, with the address restored.
    outgoing = guard.check_output(reply, prompt=incoming, redactions=store)

    # A later message: the same address gets the same placeholder from this store.
    again = guard.check_input("ada@example.com again", redactions=store)
    # Save the store encrypted between requests. associated_data ties the blob to
    # where it belongs (say, your app's name); loading needs the same bytes.
    blob = store.dumps(STORE_KEY, associated_data=b"cookbook")
    return incoming, outgoing, again, blob


if __name__ == "__main__":
    incoming, outgoing, again, blob = main(mock=parse_mock())
    show("pii", [Check("input", incoming), Check("output", outgoing), Check("again", again)])
    print(f"  model saw: {incoming.onward}")
    print(f"  model saw later (same placeholder): {again.onward}")
    print(f"  user sees: {outgoing.onward}")
    print(f"  saved store: {len(blob)} encrypted bytes")
