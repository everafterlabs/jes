"""Text maps, Unicode folding, and character properties."""

from jes.text.folding import Folded, fold, machine_identifier, nfkc
from jes.text.textmap import Edit, TextMap

__all__ = ["Edit", "Folded", "TextMap", "fold", "machine_identifier", "nfkc"]
