from __future__ import annotations

import jes


def test_version_and_import() -> None:
    assert jes.__version__ == "0.2.0"
    assert jes.Guard is not None
    assert jes.AsyncGuard is not None
    assert jes.Redactions is not None
