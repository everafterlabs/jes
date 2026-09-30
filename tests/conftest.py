import pytest
from hypothesis import settings

from jes.config import write_default_config

settings.register_profile("ci", max_examples=40, deadline=None)
settings.load_profile("ci")


@pytest.fixture(autouse=True)
def _default_guard_config(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    write_default_config()
