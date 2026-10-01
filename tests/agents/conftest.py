from pathlib import Path

import pytest

from jes.agents.config import write_default_config


@pytest.fixture(autouse=True)
def _jes_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh ~/.config with the default guard config, for every agent test."""

    home = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    write_default_config()
    return home
