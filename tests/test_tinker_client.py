from __future__ import annotations

import tinker

from tinker_client import disable_pyqwest_transport


def test_disable_pyqwest_transport_restores_service_client() -> None:
    original = tinker.ServiceClient
    with disable_pyqwest_transport():
        assert tinker.ServiceClient is not original
    assert tinker.ServiceClient is original
