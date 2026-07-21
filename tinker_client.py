from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import tinker
from tinker import types


_ORIGINAL_SERVICE_CLIENT = tinker.ServiceClient


def _service_client_with_safe_transport(*args, **kwargs):
    client_config = kwargs.pop("_client_config", None)
    if client_config is None:
        client_config = types.ClientConfigResponse(use_pyqwest_transport=False)
    else:
        client_config = client_config.model_copy(update={"use_pyqwest_transport": False})
    return _ORIGINAL_SERVICE_CLIENT(*args, _client_config=client_config, **kwargs)


@contextmanager
def disable_pyqwest_transport() -> Iterator[None]:
    original_service_client = tinker.ServiceClient
    tinker.ServiceClient = _service_client_with_safe_transport
    try:
        yield
    finally:
        tinker.ServiceClient = original_service_client
