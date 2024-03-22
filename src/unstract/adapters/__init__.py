__version__ = "0.6.0"

import logging
from logging import NullHandler
from typing import Any

logging.getLogger(__name__).addHandler(NullHandler())

AdapterDict = dict[str, dict[str, Any]]


def get_adapter_version() -> str:
    """Returns the adapter package's version."""
    return __version__
