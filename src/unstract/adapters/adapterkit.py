import logging
from typing import Any

from unstract.adapters import AdapterDict
from unstract.adapters.base import Adapter
from unstract.adapters.constants import Common
from unstract.adapters.embedding import adapters as embedding_adapters
from unstract.adapters.llm import adapters as llm_adapters
from unstract.adapters.vectordb import adapters as vectordb_adapters

logger = logging.getLogger(__name__)


class Adapterkit:
    def __init__(self) -> None:
        self._adapters: AdapterDict = (
            embedding_adapters | llm_adapters | vectordb_adapters
        )

    @property
    def adapters(self) -> AdapterDict:
        return self._adapters

    def get_adapter_class_by_adapter_id(self, adapter_id: str) -> Adapter:
        if adapter_id in self._adapters:
            adapter_class: Adapter = self._adapters[adapter_id][
                Common.METADATA
            ][Common.ADAPTER]
            return adapter_class
        else:
            raise RuntimeError(f"Couldn't obtain adapter for {adapter_id}")

    def get_adapter_by_id(
        self, adapter_id: str, *args: Any, **kwargs: Any
    ) -> Adapter:
        """Instantiates and returns a adapter.

        Args:
            adapter_id (str): Identifies adapter to create

        Raises:
            RuntimeError: If the ID is invalid/adapter is missing

        Returns:
            Adapter: Concrete impl of the `Adapter` base
        """
        adapter_class: Adapter = self.get_adapter_class_by_adapter_id(
            adapter_id
        )
        return adapter_class(*args, **kwargs)

    def get_adapters_list(self) -> list[dict[str, Any]]:
        adapters = []
        for adapter_id, adapter_registry_metadata in self._adapters.items():
            m: Adapter = adapter_registry_metadata[Common.METADATA][
                Common.ADAPTER
            ]
            _id = m.get_id()
            name = m.get_name()
            adapter_type = m.get_adapter_type().name
            json_schema = m.get_json_schema()
            desc = m.get_description()
            icon = m.get_icon()
            adapters.append(
                {
                    "id": _id,
                    "name": name,
                    "class_name": m.__name__,
                    "description": desc,
                    "icon": icon,
                    "adapter_type": adapter_type,
                    "json_schema": json_schema,
                }
            )
        return adapters
