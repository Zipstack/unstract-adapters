import logging
from abc import ABC
from typing import Any, Optional

from llama_index.llms.llm import LLM
from unstract.adapters.base import Adapter
from unstract.adapters.enums import AdapterTypes

logger = logging.getLogger(__name__)


class LLMAdapter(Adapter, ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name

    @staticmethod
    def get_id() -> str:
        return ""

    @staticmethod
    def get_name() -> str:
        return ""

    @staticmethod
    def get_description() -> str:
        return ""

    @staticmethod
    def get_icon() -> str:
        return ""

    @staticmethod
    def get_json_schema() -> str:
        return ""

    @staticmethod
    def get_adapter_type() -> AdapterTypes:
        return AdapterTypes.LLM

    def get_llm_instance(self) -> Optional[LLM]:
        """Instantiate the llama index LLM class.

        Returns:
            Optional[LLM]: llama index implementation of the LLM
        """
        return None

    def test_connection(self, llm_metadata: dict[str, Any]) -> bool:
        return False
