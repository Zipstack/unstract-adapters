import logging
from abc import ABC, abstractmethod
from typing import Any, Union

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStore,
)

from unstract.adapters.enums import AdapterTypes

logger = logging.getLogger(__name__)


class Adapter(ABC):
    def __init__(self, name: str):
        self.name = name

    @staticmethod
    @abstractmethod
    def get_id() -> str:
        return ""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        return ""

    @staticmethod
    @abstractmethod
    def get_description() -> str:
        return ""

    @staticmethod
    @abstractmethod
    def get_icon() -> str:
        return ""

    @staticmethod
    @abstractmethod
    def get_json_schema() -> str:
        return ""

    @staticmethod
    @abstractmethod
    def get_adapter_type() -> AdapterTypes:
        return ""

    @abstractmethod
    def get_llm_instance(self, llm_config: dict[str, Any]) -> LLM:
        # Overriding implementations use llm_config
        pass

    @abstractmethod
    def get_vector_db_instance(
        self, vector_db_config: dict[str, Any]
    ) -> Union[BasePydanticVectorStore, VectorStore]:
        # Overriding implementations use vector_db_config
        pass

    @abstractmethod
    def get_embedding_instance(
        self, embed_config: dict[str, Any]
    ) -> BaseEmbedding:
        # Overriding implementations use embed_config
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Override to test connection for a adapter.

        Returns:
            bool: Flag indicating if the credentials are valid or not
        """
        pass
