import logging
from abc import ABC, abstractmethod
from typing import Any, Union

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.vector_stores import SimpleVectorStore
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

    def get_llm_instance(self, llm_config: dict[str, Any]) -> LLM:
        # Overriding implementations use llm_config
        return MockLLM()

    def get_vector_db_instance(
        self, vector_db_config: dict[str, Any]
    ) -> Union[BasePydanticVectorStore, VectorStore]:
        # Overriding implementations use vector_db_config
        return SimpleVectorStore()

    def get_embedding_instance(
        self, embed_config: dict[str, Any]
    ) -> BaseEmbedding:
        # TODO: Return mock embedding, Rempve type ignore
        # Non-matching return type here to avoid the below error
        # RuntimeError: There is no current event loop in
        # thread 'ThreadPoolExecutor-0_0'

        # Overriding implementations use embed_config
        return  # type: ignore

    @abstractmethod
    def test_connection(self) -> bool:
        """Override to test connection for a adapter.

        Returns:
            bool: Flag indicating if the credentials are valid or not
        """
        pass
