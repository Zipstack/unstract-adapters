from abc import ABC
from typing import Any, Union

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStore

from unstract.adapters.base import Adapter
from unstract.adapters.enums import AdapterTypes


class VectorDBAdapter(Adapter, ABC):
    def __init__(
        self,
        name: str,
        vector_db_instance: Union[VectorStore, BasePydanticVectorStore],
    ):
        super().__init__(name)
        self.name = name
        self._vector_db_instance: Union[VectorStore, BasePydanticVectorStore] = (
            vector_db_instance
        )

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
        return AdapterTypes.VECTOR_DB

    def get_vector_db_instance(
        self, vector_db_config: dict[str, Any]
    ) -> Union[BasePydanticVectorStore, VectorStore]:
        """Instantiate the llama index VectorStore / BasePydanticVectorStore
        class.

        Returns:
            BasePydanticVectorStore / VectorStore:
                            llama index implementation of the vector store
            Raises exceptions for any error
        """
        return SimpleVectorStore()

    def close(self, **kwargs: Any) -> None:
        """Closes the client connection.

        Returns:
            None
        """
        # Overriding implementations will have the corresponding
        # library methods invoked
        pass

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete the specified docs.

        Returns:
            None
        """
        # Overriding implementations will have the corresponding
        # library methods invoked
        self._vector_db_instance.delete(
            ref_doc_id=ref_doc_id, delete_kwargs=delete_kwargs
        )

    def add(self, ref_doc_id: str, nodes: list[BaseNode]) -> list[str]:
        return self._vector_db_instance.add(nodes=nodes)
