import logging
import os
from typing import Any, Optional

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.types import BasePydanticVectorStore
from qdrant_client import QdrantClient
from unstract.adapters.exceptions import AdapterError
from unstract.adapters.vectordb.constants import VectorDbConstants
from unstract.adapters.vectordb.helper import VectorDBHelper
from unstract.adapters.vectordb.vectordb_adapter import VectorDBAdapter

logger = logging.getLogger(__name__)


class Constants:
    URL = "url"
    API_KEY = "api_key"


class Qdrant(VectorDBAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("Qdrant")
        self.config = settings
        self.client: Optional[QdrantClient] = None
        self.collection_name: str = VectorDbConstants.DEFAULT_VECTOR_DB_NAME

    @staticmethod
    def get_id() -> str:
        return "qdrant|41f64fda-2e4c-4365-89fd-9ce91bee74d0"

    @staticmethod
    def get_name() -> str:
        return "Qdrant"

    @staticmethod
    def get_description() -> str:
        return "Qdrant LLM"

    @staticmethod
    def get_icon() -> str:
        return (
            "/icons/"
            "adapter-icons/qdrant.png"
        )

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_vector_db_instance(self) -> Optional[BasePydanticVectorStore]:
        try:
            self.collection_name = VectorDBHelper.get_collection_name(
                self.config.get(VectorDbConstants.VECTOR_DB_NAME),
                self.config.get(VectorDbConstants.EMBEDDING_DIMENSION),
            )
            url = self.config.get(Constants.URL)
            if (
                self.config.get(Constants.API_KEY) is not None
                or self.config.get(Constants.API_KEY) == ""
            ):
                self.client = QdrantClient(
                    url=url, api_key=self.config.get(Constants.API_KEY)
                )
                vector_db = QdrantVectorStore(
                    collection_name=self.collection_name,
                    client=self.client,
                    api_key=self.config.get(Constants.API_KEY),
                )
            else:
                self.client = QdrantClient(url=url)
                vector_db = QdrantVectorStore(
                    collection_name=self.collection_name, client=self.client
                )
            return vector_db
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        vector_db = self.get_vector_db_instance()
        test_result: bool = VectorDBHelper.test_vector_db_instance(
            vector_store=vector_db
        )
        # Delete the collection that was created for testing
        if self.client is not None:
            self.client.delete_collection(self.collection_name)
        return test_result
