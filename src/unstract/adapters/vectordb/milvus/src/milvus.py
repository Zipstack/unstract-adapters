import os
from typing import Any, Optional

from llama_index.vector_stores import MilvusVectorStore
from llama_index.vector_stores.types import VectorStore
from pymilvus import MilvusClient

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.vectordb.constants import VectorDbConstants
from unstract.adapters.vectordb.helper import VectorDBHelper
from unstract.adapters.vectordb.vectordb_adapter import VectorDBAdapter


class Constants:
    URI = "uri"
    TOKEN = "token"
    DIM_VALUE = 1536


class Milvus(VectorDBAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("Milvus")
        self.config = settings
        self.client: Optional[MilvusClient] = None
        self.collection_name: str = VectorDbConstants.DEFAULT_VECTOR_DB_NAME

    @staticmethod
    def get_id() -> str:
        return "milvus|3f42f6f9-4b8e-4546-95f3-22ecc9aca442"

    @staticmethod
    def get_name() -> str:
        return "Milvus"

    @staticmethod
    def get_description() -> str:
        return "Milvus VectorDB"

    @staticmethod
    def get_icon() -> str:
        return (
            "/icons/"
            "adapter-icons/Milvus.png"
        )

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_vector_db_instance(self) -> Optional[VectorStore]:
        try:
            self.collection_name = VectorDBHelper.get_collection_name(
                self.config.get(VectorDbConstants.VECTOR_DB_NAME),
                self.config.get(VectorDbConstants.EMBEDDING_DIMENSION),
            )
            dimension = self.config.get(
                VectorDbConstants.EMBEDDING_DIMENSION,
                VectorDbConstants.DEFAULT_EMBEDDING_SIZE,
            )
            vector_db = MilvusVectorStore(
                uri=self.config.get(Constants.URI, ""),
                collection_name=self.collection_name,
                token=self.config.get(Constants.TOKEN, ""),
                dim=dimension,
            )
            self.client = vector_db.client
            return vector_db
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        self.config[
            VectorDbConstants.EMBEDDING_DIMENSION
        ] = VectorDbConstants.TEST_CONNECTION_EMBEDDING_SIZE
        vector_db = self.get_vector_db_instance()
        test_result: bool = VectorDBHelper.test_vector_db_instance(
            vector_store=vector_db
        )
        # Delete the collection that was created for testing
        if self.client is not None:
            self.client.drop_collection(self.collection_name)
        return test_result
