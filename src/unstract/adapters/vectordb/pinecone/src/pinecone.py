import logging
import os
from typing import Any, Optional

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import NotFoundException
from pinecone import Pinecone as LLamaIndexPinecone
from pinecone import PodSpec

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.vectordb.constants import VectorDbConstants
from unstract.adapters.vectordb.helper import VectorDBHelper
from unstract.adapters.vectordb.vectordb_adapter import VectorDBAdapter

logger = logging.getLogger(__name__)


class Constants:
    API_KEY = "api_key"
    ENVIRONMENT = "environment"
    NAMESPACE = "namespace"
    DIMENSION = 1536
    METRIC = "euclidean"


class Pinecone(VectorDBAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("Pinecone")
        self.config = settings
        self.client: Optional[LLamaIndexPinecone] = None
        self.collection_name: str = VectorDbConstants.DEFAULT_VECTOR_DB_NAME

    @staticmethod
    def get_id() -> str:
        return "pinecone|83881133-485d-4ecc-b1f7-0009f96dc74a"

    @staticmethod
    def get_name() -> str:
        return "Pinecone"

    @staticmethod
    def get_description() -> str:
        return "Pinecone VectorDB"

    @staticmethod
    def get_icon() -> str:
        return "/icons/adapter-icons/pinecone.png"

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_vector_db_instance(self) -> BasePydanticVectorStore:
        try:
            self.client = LLamaIndexPinecone(
                api_key=str(self.config.get(Constants.API_KEY))
            )
            collection_name = VectorDBHelper.get_collection_name(
                self.config.get(VectorDbConstants.VECTOR_DB_NAME),
                self.config.get(VectorDbConstants.EMBEDDING_DIMENSION),
            )
            self.collection_name = collection_name.replace("_", "-").lower()
            dimension = self.config.get(
                VectorDbConstants.EMBEDDING_DIMENSION,
                VectorDbConstants.DEFAULT_EMBEDDING_SIZE,
            )
            try:
                self.client.describe_index(name=self.collection_name)
            except NotFoundException:
                logger.info(
                    f"Index:{self.collection_name} does not exist. Creating it."
                )
                self.client.create_index(
                    name=self.collection_name,
                    dimension=dimension,
                    metric=Constants.METRIC,
                    spec=PodSpec(
                        environment=str(self.config.get(Constants.ENVIRONMENT))
                    ),
                )
            vector_db: BasePydanticVectorStore = PineconeVectorStore(
                index_name=self.collection_name,
                api_key=str(self.config.get(Constants.API_KEY)),
                environment=str(self.config.get(Constants.ENVIRONMENT)),
            )
            return vector_db
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        self.config[VectorDbConstants.EMBEDDING_DIMENSION] = (
            VectorDbConstants.TEST_CONNECTION_EMBEDDING_SIZE
        )
        vector_db = self.get_vector_db_instance()
        test_result: bool = VectorDBHelper.test_vector_db_instance(
            vector_store=vector_db
        )
        # Delete the collection that was created for testing
        if self.client:
            self.client.delete_index(self.collection_name)
        return test_result
