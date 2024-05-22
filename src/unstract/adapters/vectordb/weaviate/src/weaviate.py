import logging
import os
from typing import Any, Optional

import weaviate
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate import UnexpectedStatusCodeException

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.vectordb.constants import VectorDbConstants
from unstract.adapters.vectordb.helper import VectorDBHelper
from unstract.adapters.vectordb.vectordb_adapter import VectorDBAdapter

logger = logging.getLogger(__name__)


class Constants:
    URL = "url"
    API_KEY = "api_key"


class Weaviate(VectorDBAdapter):
    def __init__(self, settings: dict[str, Any]):
        self._config = settings
        self._client: Optional[weaviate.Client] = None
        self._collection_name: str = VectorDbConstants.DEFAULT_VECTOR_DB_NAME
        self._vector_db_instance = self._get_vector_db_instance()
        super().__init__("Weaviate", self._vector_db_instance)

    @staticmethod
    def get_id() -> str:
        return "weaviate|294e08df-4e4a-40f2-8f0d-9e4940180ccc"

    @staticmethod
    def get_name() -> str:
        return "Weaviate"

    @staticmethod
    def get_description() -> str:
        return "Weaviate VectorDB"

    @staticmethod
    def get_icon() -> str:
        return "/icons/adapter-icons/Weaviate.png"

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_vector_db_instance(self) -> BasePydanticVectorStore:
        return self._vector_db_instance

    def _get_vector_db_instance(self) -> BasePydanticVectorStore:
        try:
            collection_name = VectorDBHelper.get_collection_name(
                self._config.get(VectorDbConstants.VECTOR_DB_NAME),
                self._config.get(VectorDbConstants.EMBEDDING_DIMENSION),
            )
            # Capitalise the frst letter as Weaviate expects this
            # LLama-index throws the error if not capitalised while using
            # Weaviate
            self._collection_name = collection_name.capitalize()
            self._client = weaviate.Client(
                url=str(self._config.get(Constants.URL)),
                auth_client_secret=weaviate.AuthApiKey(
                    api_key=str(self._config.get(Constants.API_KEY))
                ),
            )

            try:
                # Class definition object. Weaviate's autoschema
                # feature will infer properties when importing.
                class_obj = {
                    "class": self._collection_name,
                    "vectorizer": "none",
                }
                # Add the class to the schema
                self._client.schema.create_class(class_obj)
            except Exception as e:
                if isinstance(e, UnexpectedStatusCodeException):
                    if "already exists" in e.message:
                        logger.warning(f"Collection already exists: {e}")
                else:
                    raise e
            vector_db: BasePydanticVectorStore = WeaviateVectorStore(
                weaviate_client=self._client,
                index_name=self._collection_name,
            )
            return vector_db
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        self._config[VectorDbConstants.EMBEDDING_DIMENSION] = (
            VectorDbConstants.TEST_CONNECTION_EMBEDDING_SIZE
        )
        vector_db = self.get_vector_db_instance()
        test_result: bool = VectorDBHelper.test_vector_db_instance(
            vector_store=vector_db
        )
        # Delete the collection that was created for testing
        if self._client is not None:
            self._client.schema.delete_class(self._collection_name)
        return test_result
