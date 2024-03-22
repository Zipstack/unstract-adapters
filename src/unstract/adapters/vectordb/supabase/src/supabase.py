import os
from typing import Any, Optional

from llama_index.vector_stores import SupabaseVectorStore
from llama_index.vector_stores.types import VectorStore
from vecs import Client

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.vectordb.constants import VectorDbConstants
from unstract.adapters.vectordb.helper import VectorDBHelper
from unstract.adapters.vectordb.vectordb_adapter import VectorDBAdapter


class Constants:
    DATABASE = "database"
    HOST = "host"
    PASSWORD = "password"
    PORT = "port"
    USER = "user"
    COLLECTION_NAME = "base_demo"


class Supabase(VectorDBAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("Supabase")
        self.config = settings
        self.client: Optional[Client] = None
        self.collection_name: str = VectorDbConstants.DEFAULT_VECTOR_DB_NAME

    @staticmethod
    def get_id() -> str:
        return "supabase|e6998e3c-3595-48c0-a190-188dbd803858"

    @staticmethod
    def get_name() -> str:
        return "Supabase"

    @staticmethod
    def get_description() -> str:
        return "Supabase VectorDB"

    @staticmethod
    def get_icon() -> str:
        return (
            "/icons/"
            "adapter-icons/supabase.png"
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
                self.config.get(
                    VectorDbConstants.EMBEDDING_DIMENSION,
                    VectorDbConstants.DEFAULT_EMBEDDING_SIZE,
                ),
            )
            user = str(self.config.get(Constants.USER))
            password = str(self.config.get(Constants.PASSWORD))
            host = str(self.config.get(Constants.HOST))
            port = str(self.config.get(Constants.PORT))
            db_name = str(self.config.get(Constants.DATABASE))

            postgres_connection_string = (
                f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            )
            dimension = self.config.get(
                VectorDbConstants.EMBEDDING_DIMENSION,
                VectorDbConstants.DEFAULT_EMBEDDING_SIZE,
            )
            vector_db = SupabaseVectorStore(
                postgres_connection_string=postgres_connection_string,
                collection_name=self.collection_name,
                dimension=dimension,
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
            self.client.delete_collection(self.collection_name)
        return test_result
