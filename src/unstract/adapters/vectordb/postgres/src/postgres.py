import os
from typing import Any, Optional

import psycopg2
from llama_index.vector_stores import PGVectorStore
from llama_index.vector_stores.types import BasePydanticVectorStore
from psycopg2._psycopg import connection
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
    SCHEMA = "schema"


class Postgres(VectorDBAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("Postgres")
        self.config = settings
        self.client: Optional[connection] = None
        self.collection_name: str = VectorDbConstants.DEFAULT_VECTOR_DB_NAME
        self.schema_name: str = VectorDbConstants.DEFAULT_VECTOR_DB_NAME

    @staticmethod
    def get_id() -> str:
        return "postgres|70ab6cc2-e86a-4e5a-896f-498a95022d34"

    @staticmethod
    def get_name() -> str:
        return "Postgres"

    @staticmethod
    def get_description() -> str:
        return "Postgres VectorDB"

    @staticmethod
    def get_icon() -> str:
        return (
            "/icons/"
            "adapter-icons/postgres.png"
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
            self.schema_name = self.config.get(
                Constants.SCHEMA,
                VectorDbConstants.DEFAULT_VECTOR_DB_NAME,
            )
            dimension = self.config.get(
                VectorDbConstants.EMBEDDING_DIMENSION,
                VectorDbConstants.DEFAULT_EMBEDDING_SIZE,
            )
            vector_db = PGVectorStore.from_params(
                database=self.config.get(Constants.DATABASE),
                schema_name=self.schema_name,
                host=self.config.get(Constants.HOST),
                password=self.config.get(Constants.PASSWORD),
                port=str(self.config.get(Constants.PORT)),
                user=self.config.get(Constants.USER),
                table_name=self.collection_name,
                embed_dim=dimension,
            )
            self.client = psycopg2.connect(
                database=self.config.get(Constants.DATABASE),
                host=self.config.get(Constants.HOST),
                user=self.config.get(Constants.USER),
                password=self.config.get(Constants.PASSWORD),
                port=str(self.config.get(Constants.PORT)),
            )

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
            self.client.cursor().execute(
                f"DROP TABLE IF EXISTS "
                f"{self.schema_name}.data_{self.collection_name} CASCADE"
            )
            self.client.cursor().execute(
                f"DROP SCHEMA IF EXISTS {self.schema_name} CASCADE"
            )
            self.client.commit()

        return test_result
