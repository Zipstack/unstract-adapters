import json
import os
from typing import Any, Optional

from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from unstract.adapters.embedding.embedding_adapter import EmbeddingAdapter
from unstract.adapters.embedding.helper import EmbeddingHelper
from unstract.adapters.exceptions import AdapterError


class Constants:
    MODEL = "model_name"
    ADAPTER_NAME = "adapter_name"


class QdrantFastEmbedM(EmbeddingAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("QdrantFastEmbedM")
        self.config = settings
        self.json_credentials = json.loads(
            settings.get("json_credentials", "{}")
        )

    @staticmethod
    def get_id() -> str:
        return "qdrantfastembed|31e83eee-a416-4c07-9c9c-02392d5bcf7f"

    @staticmethod
    def get_name() -> str:
        return "QdrantFastEmbedM"

    @staticmethod
    def get_description() -> str:
        return "QdrantFastEmbedM LLM"

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

    def get_embedding_instance(self) -> Optional[BaseEmbedding]:
        try:
            embedding = FastEmbedEmbedding(
                model_name=str(self.config.get(Constants.MODEL))
            )
            return embedding
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        embedding = self.get_embedding_instance()
        test_result: bool = EmbeddingHelper.test_embedding_instance(embedding)
        return test_result
