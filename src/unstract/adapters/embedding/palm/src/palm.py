import json
import os
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.google import GooglePaLMEmbedding

from unstract.adapters.embedding.embedding_adapter import EmbeddingAdapter
from unstract.adapters.embedding.helper import EmbeddingHelper
from unstract.adapters.exceptions import AdapterError


class Constants:
    MODEL = "model_name"
    API_KEY = "api_key"
    ADAPTER_NAME = "adapter_name"


class PaLM(EmbeddingAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("Palm")
        self.config = settings
        self.json_credentials = json.loads(
            settings.get("json_credentials", "{}")
        )

    @staticmethod
    def get_id() -> str:
        return "palm|a3fc9fda-f02f-405f-bb26-8bd2ace4317e"

    @staticmethod
    def get_name() -> str:
        return "Palm"

    @staticmethod
    def get_description() -> str:
        return "PaLM LLM"

    @staticmethod
    def get_icon() -> str:
        return "/icons/adapter-icons/PaLM.png"

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_embedding_instance(self) -> BaseEmbedding:
        try:
            embedding_batch_size = EmbeddingHelper.get_embedding_batch_size(
                config=self.config
            )
            embedding: BaseEmbedding = GooglePaLMEmbedding(
                model_name=str(self.config.get(Constants.MODEL)),
                api_key=str(self.config.get(Constants.API_KEY)),
                embed_batch_size=embedding_batch_size,
            )
            return embedding
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        embedding = self.get_embedding_instance()
        test_result: bool = EmbeddingHelper.test_embedding_instance(embedding)
        return test_result
