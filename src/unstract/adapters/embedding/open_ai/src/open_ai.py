import json
import os
from typing import Any, Optional

from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.base import BaseEmbedding
from unstract.adapters.embedding.embedding_adapter import EmbeddingAdapter
from unstract.adapters.embedding.helper import EmbeddingHelper
from unstract.adapters.exceptions import AdapterError


class Constants:
    API_KEY = "api_key"
    API_BASE_VALUE = "https://api.openai.com/v1/"
    API_BASE_KEY = "api_base"
    ADAPTER_NAME = "adapter_name"
    API_TYPE = "openai"


class OpenAI(EmbeddingAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("OpenAI")
        self.config = settings
        self.json_credentials = json.loads(
            settings.get("json_credentials", "{}")
        )

    @staticmethod
    def get_id() -> str:
        return "openai|717a0b0e-3bbc-41dc-9f0c-5689437a1151"

    @staticmethod
    def get_name() -> str:
        return "OpenAI"

    @staticmethod
    def get_description() -> str:
        return "OpenAI LLM"

    @staticmethod
    def get_icon() -> str:
        return (
            "/icons/"
            "adapter-icons/OpenAI.png"
        )

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_embedding_instance(self) -> Optional[BaseEmbedding]:
        try:
            embedding = OpenAIEmbedding(
                api_key=str(self.config.get(Constants.API_KEY)),
                api_base=str(
                    self.config.get(
                        Constants.API_BASE_KEY, Constants.API_BASE_VALUE
                    )
                ),
                api_type=Constants.API_TYPE,
            )
            return embedding
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        embedding = self.get_embedding_instance()
        test_result: bool = EmbeddingHelper.test_embedding_instance(embedding)
        return test_result
