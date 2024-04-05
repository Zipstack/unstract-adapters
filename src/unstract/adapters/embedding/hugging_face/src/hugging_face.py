import json
import os
from typing import Any, Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from unstract.adapters.embedding.embedding_adapter import EmbeddingAdapter
from unstract.adapters.embedding.helper import EmbeddingHelper
from unstract.adapters.exceptions import AdapterError


class Constants:
    ADAPTER_NAME = "adapter_name"
    MODEL = "model_name"
    TOKENIZER_NAME = "tokenizer_name"
    MAX_LENGTH = "max_length"
    NORMALIZE = "normalize"


class HuggingFace(EmbeddingAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("HuggingFace")
        self.config = settings
        self.json_credentials = json.loads(
            settings.get("json_credentials", "{}")
        )

    @staticmethod
    def get_id() -> str:
        return "huggingface|90ec9ec2-1768-4d69-8fb1-c88b95de5e5a"

    @staticmethod
    def get_name() -> str:
        return "HuggingFace"

    @staticmethod
    def get_description() -> str:
        return "HuggingFace LLM"

    @staticmethod
    def get_icon() -> str:
        return "/icons/adapter-icons/huggingface.png"

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
            max_length: Optional[int] = (
                int(self.config.get(Constants.MAX_LENGTH, 0))
                if self.config.get(Constants.MAX_LENGTH)
                else None
            )
            embedding: BaseEmbedding = HuggingFaceEmbedding(
                model_name=str(self.config.get(Constants.MODEL)),
                tokenizer_name=str(self.config.get(Constants.TOKENIZER_NAME)),
                normalize=bool(self.config.get(Constants.NORMALIZE)),
                embed_batch_size=embedding_batch_size,
                max_length=max_length,
            )

            return embedding
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        embedding = self.get_embedding_instance()
        test_result: bool = EmbeddingHelper.test_embedding_instance(embedding)
        return test_result
