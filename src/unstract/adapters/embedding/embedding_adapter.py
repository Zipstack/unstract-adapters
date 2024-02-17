from abc import ABC
from typing import Any, Optional

from llama_index.core.embeddings.base import BaseEmbedding
from unstract.adapters.base import Adapter
from unstract.adapters.enums import AdapterTypes


class EmbeddingAdapter(Adapter, ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name

    @staticmethod
    def get_id() -> str:
        return ""

    @staticmethod
    def get_name() -> str:
        return ""

    @staticmethod
    def get_description() -> str:
        return ""

    @staticmethod
    def get_icon() -> str:
        return ""

    @staticmethod
    def get_json_schema() -> str:
        return ""

    @staticmethod
    def get_adapter_type() -> AdapterTypes:
        return AdapterTypes.EMBEDDING

    def get_embedding_instance(
        self, embed_config: dict[str, Any]
    ) -> Optional[BaseEmbedding]:
        return None
