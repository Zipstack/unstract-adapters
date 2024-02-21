from abc import ABC
from typing import Any

from unstract.adapters.base import Adapter
from unstract.adapters.enums import AdapterTypes


class OCRAdapter(Adapter, ABC):
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
        return AdapterTypes.OCR

    def process(self, input_file_path: str, output_file_path: str) -> str:
        return ""

    def test_connection(self, llm_metadata: dict[str, Any]) -> bool:
        return False
