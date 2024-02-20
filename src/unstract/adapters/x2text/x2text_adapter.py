from abc import ABC

from unstract.adapters.base import Adapter
from unstract.adapters.enums import AdapterTypes


class X2TextAdapter(Adapter, ABC):
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
        return AdapterTypes.X2TEXT

    def test_connection(self) -> bool:
        return False

    def process(self, input_file_path: str, output_file_path: str) -> None:
        # Overriding methods will have the actual implementation
        pass
