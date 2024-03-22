import os
from typing import Any, Optional

from llama_index.llms import Anyscale
from llama_index.llms.llm import LLM
from unstract.adapters.exceptions import AdapterError
from unstract.adapters.llm.helper import LLMHelper
from unstract.adapters.llm.llm_adapter import LLMAdapter


class Constants:
    MODEL = "model"
    API_KEY = "api_key"
    API_BASE = "api_base"
    MAX_RETIRES = "max_retries"
    ADDITIONAL_KWARGS = "additional_kwargs"


class AnyScaleLLM(LLMAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("AnyScale")
        self.config = settings

    @staticmethod
    def get_id() -> str:
        return "anyscale|adec9815-eabc-4207-9389-79cb89952639"

    @staticmethod
    def get_name() -> str:
        return "AnyScale"

    @staticmethod
    def get_description() -> str:
        return "AnyScale LLM"

    @staticmethod
    def get_icon() -> str:
        return (
            "/icons/"
            "adapter-icons/anyscale.png"
        )

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_llm_instance(self) -> Optional[LLM]:
        try:
            if self.config.get(Constants.MAX_RETIRES) is not None:
                llm = Anyscale(
                    model=str(self.config.get(Constants.MODEL)),
                    api_key=str(self.config.get(Constants.API_KEY)),
                    api_base=str(self.config.get(Constants.API_BASE)),
                    additional_kwargs=self.config.get(
                        Constants.ADDITIONAL_KWARGS
                    ),
                    max_retries=int(self.config.get(Constants.MAX_RETIRES, 10)),
                    temperature=0,
                )
            else:
                llm = Anyscale(
                    model=str(self.config.get(Constants.MODEL)),
                    api_key=str(self.config.get(Constants.API_KEY)),
                    api_base=str(self.config.get(Constants.API_BASE)),
                    additional_kwargs=self.config.get(
                        Constants.ADDITIONAL_KWARGS
                    ),
                    temperature=0,
                )

            return llm
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        llm = self.get_llm_instance()
        test_result: bool = LLMHelper.test_llm_instance(llm=llm)
        return test_result
