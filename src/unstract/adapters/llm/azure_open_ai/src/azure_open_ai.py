import os
from typing import Any

from llama_index.core.llms import LLM
from llama_index.llms.azure_openai import AzureOpenAI

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.llm.constants import LLMKeys
from unstract.adapters.llm.helper import LLMHelper
from unstract.adapters.llm.llm_adapter import LLMAdapter


class Constants:
    MODEL = "model"
    DEPLOYMENT_NAME = "deployment_name"
    API_KEY = "api_key"
    API_VERSION = "api_version"
    MAX_RETRIES = "max_retries"
    AZURE_ENDPONT = "azure_endpoint"
    API_TYPE = "azure"
    TIMEOUT = "timeout"
    DEFAULT_MODEL = "gpt-35-turbo"


class AzureOpenAILLM(LLMAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("AzureOpenAI")
        self.config = settings

    @staticmethod
    def get_id() -> str:
        return "azureopenai|592d84b9-fe03-4102-a17e-6b391f32850b"

    @staticmethod
    def get_name() -> str:
        return "AzureOpenAI"

    @staticmethod
    def get_description() -> str:
        return "AzureOpenAI LLM"

    @staticmethod
    def get_icon() -> str:
        return "/icons/adapter-icons/AzureopenAI.png"

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_llm_instance(self) -> LLM:
        max_retries = int(
            self.config.get(Constants.MAX_RETRIES, LLMKeys.DEFAULT_MAX_RETRIES)
        )
        try:
            llm: LLM = AzureOpenAI(
                model=self.config.get(Constants.MODEL, Constants.DEFAULT_MODEL),
                deployment_name=str(self.config.get(Constants.DEPLOYMENT_NAME)),
                api_key=str(self.config.get(Constants.API_KEY)),
                api_version=str(self.config.get(Constants.API_VERSION)),
                azure_endpoint=str(self.config.get(Constants.AZURE_ENDPONT)),
                api_type=Constants.API_TYPE,
                temperature=0,
                timeout=float(
                    self.config.get(Constants.TIMEOUT, LLMKeys.DEFAULT_TIMEOUT)
                ),
                max_retries=max_retries,
            )
            return llm
        except Exception as e:
            raise AdapterError(str(e))

    def test_connection(self) -> bool:
        llm = self.get_llm_instance()
        test_result: bool = LLMHelper.test_llm_instance(llm=llm)
        return test_result
