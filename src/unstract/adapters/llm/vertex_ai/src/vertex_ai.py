import json
import os
from typing import Any

from google.auth.transport import requests as google_requests
from google.oauth2 import service_account
from llama_index.core.llms import LLM
from llama_index.llms.vertex import Vertex

from unstract.adapters.llm.constants import LLMKeys
from unstract.adapters.llm.helper import LLMHelper
from unstract.adapters.llm.llm_adapter import LLMAdapter


class Constants:
    MODEL = "model"
    PROJECT = "project"
    JSON_CREDENTIALS = "json_credentials"
    MAX_RETRIES = "max_retries"
    MAX_TOKENS = "max_tokens"
    DEFAULT_MAX_TOKENS = 2048


class VertexAILLM(LLMAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("VertexAILLM")
        self.config = settings

    @staticmethod
    def get_id() -> str:
        return "vertexai|78fa17a5-a619-47d4-ac6e-3fc1698fdb55"

    @staticmethod
    def get_name() -> str:
        return "VertexAI"

    @staticmethod
    def get_description() -> str:
        return "Vertex Gemini LLM"

    @staticmethod
    def get_icon() -> str:
        return "/icons/adapter-icons/VertexAI.png"

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def get_llm_instance(self) -> LLM:
        input_credentials = self.config.get(Constants.JSON_CREDENTIALS)
        if not input_credentials:
            input_credentials = "{}"
        json_credentials = json.loads(input_credentials)
        credentials = service_account.Credentials.from_service_account_info(
            info=json_credentials,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        credentials.refresh(google_requests.Request())
        max_retries = int(
            self.config.get(Constants.MAX_RETRIES, LLMKeys.DEFAULT_MAX_RETRIES)
        )
        max_tokens = int(
            self.config.get(Constants.MAX_TOKENS, Constants.DEFAULT_MAX_TOKENS)
        )
        llm: LLM = Vertex(
            project=str(self.config.get(Constants.PROJECT)),
            model=str(self.config.get(Constants.MODEL)),
            credentials=credentials,
            temperature=0,
            max_retries=max_retries,
            max_tokens=max_tokens,
            additional_kwargs={},
        )
        return llm

    def test_connection(self) -> bool:
        llm = self.get_llm_instance()
        test_result: bool = LLMHelper.test_llm_instance(llm=llm)
        return test_result
