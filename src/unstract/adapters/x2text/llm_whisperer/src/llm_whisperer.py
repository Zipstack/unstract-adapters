import logging
import os
from typing import Any, Optional

import requests
from requests import Response

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.x2text.constants import X2TextConstants
from unstract.adapters.x2text.llm_whisperer.src.constants import (
    ExtractionModes,
    OCRModes,
)
from unstract.adapters.x2text.x2text_adapter import X2TextAdapter

logger = logging.getLogger(__name__)


class Constants:
    URL = "url"
    TEST_CONNECTION = "test-connection"
    PROCESS = "process"
    EXTRACTION_MODE = "extraction_mode"
    OCR_MODE = "ocr_mode"


class LLMWhisperer(X2TextAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("LLMWhisperer")
        self.config = settings

    @staticmethod
    def get_id() -> str:
        return "llmwhisperer|0a1647f0-f65f-410d-843b-3d979c78350e"

    @staticmethod
    def get_name() -> str:
        return "LLMWhisperer"

    @staticmethod
    def get_description() -> str:
        return "LLMWhisperer X2Text"

    @staticmethod
    def get_icon() -> str:
        return (
            "https://storage.googleapis.com/pandora-static/"
            "adapter-icons/LLMWhisperer.png"
        )

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def _make_request(
        self, request_type: str, **kwargs: Optional[dict[Any, Any]]
    ) -> Response:
        llm_whisperer_svc_url = (
            f"{self.config.get(Constants.URL)}"
            f"/api/v1/llm-whisperer/{request_type}"
        )
        platform_service_api_key = self.config.get(
            X2TextConstants.PLATFORM_SERVICE_API_KEY
        )

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {platform_service_api_key}",
        }

        # Add files only if the request is for process
        files = None
        if "files" in kwargs:
            files = kwargs["files"] if kwargs["files"] is not None else None

        body = {
            Constants.EXTRACTION_MODE: self.config.get(
                Constants.EXTRACTION_MODE, ExtractionModes.TEXT.value
            ),
            Constants.OCR_MODE: self.config.get(
                Constants.OCR_MODE, OCRModes.LINE_PRINTER.value
            ),
        }

        response = requests.post(
            llm_whisperer_svc_url, headers=headers, files=files, data=body
        )
        return response

    def process(
        self,
        input_file_path: str,
        output_file_path: Optional[str] = None,
        **kwargs: dict[Any, Any],
    ) -> str:
        try:
            input_f = open(input_file_path, "rb")
            files = {"file": input_f}
            response = self._make_request(
                request_type=Constants.PROCESS, files=files, **kwargs
            )
            if response.status_code != 200:
                logger.error(
                    "Error in LLM Whisperer process document: "
                    f"[{response.status_code}] {response.reason}"
                )
                raise AdapterError(
                    f"{response.status_code} - {response.reason}"
                )
            else:
                if response.content is not None:
                    if isinstance(response.content, bytes):
                        output = response.content.decode("utf-8")
                    if output_file_path is not None:
                        with open(output_file_path, "w", encoding="utf-8") as f:
                            f.write(output)
                            f.close()
                    return output
        except Exception as e:
            logger.error(f"Error occured while processing document {e}")
            if not isinstance(e, AdapterError):
                raise AdapterError(str(e))
            else:
                raise e
        finally:
            if input_f is not None:
                input_f.close()

    def test_connection(self) -> bool:
        try:
            response = self._make_request(Constants.TEST_CONNECTION)
            if response.status_code != 200:
                logger.error(
                    "Error in LLM Whisperer test-connection: "
                    f"[{response.status_code}] {response.reason}"
                )

                raise AdapterError(
                    f"{response.status_code} - {response.reason}"
                )
            else:
                return True
        except Exception as e:
            logger.error(f"Error occured while testing adapter {e}")
            if not isinstance(e, AdapterError):
                raise AdapterError(str(e))
            else:
                raise e
