import logging
from typing import Any, Optional

import requests
from requests import Response

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.x2text.constants import X2TextConstants

logger = logging.getLogger(__name__)


class UnstructuredHelper:
    URL = "url"
    API_KEY = "api_key"
    TEST_CONNECTION = "test-connection"
    PROCESS = "process"

    @staticmethod
    def test_server_connection(
        unstructured_adapter_config: dict[str, Any]
    ) -> bool:
        try:
            response = UnstructuredHelper.make_request(
                unstructured_adapter_config, UnstructuredHelper.TEST_CONNECTION
            )
            if response.status_code != 200:
                logger.error(
                    "Error in unstructured IO test-connection: "
                    f"[{response.status_code}] {response.reason}"
                )

                raise AdapterError(
                    f"{response.status_code} - {response.reason}"
                )
            else:
                return True
        except Exception as e:
            if not isinstance(e, AdapterError):
                raise AdapterError(str(e))
            else:
                raise e

    @staticmethod
    def process_document(
        unstructured_adapter_config: dict[str, Any],
        input_file_path: str,
        output_file_path: Optional[str] = None,
    ) -> str:
        try:
            input_f = open(input_file_path, "rb")
            files = {"file": input_f}
            response = UnstructuredHelper.make_request(
                unstructured_adapter_config,
                UnstructuredHelper.PROCESS,
                files=files,
            )
            if response.status_code != 200:
                logger.error(
                    "Error in unstructured IO process document: "
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
                else:
                    raise AdapterError("No extracted content")
        except Exception as e:
            if not isinstance(e, AdapterError):
                raise AdapterError(str(e))
            else:
                raise e
        finally:
            if input_f is not None:
                input_f.close()

    @staticmethod
    def make_request(
        unstructured_adapter_config: dict[str, Any],
        request_type: str,
        **kwargs: dict[Any, Any],
    ) -> Response:
        unstructured_url = unstructured_adapter_config.get(
            UnstructuredHelper.URL
        )

        x2text_service_url = unstructured_adapter_config.get(
            X2TextConstants.X2TEXT_HOST
        )
        x2text_service_port = unstructured_adapter_config.get(
            X2TextConstants.X2TEXT_PORT
        )
        platform_service_api_key = unstructured_adapter_config.get(
            X2TextConstants.PLATFORM_SERVICE_API_KEY
        )
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {platform_service_api_key}",
        }
        body = {
            "unstructured-url": unstructured_url,
        }
        # Add api key only if present
        api_key = unstructured_adapter_config.get(UnstructuredHelper.API_KEY)
        if api_key is not None and api_key != "":
            body["unstructured-api-key"] = api_key

        x2text_url = (
            f"{x2text_service_url}:{x2text_service_port}"
            f"/api/v1/x2text/{request_type}"
        )
        # Add files only if the request is for process
        files = None
        if "files" in kwargs:
            files = kwargs["files"] if kwargs["files"] is not None else None
        response = requests.post(
            x2text_url, headers=headers, data=body, files=files
        )
        return response
