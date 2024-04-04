import logging
import os
from typing import Any, Optional

import requests
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, Timeout

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.utils import AdapterUtils
from unstract.adapters.x2text.constants import X2TextConstants
from unstract.adapters.x2text.helper import X2TextHelper
from unstract.adapters.x2text.llm_whisperer.src.constants import (
    HTTPMethod,
    OutputModes,
    ProcessingModes,
    WhispererConfig,
    WhispererDefaults,
    WhispererEndpoint,
    WhispererHeader,
)
from unstract.adapters.x2text.x2text_adapter import X2TextAdapter

logger = logging.getLogger(__name__)


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
        return "/icons/adapter-icons/LLMWhisperer.png"

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    def _make_request(
        self,
        request_method: HTTPMethod,
        request_endpoint: str,
        params: Optional[dict[str, Any]] = None,
        files: Optional[dict[Any, Any]] = None,
    ) -> Response:
        llm_whisperer_svc_url = (
            f"{self.config.get(WhispererConfig.URL)}" f"/v1/{request_endpoint}"
        )
        # Required when whisperer service is run locally
        platform_service_api_key = self.config.get(
            X2TextConstants.PLATFORM_SERVICE_API_KEY
        )

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {platform_service_api_key}",
            WhispererHeader.UNSTRACT_KEY: self.config.get(
                WhispererConfig.UNSTRACT_KEY
            ),
        }

        data = None
        if files is not None:
            f = files["file"]
            data = f.read()
            headers["Content-Type"] = "application/octet-stream"

        try:
            response: Response
            if request_method == HTTPMethod.GET:
                response = requests.get(
                    url=llm_whisperer_svc_url, headers=headers
                )
            elif request_method == HTTPMethod.POST:
                response = requests.post(
                    url=llm_whisperer_svc_url,
                    headers=headers,
                    params=params,
                    data=data,
                )
            else:
                raise AdapterError(
                    f"Unsupported request method: {request_method}"
                )
            response.raise_for_status()
        except ConnectionError as e:
            logger.error(f"Adapter error: {e}")
            raise AdapterError(
                "Unable to connect to LLM Whisperer service, "
                "please check the URL"
            )
        except Timeout as e:
            msg = "Request to LLM whisperer has timed out"
            logger.error(f"{msg}: {e}")
            raise AdapterError(msg)
        except HTTPError as e:
            logger.error(f"Adapter error: {e}")
            default_err = "Error while calling the LLM Whisperer service"
            msg = AdapterUtils.get_msg_from_request_exc(
                err=e, message_key="message", default_err=default_err
            )
            raise AdapterError(msg)
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

            params = {
                WhispererConfig.PROCESSING_MODE: self.config.get(
                    WhispererConfig.PROCESSING_MODE, ProcessingModes.TEXT.value
                ),
                WhispererConfig.OUTPUT_MODE: self.config.get(
                    WhispererConfig.OUTPUT_MODE, OutputModes.LINE_PRINTER.value
                ),
                WhispererConfig.FORCE_TEXT_PROCESSING: self.config.get(
                    WhispererConfig.FORCE_TEXT_PROCESSING,
                    WhispererDefaults.FORCE_TEXT_PROCESSING,
                ),
            }
            if not params[WhispererConfig.FORCE_TEXT_PROCESSING]:
                params.update(
                    {
                        WhispererConfig.MEDIAN_FILTER_SIZE: self.config.get(
                            WhispererConfig.MEDIAN_FILTER_SIZE,
                            WhispererDefaults.MEDIAN_FILTER_SIZE,
                        ),
                        WhispererConfig.GAUSSIAN_BLUR_RADIUS: self.config.get(
                            WhispererConfig.GAUSSIAN_BLUR_RADIUS,
                            WhispererDefaults.GAUSSIAN_BLUR_RADIUS,
                        ),
                    }
                )

            response = self._make_request(
                request_method=HTTPMethod.POST,
                request_endpoint=WhispererEndpoint.WHISPER,
                params=params,
                files=files,
            )
            output, is_success = X2TextHelper.parse_response(
                response=response, out_file_path=output_file_path
            )
            if not is_success:
                raise AdapterError("Couldn't extract text from file")
            return output  # type: ignore
        except OSError as e:
            msg = f"OS error while reading {input_file_path} "
            if output_file_path:
                msg += f"and writing {output_file_path}"
            msg += f": {str(e)}"
            logger.error(msg)
            raise AdapterError(str(e))
        # TODO: Review this practice and remove if unnecessary
        except Exception as e:
            logger.error(f"Error occured while processing document: {e}")
            raise AdapterError(str(e))
        finally:
            if input_f is not None:
                input_f.close()

    def test_connection(self) -> bool:
        self._make_request(
            request_method=HTTPMethod.GET,
            request_endpoint=WhispererEndpoint.TEST_CONNECTION,
        )
        return True
