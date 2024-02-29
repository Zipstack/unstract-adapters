import base64
import json
import logging
import os
from typing import Any, Optional

import requests
from filetype import filetype
from google.auth.transport import requests as google_requests
from google.oauth2.service_account import Credentials

from unstract.adapters.exceptions import AdapterError
from unstract.adapters.ocr.constants import FileType
from unstract.adapters.ocr.ocr_adapter import OCRAdapter

logger = logging.getLogger(__name__)


class GoogleDocumentAIKey:
    RAW_DOCUMENT = "rawDocument"
    MIME_TYPE = "mimeType"
    CONTENT = "content"
    SKIP_HUMAN_REVIEW = "skipHumanReview"
    FIELD_MASK = "fieldMask"


class Constants:
    URL = "url"
    CREDENTIALS = "credentials"
    CREDENTIAL_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class GoogleDocumentAI(OCRAdapter):
    def __init__(self, settings: dict[str, Any]):
        super().__init__("GoogleDocumentAI")
        self.config = settings
        google_service_account = self.config.get(Constants.CREDENTIALS)
        if not google_service_account:
            logger.error("Google service account not found")
        else:
            self.google_service_account = json.loads(google_service_account)

    @staticmethod
    def get_id() -> str:
        return "googledocumentai|1013f64b-ecc9-4e35-b986-aebd60fb55d7"

    @staticmethod
    def get_name() -> str:
        return "GoogleDocumentAI"

    @staticmethod
    def get_description() -> str:
        return "Google Document AI OCR"

    @staticmethod
    def get_icon() -> str:
        return (
            "https://storage.googleapis.com/pandora-static/"
            "adapter-icons/GoogleDocumentAI.png"
        )

    @staticmethod
    def get_json_schema() -> str:
        f = open(f"{os.path.dirname(__file__)}/static/json_schema.json")
        schema = f.read()
        f.close()
        return schema

    """ Construct the request body to be sent to Google AI Document server """

    def _get_request_body(
        self, file_type_mime: str, file_content_in_bytes: bytes
    ) -> dict[str, Any]:
        return {
            GoogleDocumentAIKey.RAW_DOCUMENT: {
                GoogleDocumentAIKey.MIME_TYPE: file_type_mime,
                GoogleDocumentAIKey.CONTENT: base64.b64encode(
                    file_content_in_bytes
                ).decode("utf-8"),
            },
            GoogleDocumentAIKey.SKIP_HUMAN_REVIEW: True,
            GoogleDocumentAIKey.FIELD_MASK: "text",
        }

    """ Construct the request headers to be sent
    to Google AI Document server """

    def _get_request_headers(self) -> dict[str, Any]:
        credentials = Credentials.from_service_account_info(
            self.google_service_account, scopes=Constants.CREDENTIAL_SCOPES
        )
        credentials.refresh(google_requests.Request())

        return {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {credentials.token}",
        }

    """ Detect the mime type from the file content """

    def _get_input_file_type_mime(self, input_file_path: str) -> str:
        with open(input_file_path, mode="rb") as file_obj:
            sample_contents = file_obj.read(100)
            file_type = filetype.guess(sample_contents)

        file_type_mime: str = (
            file_type.MIME if file_type else FileType.TEXT_PLAIN
        )

        if file_type_mime not in FileType.ALLOWED_TYPES:
            logger.error("Input file type not supported: " f"{file_type_mime}")

        logger.info(f"file: `{input_file_path} [{file_type_mime}]`\n\n")

        return file_type_mime

    def process(
        self, input_file_path: str, output_file_path: Optional[str] = None
    ) -> str:
        try:
            file_type_mime = self._get_input_file_type_mime(input_file_path)
            if os.path.isfile(input_file_path):
                with open(input_file_path, "rb") as fop:
                    file_content_in_bytes: bytes = fop.read()
            else:
                raise AdapterError(f"File not found {input_file_path}")
            processor_url = self.config.get(Constants.URL, "") + ":process"
            headers = self._get_request_headers()
            data = self._get_request_body(
                file_type_mime=file_type_mime,
                file_content_in_bytes=file_content_in_bytes,
            )
            response = requests.post(processor_url, headers=headers, json=data)
            if response.status_code != 200:
                logger.error(
                    f"Error while calling Google Document AI: {response.text}"
                )
            response_json: dict[str, Any] = response.json()
            result_text: str = response_json["document"]["text"]
            if output_file_path is not None:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(result_text)
                    f.close()
            return result_text
        except Exception as e:
            logger.error(f"Error while processing document {e}")
            if not isinstance(e, AdapterError):
                raise AdapterError(str(e))
            else:
                raise e
        finally:
            if fop is not None:
                fop.close()

    def test_connection(self) -> bool:
        try:
            url = self.config.get(Constants.URL, "")
            headers = self._get_request_headers()
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(
                    f"Error while testing Google Document AI: {response.text}"
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
