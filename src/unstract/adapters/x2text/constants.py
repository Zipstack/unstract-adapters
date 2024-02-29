from enum import Enum


class X2TextConstants:
    PLATFORM_SERVICE_API_KEY = "PLATFORM_SERVICE_API_KEY"
    X2TEXT_HOST = "X2TEXT_HOST"
    X2TEXT_PORT = "X2TEXT_PORT"


class LLMWhispererSupportedModes(Enum):
    OCR = "ocr"
    TEXT = "text"
