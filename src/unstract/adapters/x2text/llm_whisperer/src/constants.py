from enum import Enum


class ProcessingModes(Enum):
    OCR = "ocr"
    TEXT = "text"


class OutputModes(Enum):
    LINE_PRINTER = "line-printer"
    DUMP_TEXT = "dump-text"
    TEXT = "text"


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"


class WhispererHeader:
    UNSTRACT_KEY = "unstract-key"


class WhispererEndpoint:
    """Endpoints available at LLMWhisperer service."""

    TEST_CONNECTION = "test-connection"
    WHISPER = "whisper"


class WhispererConfig:
    """Dictionary keys used to configure LLMWhisperer service."""

    URL = "url"
    PROCESSING_MODE = "processing_mode"
    OUTPUT_MODE = "output_mode"
    UNSTRACT_KEY = "unstract_key"
    MEDIAN_FILTER_SIZE = "median_filter_size"
    GAUSSIAN_BLUR_RADIUS = "gaussian_blur_radius"
    FORCE_TEXT_PROCESSING = "force_text_processing"


class WhispererDefaults:
    """Defaults meant for OCR mode."""

    MEDIAN_FILTER_SIZE = 0
    GAUSSIAN_BLUR_RADIUS = 0.0
    FORCE_TEXT_PROCESSING = False
