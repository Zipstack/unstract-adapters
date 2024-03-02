from enum import Enum


class ExtractionModes(Enum):
    OCR = "ocr"
    TEXT = "text"


class OCRModes(Enum):
    LINE_PRINTER = "line-printer"
    TEXT_DUMP = "text-dump"
