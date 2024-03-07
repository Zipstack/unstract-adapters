from enum import Enum


class ProcessingModes(Enum):
    OCR = "ocr"
    TEXT = "text"


class OutputModes(Enum):
    LINE_PRINTER = "line-printer"
    TEXT_DUMP = "text-dump"


class OCRFilters(Enum):
    MEDIAN_FILTER_SIZE = 3
    GAUSSIAN_BLUR_RADIUS = 1.0
