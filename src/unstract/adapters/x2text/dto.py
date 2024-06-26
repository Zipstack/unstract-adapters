from dataclasses import dataclass
from typing import Union


@dataclass
class TextExtractionMetaData:
    whisper_hash: str


@dataclass
class TextExtractionResult:
    extracted_text: str
    extraction_metadata: Union[TextExtractionMetaData, None]
