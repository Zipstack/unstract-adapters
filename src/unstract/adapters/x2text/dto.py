from dataclasses import dataclass
from typing import Union


@dataclass
class TextExtractionMetadata:
    whisper_hash: str


@dataclass
class TextExtractionResult:
    extracted_text: str
    extraction_metadata: Union[TextExtractionMetadata, None]
