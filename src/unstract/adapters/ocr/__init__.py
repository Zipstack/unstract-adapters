from unstract.adapters import AdapterDict
from unstract.adapters.ocr.register import OCRRegistry

adapters: AdapterDict = {}
OCRRegistry.register_adapters(adapters)
