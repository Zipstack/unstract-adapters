from unstract.adapters import AdapterDict
from unstract.adapters.llm.register import LLMRegistry

adapters: AdapterDict = {}
LLMRegistry.register_adapters(adapters)
