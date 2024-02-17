from unstract.adapters import AdapterDict
from unstract.adapters.embedding.register import EmbeddingRegistry

adapters: AdapterDict = {}
EmbeddingRegistry.register_adapters(adapters)
