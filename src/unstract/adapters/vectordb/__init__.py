from unstract.adapters import AdapterDict
from unstract.adapters.vectordb.register import VectorDBRegistry


adapters: AdapterDict = {}
VectorDBRegistry.register_adapters(adapters)
