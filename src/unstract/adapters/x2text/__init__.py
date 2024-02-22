from unstract.adapters import AdapterDict
from unstract.adapters.x2text.register import X2TextRegistry

adapters: AdapterDict = {}
X2TextRegistry.register_adapters(adapters)
