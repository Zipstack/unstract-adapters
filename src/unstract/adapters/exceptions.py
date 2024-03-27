from unstract.adapters.constants import Common


class AdapterError(Exception):
    def __init__(self, message: str = Common.DEFAULT_ERR_MESSAGE):
        super().__init__(message)
        # Make it user friendly wherever possible
        self.message = message

    def __str__(self) -> str:
        return self.message
