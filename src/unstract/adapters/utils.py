from requests import Response
from requests.exceptions import RequestException

from unstract.adapters.constants import Common


class AdapterUtils:
    @staticmethod
    def get_msg_from_request_exc(
        err: RequestException,
        message_key: str,
        default_err: str = Common.DEFAULT_ERR_MESSAGE,
    ) -> str:
        """Gets the message from the RequestException.

        Args:
            err_response (Response): Error response from the exception
            message_key (str): Key from response containing error message

        Returns:
            str: Error message returned by the server
        """
        if hasattr(err, "response"):
            err_response: Response = err.response  # type: ignore
            if err_response.headers["Content-Type"] == "application/json":
                err_json = err_response.json()
                if message_key in err_json:
                    return str(err_json[message_key])
            elif err_response.headers["Content-Type"] == "text/plain":
                return err.response.text()  # type: ignore
        return default_err
