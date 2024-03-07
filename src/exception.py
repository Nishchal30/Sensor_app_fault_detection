from typing import Any
import sys

class CustomException(Exception):

    def __init__(self, error_msg : str, error_details : str) -> Any:
        self.error_msg = error_msg

        _,_,exc_info = error_details.exc_info()

        self.lineno = exc_info.tb_lineno
        self.filename = exc_info.tb_frame.f_code.co_filename

    def __str__(self) -> str:
        return (f"Error occured in file {self.filename} at lineno {self.lineno} with error {self.error_msg}")
        