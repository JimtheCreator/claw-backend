from fastapi import HTTPException
from typing import Optional

# Custom Exceptions
class DataUnavailableError(Exception):
    def __init__(self, message: str = "Market data unavailable", detail: Optional[str] = None):
        self.message = message
        self.detail = detail
        super().__init__(message)