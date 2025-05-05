# Exception Handler
from fastapi import Request
from fastapi.responses import JSONResponse
from common.custom_exceptions.data_unavailable_error import DataUnavailableError

async def data_unavailable_handler(request: Request, exc: DataUnavailableError):
    return JSONResponse(
        status_code=503,
        content={
            "error": exc.message,
            "detail": exc.detail or "Failed to retrieve OHLCV data from database"
        }
    )