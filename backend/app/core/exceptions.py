from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.logging import logger


class OmniRAGException(Exception):

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


async def omnirag_exception_handler(
    request: Request,
    exc: OmniRAGException,
):

    logger.error(f"{exc.error_code}: {exc.message}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
            },
        },
    )
