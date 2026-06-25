import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import logger


class RequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        request_id = str(uuid.uuid4())
        start_time = time.time()
        request.state.request_id = request_id
        logger.info(
            f"Incoming request "
            f"{request.method} {request.url.path} "
            f"| request_id={request_id}"
        )

        response = await call_next(request)
        process_time = round(time.time() - start_time, 4)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(
            f"Completed request "
            f"{request.method} {request.url.path} "
            f"| status={response.status_code} "
            f"| duration={process_time}s "
            f"| request_id={request_id}"
        )

        return response
