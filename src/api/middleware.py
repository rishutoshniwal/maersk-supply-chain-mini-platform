"""
Structured JSON logging middleware and request correlation.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request as a structured JSON object."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        response: Response
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            log_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
                "status_code": 500,
                "latency_ms": latency_ms,
                "client_ip": request.client.host if request.client else "unknown",
            }
            logger.error(json.dumps(log_payload))
            raise

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "client_ip": request.client.host if request.client else "unknown",
        }
        logger.info(json.dumps(log_payload))

        return response
