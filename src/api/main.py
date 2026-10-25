"""
FastAPI application entry point.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.middleware import StructuredLoggingMiddleware
from src.api.routes import router
from src.ml.predict import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up dependencies and load model artifact at startup."""
    logging.info("[*] Supply Chain Intelligence API starting up...")
    try:
        predictor = get_predictor()
        logging.info(f"[+] Loaded ML predictor model: {predictor.metadata.get('selected_model')}")
    except Exception as e:
        logging.warning(f"[-] Could not pre-load model artifact: {e}")
    yield
    logging.info("[*] Supply Chain Intelligence API shutting down...")


app = FastAPI(
    title="Supply Chain Intelligence Mini-Platform API",
    description="Production-ready REST API for container shipments, route analytics, and ML delay risk prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

# Attach structured JSON logging middleware
app.add_middleware(StructuredLoggingMiddleware)

# Include API endpoints
app.include_router(router)


# -------------------------------------------------------------
# Structured JSON Error Handlers
# -------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "RESOURCE_NOT_FOUND",
        422: "UNPROCESSABLE_ENTITY",
        500: "INTERNAL_SERVER_ERROR",
    }
    error_code = code_map.get(exc.status_code, "HTTP_ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": error_code,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": request.url.path,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request payload validation failed",
            "errors": exc.errors(),
            "error_code": "VALIDATION_ERROR",
            "status_code": 422,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": request.url.path,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "error_code": "INTERNAL_SERVER_ERROR",
            "status_code": 500,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
