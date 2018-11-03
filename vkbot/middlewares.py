import logging
from typing import Callable
from aiohttp import web

logger = logging.getLogger(__name__)


@web.middleware
async def error_middleware(request: web.Request, handler: Callable) -> web.Response:
    """
    Send exceptions and errors to Sentry. Always return "OK" response
    """
    try:
        await handler(request)
    except Exception as exc:
        extra = {"request_body": await request.text()}

        if isinstance(exc, web.HTTPException):
            extra["response_text"] = exc.text

        logger.error("Error in request handler", exc_info=True, extra=extra)

    finally:
        return web.Response(text="ok")
