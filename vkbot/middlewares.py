import logging
from typing import Callable
import aiohttp
from aiohttp import web, client

logger = logging.getLogger(__name__)


@web.middleware
async def error_middleware(request: web.Request, handler: Callable) -> web.Response:
    """
    Send exceptions and errors to Sentry. Always return "OK" response
    """
    try:
        await handler(request)
    except web.HTTPError as exc:
        extra = {"request_body": await request.text()}

        if exc.status_code >= 500:
            extra["response_text"] = exc.text
            logger.error("Error in request handler", exc_info=True, extra=extra)
    except aiohttp.ClientError:
        logger.exception("Client error")
    except Exception:
        logger.error("Error in request handler", exc_info=True)

    finally:
        return web.Response(text="ok")
