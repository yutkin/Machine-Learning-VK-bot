import logging
import os
from aiohttp import web
from raven import Client as SentryClient
from raven_aiohttp import AioHttpTransport
from raven.handlers.logging import SentryHandler
from raven.conf import setup_logging


logger = logging.getLogger(__name__)


async def _startup(app: web.Application) -> None:
    dsn = os.environ["SENTRY_DSN"]
    environment = app["config"]["app"]["environment"]

    sentry_client = SentryClient(
        dsn=dsn, transport=AioHttpTransport, environment=environment
    )

    sentry_logging_handler = SentryHandler(sentry_client)
    sentry_logging_handler.setLevel(logging.ERROR)
    setup_logging(sentry_logging_handler)

    app["sentry"] = sentry_client


async def _cleanup(app: web.Application) -> None:
    app["sentry"].remote.get_transport().close()


def init_sentry(app: web.Application) -> None:
    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
