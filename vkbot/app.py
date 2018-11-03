import logging
from typing import Dict, Optional

import click
import uvloop
import yaml
from aiohttp import web

from vkbot.connectors import setup_connectors
from vkbot.schemas import ConfigSchema
from vkbot.sentry import init_sentry
from vkbot.views import handle_vk_event
from vkbot.vkapi import setup_vk_api
from vkbot.middlewares import error_middleware

uvloop.install()

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] PID: %(process)d %(levelname)s @ "
    "%(pathname)s:%(lineno)d ~ %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
)


def read_config(config_path: str) -> Dict:
    with open(config_path, "r") as conf_file:
        raw_config = yaml.load(conf_file) or {}

    return ConfigSchema(strict=True).load(raw_config).data


def setup_routes(app: web.Application) -> None:
    app.router.add_post("/send_vk", handle_vk_event)


async def create_app(config_path: str) -> web.Application:
    app = web.Application(middlewares=[error_middleware])

    app["config"] = read_config(config_path)

    init_sentry(app)
    setup_routes(app)
    setup_connectors(app)
    setup_vk_api(app)

    return app


@click.command()
@click.option("--config", default="/app/etc/development.yml")
@click.option("--port", default=8080)
@click.option("--socket_path")
def start_app(config: str, port: int, socket_path: Optional[str]) -> None:
    web.run_app(
        create_app(config), host="0.0.0.0", port=port, access_log=None, path=socket_path
    )


if __name__ == "__main__":
    start_app()
