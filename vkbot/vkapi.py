import base64
import logging
from typing import Dict

import aiohttp
import ujson
from aiohttp import web

logger = logging.getLogger(__name__)


class VKApi:
    def __init__(
        self,
        access_token: str,
        app_secret: str,
        api_url: str,
        api_version: str,
        conn_timeout: int,
        read_timeout: int,
        img_type: str,
        **kwargs: Dict,
    ):
        self._access_token = access_token
        self._app_secret = app_secret

        self._api_url = api_url
        self._api_version = api_version
        self._conn_timeout = conn_timeout
        self._read_timeout = read_timeout
        self._img_type = img_type

        self._connector = aiohttp.TCPConnector(
            use_dns_cache=True, ttl_dns_cache=60 * 60, limit=1024
        )

        self._sess = aiohttp.ClientSession(
            connector=self._connector,
            read_timeout=self._read_timeout,
            conn_timeout=self._conn_timeout,
            json_serialize=ujson.dumps,
        )

    @property
    def app_secret(self) -> str:
        return self._app_secret

    @property
    def img_type(self) -> str:
        return self._img_type

    async def send_message(self, user_id: int, message: str, **kwargs: Dict) -> None:
        params = dict(
            access_token=self._access_token,
            v=self._api_version,
            user_id=user_id,
            message=message,
            **kwargs,
        )

        async with self._sess.get(
            f"{self._api_url}/messages.send", params=params
        ) as resp:
            resp.raise_for_status()

    async def fetch_image(self, url: str, encode_base64: bool = True) -> bytes:
        async with self._sess.get(url) as resp:
            resp.raise_for_status()

            raw_data = await resp.read()

            if encode_base64:
                return base64.b64encode(raw_data)

            return raw_data


def setup_vk_api(app: web.Application) -> None:
    app["vk"] = VKApi(**app["config"]["vk"])
