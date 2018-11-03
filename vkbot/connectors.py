import ujson
from typing import List, Dict, Any

import aiohttp
from aiohttp import web

from vkbot.const import IMAGENET_CLASSES


class BaseMLConnector:
    def __init__(self, url: str, read_timeout: int, conn_timeout: int):
        self._url = url
        self._read_timeout = read_timeout
        self._conn_timeout = conn_timeout
        self._connector = aiohttp.TCPConnector(
            use_dns_cache=True, ttl_dns_cache=60 * 60, limit=1024
        )

        self._sess = aiohttp.ClientSession(
            connector=self._connector,
            read_timeout=self._read_timeout,
            conn_timeout=self._conn_timeout,
            json_serialize=ujson.dumps,
        )


class ChatBotConnector(BaseMLConnector):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    async def send_message(self, message):
        payload = dict(message=message)

        async with self._sess.post(self._url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json(loads=ujson.loads)


class ImageNetConnector(BaseMLConnector):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    async def classify_batch_of_images(self, images: List[bytes]) -> Dict:
        payload = dict(instances=[{"b64": str(image, "utf-8")} for image in images])

        async with self._sess.post(self._url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json(loads=ujson.loads)

    @staticmethod
    def join_predictions_to_one_message(
        predictions: List[Dict], sep: str = "\n"
    ) -> str:
        class_names = [
            f"{i+1}. {IMAGENET_CLASSES[pred['classes'] - 1]}"
            for i, pred in enumerate(predictions)
        ]
        return sep.join(class_names)


def setup_connectors(app: web.Application) -> None:
    app["cv"] = ImageNetConnector(**app["config"]["cv"])
    app["nlp"] = ChatBotConnector(**app["config"]["nlp"])
