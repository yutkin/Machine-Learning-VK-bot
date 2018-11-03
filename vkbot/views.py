import logging

import ujson
from aiohttp import web

from vkbot.vkapi import VKApi
from vkbot.utils import fetch_photos
from vkbot.connectors import ImageNetConnector, ChatBotConnector

logger = logging.getLogger(__name__)


async def handle_vk_event(request: web.Request) -> None:
    try:
        request_body = await request.json(loads=ujson.loads)
    except ValueError:
        raise web.HTTPBadRequest(text="JSON is malformed")

    vk_api: VKApi = request.app["vk"]
    imagenet_conn: ImageNetConnector = request.app["cv"]
    chatbot_conn: ChatBotConnector = request.app["nlp"]

    if request_body.get("secret") != vk_api.app_secret:
        raise web.HTTPForbidden(text="Secret key is not specified or invalid")

    # Filter out non message events
    if request_body["type"] != "message_new":
        return

    event_obj = request_body["object"]
    message = event_obj["text"]

    # Trying to fetch photos from attachments
    base64_photos = await fetch_photos(vk_api, event_obj["attachments"])

    # If fetched at least one photo, try to classify it
    if base64_photos:
        res = await imagenet_conn.classify_batch_of_images(base64_photos)
        message = imagenet_conn.join_predictions_to_one_message(
            res.get("predictions", [])
        )
    elif message:
        message = await chatbot_conn.send_message(message)
        message = message["reply"]

    # Send reply to user
    await vk_api.send_message(user_id=event_obj["from_id"], message=message)
