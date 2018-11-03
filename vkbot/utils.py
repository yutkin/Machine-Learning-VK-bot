import asyncio
import logging
from typing import List, Dict

import aiohttp

from vkbot.vkapi import VKApi

logger = logging.getLogger(__name__)


def extract_image_with_size_type(size_list: List[Dict], img_type: str) -> Dict:
    """
    Return image item with corresponding size from JSON list
    """
    return next(filter(lambda x: x["type"] == img_type, size_list), {})


async def fetch_photos(api: VKApi, attachments: List[Dict]) -> List[bytes]:
    """
    Asynchronously fetching list of photos from VK.com
    """
    futures = []

    for attachment in attachments:
        if attachment.get("type") == "photo":
            sizes_list = attachment.get("photo", {}).get("sizes", [])
            image = extract_image_with_size_type(sizes_list, api.img_type)

            fut = asyncio.ensure_future(api.fetch_image(image["url"]))
            setattr(fut, "url", image["url"])
            futures.append(fut)

    base64_images = []

    for fut in futures:
        try:
            base64_images.append(await fut)
        except aiohttp.ClientError:
            logger.error(f"Cannot fetch image", exc_info=True)

    return base64_images
