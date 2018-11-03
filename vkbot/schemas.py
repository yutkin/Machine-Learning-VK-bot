import os
from typing import Dict

from marshmallow import Schema, fields, validate, post_load


class AppSchemaSection(Schema):
    debug = fields.Bool(required=True)
    environment = fields.String(
        required=True, validate=validate.OneOf(["development", "production", "testing"])
    )


class VKSchemaSection(Schema):
    api_url = fields.String(required=True)
    api_version = fields.Str(required=True)
    conn_timeout = fields.Integer(required=True)
    read_timeout = fields.Integer(required=True)
    img_type = fields.String(
        required=True,
        validate=validate.OneOf(["s", "m", "x", "o", "p", "q", "r", "y", "z", "w"]),
    )

    @post_load
    def setup_credentials(self, vk_cfg: Dict) -> Dict:
        vk_cfg["access_token"] = os.environ["VK_ACCESS_TOKEN"]
        vk_cfg["app_secret"] = os.environ["VK_APP_SECRET"]
        return vk_cfg


class ImageNetSchemaSection(Schema):
    url = fields.String(required=True)
    conn_timeout = fields.Integer(required=True)
    read_timeout = fields.Integer(required=True)


class ChatBotSchemaSection(Schema):
    url = fields.String(required=True)
    conn_timeout = fields.Integer(required=True)
    read_timeout = fields.Integer(required=True)


class ConfigSchema(Schema):
    app = fields.Nested(AppSchemaSection(), required=True)
    vk = fields.Nested(VKSchemaSection(), required=True)
    cv = fields.Nested(ImageNetSchemaSection(), required=True)
    nlp = fields.Nested(ChatBotSchemaSection(), required=True)
