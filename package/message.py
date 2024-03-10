from package.enums import *

from typing import Any


class Message:
    def __init__(self, type: MessageType, content: Any = None):
        self.type = type
        self.content = content

    def __str__(self):
        return f"Message{{'type': {self.type.value}, 'content': {self.content}}}"
