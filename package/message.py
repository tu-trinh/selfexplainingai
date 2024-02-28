from package.enums import *

from typing import Any


class Message:
    def __init__(self, type: MessageType, content: Any = None):
        self.type = type
        self.content = content
