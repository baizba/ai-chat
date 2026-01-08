from __future__ import annotations  # imported to make it possible to reuse type inside type

from enum import Enum

from pydantic import BaseModel


# fastapi request
class ChatRequest(BaseModel):
    message: str
    uiRequestId: str
    user: str


# fastapi response
class ChatResponse(BaseModel):
    documents: list[str]
    distances: list[float]


# used for nodes (node level)
class CVNodeLevel(Enum):
    ROOT = 1
    SECTION = 2
    SUBSECTION = 3


# Represents CV structure
class CVNode:
    def __init__(self):
        self.id: str | None = None
        self.title: str = ""
        self.text: str = ""
        self.parent: CVNode | None = None
        self.children: list[CVNode] = []
        self.level: CVNodeLevel | None = None

    def get_path(self) -> str:
        parts = []

        temp_node = self
        while temp_node is not None:
            # strip heading markers and whitespace
            title = temp_node.title.lstrip('#').strip()
            parts.append(title)
            temp_node = temp_node.parent

        # reverse to get root -> leaf
        parts.reverse()

        # join into path
        return "/".join(parts)

