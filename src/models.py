from __future__ import annotations  # imported to make it possible to reuse type inside type

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    uiRequestId: str
    user: str


class ChatResponse(BaseModel):
    documents: list[str]
    distances: list[float]


class CVNode:
    def __init__(self):
        self.id: str | None = None
        self.title: str = ""
        self.text: str = ""
        self.parent: CVNode | None = None
        self.children: list[CVNode] = []

