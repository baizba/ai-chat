from __future__ import annotations #imported to make it possible to reuse type inside type

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    uiRequestId: str
    user: str


class ChatResponse(BaseModel):
    documents: list[str]
    distances: list[float]


class CVNode:
    id: str
    title: str
    text: str | None
    parent: CVNode | None
    children : list[CVNode]