from __future__ import annotations  # imported to make it possible to reuse type inside type

from pydantic import BaseModel


# fastapi request
class ChatRequest(BaseModel):
    message: str
    uiRequestId: str
    user: str


# fastapi response
class ChatResponse(BaseModel):
    response: str
