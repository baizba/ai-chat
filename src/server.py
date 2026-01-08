import time
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from cv_service import CVService
from models import ChatRequest
from models import ChatResponse

log = structlog.get_logger()

app = FastAPI()

cv_service = CVService()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@contextmanager
def measure_time():
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())

    request_log = log.bind(
        request_id=request_id,
        ui_request_id=request.uiRequestId,
        user_id=request.user,
    )

    request_log.info(
        "llm.chat.request",
        message=request.message,
        message_length=len(request.message),
        model="ChromaVectorDB"
    )

    with measure_time() as elapsed_time:
        response = cv_service.query(request.message)

    request_log.info(
        "llm.chat.response",
        duration_ms=round(elapsed_time(), 2),
        documents_count=len(response.documents)
    )

    return response


@app.get("/admin/docs/raw")
async def get_chroma_docs() -> list[dict[str, Any]]:
    return cv_service.get_docs_raw()


@app.post("/admin/reindex", status_code=HTTPStatus.NO_CONTENT)
async def reindex():
    log.info("admin.reindex.started")
    cv_service.index_cv()
    log.info("admin.reindex.finished")
