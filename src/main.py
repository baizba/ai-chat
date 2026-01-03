import datetime
import time
import uuid
from contextlib import contextmanager

import structlog
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from LocalEmbeddings import LocalEmbeddings
from models import ChatRequest
from models import ChatResponse

log = structlog.get_logger()

app = FastAPI()

# start end execute embedding of local texts
local_embeddings = LocalEmbeddings()
local_embeddings.perform_embeddings()

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
async def get_hello(request: ChatRequest):
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
        response = local_embeddings.perform_query(request.message)

    request_log.info(
        "llm.chat.response",
        duration_ms=round(elapsed_time(), 2),
        documents_count=len(response.documents)
    )

    return response
