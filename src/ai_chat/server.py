import time
import uuid
from contextlib import contextmanager
from http import HTTPStatus

import structlog
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from ai_chat.intent.intent_classifier import IntentClassifier
from ai_chat.retrieval.cv_service import CvService
from ai_chat.indexing.cv_indexing_service import CvIndexingService
from ai_chat.llm.llm_service import LLMService
from ai_chat.models import ChatRequest
from ai_chat.models import ChatResponse
from ai_chat.router.query_router import QueryRouter
from ai_chat.vectordb.cv_repository import CvRepository
from ai_chat.vectordb.models import VectorItem

log = structlog.get_logger()

app = FastAPI()

repository = CvRepository()
cv_service = CvService(repository)
cv_indexing_service = CvIndexingService(repository)
llm_service = LLMService()
classifier = IntentClassifier()
query_router = QueryRouter(llm_service)

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
    question = request.message

    request_log = log.bind(
        request_id=request_id,
        ui_request_id=request.uiRequestId,
        user_id=request.user
    )

    request_log.info(
        "chat.request",
        message=question,
        message_length=len(question)
    )

    with measure_time() as elapsed_time:
        answer = query_router.route_query(question)

    request_log.info(
        "chat.response",
        duration_ms=round(elapsed_time()),
        documents_count=len(answer)
    )

    return ChatResponse(response=answer)


@app.get("/admin/docs/cv/raw")
async def get_chroma_docs() -> list[VectorItem]:
    return cv_service.get_docs_raw()


@app.get("/admin/docs/intent/raw")
async def get_intent_docs() -> list[VectorItem]:
    return classifier.get_intents_raw()


@app.post("/admin/cv/reindex", status_code=HTTPStatus.NO_CONTENT)
async def reindex_cv():
    log.info("admin.reindex.cv.started")
    cv_indexing_service.index_cv()
    log.info("admin.reindex.cv.finished")


@app.post("/admin/intents/reindex", status_code=HTTPStatus.NO_CONTENT)
async def reindex_intent():
    log.info("admin.reindex.intents.started")
    classifier.index_intents()
    log.info("admin.reindex.intents.finished")
