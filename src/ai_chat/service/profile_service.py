import structlog

from ai_chat.llm.llm_service import LLMService
from ai_chat.service import prompts
from ai_chat.vectordb.cv_repository import CvRepository

log = structlog.get_logger()


class ProfileService:
    def __init__(self, repository: CvRepository, llm_service: LLMService):
        self.repository = repository
        self.llm_service = llm_service

    def handle(self, question: str) -> str:
        metadata_profile = {"entityType": "profile"}
        result = self.repository.metadata_query(metadata_profile)
        log.info(
            "cv.retrieval.metadata.profile",
            query=metadata_profile,
            doc_ids=[r.id for r in result],
            path=[r.metadata["path"] for r in result],
        )

        context = "\n".join([r.document for r in result])
        return self.llm_service.answer(prompts.profile_prompt, question, context)
