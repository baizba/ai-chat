import structlog

from ai_chat.llm.llm_service import LLMService
from ai_chat.vectordb.cv_repository import CvRepository

log = structlog.get_logger()


class SkillsService:
    def __init__(self, repository: CvRepository, llm_service: LLMService):
        self.repository = repository
        self.llm_service = llm_service

    def handle(self, question: str) -> str:
        skills_metadata = {"entityType": "skills"}
        result = self.repository.metadata_query(skills_metadata)
        log.info(
            "cv.retrieval.metadata.skills",
            query=skills_metadata,
            doc_ids=[r.id for r in result],
            path=[r.metadata["path"] for r in result],
            category=[r.metadata["category"] for r in result]
        )

        context = "\n".join([r.document for r in result])
        return self.llm_service.answer_skills(question, context)
