from typing import Callable

import structlog

from ai_chat.intent.intent_classifier import IntentClassifier
from ai_chat.intent.models import Domain
from ai_chat.service.employment_service import EmploymentService
from ai_chat.service.skills_service import SkillsService
from ai_chat.vectordb.cv_repository import CvRepository

log = structlog.get_logger()


class QueryRouter:
    def __init__(self, repository: CvRepository) -> None:
        self.intent_classifier = IntentClassifier()
        self.employment_service = EmploymentService(repository)
        self.skills_service = SkillsService()

        self.routes: dict[Domain, Callable] = {
            Domain.EMPLOYMENT: self.employment_service.handle,
            Domain.SKILLS: self.skills_service.handle
        }

    def route_query(self, question: str) -> str:
        question = question.lower().replace("branislav", "he").replace("vidovic", "")
        domains = self.intent_classifier.get_domain_and_query_type(question)
        domain = domains[0] #decide on strategy
        if domain is None:
            return question + " -> not clear what this question is about"

        handler = self.routes.get(domain)
        log.info(
            "intent.routing",
            question=question,
            domain=domain,
            handler=handler
        )
        if handler is None:
            return "This platform answers only questions about CV of Branislav"

        return handler(question)
