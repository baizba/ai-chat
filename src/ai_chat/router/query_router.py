from typing import Callable

from ai_chat.intent.intent_classifier import IntentClassifier
from ai_chat.intent.models import Domain, QueryType
from ai_chat.service.employment_service import EmploymentService
from ai_chat.service.skills_service import SkillsService


class QueryRouter:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.employment_service = EmploymentService()
        self.skills_service = SkillsService()

        self.routes: dict[tuple[Domain, QueryType], Callable] = {
            (Domain.EMPLOYMENT, QueryType.FACT): self.employment_service.handle_fact,
            (Domain.EMPLOYMENT, QueryType.LIST): self.employment_service.handle_list,
            (Domain.SKILLS, QueryType.LIST): self.skills_service.handle_list
        }

    def route_query(self, question: str) -> str:
        domain, query_type = self.intent_classifier.get_domain_and_query_type(question)
        if domain is None or query_type is None:
            return question + " -> not clear what this question is about"

        # here goes the logic with services
        self.routes.get((domain, query_type))

        return "This platform answers only questions about VC of Branislav"
