from typing import Callable

import structlog

from ai_chat.intent.intent_classifier import IntentClassifier
from ai_chat.intent.models import Domain
from ai_chat.llm.llm_service import LLMService
from ai_chat.service.certificate_service import CertificateService
from ai_chat.service.employment_service import EmploymentService
from ai_chat.service.profile_service import ProfileService
from ai_chat.service.skills_service import SkillsService
from ai_chat.vectordb.cv_repository import CvRepository

log = structlog.get_logger()


class QueryRouter:
    def __init__(self, llm_service: LLMService) -> None:
        repository = CvRepository()
        self.intent_classifier = IntentClassifier()
        self.employment_service = EmploymentService(repository, llm_service)
        self.skills_service = SkillsService(repository, llm_service)
        self.profile_service = ProfileService(repository, llm_service)
        self.certificate_service = CertificateService(repository, llm_service)

        self.handlers: dict[Domain, Callable] = {
            Domain.EMPLOYMENT: self.employment_service.handle,
            Domain.SKILLS: self.skills_service.handle,
            Domain.PROFILE: self.profile_service.handle,
            Domain.CERTIFICATES: self.certificate_service.handle
        }

    def route_query(self, question: str) -> str:
        question = question.lower().replace("branislav", "he").replace("vidovic", "")
        intents = self.intent_classifier.get_intents(question)

        if intents is None or len(intents) < 2:
            return question + " -> not clear what this question is about"

        distance_threshold = 0.7
        absolute_dist_threshold = 0.05
        best_match = intents[0].distance
        second_best_match = intents[1].distance
        best_domain = intents[0].metadata["domain"]
        second_best_domain = intents[1].metadata["domain"]

        # simply check if the similarity is weak to avoid false domains
        if best_match > distance_threshold:
            log.error("intent.resolve.lowconfidence", best_match=best_match, distance_threshold=distance_threshold)
            return question + f" -> not clear what this question is about. Is it about {best_domain} or {second_best_domain}?"

        # if we resolve two different intents but really close then confidence is low
        if best_domain != second_best_domain and second_best_match - best_match < absolute_dist_threshold:
            log.error("intent.resolve.ambigous", best_match=best_match, second_best_match=second_best_match, absolute_dist_threshold=absolute_dist_threshold)
            return question + f" -> not clear what this question is about. Is it about {best_domain} or {second_best_domain}?"

        domain = Domain(best_domain)  # decide on strategy
        if domain is None:
            return question + " -> not clear what this question is about"

        handler = self.handlers.get(domain)
        log.info("intent.routing", question=question, domain=domain, handler=handler)
        if handler is None:
            return "This platform answers only questions about CV of Branislav"

        return handler(question)
