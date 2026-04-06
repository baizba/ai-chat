from typing import Callable

import structlog

from ai_chat.intent.intent_classifier import IntentClassifier
from ai_chat.intent.models import Domain, Intent
from ai_chat.llm.llm_service import LLMService
from ai_chat.router.models import IntentConfidence, RoutingResponse, IntentMatch
from ai_chat.service.certificate_service import CertificateService
from ai_chat.service.employment_service import EmploymentService
from ai_chat.service.profile_service import ProfileService
from ai_chat.service.skills_service import SkillsService
from ai_chat.vectordb.cv_repository import CvRepository

log = structlog.get_logger()

NAME_ALIASES = ["branislav vidovic", "branislav", "vidovic"]


def create_routing_response(answer: str, intent_confidence: IntentConfidence, first: Intent | None, second: Intent | None) -> RoutingResponse:
    intent_match = IntentMatch(first, second, intent_confidence)
    return RoutingResponse(intent_match, answer)


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

    def route_query(self, question: str) -> RoutingResponse:
        # some very light normalization (search works better without personal name)
        question = question.lower()
        for alias in NAME_ALIASES:
            question = question.replace(alias, "he")

        intents = self.intent_classifier.get_intents(question)

        if intents is None or len(intents) < 2:
            return create_routing_response(f"{question} -> not clear what this question is about", IntentConfidence.LOW, None, None)

        distance_threshold = 0.7
        absolute_dist_threshold = 0.05
        best = intents[0]
        second_best = intents[1]
        best_match = best.score
        second_best_match = second_best.score
        best_domain = best.domain
        second_best_domain = second_best.domain

        # simply check if the similarity is weak to avoid false domains
        if best_match > distance_threshold:
            log.error("intent.resolve.lowconfidence", best_match=best_match, distance_threshold=distance_threshold)
            return create_routing_response(
                f"{question} -> not clear what this question is about. Is it really about {best_domain}?",
                IntentConfidence.LOW,
                best,
                second_best
            )

        # if we resolve two different intents but really close then confidence is low
        if best_domain != second_best_domain and second_best_match - best_match < absolute_dist_threshold:
            log.error("intent.resolve.ambigous", best_match=best_match, second_best_match=second_best_match, absolute_dist_threshold=absolute_dist_threshold)
            return create_routing_response(
                f"{question} -> not clear what this question is about. Is it about {best_domain} or {second_best_domain}?",
                IntentConfidence.MEDIUM,
                best,
                second_best
            )

        handler = self.handlers.get(best_domain)
        log.info("intent.routing", question=question, domain=best_domain, handler=handler)
        if handler is None:
            # return "This platform answers only questions about CV of Branislav"
            return create_routing_response(
                "This platform answers only questions about CV of Branislav",
                IntentConfidence.LOW,
                best,
                second_best
            )

        return create_routing_response(handler(question), IntentConfidence.HIGH, best, second_best)
