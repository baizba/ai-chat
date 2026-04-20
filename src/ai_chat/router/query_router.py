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


def create_routing_response(answer: str, intent_confidence: IntentConfidence, first: Intent, second: Intent) -> RoutingResponse:
    intent_match = IntentMatch(first, second, intent_confidence)

    if intent_confidence == IntentConfidence.LOW:
        final_answer = f"I'm not fully certain, but if you mean {first.domain}, this could be the answer:\n" + answer
        return RoutingResponse(intent_match, final_answer)

    if intent_confidence == IntentConfidence.MEDIUM:
        final_answer = f"It seems you are asking about {first.domain}, this could be the answer:\n" + answer
        return RoutingResponse(intent_match, final_answer)

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

        # if we did not find any intents or have less than 2 then there is a big problem
        if intents is None:
            raise Exception("Intent retrieval failed completely")

        if len(intents) < 2:
            raise Exception("Intent retrieval returned insufficient results (expected 2)")

        distance_threshold = 0.7
        absolute_dist_threshold = 0.05
        best = intents[0]
        second_best = intents[1]
        best_match = best.score
        second_best_match = second_best.score
        best_domain = best.domain
        second_best_domain = second_best.domain

        # if no handler then there is nothing we can do
        handler = self.handlers.get(best_domain)
        log.info("intent.routing", question=question, domain=best_domain, handler=handler)

        if handler is None:
            raise Exception(f"No handler registered for domain: {best_domain}")

        # simply check if the similarity is weak to avoid false domains
        if best_match > distance_threshold:
            log.error("intent.resolve.lowconfidence", best_match=best_match, distance_threshold=distance_threshold)
            intent_confidence = IntentConfidence.LOW

        # if we resolve two different intents but really close then confidence is low
        elif best_domain != second_best_domain and second_best_match - best_match < absolute_dist_threshold:
            log.error("intent.resolve.ambigous", best_match=best_match, second_best_match=second_best_match, absolute_dist_threshold=absolute_dist_threshold)
            intent_confidence = IntentConfidence.MEDIUM

        # if none of those match then we are quite certain we have a good match
        else:
            intent_confidence = IntentConfidence.HIGH

        return create_routing_response(handler(question), intent_confidence, best, second_best)
