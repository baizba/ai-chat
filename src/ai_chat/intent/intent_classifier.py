import re

import structlog
from nltk import WordNetLemmatizer

from ai_chat.intent.models import Domain, QueryType, DOMAIN_KEYWORDS, QUERY_KEYWORDS

log = (structlog.get_logger())


def get_domain(tokens: list[str]) -> Domain | None:
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in tokens:
                return domain
    return None


def get_query_type(tokens: list[str]) -> QueryType | None:
    for qt, keywords in QUERY_KEYWORDS.items():
        for kw in keywords:
            if kw in tokens:
                return qt
    return None


class IntentClassifier:
    def __init__(self):
        # download on container build, not here
        # nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()

    def get_domain_and_query_type(self, question: str) -> tuple[Domain, QueryType]:
        tokens = self.normalize_tokens(question)
        domain = get_domain(tokens)
        query_type = get_query_type(tokens)
        log.info(
            "intent.classification",
            question=question,
            tokens=tokens,
            domain=domain.name if domain else None,
            query_type=query_type.name if query_type else None
        )
        return domain, query_type

    def normalize_tokens(self, question: str) -> list[str]:
        tokens = re.findall(r"\b\w+\b", question.lower())
        return [self.lemmatizer.lemmatize(t) for t in tokens]
