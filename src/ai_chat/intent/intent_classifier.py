import structlog

from ai_chat.intent.models import Domain
from ai_chat.vectordb.intent_repository import IntentRepository

log = structlog.get_logger()


class IntentClassifier:
    def __init__(self):
        self.intent_repository = IntentRepository()

    def get_domain_and_query_type(self, question: str) -> Domain:
        intents = self.intent_repository.query_intent(question, 3)
        log.info(
            "intent.retrieval",
            question=question,
            domain=[t.metadata["domain"] for t in intents],
            document=[t.document for t in intents],
            distance=[t.distance for t in intents]
        )
        md = intents[0].metadata
        return Domain(md["domain"])

    def index_intents(self):
        self.intent_repository.delete_intent_data()
        self.intent_repository.initialize_intents()

    def get_intents_raw(self):
        return self.intent_repository.get_intents_raw()