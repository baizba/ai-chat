import structlog

from ai_chat.intent.models import Domain
from ai_chat.vectordb.intent_repository import IntentRepository
from ai_chat.vectordb.models import RetrievalResult

log = structlog.get_logger()


class IntentClassifier:
    def __init__(self):
        self.intent_repository = IntentRepository()

    def get_intents(self, question: str) -> list[RetrievalResult]:
        intents = self.intent_repository.query_intent(question, 2)
        log.info(
            "intent.retrieval",
            question=question,
            domain=[t.metadata["domain"] for t in intents],
            document=[t.document for t in intents],
            distance=[t.distance for t in intents]
        )

        return [RetrievalResult(id=i_.id, distance=i_.distance, document=i_.document, metadata=i_.metadata) for i_ in intents]

    def index_intents(self):
        self.intent_repository.delete_intent_data()
        self.intent_repository.initialize_intents()

    def get_intents_raw(self):
        return self.intent_repository.get_intents_raw()
