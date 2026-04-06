from enum import Enum

from ai_chat.intent.models import Intent


class IntentConfidence(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IntentMatch:
    def __init__(self, first: Intent | None, second: Intent | None, intent_confidence: IntentConfidence):
        self.first = first
        self.second = second
        self.intent_confidence = intent_confidence


class RoutingResponse:
    def __init__(self, intent_match: IntentMatch, answer: str):
        self.intent_match = intent_match
        self.answer = answer
