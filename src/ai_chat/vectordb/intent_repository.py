import os
from typing import List

import chromadb
import structlog
from sentence_transformers import SentenceTransformer

from ai_chat.intent.models import Domain
from ai_chat.vectordb.custom_embedding_function import CustomEmbeddingFunction
from ai_chat.vectordb.intents import intents
from ai_chat.vectordb.models import RetrievalResult, VectorItem

log = structlog.get_logger()

INTENTS = "intents"


def get_metadata(domain: Domain) -> dict:
    return {"domain": domain.value}



class IntentRepository:
    def __init__(self):
        # chroma connection
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

        # embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # chroma collection
        self.embedding_function = CustomEmbeddingFunction(model)
        self.intent_collection = self.client.get_or_create_collection(name=INTENTS, embedding_function=self.embedding_function)

    def initialize_intents(self):
        ids = []
        documents = []
        metadatas = []

        idx = 0
        for question, domain in intents:
            ids.append(str(idx))
            documents.append(question.lower())
            metadatas.append(get_metadata(domain))
            idx += 1

        try:
            self.intent_collection = self.client.get_or_create_collection(name=INTENTS, embedding_function=self.embedding_function)
            self.intent_collection.add(ids=ids, documents=documents, metadatas=metadatas)
            log.info("Successfully indexed collection %s", INTENTS)
        except Exception:
            log.exception("Failed to create/add to collection %s", INTENTS)
            raise

    def delete_intent_data(self):
        try:
            self.client.delete_collection(INTENTS)
            log.info("Successfully deleted %s data", INTENTS)
        except ValueError:
            log.exception("Failed to delete collection %s", INTENTS)
            raise  # important: fail fast

    def query_intent(self, question: str, result_size: int) -> List[RetrievalResult]:
        result_raw = self.intent_collection.query(query_texts=[question], n_results=result_size)

        ids = result_raw["ids"][0]
        distances = result_raw["distances"][0]
        documents = result_raw["documents"][0]
        metadatas = result_raw["metadatas"][0]

        res: List[RetrievalResult] = []
        for id_, distance, document, metadata in zip(ids, distances, documents, metadatas):
            res.append(RetrievalResult(id=id_, distance=distance, document=document, metadata=metadata))

        return res

    def get_intents_raw(self):
        chroma_docs = self.intent_collection.get()
        results: List[VectorItem] = []
        for id_, doc, metadata in zip(chroma_docs["ids"], chroma_docs["documents"], chroma_docs["metadatas"]):
            results.append(VectorItem(id=id_, document=doc, metadata=metadata))
        return results