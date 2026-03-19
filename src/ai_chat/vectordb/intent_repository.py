from typing import List

import chromadb
import structlog
from sentence_transformers import SentenceTransformer

from ai_chat.intent.models import Domain
from ai_chat.vectordb.custom_embedding_function import CustomEmbeddingFunction
from ai_chat.vectordb.models import RetrievalResult

log = structlog.get_logger()

INTENTS = "intents"


def get_metadata(domain: Domain) -> dict:
    return {"domain": domain.value}


intents: list[tuple[str, Domain]] = [
    # EMPLOYMENT
    ("Where did he work in 2015?", Domain.EMPLOYMENT),
    ("Where did he work from 2015 to 2016?", Domain.EMPLOYMENT),
    ("Where was he employed in the period from 2009 to 2012", Domain.EMPLOYMENT),
    ("What was he doing in 2025?", Domain.EMPLOYMENT),
    ("Which company did he work for in 2015?", Domain.EMPLOYMENT),
    ("Who employed him in 2017?", Domain.EMPLOYMENT),
    ("What job did he have in 2018?", Domain.EMPLOYMENT),
    ("Where was he working during 2019?", Domain.EMPLOYMENT),
    ("Which employer did he have in 2020?", Domain.EMPLOYMENT),
    ("What company was he working at in 2021?", Domain.EMPLOYMENT),
    ("What companies did he work for in 2015?", Domain.EMPLOYMENT),
    ("Which employers did he have in 2018?", Domain.EMPLOYMENT),
    ("What companies did he work for between 2013 and 2014?", Domain.EMPLOYMENT),
    ("List the companies he worked for in 2016.", Domain.EMPLOYMENT),
    ("Which companies employed him during 2011?", Domain.EMPLOYMENT),
    ("Where did he work between 2009 and 2012?", Domain.EMPLOYMENT),
    ("What employers did he have in 2020?", Domain.EMPLOYMENT),
    ("Show the companies he worked for during 2017.", Domain.EMPLOYMENT),
    ("Which organizations did he work for from 2014 to 2016?", Domain.EMPLOYMENT),
    ("Give me the companies he worked for in 2019.", Domain.EMPLOYMENT),
    ("Give me all his employers.", Domain.EMPLOYMENT),
    ("List all his employers.", Domain.EMPLOYMENT),
    ("What companies did he work for?", Domain.EMPLOYMENT),
    ("Summarize his employment history.", Domain.EMPLOYMENT),
    ("Give me an overview of his career.", Domain.EMPLOYMENT),
    ("What does his work history look like?", Domain.EMPLOYMENT),
    ("Describe his professional career.", Domain.EMPLOYMENT),
    ("Can you summarize the companies he worked for?", Domain.EMPLOYMENT),
    ("Provide a summary of his professional background.", Domain.EMPLOYMENT),
    ("What is the overall timeline of his employment?", Domain.EMPLOYMENT),
    ("Explain his career path.", Domain.EMPLOYMENT),
    ("Give a short summary of his work experience.", Domain.EMPLOYMENT),
    ("How has his career developed over the years?", Domain.EMPLOYMENT),

    # SKILLS
    ("Does he know Java?", Domain.SKILLS),
    ("Is he experienced with Spring Boot?", Domain.SKILLS),
    ("Does he have experience with Camunda?", Domain.SKILLS),
    ("Is Kubernetes one of his skills?", Domain.SKILLS),
    ("Does he work with Microservices?", Domain.SKILLS),
    ("Has he used Maven?", Domain.SKILLS),
    ("Does he have experience with Junit framework?", Domain.SKILLS),
    ("Is he familiar with TDD?", Domain.SKILLS),
    ("Has he worked with Android?", Domain.SKILLS),
    ("What skills does he have?", Domain.SKILLS),
    ("Which technologies does he know?", Domain.SKILLS),
    ("List his programming skills.", Domain.SKILLS),
    ("What technologies has he worked with?", Domain.SKILLS),
    ("Which programming languages does he use?", Domain.SKILLS),
    ("Give me a list of his technical skills.", Domain.SKILLS),
    ("What frameworks is he experienced with?", Domain.SKILLS),
    ("What tools does he use as a developer?", Domain.SKILLS),
    ("Which software technologies does he work with?", Domain.SKILLS),
    ("What are his main development skills?", Domain.SKILLS),
    ("What technologies did he use?", Domain.SKILLS),

    # PROFILE
    ("Who is he?", Domain.PROFILE),
    ("Give me a summary of his profile.", Domain.PROFILE),
    ("Tell me about this person.", Domain.PROFILE),
    ("What is his professional profile?", Domain.PROFILE),
    ("Provide an overview of his background.", Domain.PROFILE),
    ("Summarize his professional experience.", Domain.PROFILE),
    ("What kind of developer is he?", Domain.PROFILE),
    ("Describe his professional background.", Domain.PROFILE),
    ("Give a short introduction about him.", Domain.PROFILE),
    ("What is his career profile?", Domain.PROFILE),
]


class IntentRepository:
    def __init__(self):
        # chroma connection
        self.client = chromadb.HttpClient(host="localhost", port=8000)

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
