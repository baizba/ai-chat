from typing import Any, Dict

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from models import ChatResponse


# wrapper matching new Chroma interface
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: SentenceTransformer):
        super().__init__()
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return self.model.encode(input).tolist()


class LocalEmbeddings:
    def __init__(self) -> None:
        # chroma collection
        self.collection = None

        # in-memory chroma
        self.client = chromadb.EphemeralClient()

        # embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Add multiple documents (this is the knowledge for chroma)
        self.documents = [
            "Apples are delicious and nutritious.",
            "Oranges contain a lot of vitamin C.",
            "Bananas are rich in potassium.",
            "Austria is a country in Europe with beautiful mountains.",
            "Brazil is famous for its coffee and carnival.",
            "Italy produces amazing pasta and wine.",
            "Basketball is a popular sport in the US.",
            "Machine learning is a branch of artificial intelligence.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "Branislav has good knowledge of Spring and Java.",
            "Branislav understands system architecture.",
            "Branislav likes to learn abot AI and LLM.",
            "Branislav is very curious.",
            "Branislav is a Private Pilot."
        ]

        self.embedding_function = MyEmbeddingFunction(self.model)

    def perform_embeddings(self) -> None:
        ids = [f"id{i}" for i in range(len(self.documents))]
        self.collection = self.client.create_collection(name="my_collection", embedding_function=self.embedding_function)
        self.collection.add(ids=ids, documents=self.documents)

    def perform_query(self, query) -> ChatResponse:
        # Example queries
        queries = [
            "I want to eat a fruit rich in vitamin C",
            "Which European country has mountains?",
            "Tell me something about AI",
            "Where does coffee come from?",
            "What technologies is Branislav good at?",
            "With whome can i fly?"
        ]
        result_raw = self.collection.query(
            query_texts=[query],  # Chroma will embed this for you
            n_results=3  # how many results to return
        )
        return ChatResponse(documents=result_raw["documents"][0], distances=result_raw["distances"][0])

#lc = LocalEmbeddings()
#lc.perform_embeddings()
#print(lc.perform_query("I want to eat a fruit rich in vitamin C"))