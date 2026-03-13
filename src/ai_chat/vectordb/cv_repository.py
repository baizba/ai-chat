# wrapper matching new Chroma interface
from typing import List, Dict

import chromadb
import structlog
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from ai_chat.vectordb.models import RetrievalResult, CvDataItem

# constants
CV_DATA = "cv_data"

log = structlog.get_logger()


# wrapper matching new Chroma interface
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: SentenceTransformer):
        super().__init__()
        self.model = model

    def __call__(self, docs: Documents) -> Embeddings:
        # embed the documents somehow
        return self.model.encode(docs).tolist()


class CvRepository:
    def __init__(self):
        # chroma connection
        self.client = chromadb.HttpClient(host="localhost", port=8000)

        # embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # chroma collection
        self.embedding_function = MyEmbeddingFunction(model)
        self.collection = self.client.get_or_create_collection(name=CV_DATA, embedding_function=self.embedding_function)

    def get_cv_docs_raw(self) -> list[CvDataItem]:
        chroma_docs = self.collection.get()
        results: List[CvDataItem] = []
        for id_, doc, metadata in zip(chroma_docs["ids"], chroma_docs["documents"], chroma_docs["metadatas"]):
            results.append(CvDataItem(id=id_, document=doc, metadata=metadata))
        return results

    def delete_cv_data(self):
        try:
            self.client.delete_collection(name=CV_DATA)
        except ValueError:
            log.exception("Failed to delete collection %s", CV_DATA)
            raise  # important: fail fast

    def add_cv_docs(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, str]]) -> None:
        try:
            self.collection = self.client.get_or_create_collection(name=CV_DATA, embedding_function=self.embedding_function)
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        except Exception:
            log.exception("Failed to create/add to collection %s", CV_DATA)
            raise

    def query(self, query: str, result_size: int) -> list[RetrievalResult]:
        # query_texts -> question(s) we wish to answer (chroma embeds it to vector)
        # n_results -> how many results we wish to get
        result_raw = self.collection.query(query_texts=[query], n_results=result_size)

        ids = result_raw["ids"][0]
        distances = result_raw["distances"][0]
        documents = result_raw["documents"][0]
        metadatas = result_raw["metadatas"][0]

        res: List[RetrievalResult] = []
        for id_, distance, document, metadata in zip(ids, distances, documents, metadatas):
            res.append(RetrievalResult(id=id_, distance=distance, document=document, metadata=metadata))

        return res
