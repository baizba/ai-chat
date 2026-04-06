from chromadb import EmbeddingFunction, Embeddings, Documents
from sentence_transformers import SentenceTransformer


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: SentenceTransformer):
        super().__init__()
        self.model = model

    def __call__(self, docs: Documents) -> Embeddings:
        # embed the documents somehow
        return self.model.encode(docs).tolist()