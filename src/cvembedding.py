from collections import deque

import chromadb
from Tools.scripts.ndiff import fail
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from cv_splitter import CvSplitter
from models import CVNode, CVNodeLevel, ChatResponse


# wrapper matching new Chroma interface
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: SentenceTransformer):
        super().__init__()
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return self.model.encode(input).tolist()


def to_chroma_documents(root_node: CVNode) -> list:
    chroma_docs = []
    nodes_to_process = deque([root_node])

    while len(nodes_to_process) > 0:
        current_node = nodes_to_process.popleft()
        if len(current_node.text.strip()) > 0:
            doc_content = ""
            metadata = {
                "path": current_node.get_path(),
                "doc_type": "cv"
            }
            if current_node.level == CVNodeLevel.ROOT:  # this means root node
                doc_content = "Document: " + current_node.title
            elif current_node.level == CVNodeLevel.SECTION:  # this means section (middle node)
                doc_content = "Document: " + current_node.parent.title + "\n" + "Section: " + current_node.title
                metadata["section"] = current_node.title
            elif current_node.level == CVNodeLevel.SUBSECTION:  # this means subsection (3rd level node)
                doc_content = "Document: " + current_node.parent.parent.title + "\n" + "Section: " + current_node.parent.title + "\n" + "Subsection: " + current_node.title
                metadata["section"] = current_node.parent.title
                metadata["subsection"] = current_node.title
            else:
                fail(f"should not be here. Node {current_node.id} is invalid")

            doc_content += "\n\n" + current_node.text.strip()  # content of the document itself

            chroma_doc = {
                "id": current_node.id,
                "document": doc_content,
                "metadata": metadata
            }
            chroma_docs.append(chroma_doc)

            if len(current_node.children) > 0:
                nodes_to_process.extend(current_node.children)

    return chroma_docs


class CVEmbedding:
    def __init__(self) -> None:
        # chroma collection
        self.collection = None

        # in-memory chroma
        self.client = chromadb.EphemeralClient()

        # embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.embedding_function = MyEmbeddingFunction(self.model)

    def perform_embeddings(self) -> None:
        with open("cv/Extended_CV.md", "r", encoding="utf-8") as f:
            content = f.read()

        cv_splitter = CvSplitter(content)
        root = cv_splitter.build_doc_tree()
        chroma_documents = to_chroma_documents(root)

        ids = []
        documents = []
        metadatas = []

        for doc in chroma_documents:
            ids.append(doc["id"])
            documents.append(doc["document"])
            metadatas.append(doc["metadata"])

        self.collection = self.client.create_collection(name="cv_data", embedding_function=self.embedding_function)
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def perform_query(self, query) -> ChatResponse:
        result_raw = self.collection.query(
            query_texts=[query],  # Chroma will embed this for you
            n_results=3  # how many results to return
        )
        return ChatResponse(documents=result_raw["documents"][0], distances=result_raw["distances"][0])

    def get_docs_raw(self) -> list:
        chroma_docs = self.collection.get()
        results = []
        for i, doc in enumerate(chroma_docs["documents"]):
            results.append({"id": chroma_docs["ids"][i], "metadata": chroma_docs["metadatas"][i], "document": doc})
        return results