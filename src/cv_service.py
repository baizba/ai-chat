from collections import deque

import chromadb
import structlog
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from cv_parser import CVParser
from models import CVNode, CVNodeLevel, ChatResponse, VectorSearchResult

N_RESULTS = 3
CV_DATA = "cv_data"

log = structlog.get_logger()


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
    nodes_to_process = deque(root_node.children)  # skip the root node completely as it does not give anything meaningful

    while len(nodes_to_process) > 0:
        current_node = nodes_to_process.popleft()
        if len(current_node.text.strip()) > 0:
            doc_content = ""
            metadata = {
                "path": current_node.get_path(),
                "doc_type": "cv"
            }

            current_node_title_clean = current_node.title.lstrip("#").strip()
            if current_node.level == CVNodeLevel.SECTION:  # this means section (middle node)
                doc_content = "Section: " + current_node_title_clean
                metadata["section"] = current_node_title_clean
            elif current_node.level == CVNodeLevel.SUBSECTION:  # this means subsection (3rd level node)
                parent_title_clean = current_node.parent.title.lstrip("#").strip()
                doc_content = "Section: " + parent_title_clean + "\n" + "Subsection: " + current_node_title_clean
                metadata["section"] = parent_title_clean
                metadata["subsection"] = current_node_title_clean
            else:
                raise RuntimeError(f"should not be here. Node {current_node.id} is invalid")

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


class CVService:
    def __init__(self) -> None:
        # chroma connection
        self.client = chromadb.HttpClient(host="localhost", port=8100)

        # embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # chroma collection
        self.embedding_function = MyEmbeddingFunction(model)
        self.collection = self.client.get_or_create_collection(name=CV_DATA, embedding_function=self.embedding_function)

    # here we embed the cv into chroma
    def index_cv(self) -> None:
        with open("cv/Extended_CV.md", "r", encoding="utf-8") as f:
            content = f.read()

        cv_splitter = CVParser(content)
        root = cv_splitter.build_tree()
        chroma_documents = to_chroma_documents(root)

        ids = []
        documents = []
        metadatas = []

        for doc in chroma_documents:
            ids.append(doc["id"])
            documents.append(doc["document"])
            metadatas.append(doc["metadata"])

        try:
            self.client.delete_collection(name=CV_DATA)
        except ValueError:
            log.exception("Failed to delete collection %s", CV_DATA)
            raise  # important: fail fast

        try:
            self.collection = self.client.get_or_create_collection(name=CV_DATA, embedding_function=self.embedding_function)
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        except Exception:
            log.exception("Failed to create/add to collection %s", CV_DATA)
            raise

    def query(self, query, request_id) -> VectorSearchResult:
        result_raw = self.collection.query(
            query_texts=[query],  # Chroma will embed this
            n_results=N_RESULTS  # how many results to return
        )

        paths = []
        for md in result_raw["metadatas"][0]:
            paths.append(md["path"])

        log.info(
            "rag.retrieval",
            request_id=request_id,
            query=query,
            top_k=N_RESULTS,
            doc_ids=result_raw["ids"],
            distances=result_raw["distances"][0],
            path=paths
        )

        return VectorSearchResult(documents=result_raw["documents"][0], distances=result_raw["distances"][0])

    def get_docs_raw(self) -> list:
        chroma_docs = self.collection.get()
        results = []
        for i, doc in enumerate(chroma_docs["documents"]):
            results.append({"id": chroma_docs["ids"][i], "metadata": chroma_docs["metadatas"][i], "document": doc})
        return results
