from collections import deque

from ai_chat.indexing.cv_parser import CVParser
from ai_chat.indexing.models import CVNode, CVNodeLevel
from ai_chat.vectordb.cv_repository import CvRepository


def to_chroma_documents(root_node: CVNode) -> list:
    chroma_docs = []
    nodes_to_process = deque(root_node.children)  # skip the root node completely as it does not give anything meaningful

    while len(nodes_to_process) > 0:
        current_node = nodes_to_process.popleft()
        if len(current_node.text.strip()) > 0:
            metadata = {
                "path": current_node.get_path()
            }

            current_node_title_clean = current_node.title.lstrip("#").strip()
            if current_node.level == CVNodeLevel.SECTION:  # this means section (middle node)
                section = current_node_title_clean
                doc_content = "Section: " + section
                metadata["section"] = section
            elif current_node.level == CVNodeLevel.SUBSECTION:  # this means subsection (3rd level node)
                section = current_node.parent.title.lstrip("#").strip()
                sub_section = current_node_title_clean
                doc_content = "Section: " + section + "\n" + "Subsection: " + sub_section
                metadata["section"] = section
                if section.lower() == "experience":
                    metadata["entityType"] = "employment"
                    metadata["company"] = sub_section
                elif section.lower() == "technical skills":
                    metadata["entityType"] = "skills"
                    metadata["category"] = sub_section
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


class CvIndexingService:
    def __init__(self, repository: CvRepository) -> None:
        self.repository = repository

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

        self.repository.delete_cv_data()
        self.repository.add_cv_docs(ids, documents, metadatas)
