import structlog

from src.models import VectorSearchResult
from src.vectordb.cv_repository import CvRepository
from src.vectordb.models import RetrievalResult

PRECISION_GAP = 0.15
N_RESULTS = 3

log = structlog.get_logger()


# filter out docs if the distance is too large from the first doc (easy way to improve contexts)
def filter_by_base_distance(results: list[RetrievalResult]) -> list[RetrievalResult]:
    if not results:
        return []

    filtered_results: list[RetrievalResult] = []
    base_distance = results[0].distance
    for res in results:
        if res.distance - base_distance < PRECISION_GAP:
            filtered_results.append(res)
    return filtered_results


def calc_separation_from_first(distances: list[float]) -> list[float]:
    result = []
    base_distance = distances[0]
    for dist in distances[1:]:
        result.append(round(dist - base_distance, 5))
    return result


class CvService:
    def __init__(self, repository: CvRepository) -> None:
        self.cv_repository = repository

    def query(self, query, request_id) -> VectorSearchResult:
        results = self.cv_repository.query(query, N_RESULTS)

        filtered_results = filter_by_base_distance(results)
        # paths = [md["path"] for md in filtered_metadatas]
        paths = [r.metadata["path"] for r in filtered_results]
        filtered_distances = [r.distance for r in filtered_results]
        filtered_documents = [r.document for r in filtered_results]

        ids = [r.id for r in results]
        distances = [r.distance for r in results]
        log.info(
            "rag.retrieval",
            request_id=request_id,
            query=query,
            top_k=N_RESULTS,
            doc_ids=ids,
            distances=distances,
            path=paths,
            kept=len(filtered_documents),
            separations=calc_separation_from_first(distances)
        )

        return VectorSearchResult(documents=filtered_documents, distances=filtered_distances)

    def get_docs_raw(self) -> list:
        return self.cv_repository.get_cv_docs_raw()
