class VectorSearchResult:
    def __init__(self, documents: list[str], distances: list[float]):
        self.documents = documents
        self.distances = distances
