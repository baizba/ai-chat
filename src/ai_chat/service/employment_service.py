import re

import structlog

from ai_chat.vectordb.cv_repository import CvRepository

log = structlog.get_logger()


class EmploymentService:
    def __init__(self, repository: CvRepository):
        self.repository = repository

    def handle(self, question: str) -> str:
        pattern = re.compile(r"\b(?:19|20)\d{2}\b")
        years = pattern.findall(question)
        years_int = [int(y) for y in years]
        years_int.sort()

        log.info("employment.range", years=years_int)

        if len(years_int) == 2:
            return self.get_experience_range(years_int[0], years_int[1])
        elif len(years_int) == 1:
            return self.get_experience_single(years_int[0])
        else:
            return self.get_experience_list()

    def get_experience_range(self, from_: int, to_: int) -> str:
        pass

    def get_experience_single(self, year: int) -> str:
        pass

    def get_experience_list(self) -> str:
        metadata_employment = {"entityType": "employment"}
        result = self.repository.metadata_query(metadata_employment)

        log.info(
            "cv.retrieval.metadata",
            query={"entityType": "employment"},
            doc_ids=[r.id for r in result],
            path=[r.metadata["path"] for r in result]
        )

        exp_list = [r.metadata["company"] for r in result]
        return f"Branislav worked in: {', '.join(exp_list)}"
