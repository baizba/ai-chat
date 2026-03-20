import re

import structlog

from ai_chat.vectordb.cv_repository import CvRepository
from ai_chat.vectordb.models import RetrievalResult

log = structlog.get_logger()


def extract_employment_period(metadata: dict) -> tuple[int, int]:
    from_year = metadata.get("fromYear")
    to_year = metadata.get("toYear")
    year_from_int = int(from_year) if from_year else None
    year_to_int = int(to_year) if to_year else float("inf")
    return year_from_int, year_to_int


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
            return self.get_experience_by_year_range(years_int[0], years_int[1])
        elif len(years_int) == 1:
            return self.get_experience_by_single_year(years_int[0])
        else:
            return self.get_experience_list()

    def get_experience_by_year_range(self, start_year: int, end_year: int) -> str:
        result = self.query_employment()

        employment_set = set()
        for id_, doc, meta in result:
            year_from_int, year_to_int = extract_employment_period(meta)

            # something is wrong in this case (employment without valid start)
            if year_from_int is None:
                continue

            # in this case we have complete employment range
            if year_from_int <= end_year and year_to_int >= start_year:
                employment_set.add(meta["company"])

        if len(employment_set) > 0:
            return f"Branislav worked in: {', '.join(employment_set)}"

        return f"No employment found for period from {start_year} to {end_year}"

    def get_experience_by_single_year(self, year: int) -> str:
        result = self.query_employment()

        employment_set = set()
        for id_, doc, meta in result:
            year_from_int, year_to_int = extract_employment_period(meta)

            # something is wrong in this case (employment without valid start)
            if year_from_int is None:
                continue

            if year_from_int <= year <= year_to_int:
                employment_set.add(meta["company"])

        if len(employment_set) > 0:
            return f"Branislav worked in: {', '.join(employment_set)}"

        return f"No employment found for year {year}"

    def get_experience_list(self) -> str:
        result = self.query_employment()
        exp_list = [r.metadata["company"] for r in result]
        return f"Branislav worked in: {', '.join(exp_list)}"

    def query_employment(self) -> list[RetrievalResult]:
        metadata_employment = {"entityType": "employment"}
        result = self.repository.metadata_query(metadata_employment)
        log.info(
            "cv.retrieval.metadata.employment",
            query=metadata_employment,
            doc_ids=[r.id for r in result],
            path=[r.metadata["path"] for r in result]
        )
        return result
