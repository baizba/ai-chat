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
            return self.get_employment_by_year_range(years_int[0], years_int[1])
        elif len(years_int) == 1:
            return self.get_employment_by_single_year(years_int[0])
        else:
            return self.get_employment_by_company_or_list(question)

    def get_employment_by_year_range(self, start_year: int, end_year: int) -> str:
        result = self.query_employment()

        employment_set = set()
        for r in result:
            year_from_int, year_to_int = extract_employment_period(r.metadata)

            # something is wrong in this case (employment without valid start)
            if year_from_int is None:
                continue

            # in this case we have complete employment range
            if year_from_int <= end_year and year_to_int >= start_year:
                employment_set.add(r.metadata["company"])

        if len(employment_set) > 0:
            return f"Branislav worked in: {', '.join(employment_set)}"

        return f"No employment found for period from {start_year} to {end_year}"

    def get_employment_by_single_year(self, year: int) -> str:
        result = self.query_employment()

        employment_set = set()
        for r in result:
            year_from_int, year_to_int = extract_employment_period(r.metadata)

            # something is wrong in this case (employment without valid start)
            if year_from_int is None:
                continue

            if year_from_int <= year <= year_to_int:
                employment_set.add(r.metadata["company"])

        if len(employment_set) > 0:
            return f"Branislav worked in: {', '.join(employment_set)}"

        return f"No employment found for year {year}"

    def get_employment_by_company_or_list(self, question: str) -> str:
        result = self.query_employment()
        question_normalized = question.lower()
        question_tokens = re.split(r"\W+", question_normalized)

        matched_companies = []
        seen = set() # used only to avoid possible duplicates in companies

        # check if user asked for company by full name or by alias
        for r in result:
            company_normalized = r.metadata["company"].lower()
            if company_normalized in question_normalized and company_normalized not in seen:
                matched_companies.append(r)
                seen.add(company_normalized)
                continue # if we matched company by the name then just move to the next company
            for alias in r.metadata["aliases"].split(","):
                if alias in question_tokens and company_normalized not in seen:
                    matched_companies.append(r)
                    break # avoid matching one company multiple times

        if len(matched_companies) > 0:
            # here goes real llm call
            return f"Explain this LLM: {matched_companies}"

        # default return only the list of employments
        return self.list_employments()

    def list_employments(self) -> str:
        result = self.query_employment()
        companies = [r.metadata["company"] for r in result]
        return f"Branislav worked in: {', '.join(companies)}"

    def query_employment(self) -> list[RetrievalResult]:
        metadata_employment = {"entityType": "employment"}
        employment = self.repository.metadata_query(metadata_employment)
        log.info(
            "cv.retrieval.metadata.employment",
            query=metadata_employment,
            doc_ids=[r.id for r in employment],
            path=[r.metadata["path"] for r in employment]
        )
        return employment
