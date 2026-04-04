import re
from datetime import date

import structlog

from ai_chat.llm.llm_service import LLMService
from ai_chat.service import prompts
from ai_chat.vectordb.cv_repository import CvRepository
from ai_chat.vectordb.models import RetrievalResult

log = structlog.get_logger()


def extract_employment_period(metadata: dict) -> tuple[int, int]:
    from_year: str = metadata.get("fromYear")
    to_year: str = metadata.get("toYear")
    year_from_int = int(from_year) if from_year else None
    year_to_int = int(to_year) if to_year else float("inf")
    return year_from_int, year_to_int


def build_employment_context(employment: RetrievalResult) -> str:
    company = employment.metadata["company"]
    aliases = employment.metadata["aliases"]
    role = employment.metadata["role"]
    description = employment.document
    return f"Company: {company} (Aliases: {aliases}) Role: {role} Description: {description}"


class EmploymentService:
    def __init__(self, repository: CvRepository, llm_service: LLMService) -> None:
        self.repository = repository
        self.llm_service = llm_service

    def handle(self, question: str) -> str:
        years = re.compile(r"\b(?:19|20)\d{2}\b").findall(question)
        years_int = [int(y) for y in years]
        years_int.sort()

        log.info("employment.range", years=years_int)

        for ref_year in years_int:
            if ref_year > date.today().year:
                return f"Can not ask question for the future: {ref_year}"

        if len(years_int) == 2:
            return self.get_employment_by_year_range(years_int[0], years_int[1])
        elif len(years_int) == 1:
            return self.get_employment_by_single_year(years_int[0], question)
        else:
            return self.get_employment_by_company_or_list(question)

    def get_employment_by_year_range(self, start_year: int, end_year: int) -> str:
        result = self.query_employment()

        employment_list = [] # do not use set to make ordering constant
        for r in result:
            year_from_int, year_to_int = extract_employment_period(r.metadata)

            # something is wrong in this case (employment without valid start)
            if year_from_int is None:
                continue

            # in this case we have complete employment range
            company = r.metadata["company"]
            if year_from_int <= end_year and year_to_int >= start_year and company not in employment_list:
                employment_list.append(company)

        if len(employment_list) > 0:
            return f"Branislav worked in: {', '.join(employment_list)}"

        return f"No employment found for period from {start_year} to {end_year}"

    def get_employment_by_single_year(self, year_ref: int, question: str) -> str:
        result = self.query_employment()
        question_normalized = question.lower()
        before = "before" in question_normalized
        after = "after" in question_normalized

        employment_list = [] # use list ot keep ordering constant
        for r in result:
            year_from_int, year_to_int = extract_employment_period(r.metadata)

            # something is wrong in this case (employment without valid start)
            if year_from_int is None:
                continue

            employment = r.metadata["company"]
            if before and year_from_int < year_ref and employment not in employment_list:
                employment_list.append(employment)
            elif after and year_to_int > year_ref and employment not in employment_list:
                employment_list.append(employment)
            elif year_from_int <= year_ref <= year_to_int and employment not in employment_list:
                employment_list.append(employment)
            else:
                log.error("employment.range.unknown", year_ref=year_ref, year_from=year_from_int, year_to=year_to_int, question=question)

        if len(employment_list) > 0:
            return f"Branislav worked in: {', '.join(employment_list)}"

        return f"No employment found for year {year_ref}"

    def get_employment_by_company_or_list(self, question: str) -> str:
        result = self.query_employment()
        question_normalized = question.lower()
        question_tokens = re.split(r"\W+", question_normalized)

        matched_employments: list[RetrievalResult] = []
        seen = set()  # used only to avoid possible duplicates in companies

        # check if user asked for company by full name or by alias
        for r in result:
            company_normalized = r.metadata["company"].lower()
            if company_normalized in question_normalized and company_normalized not in seen:
                matched_employments.append(r)
                seen.add(company_normalized)
                continue  # if we matched company by the name then just move to the next company
            for alias in r.metadata["aliases"].split(","):
                if alias in question_tokens and company_normalized not in seen:
                    matched_employments.append(r)
                    seen.add(company_normalized)
                    break  # avoid matching one company multiple times


        if len(matched_employments) > 0:
            log.info("employment.list_employments", companies=[e.metadata["company"] for e in matched_employments])
            # here goes real llm call
            answers = []
            for employment in matched_employments:
                context = build_employment_context(employment)
                question_single_employment = f"What did he do in {employment.metadata['company']}?"
                answer = self.llm_service.answer(prompts.company_role_prompt, question_single_employment, context)
                answers.append(answer)
            return "\n".join(answers)

        # default return only the list of employments
        return self.list_employments()

    def list_employments(self) -> str:
        result = self.query_employment()
        companies: list = [r.metadata["company"] for r in result]
        log.info("employment.list_employments", companies=companies)
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
