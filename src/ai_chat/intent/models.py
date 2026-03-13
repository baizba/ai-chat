from enum import Enum


class Domain(Enum):
    EMPLOYMENT = "employment"
    SKILLS = "skills"
    PROFILE = "profile"
    PROJECTS = "projects"

DOMAIN_KEYWORDS = {
    Domain.EMPLOYMENT: [
        "work",
        "worked",
        "employed",
        "company",
        "job",
        "career"
    ],
    Domain.SKILLS: [
        "skill",
        "technology",
        "tech",
        "know",
        "experience with"
    ],
    Domain.PROFILE: [
        "profile",
        "who is",
        "about"
    ],
    Domain.PROJECTS: [
        "project",
        "system",
        "built"
    ]
}


class QueryType(Enum):
    FACT = "fact"
    LIST = "list"
    SUMMARY = "summary"
    EXPLANATION = "explanation"

QUERY_KEYWORDS = {
    QueryType.LIST: [
        "list",
        "all",
        "which"
    ],
    QueryType.SUMMARY: [
        "summary",
        "summarize"
    ],
    QueryType.EXPLANATION: [
        "explain",
        "describe"
    ]
}