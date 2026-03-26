from enum import Enum


class Domain(Enum):
    EMPLOYMENT = "employment"
    SKILLS = "skills"
    PROFILE = "profile"
    EDUCATION = "education"
    CERTIFICATES = "certificates"


class QueryType(Enum):
    FACT = "fact"
    LIST = "list"
    SUMMARY = "summary"
