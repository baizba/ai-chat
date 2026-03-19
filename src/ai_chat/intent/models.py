from enum import Enum


class Domain(Enum):
    EMPLOYMENT = "employment"
    SKILLS = "skills"
    PROFILE = "profile"
    PROJECTS = "projects"


class QueryType(Enum):
    FACT = "fact"
    LIST = "list"
    SUMMARY = "summary"
