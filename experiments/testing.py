import re

from ai_chat.intent.intent_classifier import IntentClassifier
from ai_chat.llm.llm_service import LLMService
from ai_chat.router.query_router import QueryRouter
from ai_chat.service.employment_service import EmploymentService
from ai_chat.service.profile_service import ProfileService
from ai_chat.service.skills_service import SkillsService
from ai_chat.vectordb.cv_repository import CvRepository


def test_employment_service():
    employment_service = EmploymentService(CvRepository())
    experience_list = employment_service.handle("List all companies where he worked.")
    print(experience_list)

def test_index_intent():
    IntentClassifier().index_intents()

def test_skills_service():
    skills_service = SkillsService(CvRepository(), LLMService())
    answer = skills_service.handle("Did he ever use test driven design (TDD)?")
    print(answer)

def test_profile_service():
    profile_service = ProfileService(CvRepository(), LLMService())
    answer = profile_service.handle("Tell me something about Johny Bravo")
    print(answer)

def test_query_router():
    query_router = QueryRouter(CvRepository(), LLMService())
    result = query_router.route_query("Give me all his certifications.")
    print(result)


# test_index_intent()
# test_skills_service()
# test_profile_service()
test_query_router()

# pattern = re.compile(r"\bperiod of employment\b\W+([A-Za-z]+\s+\d{4}\s*[-–—]\s*[A-Za-z]+\s+\d{4})", re.IGNORECASE)
# res = pattern.findall("**Period of employment:** March 2017 – September 2020")
# pattern = re.compile(r"\bperiod of employment\b\W+since\s([A-Za-z]+\s+\d{4})", re.IGNORECASE)
# res = pattern.findall("**Period of employment:** since March 2021 (ongoing employment)")
# print(res)
