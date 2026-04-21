import unittest

from ai_chat.intent.intent_classifier import IntentClassifier
from ai_chat.llm.llm_service import LLMService
from ai_chat.router.query_router import QueryRouter
from ai_chat.vectordb.cv_repository import CvRepository

ALL_COMPANIES = ["JIT-Dienstleistungs", "ecx.io", "Netconomy", "Codecentric"]


class RagIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.query_router = QueryRouter(LLMService(), CvRepository(), IntentClassifier())

    def test_employment_all(self):
        all_employments = "JIT-Dienstleistungs GmbH, ecx.io - IBM Company, Netconomy GmbH, Codecentric doo"

        response = self.query_router.route_query("What is his employment history?")
        self.assertIn(all_employments, response.answer)

        response = self.query_router.route_query("What companies he worked for?")
        self.assertIn(all_employments, response.answer)

        response = self.query_router.route_query("List his employers.")
        self.assertIn(all_employments, response.answer)

    def test_employment_range(self):
        response = self.query_router.route_query("Where did he work from 2018 to 2022?")
        self.assertIn("JIT-Dienstleistungs GmbH, ecx.io - IBM Company", response.answer)

        response = self.query_router.route_query("Where was he employed between 2015 and 2017?")
        self.assertIn("ecx.io - IBM Company, Netconomy GmbH", response.answer)

        response = self.query_router.route_query("List his employments between 2005 and 2008?")
        self.assertIn("No employment found for period from 2005 to 2008", response.answer)

    def test_employment_single_year(self):
        response = self.query_router.route_query("Where did he work 2022?")
        self.assertIn("JIT-Dienstleistungs GmbH", response.answer)

        response = self.query_router.route_query("Where was he employed in 2010")
        self.assertIn("Codecentric doo", response.answer)

        response = self.query_router.route_query("List his employments between 2005 and 2008?")
        self.assertIn("No employment found for period from 2005 to 2008", response.answer)

    def test_employment_before_after(self):
        response = self.query_router.route_query("Where did he work before 2022?")
        self.assertIn("JIT-Dienstleistungs GmbH, ecx.io - IBM Company, Netconomy GmbH, Codecentric doo", response.answer)

        response = self.query_router.route_query("Where was he employed before 2010")
        self.assertIn("Codecentric doo", response.answer)

        response = self.query_router.route_query("List his employments after 2018?")
        self.assertIn("JIT-Dienstleistungs GmbH, ecx.io - IBM Company", response.answer)

        response = self.query_router.route_query("List his employments after 2050?")
        self.assertIn("Can not ask question for the future: 2050", response.answer)

        response = self.query_router.route_query("List his employments before 2006?")
        self.assertIn("No employment found for year 2006", response.answer)

    def test_employment_questions(self):
        response = self.query_router.route_query("What did he do in netconomy?")
        self.assertIn("Java Developer", response.answer)
        self.assertIn("SAP CX", response.answer)

        response = self.query_router.route_query("What were his duties in ibm?")
        self.assertIn("Senior Java Developer", response.answer)
        self.assertIn("SAP CX", response.answer)

        response = self.query_router.route_query("Did he work in Google?")
        self.assertTrue("could not find the answer" in response.answer, response.answer)
        self.assertTrue(all([cmp in response.answer for cmp in ALL_COMPANIES]), response.answer)


if __name__ == '__main__':
    unittest.main()
