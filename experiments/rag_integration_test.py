import unittest

from ai_chat.llm.llm_service import LLMService
from ai_chat.router.query_router import QueryRouter


class RagIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.query_router = QueryRouter(LLMService())

    def test_employment_all(self):
        all_employments = "JIT-Dienstleistungs GmbH, ecx.io - IBM Company, Netconomy GmbH, Codecentric doo"

        result = self.query_router.route_query("What is his employment history?")
        self.assertIn(all_employments, result)

        result = self.query_router.route_query("What companies he worked for?")
        self.assertIn(all_employments, result)

        result = self.query_router.route_query("List his employers.")
        self.assertIn(all_employments, result)

    def test_employment_range(self):
        result = self.query_router.route_query("Where did he work from 2018 to 2022?")
        self.assertIn("JIT-Dienstleistungs GmbH, ecx.io - IBM Company", result)

        result = self.query_router.route_query("Where was he employed between 2015 and 2017?")
        self.assertIn("ecx.io - IBM Company, Netconomy GmbH", result)

        result = self.query_router.route_query("List his employments between 2005 and 2008?")
        self.assertIn("No employment found for period from 2005 to 2008", result)

    def test_employment_single_year(self):
        result = self.query_router.route_query("Where did he work 2022?")
        self.assertIn("JIT-Dienstleistungs GmbH", result)

        result = self.query_router.route_query("Where was he employed in 2010")
        self.assertIn("Codecentric doo", result)

        result = self.query_router.route_query("List his employments between 2005 and 2008?")
        self.assertIn("No employment found for period from 2005 to 2008", result)

    def test_employment_before_after(self):
        result = self.query_router.route_query("Where did he work before 2022?")
        self.assertIn("JIT-Dienstleistungs GmbH, ecx.io - IBM Company, Netconomy GmbH, Codecentric doo", result)

        result = self.query_router.route_query("Where was he employed before 2010")
        self.assertIn("Codecentric doo", result)

        result = self.query_router.route_query("List his employments after 2018?")
        self.assertIn("JIT-Dienstleistungs GmbH, ecx.io - IBM Company", result)

        result = self.query_router.route_query("List his employments after 2050?")
        self.assertIn("Can not ask question for the future: 2050", result)

        result = self.query_router.route_query("List his employments before 2006?")
        self.assertIn("No employment found for year 2006", result)


if __name__ == '__main__':
    unittest.main()
