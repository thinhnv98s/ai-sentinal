import unittest

from data_providers import TavilyClient
from validators import FactCheckPipeline


class QueryLengthGuardsTest(unittest.TestCase):
    def test_tavily_query_is_sanitized_to_400_chars(self):
        client = TavilyClient(api_key="")
        long_query = "AMZN " + ("capex guidance aws trainium backlog " * 40)
        safe_query = client._sanitize_query(long_query)
        self.assertLessEqual(len(safe_query), 400)
        self.assertTrue(safe_query.startswith("AMZN"))

    def test_factcheck_query_builder_respects_limit(self):
        pipeline = FactCheckPipeline(tavily_client=None)
        claim = (
            "Amazon announced multiple infrastructure investments, future capacity constraints, "
            "new chip roadmap, margin inflection thesis, and reiterated revenue outlook "
            "across retail advertising and cloud services with additional commentary "
            "from management on timeline and monetization impact."
        ) * 3
        query = pipeline._build_verification_query("AMZN", claim, "event")
        self.assertLessEqual(len(query), 400)
        self.assertIn("AMZN", query)
        self.assertIn("event", query)


if __name__ == "__main__":
    unittest.main()
