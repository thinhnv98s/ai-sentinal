import unittest

from config import get_config
from data_providers import DataScoutAgent, MarketData


class MockMarketProvider:
    def get_market_data(self, symbol: str, period: str = "6mo") -> MarketData:
        prices = [100 + i * 0.4 for i in range(70)]
        prices[-1] = prices[-2] * 1.03
        volumes = [1_000_000 for _ in range(70)]
        volumes[-1] = 2_200_000
        return MarketData(
            symbol=symbol,
            prices=prices,
            highs=[p * 1.01 for p in prices],
            lows=[p * 0.99 for p in prices],
            volumes=volumes,
            dates=[f"2026-01-{(i % 28) + 1:02d}" for i in range(70)],
            eps_growth=0.25,
            annual_eps_growth=0.30,
            roe=0.22,
            roic=0.15,
            pe_ratio=27,
            market_cap=2_000_000_000_000,
            industry="Internet Retail",
            source="mock",
        )

    def get_index_data(self, index: str = "SPY") -> MarketData:
        return self.get_market_data(index, period="1y")


class MockTavily:
    def search_news(self, **kwargs):
        return {
            "query": kwargs.get("query", ""),
            "answer": "Amazon faced a capex debate while AWS demand stayed robust.",
            "results": [
                {
                    "title": "Amazon expands AI infrastructure spending",
                    "url": "https://reuters.com/mock/amazon-capex",
                    "content": "Amazon plans higher capex while reiterating demand visibility in cloud and chips.",
                    "published_at": "2026-02-24T08:30:00",
                },
                {
                    "title": "AWS backlog remains strong",
                    "url": "https://seekingalpha.com/mock/aws-backlog",
                    "content": "Commentary suggests multi-year demand and capacity constraints...",
                    "published_at": "2026-02-24T09:30:00",
                },
                {
                    "title": "Street split on valuation",
                    "url": "https://bloomberg.com/mock/amzn-valuation",
                    "content": "Analysts debate whether capex pressure is transitory or structural.",
                    "published_at": "2026-02-23T22:00:00",
                },
            ],
        }

    def search_company_deep(self, company, domains=None):
        return {
            "query": f"{company} moat",
            "answer": "Amazon has strong logistics, ad flywheel, and AWS ecosystem leverage.",
            "results": [],
        }

    def extract(self, urls, extract_depth="advanced"):
        return {
            "results": [
                {
                    "url": "https://reuters.com/mock/amazon-capex",
                    "content": (
                        "Amazon detailed capacity expansion, data center buildout, and chip roadmap. "
                        "Management said utilization is expected to stay high through next year. "
                        "Retail margins also improved on logistics efficiency."
                    ),
                },
                {
                    "url": "https://seekingalpha.com/mock/aws-backlog",
                    "content": (
                        "AWS backlog growth remained resilient and management expects demand to outpace "
                        "available supply in near term. Monetization from prior capex is expected to "
                        "improve operating leverage."
                    ),
                },
            ]
        }


class DocumentPipelineTest(unittest.TestCase):
    def setUp(self):
        cfg = get_config()
        cfg.document_intel.enable_two_tier_pipeline = True
        cfg.document_intel.deep_dive_enabled = True
        cfg.document_intel.deep_dive_top_k = 3
        cfg.document_intel.deep_dive_min_priority = 0.40
        cfg.document_intel.deep_dive_min_uncertainty = 0.30
        cfg.document_intel.deep_dive_token_budget = 6000
        cfg.document_intel.enable_tavily_extract = True
        cfg.document_intel.enable_jina_reader_fallback = False
        cfg.document_intel.rag_enabled = True
        cfg.document_intel.rag_top_chunks = 5

    def test_two_tier_pipeline_outputs_document_intel(self):
        scout = DataScoutAgent(tavily_client=MockTavily(), market_provider=MockMarketProvider())
        payload = scout.fetch_all_data("AMZN", include_news=True)

        self.assertIn("document_intel", payload)
        doc_intel = payload["document_intel"]
        self.assertIsInstance(doc_intel, dict)
        self.assertGreater(doc_intel.get("quality", {}).get("radar_count", 0), 0)
        self.assertGreater(doc_intel.get("quality", {}).get("deep_dive_count", 0), 0)
        self.assertGreater(len(doc_intel.get("evidence_chunks", [])), 0)

        deep_docs = doc_intel.get("deep_dive_docs", [])
        self.assertTrue(any(doc.get("content_mode") == "full" for doc in deep_docs))
        self.assertIn("high_priority_docs", payload.get("event_flags", {}))

    def test_company_research_is_enriched_with_deep_dive(self):
        scout = DataScoutAgent(tavily_client=MockTavily(), market_provider=MockMarketProvider())
        payload = scout.fetch_all_data("AMZN", include_news=True)
        research = payload.get("company_research", {})
        synthetic = [
            item for item in research.get("results", [])
            if item.get("source") == "document_intel_deep_dive"
        ]
        self.assertGreater(len(synthetic), 0)


if __name__ == "__main__":
    unittest.main()
