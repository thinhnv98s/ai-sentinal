import unittest

from blackboard import Blackboard
from orchestrator import BayesianResolver, PortfolioManager


class BayesianDirectionalityTest(unittest.TestCase):
    def test_sell_signals_do_not_increase_buy_odds(self):
        bb = Blackboard()
        resolver = BayesianResolver(bb)

        signals = {
            "wyckoff": {"signal": "HOLD", "confidence": 0.50},
            "canslim": {"signal": "SELL", "confidence": 0.60},
            "fourm": {"signal": "SELL", "confidence": 0.32},
            "news": {"signal": "HOLD", "confidence": 0.20},
        }
        result = resolver.resolve("TSLA", signals)

        # Không được ra >50% trong case chỉ có HOLD/SELL
        self.assertLess(result["final_probability"], 0.5)

        # LR cho SELL phải không vượt 1 (không đảo chiều thành tín hiệu BUY)
        self.assertLessEqual(result["likelihood_ratios"]["canslim"], 1.0)
        self.assertLessEqual(result["likelihood_ratios"]["fourm"], 1.0)

    def test_guardrail_blocks_buy_without_any_bullish_signal(self):
        bb = Blackboard()
        pm = PortfolioManager(bb, llm_client=None, portfolio_value=100000)
        bayesian_result = {"final_probability": 0.555, "decision": "EXECUTE_TRADE"}
        analyses = {
            "wyckoff": {"signal": "HOLD", "confidence": 0.5},
            "canslim": {"signal": "SELL", "confidence": 0.6},
            "fourm": {"signal": "SELL", "confidence": 0.32},
            "news": {"signal": "HOLD", "confidence": 0.2},
        }

        pm._apply_directional_guardrail("TSLA", analyses, bayesian_result)
        self.assertEqual(bayesian_result["decision"], "WATCH")
        self.assertTrue(bayesian_result.get("guardrail_triggered", False))


if __name__ == "__main__":
    unittest.main()
