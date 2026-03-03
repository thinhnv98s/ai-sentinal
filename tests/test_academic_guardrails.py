import unittest

from validators import VERAFIValidator


class AcademicGuardrailsTest(unittest.TestCase):
    def setUp(self):
        self.validator = VERAFIValidator()

    def test_news_material_requires_sources(self):
        output = {
            "is_material": True,
            "sources": [],
            "confidence": 0.85,
            "sentiment_score": 0.4,
        }
        result = self.validator.validate("news", output, {})
        self.assertFalse(result.is_valid)
        self.assertTrue(any("sources" in violation.lower() for violation in result.violations))
        self.assertGreater(result.confidence_penalty, 0.0)

    def test_canslim_eps_rule_still_enforced(self):
        output = {
            "scores": {"C": 12, "A": 10, "N": 8, "S": 8, "L": 8, "I": 8, "M": 8},
            "total_score": 62,
            "signal": "BUY",
            "confidence": 0.8,
        }
        actual_data = {
            "eps_growth": 0.10,      # < 18%
            "annual_eps_growth": 0.30,
            "roe": 0.22,
            "rs_rating": 90,
        }
        result = self.validator.validate("canslim", output, actual_data)
        self.assertFalse(result.is_valid)
        self.assertIn("scores", result.adjustments)
        self.assertLessEqual(result.adjustments["scores"].get("C", 99), 7)


if __name__ == "__main__":
    unittest.main()
