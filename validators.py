"""
VERAFI VALIDATOR - Neurosymbolic Verification
=============================================
Kiểm tra đầu ra LLM bằng quy tắc tất định
Tham chiếu: base.txt Section 4.2, 4.3
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from config import get_config

logger = logging.getLogger("Sentinel.VERAFI")


@dataclass
class ValidationResult:
    """Kết quả validation"""
    is_valid: bool
    violations: List[str]
    adjustments: Dict[str, Any]
    confidence_penalty: float
    reasoning: str


class VERAFIValidator:
    """
    VERAFI - Verified Agentic Financial Intelligence

    Tích hợp lớp "thần kinh - ký hiệu", nơi đầu ra của LLM được kiểm tra
    dựa trên các quy tắc tất định.

    Tham chiếu: base.txt Section 4.2
    """

    def __init__(self):
        self.config = get_config()
        self.validation_rules = self._build_rules()

    def _build_rules(self) -> Dict[str, List[callable]]:
        """Xây dựng các quy tắc validation"""
        return {
            "canslim": [
                self._rule_canslim_eps_growth,
                self._rule_canslim_annual_growth,
                self._rule_canslim_roe,
                self._rule_canslim_rs_rating,
                self._rule_canslim_score_consistency,
            ],
            "fourm": [
                self._rule_fourm_mos_math,
                self._rule_fourm_roic,
                self._rule_fourm_score_bounds,
            ],
            "wyckoff": [
                self._rule_wyckoff_phase_consistency,
                self._rule_wyckoff_spring_volume,
            ],
            "news": [
                self._rule_news_sentiment_bounds,
                self._rule_news_source_required,
            ],
            "bayesian": [
                self._rule_bayesian_probability_bounds,
                self._rule_bayesian_odds_consistency,
            ]
        }

    def validate(self,
                 agent_type: str,
                 llm_output: Dict[str, Any],
                 actual_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate LLM output against deterministic rules

        Args:
            agent_type: loại agent (canslim, fourm, wyckoff, news, bayesian)
            llm_output: Output từ LLM
            actual_data: Dữ liệu thực tế để cross-check

        Returns:
            ValidationResult
        """
        violations = []
        adjustments = {}
        confidence_penalty = 0.0

        rules = self.validation_rules.get(agent_type, [])

        for rule in rules:
            try:
                result = rule(llm_output, actual_data)
                if result:
                    violation, adjustment, penalty = result
                    violations.append(violation)
                    if adjustment:
                        adjustments.update(adjustment)
                    confidence_penalty += penalty
            except Exception as e:
                logger.warning(f"Rule execution error: {e}")

        is_valid = len(violations) == 0
        reasoning = f"Validated with {len(rules)} rules. " + \
                   (f"Found {len(violations)} violations." if violations else "All rules passed.")

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            adjustments=adjustments,
            confidence_penalty=min(confidence_penalty, 0.5),  # Cap at 50%
            reasoning=reasoning
        )

    # === CANSLIM RULES ===

    def _rule_canslim_eps_growth(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Nếu EPS Growth < 18%, score C phải <= 7"""
        eps_growth = data.get("eps_growth")
        c_score = output.get("scores", {}).get("C", 0)

        if eps_growth is not None and eps_growth < 0.18 and c_score > 7:
            return (
                f"EPS Growth ({eps_growth:.1%}) < 18% nhưng C score = {c_score} (phải <= 7)",
                {"scores": {"C": min(c_score, 7)}},
                0.1
            )
        return None

    def _rule_canslim_annual_growth(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Nếu Annual Growth < 25%, score A phải <= 8"""
        annual_growth = data.get("annual_eps_growth")
        a_score = output.get("scores", {}).get("A", 0)

        if annual_growth is not None and annual_growth < 0.25 and a_score > 8:
            return (
                f"Annual Growth ({annual_growth:.1%}) < 25% nhưng A score = {a_score}",
                {"scores": {"A": min(a_score, 8)}},
                0.1
            )
        return None

    def _rule_canslim_roe(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Nếu ROE < 17%, phải phản ánh trong A score"""
        roe = data.get("roe")
        a_score = output.get("scores", {}).get("A", 0)

        if roe is not None and roe < 0.17 and a_score > 10:
            return (
                f"ROE ({roe:.1%}) < 17% nhưng A score = {a_score} (phải xem xét giảm)",
                None,
                0.05
            )
        return None

    def _rule_canslim_rs_rating(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Nếu RS < 80, score L phải <= 7"""
        rs = data.get("rs_rating")
        l_score = output.get("scores", {}).get("L", 0)

        if rs is not None and rs < 80 and l_score > 10:
            return (
                f"RS Rating ({rs}) < 80 nhưng L score = {l_score} (Laggard)",
                {"scores": {"L": min(l_score, 7)}},
                0.1
            )
        return None

    def _rule_canslim_score_consistency(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Tổng điểm phải bằng tổng các tiêu chí"""
        scores = output.get("scores", {})
        total = output.get("total_score", 0)
        calculated = sum(scores.values())

        if abs(total - calculated) > 1:
            return (
                f"Total score ({total}) không khớp tổng tiêu chí ({calculated})",
                {"total_score": calculated},
                0.05
            )
        return None

    # === 4M RULES ===

    def _rule_fourm_mos_math(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Verify MOS calculation math"""
        mos = output.get("mos_analysis", {})
        sticker = mos.get("sticker_price")
        buy = mos.get("buy_price")

        if sticker and buy:
            expected_buy = sticker * self.config.fourm.mos_discount
            if abs(buy - expected_buy) > expected_buy * 0.1:  # 10% tolerance
                return (
                    f"Buy price ({buy}) không đúng với MOS 50% của Sticker ({sticker})",
                    {"mos_analysis": {"buy_price": round(expected_buy, 2)}},
                    0.1
                )
        return None

    def _rule_fourm_roic(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Nếu ROIC < 10%, Moat score phải thấp"""
        roic = data.get("roic") or data.get("roe")
        moat_score = output.get("scores", {}).get("moat", 0)

        if roic is not None and roic < 0.10 and moat_score > 15:
            return (
                f"ROIC ({roic:.1%}) < 10% nhưng Moat score = {moat_score} (nên <= 15)",
                {"scores": {"moat": min(moat_score, 15)}},
                0.15
            )
        return None

    def _rule_fourm_score_bounds(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Mỗi M score phải trong khoảng 0-25"""
        scores = output.get("scores", {})
        violations = []

        for key in ["meaning", "moat", "management"]:
            score = scores.get(key, 0)
            if score < 0 or score > 25:
                violations.append(f"{key}={score}")

        if violations:
            return (
                f"Score out of bounds (0-25): {', '.join(violations)}",
                None,
                0.1
            )
        return None

    # === WYCKOFF RULES ===

    def _rule_wyckoff_phase_consistency(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Phase và Signal phải nhất quán"""
        phase = output.get("phase", "")
        signal = output.get("signal", "")

        inconsistent = [
            (phase == "DISTRIBUTION" and signal in ["STRONG_BUY", "BUY"]),
            (phase == "MARKDOWN" and signal in ["STRONG_BUY", "BUY"]),
            (phase == "ACCUMULATION" and signal == "STRONG_SELL"),
            (phase == "MARKUP" and signal == "STRONG_SELL"),
        ]

        if any(inconsistent):
            return (
                f"Phase ({phase}) và Signal ({signal}) không nhất quán",
                None,
                0.2
            )
        return None

    def _rule_wyckoff_spring_volume(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Spring phải kèm volume cao"""
        spring = output.get("spring_detected", False)
        spring_details = output.get("spring_details", {})
        volume_ratio = spring_details.get("volume_ratio", 0)

        if spring and volume_ratio < 1.5:
            return (
                f"Spring detected nhưng volume ratio ({volume_ratio:.1f}x) < 1.5x",
                {"spring_detected": False, "confidence": output.get("confidence", 0) * 0.7},
                0.15
            )
        return None

    # === NEWS RULES ===

    def _rule_news_sentiment_bounds(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Sentiment score phải trong -1 đến 1"""
        sentiment = output.get("sentiment_score", 0)

        if sentiment < -1 or sentiment > 1:
            return (
                f"Sentiment score ({sentiment}) ngoài phạm vi [-1, 1]",
                {"sentiment_score": max(-1, min(1, sentiment))},
                0.1
            )
        return None

    def _rule_news_source_required(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Nếu is_material=True, phải có sources"""
        is_material = output.get("is_material", False)
        sources = output.get("sources", [])

        if is_material and not sources:
            return (
                "Material news declared nhưng không có sources (Source Attribution required)",
                {"confidence": output.get("confidence", 0) * 0.8},
                0.15
            )
        return None

    # === BAYESIAN RULES ===

    def _rule_bayesian_probability_bounds(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Probability phải trong 0-1"""
        prob = output.get("final_probability", 0)

        if prob < 0 or prob > 1:
            return (
                f"Final probability ({prob}) ngoài phạm vi [0, 1]",
                {"final_probability": max(0, min(1, prob))},
                0.2
            )
        return None

    def _rule_bayesian_odds_consistency(self, output: Dict, data: Dict) -> Optional[tuple]:
        """Verify: Probability = Odds / (1 + Odds)"""
        odds = output.get("posterior_odds", 0)
        prob = output.get("final_probability", 0)

        if odds > 0:
            expected_prob = odds / (1 + odds)
            if abs(prob - expected_prob) > 0.05:  # 5% tolerance
                return (
                    f"Probability ({prob:.3f}) không khớp với Odds ({odds:.3f})",
                    {"final_probability": round(expected_prob, 4)},
                    0.1
                )
        return None


class FactCheckPipeline:
    """
    Đường ống Kiểm chứng Sự thật

    Xác minh thông tin từ LLM output bằng cách tìm kiếm lại

    Tham chiếu: base.txt Section 4.3
    """

    def __init__(self, tavily_client=None):
        self.tavily = tavily_client
        self.max_query_length = 400

    def _build_verification_query(self, symbol: str, claim: str, claim_type: str) -> str:
        """
        Dùng claim từ LLM làm lõi truy vấn, nhưng rút gọn để phù hợp giới hạn Tavily.
        """
        base = f"{symbol} {claim} {claim_type} official announcement filing press release"
        normalized = " ".join(str(base or "").split())
        if len(normalized) <= self.max_query_length:
            return normalized

        claim_tokens = [t for t in str(claim or "").split() if len(t) > 2]
        # Giữ các token đầu tiên để vẫn bám theo ý chính do LLM sinh ra
        compact_claim = " ".join(claim_tokens[:40])
        compact = f"{symbol} {compact_claim} {claim_type} official announcement filing"
        compact = " ".join(compact.split())

        if len(compact) <= self.max_query_length:
            return compact

        clipped = compact[: self.max_query_length + 1]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        return clipped[: self.max_query_length]

    def verify_claim(self,
                     claim: str,
                     symbol: str,
                     claim_type: str = "product") -> Dict[str, Any]:
        """
        Xác minh một claim từ LLM

        Args:
            claim: Nội dung cần xác minh
            symbol: Mã cổ phiếu liên quan
            claim_type: Loại claim (product, earnings, management, etc.)

        Returns:
            Dict với verified, confidence, sources
        """
        if not self.tavily:
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": "No Tavily client available for verification"
            }

        # Construct verification query (giữ claim do LLM quyết định, nhưng enforce giới hạn 400 ký tự)
        verification_query = self._build_verification_query(symbol, claim, claim_type)

        try:
            result = self.tavily.search(
                query=verification_query,
                search_depth="basic",
                topic="news",
                days=30,
                max_results=5
            )

            # Check if results contain relevant information
            results = result.get("results", [])

            if not results:
                return {
                    "verified": False,
                    "confidence": 0.2,
                    "reason": "No verification sources found"
                }

            # Simple verification: check if claim keywords appear in results
            claim_words = set(claim.lower().split())
            match_count = 0

            for r in results:
                content = (r.get("title", "") + " " + r.get("content", "")).lower()
                matches = sum(1 for word in claim_words if word in content)
                match_count += matches

            # Calculate verification confidence
            confidence = min(1.0, match_count / (len(claim_words) * 3))

            return {
                "verified": confidence > 0.5,
                "confidence": round(confidence, 2),
                "sources": [r.get("url") for r in results[:3]],
                "reason": f"Found {match_count} keyword matches in {len(results)} sources"
            }

        except Exception as e:
            logger.error(f"Fact-check error: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": f"Verification error: {str(e)}"
            }

    def verify_news_claims(self,
                          news_output: Dict[str, Any],
                          symbol: str) -> Dict[str, Any]:
        """
        Xác minh tất cả claims trong News Agent output

        Args:
            news_output: Output từ News Agent
            symbol: Mã cổ phiếu

        Returns:
            Adjusted news output với verification results
        """
        material_events = news_output.get("material_events", [])
        new_factor = news_output.get("new_factor", "")

        verified_events = []
        confidence_adjustment = 0.0

        # Verify material events
        for event in material_events:
            verification = self.verify_claim(event, symbol, "event")
            if verification["verified"]:
                verified_events.append({
                    "event": event,
                    "verified": True,
                    "sources": verification.get("sources", [])
                })
            else:
                confidence_adjustment -= 0.1  # Penalty for unverified claims

        # Verify N factor
        n_verified = None
        if new_factor:
            n_verification = self.verify_claim(new_factor, symbol, "product")
            n_verified = n_verification["verified"]
            if not n_verified:
                confidence_adjustment -= 0.15

        # Adjust output
        original_confidence = news_output.get("confidence", 0.5)
        adjusted_confidence = max(0.2, original_confidence + confidence_adjustment)

        return {
            **news_output,
            "verified_events": verified_events,
            "n_factor_verified": n_verified,
            "confidence": round(adjusted_confidence, 3),
            "verification_applied": True
        }


# Singleton instances
_verafi_instance = None
_factcheck_instance = None


def get_verafi() -> VERAFIValidator:
    """Get VERAFI singleton"""
    global _verafi_instance
    if _verafi_instance is None:
        _verafi_instance = VERAFIValidator()
    return _verafi_instance


def get_factcheck(tavily_client=None) -> FactCheckPipeline:
    """Get FactCheck singleton"""
    global _factcheck_instance
    if _factcheck_instance is None:
        _factcheck_instance = FactCheckPipeline(tavily_client)
    elif tavily_client is not None and _factcheck_instance.tavily is None:
        _factcheck_instance.tavily = tavily_client
    return _factcheck_instance
