"""
SPECIALIST AGENTS - CÁC TÁC TỬ CHUYÊN GIA (BINH LÍNH)
=====================================================
Triển khai logic phân tích theo Wyckoff, CANSLIM, 4M
Tham chiếu: base.txt Section 2 & 3.1.3
"""

import abc
import logging
import math
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from blackboard import (
    Blackboard, AgentMessage, MessageType,
    TradingSignal, SignalType, WyckoffPhase, MarketRegime
)
from data_providers import MarketData
from llm_client import ClaudeClient
from config import get_config

logger = logging.getLogger("Sentinel.Agents")

# --- BASE AGENT CLASS ---

class BaseAgent(abc.ABC):
    """Lớp cơ sở cho tất cả các Tác tử"""

    def __init__(self, name: str, blackboard: Blackboard, llm_client: Optional[ClaudeClient] = None):
        self.name = name
        self.blackboard = blackboard
        self.llm = llm_client
        self.config = get_config()

    @abc.abstractmethod
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Logic phân tích chính - cần được override"""
        pass

    def run(self, symbols: List[str] = None):
        """
        Thực thi phân tích cho danh sách symbols
        Đọc dữ liệu từ Blackboard và ghi kết quả lên Blackboard
        """
        if symbols is None:
            # Lấy tất cả symbols có dữ liệu
            market_data = self.blackboard.read_memory("market_data")
            symbols = list(market_data.keys()) if market_data else []

        results = {}
        if not symbols:
            return results

        max_workers = max(1, min(self.config.system.max_parallel_workers, len(symbols)))

        def _analyze_symbol(symbol: str):
            data = self.blackboard.read_memory("market_data", symbol)
            if not data:
                return symbol, None
            logger.info(f"[{self.name}] Đang phân tích {symbol}...")
            return symbol, self.analyze(symbol, data)

        if max_workers == 1:
            for symbol in symbols:
                sym, result = _analyze_symbol(symbol)
                if result is None:
                    continue
                results[sym] = result
                self.send_analysis(sym, result)
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_analyze_symbol, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    symbol, result = future.result()
                except Exception as exc:
                    logger.error(f"[{self.name}] Lỗi phân tích {sym}: {exc}")
                    continue
                if result is None:
                    continue
                results[symbol] = result
                self.send_analysis(symbol, result)

        return results

    def send_analysis(self, symbol: str, result: Dict[str, Any]):
        """Gửi kết quả phân tích lên Blackboard"""
        result["symbol"] = symbol
        result["source"] = self.name
        result["timestamp"] = datetime.now().isoformat()

        msg = AgentMessage(
            sender=self.name,
            receiver="Blackboard",
            msg_type=MessageType.ANALYSIS_RESULT,
            content=result
        )
        self.blackboard.post_message(msg)


# --- WYCKOFF AGENT ---

class WyckoffAgent(BaseAgent):
    """
    TÁC TỬ WYCKOFF

    Phân tích cấu trúc thị trường theo phương pháp Wyckoff
    Phát hiện: Spring, SOS, Absorption, Phase detection

    Tham chiếu: base.txt Section 2.1
    """

    def __init__(self, blackboard: Blackboard, llm_client: Optional[ClaudeClient] = None):
        super().__init__("Wyckoff_Agent", blackboard, llm_client)

    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phân tích Wyckoff chính
        """
        # Trích xuất dữ liệu
        if isinstance(data, dict) and "market_data" in data:
            market_data = data["market_data"]
        elif hasattr(data, 'prices'):
            market_data = data
        else:
            market_data = data

        prices = self._get_prices(market_data)
        volumes = self._get_volumes(market_data)

        if not prices or len(prices) < 50:
            return {
                "signal": SignalType.NO_SIGNAL.value,
                "confidence": 0.0,
                "phase": WyckoffPhase.UNKNOWN.value,
                "reasoning": "Không đủ dữ liệu để phân tích"
            }

        # 1. Phát hiện Spring (Cú rũ bỏ)
        spring_result = self._detect_spring(prices, volumes)

        # 2. Xác định pha hiện tại
        phase = self._detect_phase(prices, volumes)

        # 3. Phân tích Volume-Price
        vpa_result = self._volume_price_analysis(prices, volumes)

        # Tổng hợp tín hiệu
        confidence = 0.5
        signal = SignalType.HOLD
        reasoning_parts = []

        if spring_result["detected"]:
            signal = SignalType.BUY
            confidence = max(confidence, spring_result["confidence"])
            reasoning_parts.append(f"Spring: {spring_result['reason']}")

        if vpa_result["bullish"]:
            confidence = min(1.0, confidence + 0.15)
            reasoning_parts.append(f"VPA: {vpa_result['reason']}")

        # Nếu đang ở pha tích lũy với tín hiệu tốt
        if phase == WyckoffPhase.ACCUMULATION and confidence > 0.6:
            signal = SignalType.BUY
            reasoning_parts.append(f"Pha: {phase.value}")
        elif phase == WyckoffPhase.DISTRIBUTION:
            signal = SignalType.SELL
            confidence = 0.65
            reasoning_parts.append(f"Pha phân phối - Cảnh báo")

        # Luôn gọi LLM (nếu có) để review tín hiệu, rule-based chỉ là baseline
        if self.llm:
            llm_result = self._llm_analyze(symbol, prices, volumes, spring_result, phase)
            if llm_result:
                # Cập nhật từ LLM
                if "confidence" in llm_result:
                    confidence = (confidence + float(llm_result["confidence"])) / 2
                llm_signal = str(llm_result.get("signal", "")).upper()
                if llm_signal in {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}:
                    signal = SignalType(llm_signal)
                reasoning_parts.append(f"LLM: {llm_result.get('reasoning', '')[:100]}")

        return {
            "signal": signal.value,
            "confidence": round(confidence, 3),
            "phase": phase.value,
            "spring_detected": spring_result["detected"],
            "spring_details": spring_result,
            "vpa_analysis": vpa_result,
            "reasoning": " | ".join(reasoning_parts)
        }

    def _get_prices(self, data) -> List[float]:
        """Trích xuất giá từ các định dạng khác nhau"""
        if hasattr(data, 'prices'):
            return data.prices
        if isinstance(data, dict):
            return data.get("prices", [])
        return []

    def _get_volumes(self, data) -> List[float]:
        """Trích xuất volume từ các định dạng khác nhau"""
        if hasattr(data, 'volumes'):
            return data.volumes
        if isinstance(data, dict):
            return data.get("volumes", [])
        return []

    def _detect_spring(self, prices: List[float], volumes: List[float]) -> Dict:
        """
        Phát hiện Spring (Cú rũ bỏ) - Pha C trong Accumulation

        Logic: base.txt Section 2.1.1
        1. Xác định Hỗ trợ (Min trong 50 thanh, trừ 5 thanh cuối)
        2. Giá thấp gần nhất < Hỗ trợ (Phá vỡ)
        3. Giá đóng cửa hiện tại > Hỗ trợ (Từ chối)
        4. Volume hiện tại > 1.5 × Volume trung bình
        """
        config = self.config.wyckoff

        if len(prices) < config.support_lookback:
            return {"detected": False, "confidence": 0, "reason": "Không đủ dữ liệu"}

        # 1. Xác định mức Hỗ trợ
        lookback_start = len(prices) - config.support_lookback - 5
        lookback_end = len(prices) - 5
        support_range = prices[max(0, lookback_start):lookback_end]

        if not support_range:
            return {"detected": False, "confidence": 0, "reason": "Không thể xác định hỗ trợ"}

        support_level = min(support_range)

        # 2. Kiểm tra hành động giá gần đây
        recent_prices = prices[-5:]
        recent_low = min(recent_prices)
        current_close = prices[-1]

        # 3. Kiểm tra Volume
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
        current_volume = volumes[-1]
        is_high_volume = current_volume > (config.volume_multiplier * avg_volume)

        # Logic Spring
        tolerance = support_level * config.spring_tolerance
        breakdown = recent_low < (support_level - tolerance)  # Đã thủng hỗ trợ
        reclaim = current_close > support_level  # Đã lấy lại hỗ trợ

        if breakdown and reclaim and is_high_volume:
            confidence = 0.85
            # Điều chỉnh confidence dựa trên volume spike
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 2.5:
                confidence = 0.90
            elif volume_ratio < 1.8:
                confidence = 0.75

            return {
                "detected": True,
                "confidence": confidence,
                "support_level": round(support_level, 2),
                "recent_low": round(recent_low, 2),
                "current_close": round(current_close, 2),
                "volume_ratio": round(volume_ratio, 2),
                "reason": f"Spring phát hiện: Giá nhúng xuống {recent_low:.2f} (dưới hỗ trợ {support_level:.2f}) rồi hồi phục lên {current_close:.2f} với volume gấp {volume_ratio:.1f}x trung bình"
            }

        return {
            "detected": False,
            "confidence": 0,
            "support_level": round(support_level, 2),
            "reason": "Không phát hiện mẫu hình Spring"
        }

    def _detect_phase(self, prices: List[float], volumes: List[float]) -> WyckoffPhase:
        """
        Xác định pha hiện tại trong chu kỳ Wyckoff
        """
        if len(prices) < 60:
            return WyckoffPhase.UNKNOWN

        # Tính xu hướng dài hạn (60 ngày)
        long_term_change = (prices[-1] - prices[-60]) / prices[-60]

        # Tính xu hướng ngắn hạn (20 ngày)
        short_term_change = (prices[-1] - prices[-20]) / prices[-20]

        # Độ biến động (Choppiness)
        recent_range = max(prices[-20:]) - min(prices[-20:])
        avg_price = sum(prices[-20:]) / 20
        volatility = recent_range / avg_price

        # Phát hiện pha
        if long_term_change < -0.10 and volatility < 0.08:
            # Giảm mạnh trước đó, giờ đang đi ngang
            return WyckoffPhase.ACCUMULATION
        elif long_term_change > 0.15 and short_term_change > 0.05:
            # Đang trong xu hướng tăng mạnh
            return WyckoffPhase.MARKUP
        elif long_term_change > 0.20 and volatility < 0.08 and short_term_change < 0.02:
            # Tăng mạnh trước đó, giờ đang đi ngang ở đỉnh
            return WyckoffPhase.DISTRIBUTION
        elif short_term_change < -0.10:
            # Đang giảm mạnh
            return WyckoffPhase.MARKDOWN

        return WyckoffPhase.UNKNOWN

    def _volume_price_analysis(self, prices: List[float], volumes: List[float]) -> Dict:
        """
        Phân tích Volume-Price theo Quy luật Nỗ lực vs Kết quả
        base.txt: Khối lượng lớn nhưng giá ít thay đổi = Absorption
        """
        if len(prices) < 10 or len(volumes) < 10:
            return {"bullish": False, "reason": "Không đủ dữ liệu"}

        # Tính các chỉ số
        avg_volume = sum(volumes[-20:]) / 20
        recent_volume = volumes[-1]
        price_change = abs(prices[-1] - prices[-2]) / prices[-2]

        # Quy luật Nỗ lực vs Kết quả
        effort = recent_volume / avg_volume  # Nỗ lực (Volume)
        result = price_change * 100  # Kết quả (Price change %)

        # Phân kỳ Volume-Price (Absorption)
        if effort > 1.5 and result < 1.0:
            # Volume cao nhưng giá ít thay đổi
            if prices[-1] > prices[-2]:
                return {
                    "bullish": True,
                    "absorption": True,
                    "reason": f"Hấp thụ cung: Volume gấp {effort:.1f}x nhưng giá chỉ tăng {result:.1f}% - Dòng tiền thông minh đang mua"
                }
            else:
                return {
                    "bullish": False,
                    "absorption": True,
                    "reason": f"Hấp thụ cầu: Volume gấp {effort:.1f}x nhưng giá chỉ giảm {result:.1f}% - Có thể có bán tháo"
                }

        # Xu hướng Volume trong 5 ngày
        vol_trend = sum(volumes[-5:]) / 5 > avg_volume
        price_up = prices[-1] > prices[-5]

        if vol_trend and price_up:
            return {
                "bullish": True,
                "absorption": False,
                "reason": "Volume tăng kèm giá tăng - Xu hướng tăng khỏe"
            }

        return {"bullish": False, "absorption": False, "reason": "Không có tín hiệu đặc biệt"}

    def _llm_analyze(self, symbol: str, prices: List[float],
                     volumes: List[float], spring_result: Dict,
                     phase: WyckoffPhase) -> Optional[Dict]:
        """Sử dụng LLM để phân tích sâu hơn"""
        if not self.llm:
            return None

        prompt = f"""Phân tích kỹ thuật Wyckoff cho {symbol}:

DỮ LIỆU:
- Giá 5 ngày gần nhất: {prices[-5:]}
- Volume 5 ngày gần nhất: {volumes[-5:]}
- Pha hiện tại: {phase.value}
- Spring detection: {spring_result}

Hãy đánh giá và đưa ra tín hiệu giao dịch theo phương pháp Wyckoff."""

        return self.llm.analyze("wyckoff_analyst", prompt, symbol=symbol)


# --- CANSLIM AGENT ---

class CANSLIMAgent(BaseAgent):
    """
    TÁC TỬ CANSLIM

    Đánh giá cổ phiếu theo 7 tiêu chí CANSLIM của William O'Neil

    Tham chiếu: base.txt Section 2.2
    """

    def __init__(self, blackboard: Blackboard, llm_client: Optional[ClaudeClient] = None):
        super().__init__("CANSLIM_Agent", blackboard, llm_client)

    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích CANSLIM chính"""

        # Trích xuất dữ liệu
        if isinstance(data, dict) and "market_data" in data:
            market_data = data["market_data"]
        elif hasattr(data, 'prices'):
            market_data = data
        else:
            market_data = data

        config = self.config.canslim
        scores = {}
        details = {}

        # C - Current Earnings
        c_score, c_detail = self._score_current_earnings(market_data, config)
        scores["C"] = c_score
        details["C"] = c_detail

        # A - Annual Earnings
        a_score, a_detail = self._score_annual_earnings(market_data, config)
        scores["A"] = a_score
        details["A"] = a_detail

        # N - New (Products, Management, Price High)
        n_score, n_detail = self._score_new_factor(market_data, data, config)
        scores["N"] = n_score
        details["N"] = n_detail

        # S - Supply/Demand
        s_score, s_detail = self._score_supply_demand(market_data, config)
        scores["S"] = s_score
        details["S"] = s_detail

        # L - Leader
        l_score, l_detail = self._score_leader(market_data, config)
        scores["L"] = l_score
        details["L"] = l_detail

        # I - Institutional
        i_score, i_detail = self._score_institutional(market_data, config)
        scores["I"] = i_score
        details["I"] = i_detail

        # M - Market Direction (Lấy từ Blackboard - Tướng quân Vĩ mô)
        m_score, m_detail = self._score_market_direction()
        scores["M"] = m_score
        details["M"] = m_detail

        # Tổng điểm
        total_score = sum(scores.values())

        # Xác định tín hiệu
        if total_score >= 85:
            signal = SignalType.STRONG_BUY
            confidence = 0.90
        elif total_score >= 70:
            signal = SignalType.BUY
            confidence = 0.75
        elif total_score >= 55:
            signal = SignalType.HOLD
            confidence = 0.55
        else:
            signal = SignalType.SELL
            confidence = 0.60

        baseline = {
            "signal": signal.value,
            "confidence": round(confidence, 3),
            "total_score": total_score,
            "scores": scores,
            "details": details,
            "reasoning": f"CANSLIM Score: {total_score}/100 | C={scores['C']}, A={scores['A']}, N={scores['N']}, S={scores['S']}, L={scores['L']}, I={scores['I']}, M={scores['M']}"
        }

        if not self.llm:
            return baseline

        llm_overlay = self._llm_refine(symbol, baseline, market_data)
        return self._merge_canslim_result(baseline, llm_overlay)

    def _llm_refine(self, symbol: str, baseline: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
        """LLM review cho CANSLIM (Sonnet)."""
        if not self.llm:
            return None

        prompt = f"""Đánh giá CANSLIM cho {symbol}. Baseline rule-based:
{baseline}

Market data snapshot:
- eps_growth: {self._get_attr(market_data, "eps_growth")}
- annual_eps_growth: {self._get_attr(market_data, "annual_eps_growth")}
- roe: {self._get_attr(market_data, "roe")}
- rs_rating: {self._get_attr(market_data, "rs_rating")}
- institutional_ownership: {self._get_attr(market_data, "institutional_ownership")}
- high_52w: {self._get_attr(market_data, "high_52w")}
- last_price: {(self._get_attr(market_data, "prices", []) or [None])[-1]}

Trả JSON đúng schema canslim_analyst, dùng baseline làm tham chiếu, sửa nếu cần."""
        try:
            return self.llm.analyze("canslim_analyst", prompt, symbol=symbol)
        except Exception as exc:
            logger.warning(f"[{self.name}] LLM CANSLIM refine failed for {symbol}: {exc}")
            return None

    def _merge_canslim_result(self, baseline: Dict[str, Any], llm_overlay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Hợp nhất kết quả LLM với baseline; fallback an toàn khi JSON lỗi."""
        if not llm_overlay or not isinstance(llm_overlay, dict):
            return baseline

        merged = dict(baseline)

        llm_scores = llm_overlay.get("scores")
        if isinstance(llm_scores, dict):
            norm_scores = {}
            for k in ["C", "A", "N", "S", "L", "I", "M"]:
                if k in llm_scores:
                    try:
                        norm_scores[k] = max(0, min(15, int(round(float(llm_scores[k])))))
                    except Exception:
                        continue
            if len(norm_scores) >= 4:
                # LLM là nguồn chính, baseline làm fallback cho key thiếu
                merged_scores = dict(merged.get("scores", {}))
                merged_scores.update(norm_scores)
                merged["scores"] = merged_scores
                merged["total_score"] = int(sum(merged_scores.values()))

        llm_signal = str(llm_overlay.get("signal", "")).upper().strip()
        if llm_signal in {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}:
            merged["signal"] = llm_signal

        if "confidence" in llm_overlay:
            try:
                merged["confidence"] = round(max(0.05, min(0.99, float(llm_overlay["confidence"]))), 4)
            except Exception:
                pass

        llm_reasoning = str(llm_overlay.get("reasoning", "")).strip()
        if llm_reasoning:
            merged["reasoning"] = f"{merged.get('reasoning', '')} | LLM: {llm_reasoning[:300]}"
        merged["llm_review_applied"] = True
        return merged

    def _score_current_earnings(self, data, config) -> tuple:
        """C - Current Earnings: EPS quý hiện tại tăng > 18-20% YoY"""
        eps_growth = self._get_attr(data, "eps_growth")

        if eps_growth is None:
            return 5, "Không có dữ liệu EPS"

        if eps_growth >= config.min_eps_growth:
            if eps_growth >= 0.25:
                return 15, f"Xuất sắc: EPS tăng {eps_growth*100:.1f}%"
            return 12, f"Tốt: EPS tăng {eps_growth*100:.1f}%"
        elif eps_growth > 0:
            return 7, f"Trung bình: EPS tăng {eps_growth*100:.1f}%"
        return 0, f"Kém: EPS giảm {eps_growth*100:.1f}%"

    def _score_annual_earnings(self, data, config) -> tuple:
        """A - Annual Earnings: CAGR > 25% trong 3-5 năm, ROE > 17%"""
        annual_growth = self._get_attr(data, "annual_eps_growth")
        roe = self._get_attr(data, "roe")

        score = 0
        details = []

        if annual_growth is not None:
            if annual_growth >= config.min_annual_growth:
                score += 8
                details.append(f"CAGR xuất sắc: {annual_growth*100:.1f}%")
            elif annual_growth > 0.15:
                score += 5
                details.append(f"CAGR tốt: {annual_growth*100:.1f}%")
        else:
            details.append("Không có dữ liệu CAGR")

        if roe is not None:
            if roe >= config.min_roe:
                score += 7
                details.append(f"ROE xuất sắc: {roe*100:.1f}%")
            elif roe > 0.12:
                score += 4
                details.append(f"ROE tốt: {roe*100:.1f}%")
        else:
            details.append("Không có dữ liệu ROE")

        return score, " | ".join(details)

    def _score_new_factor(self, market_data, full_data, config) -> tuple:
        """N - New: Sản phẩm mới, quản lý mới, hoặc đỉnh giá mới"""
        score = 0
        details = []

        # Kiểm tra đỉnh giá mới
        prices = self._get_attr(market_data, "prices", [])
        high_52w = self._get_attr(market_data, "high_52w")

        if prices and high_52w:
            current = prices[-1]
            distance = (high_52w - current) / high_52w
            if distance <= config.high_52w_tolerance:
                score += 10
                details.append(f"Gần đỉnh 52 tuần (cách {distance*100:.1f}%)")

        # Kiểm tra từ tin tức (nếu có)
        news_data = full_data.get("news_data", {}) if isinstance(full_data, dict) else {}
        if news_data and not news_data.get("is_mock"):
            news_count = len(news_data.get("results", []))
            if news_count >= 3:
                score += 5
                details.append(f"Có catalyst/news material ({news_count} tin)")
            elif news_count > 0:
                score += 2
                details.append(f"Có tin tức mới ({news_count} tin)")

        if not details:
            details.append("Không có yếu tố mới rõ ràng")

        return min(15, score), " | ".join(details)

    def _score_supply_demand(self, data, config) -> tuple:
        """S - Supply/Demand: Phân tích khối lượng"""
        volumes = self._get_attr(data, "volumes", [])
        prices = self._get_attr(data, "prices", [])

        if not volumes or len(volumes) < 20:
            return 5, "Không đủ dữ liệu volume"

        # Kiểm tra volume trend
        avg_volume = sum(volumes[-20:]) / 20
        recent_avg = sum(volumes[-5:]) / 5

        # Kiểm tra giá tăng kèm volume tăng
        if len(prices) >= 5:
            price_up = prices[-1] > prices[-5]
            vol_up = recent_avg > avg_volume

            if price_up and vol_up:
                return 12, "Giá tăng kèm volume tăng - Cầu mạnh"
            elif price_up and not vol_up:
                return 8, "Giá tăng nhưng volume thấp"
            elif not price_up and vol_up:
                return 5, "Giá giảm với volume cao - Có áp lực bán"

        return 7, "Volume bình thường"

    def _score_leader(self, data, config) -> tuple:
        """L - Leader: Relative Strength > 80"""
        rs = self._get_attr(data, "rs_rating")

        if rs is None:
            return 5, "Không có dữ liệu RS"

        if rs >= 90:
            return 15, f"Dẫn đầu tuyệt đối: RS = {rs}"
        elif rs >= config.min_rs_rating:
            return 12, f"Cổ phiếu dẫn đầu: RS = {rs}"
        elif rs >= 60:
            return 7, f"Trung bình: RS = {rs}"
        return 3, f"Tụt hậu: RS = {rs}"

    def _score_institutional(self, data, config) -> tuple:
        """I - Institutional Sponsorship"""
        inst_own = self._get_attr(data, "institutional_ownership")
        insider_own = self._get_attr(data, "insider_ownership")
        market_cap = self._get_attr(data, "market_cap")

        if inst_own is not None:
            inst_pct = inst_own * 100
            if inst_own >= 0.65:
                score = 15
                detail = f"Sở hữu tổ chức rất cao: {inst_pct:.1f}%"
            elif inst_own >= 0.45:
                score = 12
                detail = f"Sở hữu tổ chức tốt: {inst_pct:.1f}%"
            elif inst_own >= 0.20:
                score = 8
                detail = f"Sở hữu tổ chức trung bình: {inst_pct:.1f}%"
            else:
                score = 5
                detail = f"Sở hữu tổ chức thấp: {inst_pct:.1f}%"

            if insider_own is not None and insider_own >= 0.05:
                score = min(15, score + 1)
                detail += f" | Insider ownership: {insider_own*100:.1f}%"
            return score, detail

        if market_cap is None:
            return 5, "Không có dữ liệu Institutional ownership"

        if market_cap >= 100e9:  # > $100B
            return 12, "Công ty lớn - Nhiều quỹ sở hữu"
        elif market_cap >= 10e9:  # > $10B
            return 10, "Mid-cap - Quỹ tăng trưởng quan tâm"
        elif market_cap >= 1e9:  # > $1B
            return 7, "Small-cap - Ít quỹ theo dõi"
        return 4, "Micro-cap - Rủi ro thanh khoản"

    def _score_market_direction(self) -> tuple:
        """M - Market Direction: Lấy từ Tướng quân Vĩ mô"""
        regime = self.blackboard.get_current_regime()

        if regime["current"] == MarketRegime.RISK_ON.value:
            return 15, "Thị trường Tăng (Risk-On)"
        elif regime["current"] == MarketRegime.SIDEWAYS.value:
            return 8, "Thị trường Đi ngang"
        elif regime["current"] == MarketRegime.RISK_OFF.value:
            return 3, "Thị trường Giảm (Risk-Off) - CẨN THẬN"
        return 7, "Chế độ thị trường chưa xác định"

    def _get_attr(self, data, attr: str, default=None):
        """Lấy thuộc tính từ data (hỗ trợ cả dict và object)"""
        if hasattr(data, attr):
            return getattr(data, attr)
        if isinstance(data, dict):
            return data.get(attr, default)
        return default


# --- 4M AGENT ---

class FourMAgent(BaseAgent):
    """
    TÁC TỬ 4M

    Đánh giá theo khung đầu tư giá trị 4M của Phil Town
    Meaning, Moat, Management, Margin of Safety

    Tham chiếu: base.txt Section 2.3
    """

    def __init__(self, blackboard: Blackboard, llm_client: Optional[ClaudeClient] = None):
        super().__init__("FourM_Agent", blackboard, llm_client)

    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích 4M chính"""

        # Trích xuất dữ liệu
        if isinstance(data, dict) and "market_data" in data:
            market_data = data["market_data"]
        elif hasattr(data, 'prices'):
            market_data = data
        else:
            market_data = data

        config = self.config.fourm

        # 1. Meaning (Ý nghĩa)
        meaning_score, meaning_detail = self._score_meaning(market_data, data)

        # 2. Moat (Lợi thế cạnh tranh)
        moat_score, moat_detail = self._score_moat(market_data, data, config)

        # 3. Management (Ban lãnh đạo)
        mgmt_score, mgmt_detail = self._score_management(market_data, data)

        # 4. Margin of Safety (Biên độ an toàn)
        mos_result = self._calculate_mos(market_data, config)

        # Tổng hợp
        total_score = meaning_score + moat_score + mgmt_score

        # Điều chỉnh dựa trên MOS
        if mos_result["has_mos"]:
            if mos_result["discount_pct"] >= 50:
                total_score += 25
                signal = SignalType.STRONG_BUY
            elif mos_result["discount_pct"] >= 30:
                total_score += 15
                signal = SignalType.BUY
            else:
                total_score += 5
                signal = SignalType.HOLD
        else:
            signal = SignalType.HOLD if total_score >= 50 else SignalType.SELL

        confidence = min(0.95, total_score / 100)

        baseline = {
            "signal": signal.value,
            "confidence": round(confidence, 3),
            "total_score": total_score,
            "scores": {
                "meaning": meaning_score,
                "moat": moat_score,
                "management": mgmt_score,
                "mos_discount": mos_result["discount_pct"]
            },
            "mos_analysis": mos_result,
            "details": {
                "meaning": meaning_detail,
                "moat": moat_detail,
                "management": mgmt_detail
            },
            "reasoning": f"4M Score: {total_score}/100 | Meaning={meaning_score}, Moat={moat_score}, Mgmt={mgmt_score}, MOS Discount={mos_result['discount_pct']:.1f}%"
        }

        if not self.llm:
            return baseline

        llm_overlay = self._llm_refine(symbol, baseline, market_data)
        return self._merge_fourm_result(baseline, llm_overlay)

    def _llm_refine(self, symbol: str, baseline: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
        """LLM review cho 4M (Sonnet)."""
        if not self.llm:
            return None

        current_price = (self._get_attr(market_data, "prices", []) or [None])[-1]
        prompt = f"""Đánh giá 4M cho {symbol}. Baseline rule-based:
{baseline}

Market data snapshot:
- current_price: {current_price}
- pe_ratio: {self._get_attr(market_data, "pe_ratio")}
- roe: {self._get_attr(market_data, "roe")}
- roic: {self._get_attr(market_data, "roic")}
- annual_eps_growth: {self._get_attr(market_data, "annual_eps_growth")}

Trả JSON đúng schema fourm_analyst, dùng baseline làm tham chiếu, sửa nếu cần."""
        try:
            return self.llm.analyze("fourm_analyst", prompt, symbol=symbol)
        except Exception as exc:
            logger.warning(f"[{self.name}] LLM 4M refine failed for {symbol}: {exc}")
            return None

    def _merge_fourm_result(self, baseline: Dict[str, Any], llm_overlay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Hợp nhất kết quả 4M từ LLM với baseline."""
        if not llm_overlay or not isinstance(llm_overlay, dict):
            return baseline

        merged = dict(baseline)

        # map hỗ trợ nhiều format output
        if "scores" in llm_overlay and isinstance(llm_overlay["scores"], dict):
            llm_scores = llm_overlay["scores"]
            merged_scores = dict(merged.get("scores", {}))
            for key in ["meaning", "moat", "management"]:
                if key in llm_scores:
                    try:
                        merged_scores[key] = max(0, min(25, int(round(float(llm_scores[key])))))
                    except Exception:
                        pass
            merged["scores"] = merged_scores
        else:
            for src, dst in [("meaning_score", "meaning"), ("moat_score", "moat"), ("management_score", "management")]:
                if src in llm_overlay:
                    try:
                        merged["scores"][dst] = max(0, min(25, int(round(float(llm_overlay[src])))))
                    except Exception:
                        pass

        # MOS
        llm_mos = llm_overlay.get("mos_analysis")
        if isinstance(llm_mos, dict):
            merged_mos = dict(merged.get("mos_analysis", {}))
            for k in ["sticker_price", "buy_price", "current_price", "discount_pct", "has_mos"]:
                if k in llm_mos:
                    merged_mos[k] = llm_mos[k]
            merged["mos_analysis"] = merged_mos

        # recompute score if possible
        try:
            part_score = int(merged["scores"].get("meaning", 0) + merged["scores"].get("moat", 0) + merged["scores"].get("management", 0))
            mos_bonus = 0
            mos = merged.get("mos_analysis", {})
            if mos.get("has_mos"):
                discount = float(mos.get("discount_pct", 0))
                mos_bonus = 25 if discount >= 50 else (15 if discount >= 30 else 5)
            merged["total_score"] = part_score + mos_bonus
        except Exception:
            pass

        llm_signal = str(llm_overlay.get("signal", "")).upper().strip()
        if llm_signal in {"STRONG_BUY", "BUY", "HOLD", "SELL", "NO_DEAL"}:
            merged["signal"] = llm_signal

        if "confidence" in llm_overlay:
            try:
                merged["confidence"] = round(max(0.05, min(0.99, float(llm_overlay["confidence"]))), 4)
            except Exception:
                pass

        llm_reasoning = str(llm_overlay.get("reasoning", "")).strip()
        if llm_reasoning:
            merged["reasoning"] = f"{merged.get('reasoning', '')} | LLM: {llm_reasoning[:300]}"
        merged["llm_review_applied"] = True
        return merged

    def _score_meaning(self, market_data, full_data) -> tuple:
        """Meaning: Có hiểu công ty này không?"""
        industry = self._get_attr(market_data, "industry")

        # Các ngành dễ hiểu (Vòng tròn năng lực)
        understandable = ["Technology", "Consumer", "Healthcare", "Finance", "Retail"]

        if industry:
            if any(ind.lower() in industry.lower() for ind in understandable):
                return 20, f"Ngành dễ hiểu: {industry}"
            return 10, f"Ngành: {industry} - Cần nghiên cứu thêm"

        return 15, "Không có thông tin ngành"

    def _score_moat(self, data, full_data, config) -> tuple:
        """Moat: ROIC + tín hiệu định tính từ nghiên cứu công ty"""
        roic = self._get_attr(data, "roic")
        roe = self._get_attr(data, "roe")
        debt_to_equity = self._get_attr(data, "debt_to_equity")
        research = full_data.get("company_research", {}) if isinstance(full_data, dict) else {}

        score = 0
        details = []

        # Sử dụng ROIC hoặc ROE
        profitability = roic if roic else roe

        if profitability is not None:
            if profitability >= config.min_roic:
                score = 22
                details.append(f"ROIC/ROE xuất sắc: {profitability*100:.1f}%")
            elif profitability >= 0.08:
                score = 15
                details.append(f"ROIC/ROE tốt: {profitability*100:.1f}%")
            else:
                score = 8
                details.append(f"ROIC/ROE thấp: {profitability*100:.1f}%")
        else:
            score = 10
            details.append("Không có dữ liệu ROIC/ROE")

        # Tăng/giảm điểm theo chất lượng bảng cân đối
        if debt_to_equity is not None:
            if debt_to_equity <= 80:
                score += 1
                details.append(f"Đòn bẩy thấp: D/E={debt_to_equity:.1f}")
            elif debt_to_equity >= 220:
                score -= 2
                details.append(f"Đòn bẩy cao: D/E={debt_to_equity:.1f}")

        # Bổ sung phân tích moat định tính từ nghiên cứu
        text = ""
        if research:
            text = " ".join([
                str(research.get("answer", "")),
                " ".join(
                    f"{r.get('title', '')} {r.get('content', '')}"
                    for r in research.get("results", [])[:5]
                )
            ]).lower()

        moat_keywords = [
            "network effect", "switching cost", "brand", "ecosystem",
            "patent", "proprietary", "distribution advantage", "cost advantage"
        ]
        risk_keywords = [
            "commoditized", "price war", "margin pressure", "regulatory risk", "competitive pressure"
        ]

        moat_hits = sum(1 for kw in moat_keywords if kw in text)
        risk_hits = sum(1 for kw in risk_keywords if kw in text)
        if moat_hits:
            score += min(4, moat_hits)
            details.append(f"Moat định tính: +{moat_hits} tín hiệu")
        if risk_hits:
            score -= min(3, risk_hits)
            details.append(f"Rủi ro cạnh tranh: {risk_hits} tín hiệu")

        score = max(0, min(25, score))
        return score, " | ".join(details)

    def _score_management(self, market_data, full_data) -> tuple:
        """Management: Đánh giá ban lãnh đạo"""
        roe = self._get_attr(market_data, "roe")
        insider = self._get_attr(market_data, "insider_ownership")
        news_data = full_data.get("news_data", {}) if isinstance(full_data, dict) else {}
        research = full_data.get("company_research", {}) if isinstance(full_data, dict) else {}

        score = 8
        details = []

        if roe and roe > 0.15:
            score += 8
            details.append("ROE cao cho thấy quản lý hiệu quả")
        elif roe and roe > 0.10:
            score += 4
            details.append("ROE trung bình")
        else:
            details.append("ROE thấp/không đủ dữ liệu")

        if insider is not None:
            if insider >= 0.05:
                score += 4
                details.append(f"Insider ownership tích cực: {insider*100:.1f}%")
            elif insider <= 0.005:
                score -= 1
                details.append(f"Insider ownership thấp: {insider*100:.2f}%")

        text = ""
        if news_data:
            text += " ".join(
                f"{r.get('title', '')} {r.get('content', '')}" for r in news_data.get("results", [])[:8]
            ).lower()
        if research:
            text += " " + str(research.get("answer", "")).lower()

        positive_mgmt_words = ["transparent", "execution", "discipline", "guidance raised", "shareholder return"]
        negative_mgmt_words = ["restatement", "investigation", "lawsuit", "resign", "missed guidance"]
        pos_hits = sum(1 for w in positive_mgmt_words if w in text)
        neg_hits = sum(1 for w in negative_mgmt_words if w in text)

        if pos_hits:
            score += min(4, pos_hits)
            details.append(f"Tín hiệu quản trị tích cực: {pos_hits}")
        if neg_hits:
            score -= min(6, neg_hits * 2)
            details.append(f"Tín hiệu quản trị tiêu cực: {neg_hits}")

        score = max(0, min(25, score))
        return score, " | ".join(details)

    def _calculate_mos(self, data, config) -> Dict:
        """
        Tính Margin of Safety

        Công thức (base.txt Section 2.3):
        1. Future_EPS = Current_EPS × (1 + Growth)^10
        2. Future_PE = min(Historical_PE, Growth × 200, 40)
        3. Sticker_Price = Future_EPS × Future_PE / (1.15)^10
        4. Buy_Price = Sticker × 0.50
        """
        current_eps = self._get_attr(data, "eps_current")
        growth = self._get_attr(data, "annual_eps_growth")
        pe_ratio = self._get_attr(data, "pe_ratio")
        prices = self._get_attr(data, "prices", [])

        result = {
            "has_mos": False,
            "sticker_price": None,
            "buy_price": None,
            "current_price": None,
            "discount_pct": 0
        }

        if not current_eps or not prices:
            result["reason"] = "Thiếu dữ liệu EPS hoặc giá"
            return result

        current_price = prices[-1]
        result["current_price"] = round(current_price, 2)

        # Default growth nếu không có
        if growth is None or growth <= 0:
            growth = 0.08  # Conservative 8%

        # Giới hạn growth (bảo thủ)
        growth = min(growth, 0.20)

        # Tính Future EPS (10 năm)
        future_eps = current_eps * ((1 + growth) ** 10)

        # Tính Future P/E
        if pe_ratio:
            future_pe = min(pe_ratio, growth * 200, config.max_pe)
        else:
            future_pe = min(growth * 200, config.max_pe)

        future_pe = max(future_pe, 10)  # Tối thiểu P/E = 10

        # Tính giá tương lai
        future_price = future_eps * future_pe

        # Chiết khấu về hiện tại (MARR 15%)
        sticker_price = future_price / ((1 + config.marr) ** 10)

        # Giá mua với MOS 50%
        buy_price = sticker_price * config.mos_discount

        result["sticker_price"] = round(sticker_price, 2)
        result["buy_price"] = round(buy_price, 2)

        # Tính discount
        if sticker_price > 0:
            discount = (sticker_price - current_price) / sticker_price * 100
            result["discount_pct"] = round(discount, 1)
            result["has_mos"] = current_price < buy_price

            if current_price < buy_price:
                result["reason"] = f"Giá hiện tại ({current_price:.2f}) < Giá mua MOS ({buy_price:.2f})"
            elif current_price < sticker_price:
                result["reason"] = f"Giá hiện tại ({current_price:.2f}) < Giá trị nội tại ({sticker_price:.2f}) nhưng chưa đủ MOS"
            else:
                result["reason"] = f"Giá hiện tại ({current_price:.2f}) > Giá trị nội tại ({sticker_price:.2f}) - KHÔNG MUA"

        return result

    def _get_attr(self, data, attr: str, default=None):
        """Lấy thuộc tính từ data"""
        if hasattr(data, attr):
            return getattr(data, attr)
        if isinstance(data, dict):
            return data.get(attr, default)
        return default


# --- NEWS/SENTIMENT AGENT ---

class NewsSentimentAgent(BaseAgent):
    """
    TÁC TỬ TIN TỨC/CẢM XÚC

    Phân tích tin tức phi cấu trúc và đánh giá cảm xúc thị trường
    Sử dụng LLM để xử lý ngôn ngữ tự nhiên

    Tham chiếu: base.txt Section 2.3 (Management), 4.3 (Fact-Check Pipeline)
    """

    def __init__(self, blackboard: Blackboard, llm_client: Optional[ClaudeClient] = None):
        super().__init__("News_Agent", blackboard, llm_client)

    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích tin tức"""

        news_data = data.get("news_data", {}) if isinstance(data, dict) else {}
        company_research = data.get("company_research", {}) if isinstance(data, dict) else {}
        document_intel = data.get("document_intel", {}) if isinstance(data, dict) else {}

        if not news_data and not company_research and not document_intel:
            return {
                "signal": SignalType.HOLD.value,
                "confidence": 0.3,
                "sentiment_score": 0,
                "is_material": False,
                "reasoning": "Không có dữ liệu tin tức"
            }

        # Nếu có LLM, sử dụng để phân tích sâu
        if self.llm:
            return self._llm_analyze(symbol, news_data, company_research, document_intel)

        # Fallback: Phân tích cơ bản
        return self._basic_analyze(news_data, document_intel)

    def _basic_analyze(self, news_data: Dict, document_intel: Dict) -> Dict:
        """Phân tích cơ bản không cần LLM"""

        results = news_data.get("results", [])
        deep_docs = document_intel.get("deep_dive_docs", []) if document_intel else []
        evidence_chunks = document_intel.get("evidence_chunks", []) if document_intel else []
        quality = document_intel.get("quality", {}) if document_intel else {}

        if not results and not deep_docs and not evidence_chunks:
            return {
                "signal": SignalType.HOLD.value,
                "confidence": 0.4,
                "sentiment_score": 0,
                "is_material": False,
                "reasoning": "Không tìm thấy tin tức"
            }

        # Đếm từ khóa tích cực/tiêu cực
        positive_words = ["growth", "beat", "exceed", "strong", "profit", "gain", "upgrade"]
        negative_words = ["loss", "miss", "decline", "weak", "downgrade", "concern", "risk"]

        pos_count = 0
        neg_count = 0

        corpus = []
        for result in results:
            corpus.append(result)
        for doc in deep_docs[:5]:
            corpus.append({
                "title": doc.get("title", ""),
                "content": doc.get("full_text") or doc.get("snippet", "")
            })
        for chunk in evidence_chunks[:6]:
            corpus.append({"title": "evidence_chunk", "content": chunk.get("text", "")})

        for result in corpus:
            content = (result.get("title", "") + " " + result.get("content", "")).lower()
            pos_count += sum(1 for word in positive_words if word in content)
            neg_count += sum(1 for word in negative_words if word in content)

        # Tính sentiment score
        total = pos_count + neg_count
        if total > 0:
            sentiment = (pos_count - neg_count) / total
        else:
            sentiment = 0

        # Xác định tín hiệu
        if sentiment > 0.3:
            signal = SignalType.BUY
            confidence = 0.6 + sentiment * 0.2
        elif sentiment < -0.3:
            signal = SignalType.SELL
            confidence = 0.6 + abs(sentiment) * 0.2
        else:
            signal = SignalType.HOLD
            confidence = 0.5

        avg_evidence_conf = float(quality.get("avg_evidence_confidence", 0.0))
        snippet_ratio = float(quality.get("snippet_only_ratio", 1.0))
        confidence = max(0.2, min(0.95, confidence * (0.8 + 0.4 * avg_evidence_conf - 0.15 * snippet_ratio)))
        sources = [d.get("url") for d in deep_docs[:5] if d.get("url")] or [r.get("url") for r in results[:5] if r.get("url")]

        return {
            "signal": signal.value,
            "confidence": round(confidence, 3),
            "sentiment_score": round(sentiment, 3),
            "is_material": (len(results) > 3) or int(quality.get("high_priority_docs", 0)) > 0,
            "news_count": len(results),
            "sources": sources,
            "evidence_quality": quality,
            "reasoning": f"Sentiment: {sentiment:.2f} (Pos: {pos_count}, Neg: {neg_count})"
        }

    @staticmethod
    def _format_url_list(urls: List[str]) -> str:
        if not urls:
            return "[]"
        return ", ".join(urls[:8])

    def _build_news_context(self, news_data: Dict, document_intel: Dict) -> str:
        lines = []
        answer = news_data.get("answer")
        if answer:
            lines.append(f"- Radar answer: {str(answer)[:1200]}")

        radar_results = news_data.get("results", [])[:6]
        for idx, item in enumerate(radar_results, start=1):
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            url = str(item.get("url", "")).strip()
            lines.append(f"- Radar {idx}: {title} | {content[:500]} | {url}")

        deep_docs = document_intel.get("deep_dive_docs", []) if document_intel else []
        for idx, doc in enumerate(deep_docs[:4], start=1):
            body = doc.get("full_text") or doc.get("snippet", "")
            lines.append(
                f"- DeepDive {idx}: {doc.get('title', '')} | mode={doc.get('content_mode')} "
                f"| conf={doc.get('evidence_confidence', 0)} | {str(body)[:1000]} | {doc.get('url', '')}"
            )

        chunks = document_intel.get("evidence_chunks", []) if document_intel else []
        for idx, chunk in enumerate(chunks[:6], start=1):
            lines.append(
                f"- EvidenceChunk {idx}: tags={chunk.get('query_tags', [])} "
                f"| score={chunk.get('score')} | {str(chunk.get('text', ''))[:700]} | {chunk.get('url', '')}"
            )

        return "\n".join(lines) if lines else "Không có dữ liệu news."

    def _build_research_context(self, research: Dict) -> str:
        lines = []
        if research.get("answer"):
            lines.append(str(research.get("answer"))[:1600])
        for idx, item in enumerate(research.get("results", [])[:5], start=1):
            lines.append(
                f"{idx}. {item.get('title', '')} | {str(item.get('content', ''))[:800]} | {item.get('url', '')}"
            )
        return "\n".join(lines) if lines else "Không có dữ liệu."

    def _llm_analyze(self, symbol: str, news_data: Dict, research: Dict, document_intel: Dict) -> Dict:
        """Phân tích sâu bằng LLM"""

        quality = document_intel.get("quality", {}) if document_intel else {}
        news_context = self._build_news_context(news_data, document_intel or {})
        research_context = self._build_research_context(research or {})

        prompt = f"""Phân tích tin tức và cảm xúc thị trường cho {symbol} theo quy trình 2 tầng (Radar + DeepDive).

YÊU CẦU CHUYÊN MÔN:
1. Tách bạch tín hiệu Material vs Noise.
2. Đánh giá yếu tố N (CANSLIM) và Management (4M).
3. Nếu nguồn chỉ là snippet hoặc thiếu corroboration, phải giảm confidence.
4. Bắt buộc nêu sources (URL).

CHẤT LƯỢNG DỮ LIỆU:
- radar_count={quality.get("radar_count", 0)}
- deep_dive_count={quality.get("deep_dive_count", 0)}
- high_priority_docs={quality.get("high_priority_docs", 0)}
- snippet_only_ratio={quality.get("snippet_only_ratio", 1.0)}
- avg_evidence_confidence={quality.get("avg_evidence_confidence", 0.0)}

TIN TỨC VÀ BẰNG CHỨNG:
{news_context}

NGHIÊN CỨU CÔNG TY:
{research_context}

Trả về JSON đúng schema của News Agent."""

        result = self.llm.analyze("news_sentiment", prompt, symbol=symbol)

        # Đảm bảo có các trường cần thiết
        if "signal" not in result:
            result["signal"] = SignalType.HOLD.value
        if "confidence" not in result:
            result["confidence"] = 0.5
        if "sentiment_score" not in result:
            result["sentiment_score"] = 0

        if "sources" not in result or not result.get("sources"):
            deep_sources = [d.get("url") for d in document_intel.get("deep_dive_docs", [])[:6] if d.get("url")]
            radar_sources = [r.get("url") for r in news_data.get("results", [])[:6] if r.get("url")]
            result["sources"] = deep_sources or radar_sources

        if "material_events" not in result:
            result["material_events"] = []
        if "is_material" not in result:
            result["is_material"] = bool(int(quality.get("high_priority_docs", 0)) > 0)

        avg_evidence_conf = float(quality.get("avg_evidence_confidence", 0.0))
        snippet_ratio = float(quality.get("snippet_only_ratio", 1.0))
        base_conf = float(result.get("confidence", 0.5))
        quality_mult = max(0.35, min(1.05, 0.75 + 0.45 * avg_evidence_conf - 0.20 * snippet_ratio))
        result["confidence"] = round(max(0.05, min(0.98, base_conf * quality_mult)), 4)
        result["evidence_quality"] = quality

        return result
