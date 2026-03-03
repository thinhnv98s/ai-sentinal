# Source Code Snapshot

Generated from workspace: /Users/Nero/PycharmProjects/ai-sentinel

## `./.env.example`

```dotenv
# ==============================================
# SENTINEL AI TRADING SYSTEM - Environment Variables
# ==============================================
# Copy file này thành .env và điền API keys của bạn
# ==============================================

# ----------------------------------------------
# TAVILY API (Tìm kiếm tin tức tài chính)
# ----------------------------------------------
# Đăng ký tại: https://tavily.com/
TAVILY_API_KEY=
TAVILY_API_BASE_URL=https://api.tavily.com

# ----------------------------------------------
# LLM API (OpenAI-compatible endpoint)
# ----------------------------------------------
# Endpoint mặc định theo yêu cầu: https://vertex-key.com/api/v1
LLM_API_KEY=
LLM_BASE_URL=https://vertex-key.com/api/v1

# ----------------------------------------------
# CẤU HÌNH HỆ THỐNG
# ----------------------------------------------
# Chọn provider: "openai_compatible"
LLM_PROVIDER=openai_compatible

# Model mặc định cho từng vai trò
LLM_MODEL_GENERAL=flash/claude-opus-4-6
LLM_MODEL_ANALYST=flash/claude-sonnet-4-6
LLM_MODEL_WORKER=flash/claude-sonnet-4-6

# Optional: thư mục xuất file markdown review chất lượng mỗi run
SENTINEL_RUN_LOG_DIR=./logs/run_reports

# ==============================================
# HƯỚNG DẪN:
# 1. Copy file này: cp .env.example .env
# 2. Điền API keys vào file .env
# 3. Chạy: python main.py
# ==============================================

```

## `./Makefile`

```makefile
run:
	python3 main.py
```

## `./agents.py`

```python
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

```

## `./blackboard.py`

```python
"""
BẢNG ĐEN TOÀN CỤC (GLOBAL BLACKBOARD)
=====================================
Quản lý trạng thái chia sẻ và giao tiếp giữa các tác tử
Theo mẫu thiết kế Blackboard Architecture trong base.txt
"""

import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import Lock

logger = logging.getLogger("Sentinel.Blackboard")

try:
    from run_logger import get_run_logger
    RUN_LOGGER_AVAILABLE = True
except Exception:
    RUN_LOGGER_AVAILABLE = False

# --- ĐỊNH NGHĨA GIAO THỨC GIAO TIẾP ---

class MessageType(Enum):
    """
    Các loại hành động giao tiếp (Performatives) theo chuẩn FIPA-ACL
    Tham chiếu: base.txt Section 3.2
    """
    TASK_ASSIGNMENT = "TASK_ASSIGNMENT"      # Tướng quân giao việc
    DATA_REPORT = "DATA_REPORT"              # Trinh sát báo cáo dữ liệu thô
    ANALYSIS_RESULT = "ANALYSIS_RESULT"      # Chuyên gia báo cáo kết quả phân tích
    SIGNAL = "SIGNAL"                        # Tín hiệu giao dịch
    REGIME_UPDATE = "REGIME_UPDATE"          # Cập nhật chế độ thị trường
    RISK_ALERT = "RISK_ALERT"                # Cảnh báo rủi ro
    VETO = "VETO"                            # Quyền phủ quyết từ Risk Guardian
    FINAL_DECISION = "FINAL_DECISION"        # Quyết định cuối cùng
    ERROR = "ERROR"                          # Thông báo lỗi
    INFO = "INFO"                            # Thông tin chung

class SignalType(Enum):
    """Các loại tín hiệu giao dịch"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_SIGNAL = "NO_SIGNAL"

class MarketRegime(Enum):
    """
    Chế độ thị trường (Market Regime)
    Tham chiếu: base.txt Section 3.1.1 - Tướng quân Vĩ mô
    """
    RISK_ON = "RISK_ON"          # Thị trường tăng - Chấp nhận rủi ro
    RISK_OFF = "RISK_OFF"        # Thị trường giảm - Né tránh rủi ro
    SIDEWAYS = "SIDEWAYS"        # Đi ngang
    UNKNOWN = "UNKNOWN"

class WyckoffPhase(Enum):
    """
    Các pha của chu kỳ Wyckoff
    Tham chiếu: base.txt Section 2.1
    """
    ACCUMULATION = "ACCUMULATION"    # Tích lũy
    MARKUP = "MARKUP"                # Tăng giá
    DISTRIBUTION = "DISTRIBUTION"   # Phân phối
    MARKDOWN = "MARKDOWN"           # Giảm giá
    UNKNOWN = "UNKNOWN"

@dataclass
class AgentMessage:
    """
    Cấu trúc tin nhắn chuẩn JSON-schema
    Tham chiếu: base.txt Section 3.2 - Giao thức giao tiếp
    """
    sender: str
    receiver: str  # "ALL" cho broadcast, hoặc tên tác tử cụ thể
    msg_type: MessageType
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: str = "NORMAL"  # LOW, NORMAL, HIGH, CRITICAL

    def to_dict(self) -> Dict:
        """Chuyển đổi sang dictionary"""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "msg_type": self.msg_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "id": self.id,
            "priority": self.priority
        }

    def to_json(self) -> str:
        """Chuyển đổi sang JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

@dataclass
class TradingSignal:
    """
    Cấu trúc tín hiệu giao dịch
    """
    symbol: str
    signal: SignalType
    confidence: float  # 0.0 - 1.0
    source: str  # Tác tử phát tín hiệu
    reasoning: str  # Lý do
    artifacts: Dict[str, Any] = field(default_factory=dict)  # Dữ liệu bổ sung
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "confidence": self.confidence,
            "source": self.source,
            "reasoning": self.reasoning,
            "artifacts": self.artifacts,
            "timestamp": self.timestamp
        }

@dataclass
class TradeOrder:
    """
    Cấu trúc lệnh giao dịch (Output của hệ thống)
    """
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: float
    order_type: str  # MARKET, LIMIT
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

class Blackboard:
    """
    BẢNG ĐEN TOÀN CỤC

    Nơi lưu trữ trạng thái chung và luân chuyển tin nhắn giữa các tác tử.
    Các tác tử ghi kết quả vào Bảng đen thay vì gửi tin nhắn trực tiếp,
    giúp tránh "hiệu ứng tam sao thất bản" và giảm chi phí token.

    Tham chiếu: base.txt Section 3.2
    """

    def __init__(self):
        self._lock = Lock()  # Thread-safe

        # Cấu trúc bộ nhớ phân cấp
        self.memory: Dict[str, Any] = {
            # Dữ liệu thị trường thô
            "market_data": {},  # {symbol: {prices, volumes, fundamentals}}

            # Trạng thái các tác tử
            "agent_state": {},  # {agent_name: {status, last_active}}

            # Kết quả phân tích từ các Binh lính
            "analysis_layer": {
                "wyckoff": {},   # {symbol: WyckoffAnalysis}
                "canslim": {},   # {symbol: CANSLIMAnalysis}
                "fourm": {},     # {symbol: FourMAnalysis}
                "news": {},      # {symbol: NewsAnalysis}
            },

            # Chế độ thị trường (từ Tướng quân Vĩ mô)
            "regime": {
                "current": MarketRegime.UNKNOWN.value,
                "confidence": 0.0,
                "last_update": None
            },

            # Kết quả đồng thuận Bayesian
            "consensus": {},  # {symbol: {probability, signals, decision}}

            # Các tín hiệu giao dịch tổng hợp
            "signals": [],  # List[TradingSignal]

            # Danh sách lệnh giao dịch output
            "trade_orders": [],  # List[TradeOrder]

            # Lịch sử quyết định
            "decision_history": []
        }

        # Hàng đợi tin nhắn (Message Bus)
        self.message_bus: List[AgentMessage] = []

        # Các tin nhắn đã xử lý (để tránh xử lý trùng)
        self.processed_message_ids: set = set()

        logger.info("Blackboard khởi tạo thành công")

    def post_message(self, message: AgentMessage):
        """
        Tác tử ghi tin nhắn lên Bảng đen
        """
        with self._lock:
            self.message_bus.append(message)
            logger.debug(f"[BLACKBOARD] Nhận tin từ {message.sender} [{message.msg_type.value}]")

            if RUN_LOGGER_AVAILABLE:
                try:
                    get_run_logger().log_blackboard_message(message.to_dict())
                except Exception:
                    # Logging không được phép ảnh hưởng logic chính
                    pass

            # Tự động cập nhật bộ nhớ tương ứng
            self._process_message(message)

    def _process_message(self, message: AgentMessage):
        """Xử lý và lưu trữ tin nhắn vào bộ nhớ phù hợp"""

        content = message.content

        if message.msg_type == MessageType.DATA_REPORT:
            # Dữ liệu thị trường thô
            symbol = content.get("symbol", "UNKNOWN")
            self.memory["market_data"][symbol] = content

        elif message.msg_type == MessageType.ANALYSIS_RESULT:
            # Kết quả phân tích từ các tác tử
            symbol = content.get("symbol", "UNKNOWN")
            source = content.get("source", message.sender).lower()

            # Xác định layer phù hợp
            if "wyckoff" in source:
                self.memory["analysis_layer"]["wyckoff"][symbol] = content
            elif "canslim" in source:
                self.memory["analysis_layer"]["canslim"][symbol] = content
            elif "fourm" in source or "4m" in source:
                self.memory["analysis_layer"]["fourm"][symbol] = content
            elif "news" in source or "sentiment" in source:
                self.memory["analysis_layer"]["news"][symbol] = content

        elif message.msg_type == MessageType.REGIME_UPDATE:
            # Cập nhật chế độ thị trường
            self.memory["regime"] = {
                "current": content.get("regime", MarketRegime.UNKNOWN.value),
                "confidence": content.get("confidence", 0.0),
                "last_update": message.timestamp
            }

        elif message.msg_type == MessageType.SIGNAL:
            # Tín hiệu giao dịch
            self.memory["signals"].append(content)

        elif message.msg_type == MessageType.FINAL_DECISION:
            # Quyết định cuối cùng
            self.memory["decision_history"].append(content)

            # Tạo TradeOrder nếu cần
            if content.get("action") in ["BUY", "SELL"]:
                order = TradeOrder(
                    symbol=content.get("symbol"),
                    action=content.get("action"),
                    quantity=content.get("quantity", 0),
                    order_type=content.get("order_type", "MARKET"),
                    stop_loss=content.get("stop_loss"),
                    take_profit=content.get("take_profit"),
                    reasoning=content.get("reasoning", ""),
                    confidence=content.get("confidence", 0)
                )
                self.memory["trade_orders"].append(order)

    def read_memory(self, section: str, key: Optional[str] = None) -> Any:
        """
        Tác tử đọc dữ liệu từ bộ nhớ chia sẻ
        """
        with self._lock:
            if section not in self.memory:
                return None
            if key:
                return self.memory[section].get(key)
            return self.memory[section]

    def write_memory(self, section: str, key: str, value: Any):
        """
        Ghi trực tiếp vào bộ nhớ (cho các trường hợp đặc biệt)
        """
        with self._lock:
            if section not in self.memory:
                self.memory[section] = {}
            self.memory[section][key] = value

    def get_messages_for(self, receiver_name: str,
                         unprocessed_only: bool = True) -> List[AgentMessage]:
        """
        Lọc tin nhắn dành cho một tác tử cụ thể
        """
        with self._lock:
            messages = []
            for msg in self.message_bus:
                # Kiểm tra người nhận
                if msg.receiver not in [receiver_name, "ALL", "Blackboard"]:
                    continue

                # Kiểm tra đã xử lý chưa
                if unprocessed_only and msg.id in self.processed_message_ids:
                    continue

                messages.append(msg)

                # Đánh dấu đã xử lý
                if unprocessed_only:
                    self.processed_message_ids.add(msg.id)

            return messages

    def get_all_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Lấy tất cả kết quả phân tích cho một mã cổ phiếu
        """
        with self._lock:
            analysis = {}
            for source in ["wyckoff", "canslim", "fourm", "news"]:
                data = self.memory["analysis_layer"][source].get(symbol)
                if data:
                    analysis[source] = data
            return analysis

    def get_current_regime(self) -> Dict[str, Any]:
        """
        Lấy chế độ thị trường hiện tại
        """
        return self.memory["regime"]

    def get_trade_orders(self) -> List[TradeOrder]:
        """
        Lấy danh sách lệnh giao dịch (Output chính của hệ thống)
        """
        return self.memory["trade_orders"]

    def clear_orders(self):
        """Xóa danh sách lệnh sau khi đã xử lý"""
        with self._lock:
            self.memory["trade_orders"] = []

    def get_summary(self) -> Dict:
        """
        Lấy tóm tắt trạng thái Blackboard
        """
        return {
            "market_data_symbols": list(self.memory["market_data"].keys()),
            "regime": self.memory["regime"],
            "total_signals": len(self.memory["signals"]),
            "pending_orders": len(self.memory["trade_orders"]),
            "total_messages": len(self.message_bus)
        }

```

## `./config.py`

```python
"""
CẤU HÌNH HỆ THỐNG SENTINEL AI TRADING
=====================================
Cấu hình API keys và các thông số hệ thống
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum

# --- API KEYS (Để trống cho người dùng tự cấu hình) ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_API_BASE_URL = os.environ.get("TAVILY_API_BASE_URL", "https://api.tavily.com")

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://vertex-key.com/api/v1")

# Backward-compatible aliases (để không vỡ các import cũ)
ZAI_API_KEY = os.environ.get("ZAI_API_KEY", LLM_API_KEY)
ZAI_CHAT_COMPLETIONS_URL = os.environ.get("ZAI_CHAT_COMPLETIONS_URL", f"{LLM_BASE_URL.rstrip('/')}/chat/completions")

# --- LLM PROVIDER SELECTION ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai_compatible")

# --- MODEL CONFIG ---
LLM_MODEL_GENERAL = os.environ.get("LLM_MODEL_GENERAL", "flash/claude-opus-4-6")
LLM_MODEL_ANALYST = os.environ.get("LLM_MODEL_ANALYST", "flash/claude-sonnet-4-6")
LLM_MODEL_WORKER = os.environ.get("LLM_MODEL_WORKER", "flash/claude-sonnet-4-6")

# Backward-compatible model aliases
ZAI_MODEL_GENERAL = os.environ.get("ZAI_MODEL_GENERAL", LLM_MODEL_GENERAL)
ZAI_MODEL_ANALYST = os.environ.get("ZAI_MODEL_ANALYST", LLM_MODEL_ANALYST)
ZAI_MODEL_WORKER = os.environ.get("ZAI_MODEL_WORKER", LLM_MODEL_WORKER)

# --- CẤU HÌNH MÔ HÌNH LLM ---
class LLMProvider(Enum):
    """Lựa chọn provider cho LLM"""
    OPENAI_COMPATIBLE = "openai_compatible"

class LLMModel(Enum):
    """Định nghĩa các mô hình LLM sử dụng trong hệ thống"""
    CLAUDE_SONNET_46 = "flash/claude-sonnet-4-6"
    CLAUDE_OPUS_46 = "flash/claude-opus-4-6"

@dataclass
class LLMConfig:
    """Cấu hình cho LLM"""
    provider: str = LLM_PROVIDER
    api_key: str = os.environ.get("LLM_API_KEY", os.environ.get("ZAI_API_KEY", ""))
    base_url: str = os.environ.get("LLM_BASE_URL", "https://vertex-key.com/api/v1")

    # Models cho từng vai trò
    model_general: str = LLM_MODEL_GENERAL     # Tướng quân (Opus)
    model_analyst: str = LLM_MODEL_ANALYST     # Binh lính phân tích (Sonnet)
    model_worker: str = LLM_MODEL_WORKER       # Worker tasks (Sonnet)

    max_tokens: int = 4096
    temperature: float = 0.3  # Thấp để đảm bảo tính nhất quán
    # Tùy chọn giới hạn tần suất gọi API (0 = tắt)
    min_call_interval_seconds: float = float(os.environ.get("LLM_MIN_CALL_INTERVAL_SECONDS", "0"))

@dataclass
class TavilyConfig:
    """Cấu hình cho Tavily API"""
    search_depth: str = "advanced"  # "basic" hoặc "advanced"
    max_results: int = 10
    include_answer: bool = True
    include_raw_content: bool = False
    # Nguồn tin tài chính uy tín (theo base.txt)
    trusted_domains_vn: List[str] = field(default_factory=lambda: [
        "vietstock.vn", "cafef.vn", "tinnhanhchungkhoan.vn",
        "hsx.vn", "hnx.vn", "vneconomy.vn", "stockbiz.vn"
    ])
    trusted_domains_us: List[str] = field(default_factory=lambda: [
        "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
        "seekingalpha.com", "yahoo.com/finance", "cnbc.com"
    ])

# --- CẤU HÌNH CHIẾN LƯỢC ---
@dataclass
class WyckoffConfig:
    """Cấu hình cho phân tích Wyckoff"""
    support_lookback: int = 50  # Số thanh để xác định hỗ trợ
    volume_multiplier: float = 1.5  # Ngưỡng volume cao (150% trung bình)
    spring_tolerance: float = 0.02  # Sai số cho phép (2%)

@dataclass
class CANSLIMConfig:
    """Cấu hình cho chiến lược CANSLIM"""
    min_eps_growth: float = 0.05
    min_annual_growth: float = 0.10  # CAGR 25% trong 3-5 năm
    min_roe: float = 0.08  # ROE > 17%
    min_rs_rating: int = 70
    high_52w_tolerance: float = 0.15  # Trong 15% so với đỉnh 52 tuần

@dataclass
class FourMConfig:
    """Cấu hình cho khung 4M (Phil Town)"""
    min_roic: float = 0.05  # ROIC > 10% trong 10 năm
    marr: float = 0.15  # Minimum Acceptable Rate of Return (15%)
    mos_discount: float = 0.85  # Biên độ an toàn 50%
    max_pe: int = 40  # P/E tối đa chấp nhận

@dataclass
class RiskConfig:
    """Cấu hình quản trị rủi ro"""
    max_position_pct: float = 0.15  # Tối đa 5% vốn mỗi vị thế
    max_sector_exposure: float = 0.25  # Tối đa 25% vốn mỗi ngành
    max_drawdown: float = 0.35  # Drawdown tối đa 20%
    stop_loss_atr_multiplier: float = 1.8  # Stop-loss = 2 x ATR
    risk_per_trade_pct: float = 0.02  # Rủi ro vốn tối đa mỗi lệnh (1%)
    allow_short: bool = os.environ.get("ALLOW_SHORT", "false").lower() == "true"

@dataclass
class BayesianConfig:
    """Cấu hình cho Bayesian Consensus"""
    prior_probability: float = 0.5  # Xác suất tiên nghiệm 50%
    min_probability_threshold: float = 0.48
    # Hồ sơ độ tin cậy của các tác tử (Sensitivity/Specificity)
    agent_profiles: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "wyckoff": {"sensitivity": 0.70, "specificity": 0.60},
        "canslim": {"sensitivity": 0.75, "specificity": 0.65},
        "fourm": {"sensitivity": 0.60, "specificity": 0.90},
        "news": {"sensitivity": 0.65, "specificity": 0.55}
    })
    # Điều chỉnh trọng số theo chế độ thị trường (regime-aware)
    regime_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "RISK_ON": {
            "wyckoff": 1.10,
            "canslim": 1.10,
            "fourm": 0.95,
            "news": 1.00
        },
        "RISK_OFF": {
            "wyckoff": 0.90,
            "canslim": 0.70,
            "fourm": 1.15,
            "news": 1.15
        },
        "SIDEWAYS": {
            "wyckoff": 1.00,
            "canslim": 0.95,
            "fourm": 1.05,
            "news": 1.00
        },
        "UNKNOWN": {}
    })


@dataclass
class SystemConfig:
    """Cấu hình vận hành hệ thống (tối ưu throughput)"""
    max_parallel_workers: int = int(os.environ.get("MAX_PARALLEL_WORKERS", "8"))
    enable_regime_gating: bool = os.environ.get("ENABLE_REGIME_GATING", "true").lower() == "true"
    risk_off_skip_news: bool = os.environ.get("RISK_OFF_SKIP_NEWS", "true").lower() == "true"
    risk_off_skip_fundamental_agents: bool = os.environ.get("RISK_OFF_SKIP_FUND_AGENTS", "true").lower() == "true"
    enable_event_driven_filter: bool = os.environ.get("ENABLE_EVENT_FILTER", "true").lower() == "true"
    event_price_move_threshold: float = float(os.environ.get("EVENT_PRICE_MOVE_THRESHOLD", "0.015"))
    event_volume_spike_multiplier: float = float(os.environ.get("EVENT_VOLUME_SPIKE_MULTIPLIER", "1.3"))
    enable_fact_check: bool = os.environ.get("ENABLE_FACT_CHECK", "true").lower() == "true"


@dataclass
class DocumentIntelligenceConfig:
    """Cấu hình pipeline 2 tầng Radar -> DeepDive -> Evidence"""
    enable_two_tier_pipeline: bool = os.environ.get("ENABLE_TWO_TIER_PIPELINE", "true").lower() == "true"

    # Tier-1: Radar scan (rẻ, nhanh)
    radar_search_depth: str = os.environ.get("RADAR_SEARCH_DEPTH", "basic")
    radar_max_results: int = int(os.environ.get("RADAR_MAX_RESULTS", "12"))

    # Tier-2: DeepDive (đọc sâu có điều kiện)
    deep_dive_enabled: bool = os.environ.get("DEEP_DIVE_ENABLED", "true").lower() == "true"
    deep_dive_top_k: int = int(os.environ.get("DEEP_DIVE_TOP_K", "4"))
    deep_dive_min_priority: float = float(os.environ.get("DEEP_DIVE_MIN_PRIORITY", "0.55"))
    deep_dive_min_uncertainty: float = float(os.environ.get("DEEP_DIVE_MIN_UNCERTAINTY", "0.45"))
    deep_dive_token_budget: int = int(os.environ.get("DEEP_DIVE_TOKEN_BUDGET", "5000"))
    deep_dive_target_tokens_per_doc: int = int(os.environ.get("DEEP_DIVE_TARGET_TOKENS_PER_DOC", "1200"))
    max_fulltext_chars: int = int(os.environ.get("MAX_FULLTEXT_CHARS", "20000"))

    # Fetch strategy
    enable_tavily_extract: bool = os.environ.get("ENABLE_TAVILY_EXTRACT", "true").lower() == "true"
    enable_jina_reader_fallback: bool = os.environ.get("ENABLE_JINA_READER_FALLBACK", "false").lower() == "true"
    jina_timeout_seconds: float = float(os.environ.get("JINA_TIMEOUT_SECONDS", "8"))

    # Evidence quality weighting
    snippet_confidence_penalty: float = float(os.environ.get("SNIPPET_CONFIDENCE_PENALTY", "0.65"))
    corroboration_min_sources: int = int(os.environ.get("CORROBORATION_MIN_SOURCES", "2"))

    # Mini-RAG settings cho tài liệu dài
    rag_enabled: bool = os.environ.get("DOC_RAG_ENABLED", "true").lower() == "true"
    rag_chunk_words: int = int(os.environ.get("DOC_RAG_CHUNK_WORDS", "180"))
    rag_chunk_overlap_words: int = int(os.environ.get("DOC_RAG_CHUNK_OVERLAP_WORDS", "40"))
    rag_top_chunks: int = int(os.environ.get("DOC_RAG_TOP_CHUNKS", "8"))
    rag_rrf_k: int = int(os.environ.get("DOC_RAG_RRF_K", "60"))
    rag_mmr_lambda: float = float(os.environ.get("DOC_RAG_MMR_LAMBDA", "0.7"))

# --- CẤU HÌNH HỆ THỐNG CHÍNH ---
@dataclass
class SentinelConfig:
    """Cấu hình tổng thể cho Sentinel"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    wyckoff: WyckoffConfig = field(default_factory=WyckoffConfig)
    canslim: CANSLIMConfig = field(default_factory=CANSLIMConfig)
    fourm: FourMConfig = field(default_factory=FourMConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    document_intel: DocumentIntelligenceConfig = field(default_factory=DocumentIntelligenceConfig)

    # Chế độ thị trường (Market Regime)
    market_regime: str = "UNKNOWN"  # BULL, BEAR, SIDEWAYS, UNKNOWN

    # Debug mode
    debug: bool = False


    # Watchlist mặc định
    default_watchlist: List[str] = field(default_factory=lambda: [
        "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN"
    ])

# Singleton config instance
_config_instance = None

def get_config() -> SentinelConfig:
    """Lấy instance cấu hình (Singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = SentinelConfig()
    return _config_instance

def update_config(**kwargs):
    """Cập nhật cấu hình"""
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

```

## `./cost_tracker.py`

```python
"""
COST TRACKER - Theo dõi Chi phí API
===================================
Giám sát chi phí LLM và tối ưu hóa
Tham chiếu: base.txt Section 7
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger("Sentinel.CostTracker")


@dataclass
class APICall:
    """Ghi nhận một lần gọi API"""
    timestamp: str
    model: str
    provider: str  # anthropic, vertex, zai
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    role: str = ""  # macro_general, wyckoff, etc.
    symbol: str = ""
    latency_ms: float = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class CostSummary:
    """Tổng kết chi phí"""
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    estimated_cost_usd: float
    savings_from_cache_usd: float
    average_latency_ms: float
    success_rate: float


# Bảng giá Claude API (2026) - Tham chiếu base.txt Section 7.2
PRICING = {
    "anthropic": {
        "claude-opus-4-20250514": {
            "input": 5.00 / 1_000_000,      # $5.00/1M tokens
            "output": 25.00 / 1_000_000,    # $25.00/1M tokens
            "cache_write": 6.25 / 1_000_000,  # $6.25/1M tokens
            "cache_read": 0.50 / 1_000_000,   # $0.50/1M tokens
        },
        "claude-sonnet-4-20250514": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
            "cache_write": 3.75 / 1_000_000,
            "cache_read": 0.30 / 1_000_000,
        },
        "claude-haiku-4-5-20250514": {
            "input": 1.00 / 1_000_000,
            "output": 5.00 / 1_000_000,
            "cache_write": 1.25 / 1_000_000,
            "cache_read": 0.10 / 1_000_000,
        },
    },
    "vertex": {
        # Vertex AI pricing (tương tự Anthropic)
        "claude-opus-4-6": {
            "input": 5.00 / 1_000_000,
            "output": 25.00 / 1_000_000,
            "cache_write": 6.25 / 1_000_000,
            "cache_read": 0.50 / 1_000_000,
        },
        "claude-sonnet-4-5": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
            "cache_write": 3.75 / 1_000_000,
            "cache_read": 0.30 / 1_000_000,
        },
        "claude-haiku-4-5": {
            "input": 1.00 / 1_000_000,
            "output": 5.00 / 1_000_000,
            "cache_write": 1.25 / 1_000_000,
            "cache_read": 0.10 / 1_000_000,
        },
    },
    # Z.AI pricing cấu hình qua ENV để dễ cập nhật mà không sửa code.
    # Đơn vị: USD / 1M token.
    "zai": {
        "glm-4.7-flash": {
            "input": float(os.environ.get("ZAI_GLM47_FLASH_INPUT_PER_M", "1.0")) / 1_000_000,
            "output": float(os.environ.get("ZAI_GLM47_FLASH_OUTPUT_PER_M", "5.0")) / 1_000_000,
            "cache_write": float(os.environ.get("ZAI_GLM47_FLASH_CACHE_WRITE_PER_M", "1.25")) / 1_000_000,
            "cache_read": float(os.environ.get("ZAI_GLM47_FLASH_CACHE_READ_PER_M", "0.1")) / 1_000_000,
        },
        "glm-4.7": {
            "input": float(os.environ.get("ZAI_GLM47_INPUT_PER_M", "3.0")) / 1_000_000,
            "output": float(os.environ.get("ZAI_GLM47_OUTPUT_PER_M", "15.0")) / 1_000_000,
            "cache_write": float(os.environ.get("ZAI_GLM47_CACHE_WRITE_PER_M", "3.75")) / 1_000_000,
            "cache_read": float(os.environ.get("ZAI_GLM47_CACHE_READ_PER_M", "0.3")) / 1_000_000,
        },
    },
    "openai_compatible": {
        "flash/claude-opus-4-6": {
            "input": 5.00 / 1_000_000,
            "output": 25.00 / 1_000_000,
            "cache_write": 6.25 / 1_000_000,
            "cache_read": 0.50 / 1_000_000,
        },
        "flash/claude-sonnet-4-6": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
            "cache_write": 3.75 / 1_000_000,
            "cache_read": 0.30 / 1_000_000,
        },
    },
}

# Tavily pricing
TAVILY_PRICING = {
    "basic_search": 0.004,    # 1 credit = $0.004 (pay-as-you-go)
    "advanced_search": 0.008,  # 2 credits = $0.008
}


class CostTracker:
    """
    Theo dõi và phân tích chi phí API

    Tính năng:
    - Ghi nhận từng lần gọi API
    - Tính toán chi phí thực tế vs dự kiến
    - Theo dõi tiết kiệm từ caching
    - Báo cáo theo thời gian/agent/symbol
    """

    def __init__(self, budget_daily_usd: float = 50.0):
        self.budget_daily_usd = budget_daily_usd
        self.calls: List[APICall] = []
        self.tavily_calls: List[Dict] = []

        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0

        # Alerts
        self.alerts: List[str] = []

    def track_llm_call(self,
                       model: str,
                       provider: str,
                       input_tokens: int,
                       output_tokens: int,
                       cached_tokens: int = 0,
                       role: str = "",
                       symbol: str = "",
                       latency_ms: float = 0,
                       success: bool = True,
                       error: str = None):
        """
        Ghi nhận một lần gọi LLM API
        """
        call = APICall(
            timestamp=datetime.now().isoformat(),
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            role=role,
            symbol=symbol,
            latency_ms=latency_ms,
            success=success,
            error=error
        )
        self.calls.append(call)

        # Update cache metrics
        if cached_tokens > 0:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Check budget
        daily_cost = self.get_daily_cost()
        if daily_cost > self.budget_daily_usd * 0.8:
            alert = f"⚠️ Chi phí đạt {daily_cost:.2f}/{self.budget_daily_usd:.2f} (80% budget)"
            self.alerts.append(alert)
            logger.warning(alert)

        logger.debug(f"API call tracked: {model} - {input_tokens}+{output_tokens} tokens")

    def track_tavily_call(self,
                          query: str,
                          search_depth: str,
                          credits_used: int = 1):
        """
        Ghi nhận một lần gọi Tavily API
        """
        cost = credits_used * TAVILY_PRICING.get(f"{search_depth}_search", 0.004)

        self.tavily_calls.append({
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "search_depth": search_depth,
            "credits": credits_used,
            "cost_usd": cost
        })

    def calculate_call_cost(self, call: APICall) -> Dict[str, float]:
        """
        Tính chi phí cho một lần gọi
        """
        pricing = PRICING.get(call.provider, {}).get(call.model, {})

        if not pricing:
            logger.warning(
                "Không có pricing cho provider=%s model=%s, tạm tính 0 USD",
                call.provider,
                call.model,
            )
            pricing = {"input": 0.0, "output": 0.0, "cache_read": 0.0}

        # Input cost (with cache consideration)
        non_cached_input = call.input_tokens - call.cached_tokens
        input_cost = non_cached_input * pricing.get("input", 0)
        cache_cost = call.cached_tokens * pricing.get("cache_read", 0)

        # Output cost
        output_cost = call.output_tokens * pricing.get("output", 0)

        # Cost without cache (for comparison)
        cost_without_cache = call.input_tokens * pricing.get("input", 0) + output_cost

        total_cost = input_cost + cache_cost + output_cost
        savings = cost_without_cache - total_cost

        return {
            "input_cost": input_cost,
            "cache_cost": cache_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_without_cache": cost_without_cache,
            "savings": savings
        }

    def get_daily_cost(self, date: datetime = None) -> float:
        """Tính tổng chi phí trong ngày"""
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y-%m-%d")

        total = 0.0

        # LLM costs
        for call in self.calls:
            if call.timestamp.startswith(date_str):
                cost = self.calculate_call_cost(call)
                total += cost["total_cost"]

        # Tavily costs
        for tcall in self.tavily_calls:
            if tcall["timestamp"].startswith(date_str):
                total += tcall["cost_usd"]

        return total

    def get_summary(self,
                    start_date: datetime = None,
                    end_date: datetime = None) -> CostSummary:
        """
        Tạo báo cáo tổng hợp
        """
        # Filter by date range
        calls = self.calls
        if start_date or end_date:
            calls = [c for c in self.calls if self._in_range(c.timestamp, start_date, end_date)]

        if not calls:
            return CostSummary(
                total_calls=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cached_tokens=0,
                estimated_cost_usd=0,
                savings_from_cache_usd=0,
                average_latency_ms=0,
                success_rate=1.0
            )

        total_input = sum(c.input_tokens for c in calls)
        total_output = sum(c.output_tokens for c in calls)
        total_cached = sum(c.cached_tokens for c in calls)

        total_cost = 0.0
        total_savings = 0.0

        for call in calls:
            cost_breakdown = self.calculate_call_cost(call)
            total_cost += cost_breakdown["total_cost"]
            total_savings += cost_breakdown["savings"]

        avg_latency = sum(c.latency_ms for c in calls) / len(calls)
        success_count = sum(1 for c in calls if c.success)

        return CostSummary(
            total_calls=len(calls),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cached_tokens=total_cached,
            estimated_cost_usd=round(total_cost, 4),
            savings_from_cache_usd=round(total_savings, 4),
            average_latency_ms=round(avg_latency, 2),
            success_rate=round(success_count / len(calls), 3)
        )

    def get_cost_by_role(self) -> Dict[str, float]:
        """Chi phí theo từng role (agent)"""
        costs = defaultdict(float)

        for call in self.calls:
            cost = self.calculate_call_cost(call)
            role = call.role or "unknown"
            costs[role] += cost["total_cost"]

        return dict(costs)

    def get_cost_by_symbol(self) -> Dict[str, float]:
        """Chi phí theo từng symbol"""
        costs = defaultdict(float)

        for call in self.calls:
            cost = self.calculate_call_cost(call)
            symbol = call.symbol or "unknown"
            costs[symbol] += cost["total_cost"]

        return dict(costs)

    def get_cache_efficiency(self) -> Dict[str, Any]:
        """Thống kê hiệu quả caching"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        total_cached_tokens = sum(c.cached_tokens for c in self.calls)
        total_input_tokens = sum(c.input_tokens for c in self.calls)

        cache_ratio = total_cached_tokens / total_input_tokens if total_input_tokens > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 3),
            "total_cached_tokens": total_cached_tokens,
            "cache_token_ratio": round(cache_ratio, 3),
            "estimated_savings_pct": round(cache_ratio * 0.9, 3)  # ~90% savings potential
        }

    def get_report(self) -> Dict[str, Any]:
        """
        Báo cáo chi tiết cho display
        """
        summary = self.get_summary()
        cache_stats = self.get_cache_efficiency()

        return {
            "summary": {
                "total_calls": summary.total_calls,
                "total_tokens": summary.total_input_tokens + summary.total_output_tokens,
                "estimated_cost_usd": summary.estimated_cost_usd,
                "savings_usd": summary.savings_from_cache_usd,
                "success_rate": f"{summary.success_rate:.1%}",
                "avg_latency_ms": summary.average_latency_ms,
            },
            "cache_efficiency": cache_stats,
            "cost_by_role": self.get_cost_by_role(),
            "cost_by_symbol": self.get_cost_by_symbol(),
            "daily_cost": self.get_daily_cost(),
            "budget_remaining": self.budget_daily_usd - self.get_daily_cost(),
            "alerts": self.alerts[-5:],  # Last 5 alerts
            "tavily_cost": sum(t["cost_usd"] for t in self.tavily_calls),
            "timestamp": datetime.now().isoformat()
        }

    def _in_range(self, timestamp: str, start: datetime, end: datetime) -> bool:
        """Check if timestamp is in range"""
        ts = datetime.fromisoformat(timestamp)
        if start and ts < start:
            return False
        if end and ts > end:
            return False
        return True

    def export_json(self, filepath: str = None) -> str:
        """Export tracking data to JSON"""
        data = {
            "report": self.get_report(),
            "calls": [
                {
                    "timestamp": c.timestamp,
                    "model": c.model,
                    "provider": c.provider,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cached_tokens": c.cached_tokens,
                    "role": c.role,
                    "symbol": c.symbol,
                    "cost": self.calculate_call_cost(c)["total_cost"]
                }
                for c in self.calls
            ],
            "tavily_calls": self.tavily_calls
        }

        json_str = json.dumps(data, indent=2, ensure_ascii=False)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            logger.info(f"Cost report exported to {filepath}")

        return json_str

    def print_report(self):
        """In báo cáo chi phí ra console"""
        report = self.get_report()

        print("\n" + "="*60)
        print("💰 BÁO CÁO CHI PHÍ API")
        print("="*60)

        print(f"\n📊 TỔNG QUAN:")
        print(f"   Tổng số calls: {report['summary']['total_calls']}")
        print(f"   Tổng tokens: {report['summary']['total_tokens']:,}")
        print(f"   Chi phí ước tính: ${report['summary']['estimated_cost_usd']:.4f}")
        print(f"   Tiết kiệm từ cache: ${report['summary']['savings_usd']:.4f}")
        print(f"   Success rate: {report['summary']['success_rate']}")

        cache = report['cache_efficiency']
        print(f"\n🗃️ CACHE EFFICIENCY:")
        print(f"   Hit rate: {cache['hit_rate']:.1%}")
        print(f"   Cached tokens: {cache['total_cached_tokens']:,}")
        print(f"   Tiềm năng tiết kiệm: {cache['estimated_savings_pct']:.1%}")

        print(f"\n📈 CHI PHÍ THEO ROLE:")
        for role, cost in report['cost_by_role'].items():
            print(f"   {role}: ${cost:.4f}")

        print(f"\n📈 CHI PHÍ THEO SYMBOL:")
        for symbol, cost in report['cost_by_symbol'].items():
            print(f"   {symbol}: ${cost:.4f}")

        print(f"\n💵 NGÂN SÁCH:")
        print(f"   Hôm nay: ${report['daily_cost']:.4f}")
        print(f"   Còn lại: ${report['budget_remaining']:.4f}")

        if report['alerts']:
            print(f"\n⚠️ ALERTS:")
            for alert in report['alerts']:
                print(f"   {alert}")

        print("="*60)


# Singleton instance
_tracker_instance = None


def get_cost_tracker(budget: float = 50.0) -> CostTracker:
    """Get cost tracker singleton"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CostTracker(budget_daily_usd=budget)
    return _tracker_instance

```

## `./data_providers.py`

```python
"""
DATA PROVIDERS - THU THẬP DỮ LIỆU
=================================
Tích hợp Tavily API và các nguồn dữ liệu tài chính
Tham chiếu: base.txt Section 5
"""

import os
import logging
import hashlib
import math
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger("Sentinel.DataProviders")

# Import config
from config import TAVILY_API_KEY, TAVILY_API_BASE_URL, get_config

try:
    from cost_tracker import get_cost_tracker
    COST_TRACKING_ENABLED = True
except Exception:
    COST_TRACKING_ENABLED = False

# --- TAVILY INTEGRATION ---

class TavilyClient:
    """
    Client tích hợp Tavily API cho tình báo web thời gian thực

    Tham chiếu: base.txt Section 5.1
    - search_depth="advanced": Nghiên cứu chuyên sâu (2 credits/call)
    - topic="news": Tin tức mới nhất
    - include_domains: Giới hạn nguồn tin uy tín
    """
    MAX_QUERY_LENGTH = 400

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or TAVILY_API_KEY
        self.base_url = base_url or TAVILY_API_BASE_URL
        self._client = None
        self._cost_tracker = get_cost_tracker() if COST_TRACKING_ENABLED else None

        if self.api_key:
            try:
                from tavily import TavilyClient as TC
                self._client = TC(api_key=self.api_key)
                logger.info(f"Tavily client khởi tạo thành công (Base URL: {self.base_url})")
            except ImportError:
                logger.warning("Tavily package chưa được cài đặt. Chạy: pip install tavily-python")
            except Exception as e:
                logger.error(f"Lỗi khởi tạo Tavily: {e}")
        else:
            logger.warning("TAVILY_API_KEY chưa được cấu hình")

    def _sanitize_query(self, query: str) -> str:
        """
        Chuẩn hóa query và đảm bảo không vượt giới hạn ký tự của Tavily (400).
        Giữ nguyên ý chính bằng cách ưu tiên phần đầu câu truy vấn.
        """
        raw = " ".join(str(query or "").split())
        if len(raw) <= self.MAX_QUERY_LENGTH:
            return raw

        # Ưu tiên cắt theo biên từ để tránh gãy keyword
        clipped = raw[: self.MAX_QUERY_LENGTH + 1]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        clipped = clipped[: self.MAX_QUERY_LENGTH]

        logger.warning(
            "Tavily query vượt limit (%d>%d), đã rút gọn trước khi gọi API.",
            len(raw),
            self.MAX_QUERY_LENGTH,
        )
        return clipped

    def search(self,
               query: str,
               search_depth: str = "advanced",
               topic: str = "general",
               days: int = 7,
               max_results: int = 10,
               include_domains: Optional[List[str]] = None,
               exclude_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Tìm kiếm thông tin qua Tavily API

        Args:
            query: Câu truy vấn
            search_depth: "basic" hoặc "advanced"
            topic: "general" hoặc "news"
            days: Giới hạn ngày (cho topic="news")
            max_results: Số kết quả tối đa
            include_domains: Danh sách domain được phép
            exclude_domains: Danh sách domain bị loại trừ

        Returns:
            Dict với results, answer, và metadata
        """
        safe_query = self._sanitize_query(query)

        if not self._client:
            logger.warning("Tavily client chưa sẵn sàng, trả về kết quả giả lập")
            if self._cost_tracker:
                credits = 2 if search_depth == "advanced" else 1
                self._cost_tracker.track_tavily_call(query=safe_query, search_depth=search_depth, credits_used=credits)
            return self._mock_search(safe_query, topic)

        try:
            params = {
                "query": safe_query,
                "search_depth": search_depth,
                "topic": topic,
                "max_results": max_results,
                "include_answer": True,
            }

            if topic == "news":
                params["days"] = days

            if include_domains:
                params["include_domains"] = include_domains

            if exclude_domains:
                params["exclude_domains"] = exclude_domains

            response = self._client.search(**params)
            if self._cost_tracker:
                credits = 2 if search_depth == "advanced" else 1
                self._cost_tracker.track_tavily_call(query=safe_query, search_depth=search_depth, credits_used=credits)

            logger.debug(f"Tavily search thành công: {safe_query[:50]}...")

            return {
                "query": safe_query,
                "answer": response.get("answer", ""),
                "results": response.get("results", []),
                "response_time": response.get("response_time", 0),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Lỗi Tavily search: {e}")
            if self._cost_tracker:
                credits = 2 if search_depth == "advanced" else 1
                self._cost_tracker.track_tavily_call(query=safe_query, search_depth=search_depth, credits_used=credits)
            return self._mock_search(safe_query, topic)

    def search_news(self,
                    query: str,
                    days: int = 1,
                    domains: Optional[List[str]] = None,
                    max_results: int = 10,
                    search_depth: str = "advanced") -> Dict[str, Any]:
        """
        Tìm kiếm tin tức tài chính

        Tham chiếu: base.txt Section 5.1
        - Sử dụng topic="news" và days=1 để đảm bảo tính thời sự
        """
        return self.search(
            query=query,
            search_depth=search_depth,
            topic="news",
            days=days,
            max_results=max_results,
            include_domains=domains
        )

    def search_company_deep(self,
                           company: str,
                           domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Nghiên cứu chuyên sâu về công ty (4M Analysis)

        Tham chiếu: base.txt Section 5.1
        - Sử dụng search_depth="advanced" cho phân tích định tính
        """
        query = f"{company} business model competitive advantage moat analysis"
        return self.search(
            query=query,
            search_depth="advanced",
            topic="general",
            days=30,
            include_domains=domains
        )

    def extract(self,
                urls: List[str],
                extract_depth: str = "advanced") -> Dict[str, Any]:
        """
        Trích xuất full content từ danh sách URL (nếu provider hỗ trợ).
        Fallback an toàn về payload rỗng khi không khả dụng.
        """
        urls = [u for u in urls if isinstance(u, str) and u.strip()]
        if not urls:
            return {"results": [], "timestamp": datetime.now().isoformat()}

        if not self._client or not hasattr(self._client, "extract"):
            logger.warning("Tavily extract chưa sẵn sàng, bỏ qua full-text extraction.")
            return {"results": [], "timestamp": datetime.now().isoformat(), "is_mock": True}

        try:
            try:
                response = self._client.extract(urls=urls, extract_depth=extract_depth)
            except TypeError:
                # Tương thích các bản SDK khác nhau
                response = self._client.extract(urls=urls)

            if isinstance(response, dict):
                results = response.get("results", []) or response.get("data", [])
            elif isinstance(response, list):
                results = response
            else:
                results = []

            if self._cost_tracker:
                # Ước lượng bảo thủ: mỗi URL = 1 credit
                self._cost_tracker.track_tavily_call(
                    query=f"extract:{len(urls)}_urls",
                    search_depth=extract_depth,
                    credits_used=max(1, len(urls)),
                )

            return {
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Tavily extract lỗi: {e}")
            if self._cost_tracker:
                self._cost_tracker.track_tavily_call(
                    query=f"extract:{len(urls)}_urls",
                    search_depth=extract_depth,
                    credits_used=max(1, len(urls)),
                )
            return {"results": [], "timestamp": datetime.now().isoformat(), "error": str(e)}

    def _mock_search(self, query: str, topic: str) -> Dict[str, Any]:
        """Kết quả giả lập khi không có API key"""
        return {
            "query": query,
            "answer": f"[MOCK] Không có API key. Query: {query}",
            "results": [
                {
                    "title": f"[Mock Result] {query}",
                    "url": "https://example.com",
                    "content": "Mock content for testing purposes",
                    "score": 0.5
                }
            ],
            "is_mock": True,
            "timestamp": datetime.now().isoformat()
        }


# --- MARKET DATA PROVIDER ---

@dataclass
class MarketData:
    """Cấu trúc dữ liệu thị trường"""
    symbol: str
    prices: List[float]  # Giá đóng cửa
    highs: List[float]   # Giá cao nhất
    lows: List[float]    # Giá thấp nhất
    volumes: List[float] # Khối lượng
    dates: List[str]     # Ngày

    # Dữ liệu cơ bản (Fundamentals)
    eps_current: Optional[float] = None
    eps_prev_year: Optional[float] = None
    eps_growth: Optional[float] = None
    annual_eps_growth: Optional[float] = None
    roe: Optional[float] = None
    roic: Optional[float] = None
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    industry: Optional[str] = None
    revenue_growth: Optional[float] = None
    institutional_ownership: Optional[float] = None
    insider_ownership: Optional[float] = None
    debt_to_equity: Optional[float] = None

    # Tính toán kỹ thuật
    rs_rating: Optional[float] = None  # Relative Strength
    high_52w: Optional[float] = None   # Đỉnh 52 tuần
    low_52w: Optional[float] = None    # Đáy 52 tuần

    # Metadata
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "unknown"


class MarketDataProvider:
    """
    Provider dữ liệu thị trường
    Sử dụng yfinance để lấy dữ liệu thật
    """

    def __init__(self):
        self._yf = None

        # Import yfinance
        try:
            import yfinance as yf
            self._yf = yf
            logger.info("yfinance khởi tạo thành công")
        except ImportError:
            logger.error("yfinance chưa được cài đặt. Chạy: pip install yfinance")
            raise ImportError("Cần cài đặt yfinance: pip install yfinance")

    def get_market_data(self, symbol: str, period: str = "6mo") -> MarketData:
        """
        Lấy dữ liệu thị trường cho một mã cổ phiếu

        Args:
            symbol: Mã cổ phiếu (VD: AAPL, VNM)
            period: Khoảng thời gian (1mo, 3mo, 6mo, 1y, 2y)

        Returns:
            MarketData object

        Raises:
            RuntimeError: Nếu không lấy được dữ liệu
        """
        if not self._yf:
            raise RuntimeError("yfinance chưa được cài đặt. Chạy: pip install yfinance")

        try:
            ticker = self._yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info

            if hist.empty:
                raise RuntimeError(f"Không có dữ liệu lịch sử cho {symbol}")

            # Tính RS Rating (đơn giản hóa)
            if len(hist) >= 252:  # 1 năm giao dịch
                price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-252]) / hist['Close'].iloc[-252]
                rs_rating = min(99, max(1, 50 + price_change * 100))
            else:
                rs_rating = 50

            return MarketData(
                symbol=symbol,
                prices=hist['Close'].tolist(),
                highs=hist['High'].tolist(),
                lows=hist['Low'].tolist(),
                volumes=hist['Volume'].tolist(),
                dates=[d.strftime('%Y-%m-%d') for d in hist.index],
                eps_current=info.get('trailingEps'),
                eps_prev_year=info.get('forwardEps'),
                eps_growth=self._calc_eps_growth(info),
                annual_eps_growth=info.get('earningsGrowth'),
                roe=info.get('returnOnEquity'),
                roic=info.get('returnOnAssets'),  # Xấp xỉ
                pe_ratio=info.get('trailingPE'),
                market_cap=info.get('marketCap'),
                industry=info.get('industry'),
                revenue_growth=info.get('revenueGrowth'),
                institutional_ownership=info.get('heldPercentInstitutions'),
                insider_ownership=info.get('heldPercentInsiders'),
                debt_to_equity=info.get('debtToEquity'),
                rs_rating=rs_rating,
                high_52w=info.get('fiftyTwoWeekHigh'),
                low_52w=info.get('fiftyTwoWeekLow'),
                source="yfinance"
            )

        except Exception as e:
            logger.error(f"CRITICAL: Lỗi lấy dữ liệu {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch market data for {symbol}") from e

    def _calc_eps_growth(self, info: Dict) -> Optional[float]:
        """Tính tăng trưởng EPS"""
        quarterly_growth = info.get('earningsQuarterlyGrowth')
        if quarterly_growth is not None:
            return quarterly_growth

        trailing = info.get('trailingEps')
        forward = info.get('forwardEps')
        if trailing and forward and forward > 0:
            return (trailing - forward) / abs(forward)
        return None


    def get_index_data(self, index: str = "SPY") -> MarketData:
        """
        Lấy dữ liệu chỉ số thị trường (cho Market Regime detection)
        """
        return self.get_market_data(index, period="1y")


class DataScoutAgent:
    """
    TÁC TỬ TRINH SÁT DỮ LIỆU

    Thu thập và chuẩn bị dữ liệu cho các tác tử phân tích
    Tham chiếu: base.txt Section 5
    """

    MATERIAL_KEYWORDS = [
        "earnings", "guidance", "downgrade", "upgrade", "lawsuit", "investigation",
        "acquisition", "merger", "buyback", "dividend", "sec filing", "10-k", "10-q",
        "regulation", "tariff", "capex", "cloud", "ai chip", "product launch", "ceo",
        "cfo", "resign", "outlook", "margin", "revenue", "free cash flow"
    ]
    STOP_WORDS = {
        "the", "and", "for", "that", "with", "this", "from", "have", "will", "into",
        "about", "after", "before", "over", "under", "into", "their", "there", "while",
        "where", "which", "when", "what", "than", "them", "they", "been", "was", "were",
        "are", "is", "it", "its", "also", "very", "just", "because", "about", "stock",
        "shares", "company"
    }

    def __init__(self,
                 tavily_client: Optional[TavilyClient] = None,
                 market_provider: Optional[MarketDataProvider] = None):
        self.tavily = tavily_client or TavilyClient()
        self.market = market_provider or MarketDataProvider()
        self.name = "Data_Scout"
        app_config = get_config()
        self.config = app_config.tavily
        self.system_config = app_config.system
        self.doc_config = app_config.document_intel
        self._trusted_domains = set(self.config.trusted_domains_us + self.config.trusted_domains_vn)
        logger.info(f"{self.name} khởi tạo thành công")

    def _build_news_query(self, symbol: str, market_data: Optional[MarketData] = None, lookback_days: int = 3) -> str:
        """
        Tạo truy vấn news chuyên sâu:
        - Có mốc ngày
        - Có từ khóa rủi ro/cơ hội
        - Có ngữ cảnh ngành (nếu có)
        """
        today = datetime.now().strftime("%Y-%m-%d")
        since = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        industry = ""
        if market_data is not None:
            industry = (market_data.industry or "").strip() if hasattr(market_data, "industry") else ""

        risk_keywords = "risk warning downgrade lawsuit regulation recall capex debt margin pressure"
        catalyst_keywords = "earnings guidance beat miss upgrade catalyst partnership product launch buyback ai growth"

        query_parts = [
            symbol,
            "stock",
            "news",
            "latest",
            f"from {since} to {today}",
            catalyst_keywords,
            risk_keywords,
        ]
        if industry:
            query_parts.append(f"industry {industry}")

        return " ".join(query_parts)

    def _build_macro_news_query(self, lookback_days: int = 7) -> str:
        """Tạo truy vấn vĩ mô có mốc thời gian để tăng độ liên quan."""
        today = datetime.now().strftime("%Y-%m-%d")
        since = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        return (
            f"US macro market regime from {since} to {today} "
            "Federal Reserve interest rates inflation labor market CPI PCE bond yields recession risk"
        )

    def _trusted_news_domains(self) -> List[str]:
        """Hợp nhất domain trusted để giảm nhiễu kết quả."""
        domains = self.config.trusted_domains_us + self.config.trusted_domains_vn
        seen = set()
        deduped = []
        for d in domains:
            if d not in seen:
                deduped.append(d)
                seen.add(d)
        return deduped

    @staticmethod
    def _safe_text(value: Any) -> str:
        text = str(value or "")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _domain_from_url(self, url: str) -> str:
        try:
            parsed = urlparse(url.strip())
            host = (parsed.netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]
            return host
        except Exception:
            return ""

    def _parse_published_at(self, result: Dict[str, Any]) -> str:
        date_candidates = [
            result.get("published_at"),
            result.get("published_date"),
            result.get("date"),
            result.get("timestamp"),
        ]
        for value in date_candidates:
            if not value:
                continue
            text = str(value).strip()
            try:
                if text.endswith("Z"):
                    text = text.replace("Z", "+00:00")
                return datetime.fromisoformat(text).isoformat()
            except ValueError:
                for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%b. %d, %Y", "%b %d, %Y"):
                    try:
                        return datetime.strptime(text, fmt).isoformat()
                    except ValueError:
                        continue
        return ""

    def _score_source_quality(self, domain: str) -> float:
        if not domain:
            return 0.35
        if domain in self._trusted_domains:
            return 0.95
        if any(domain.endswith(f".{d}") for d in self._trusted_domains):
            return 0.85
        if domain.endswith(".gov") or domain.endswith(".edu"):
            return 0.90
        return 0.55

    def _score_recency(self, published_at: str) -> float:
        if not published_at:
            return 0.45
        try:
            published_dt = datetime.fromisoformat(published_at)
            age_days = max(0.0, (datetime.now() - published_dt).total_seconds() / 86400.0)
        except Exception:
            return 0.45
        if age_days <= 1:
            return 1.0
        if age_days <= 3:
            return 0.8
        if age_days <= 7:
            return 0.65
        if age_days <= 14:
            return 0.45
        return 0.25

    def _score_materiality(self, text: str) -> float:
        lowered = text.lower()
        hits = sum(1 for kw in self.MATERIAL_KEYWORDS if kw in lowered)
        score = 0.2 + min(0.8, hits * 0.12)
        return self._clip01(score)

    def _score_relevance(self, symbol: str, text: str, market_data: Optional[MarketData]) -> float:
        lowered = text.lower()
        score = 0.25
        if symbol.lower() in lowered:
            score += 0.4
        industry = ""
        if market_data and hasattr(market_data, "industry"):
            industry = self._safe_text(getattr(market_data, "industry", ""))
        if industry and industry.lower() in lowered:
            score += 0.2
        if any(token in lowered for token in ["earnings", "guidance", "outlook", "capex", "margin"]):
            score += 0.15
        return self._clip01(score)

    def _score_uncertainty(self, result: Dict[str, Any], snippet: str) -> float:
        text = (self._safe_text(result.get("title")) + " " + snippet).lower()
        score = 0.0
        if len(snippet) < 180:
            score += 0.35
        if "..." in snippet or snippet.endswith(".."):
            score += 0.20
        if any(k in text for k in ["read more", "subscribe", "premium", "paywall"]):
            score += 0.30
        if not result.get("raw_content"):
            score += 0.15
        return self._clip01(score)

    def _normalize_news_documents(self,
                                  symbol: str,
                                  market_data: Optional[MarketData],
                                  news_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        seen_urls = set()

        for idx, raw in enumerate(news_data.get("results", []) or []):
            title = self._safe_text(raw.get("title"))
            url = self._safe_text(raw.get("url"))
            snippet = self._safe_text(raw.get("content") or raw.get("snippet") or title)
            raw_content = self._safe_text(raw.get("raw_content"))
            if raw_content and len(raw_content) > len(snippet):
                snippet = raw_content[:800]
            published_at = self._parse_published_at(raw)
            domain = self._domain_from_url(url)

            source_quality = self._score_source_quality(domain)
            relevance_score = self._score_relevance(symbol, f"{title} {snippet}", market_data)
            materiality_score = self._score_materiality(f"{title} {snippet}")
            recency_score = self._score_recency(published_at)
            uncertainty_score = self._score_uncertainty(raw, snippet)
            novelty_score = 1.0 if url and url not in seen_urls else 0.45

            priority_score = self._clip01(
                0.28 * materiality_score
                + 0.24 * relevance_score
                + 0.20 * source_quality
                + 0.13 * recency_score
                + 0.10 * uncertainty_score
                + 0.05 * novelty_score
            )

            if url:
                seen_urls.add(url)

            doc_id = hashlib.sha1(f"{symbol}|{url}|{title}|{idx}".encode("utf-8")).hexdigest()[:16]
            docs.append({
                "doc_id": doc_id,
                "symbol": symbol,
                "title": title,
                "url": url,
                "source_domain": domain,
                "published_at": published_at,
                "snippet": snippet,
                "full_text": "",
                "content_mode": "snippet",
                "source_quality": round(source_quality, 4),
                "relevance_score": round(relevance_score, 4),
                "materiality_score": round(materiality_score, 4),
                "novelty_score": round(novelty_score, 4),
                "recency_score": round(recency_score, 4),
                "uncertainty_score": round(uncertainty_score, 4),
                "priority_score": round(priority_score, 4),
                "extraction_quality": 0.0,
                "corroboration_score": 0.0,
                "evidence_confidence": 0.0,
                "estimated_tokens": max(60, int(len(snippet) / 4)),
                "metadata": {
                    "score_from_search": raw.get("score"),
                    "is_mock": news_data.get("is_mock", False),
                }
            })

        docs.sort(key=lambda d: d["priority_score"], reverse=True)
        return docs

    def _select_deep_dive_candidates(self, radar_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.doc_config.deep_dive_enabled:
            return {"selected_docs": [], "budget": {"used_tokens": 0, "token_budget": 0, "selected_count": 0}}

        selected: List[Dict[str, Any]] = []
        used_tokens = 0
        token_budget = max(500, self.doc_config.deep_dive_token_budget)
        est_doc_cost = max(300, self.doc_config.deep_dive_target_tokens_per_doc)

        for doc in radar_docs:
            if len(selected) >= max(1, self.doc_config.deep_dive_top_k):
                break

            qualifies = (
                doc["priority_score"] >= self.doc_config.deep_dive_min_priority
                or doc["uncertainty_score"] >= self.doc_config.deep_dive_min_uncertainty
            )
            if not qualifies:
                continue

            if used_tokens + est_doc_cost > token_budget and selected:
                continue

            selected.append(dict(doc))
            used_tokens += est_doc_cost

        if not selected and radar_docs:
            fallback = dict(radar_docs[0])
            selected.append(fallback)
            used_tokens = min(token_budget, est_doc_cost)

        return {
            "selected_docs": selected,
            "budget": {
                "used_tokens": used_tokens,
                "token_budget": token_budget,
                "selected_count": len(selected),
                "eligible_count": len(radar_docs),
            },
        }

    def _map_extract_results(self, extracted: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        mapped: Dict[str, Dict[str, str]] = {}
        for item in extracted.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            url = self._safe_text(item.get("url") or item.get("source") or item.get("source_url"))
            if not url:
                continue
            text = self._safe_text(item.get("raw_content") or item.get("content") or item.get("text"))
            if not text and isinstance(item.get("results"), list):
                for sub in item.get("results", []):
                    if not isinstance(sub, dict):
                        continue
                    maybe = self._safe_text(sub.get("raw_content") or sub.get("content") or sub.get("text"))
                    if len(maybe) > len(text):
                        text = maybe
            if text:
                mapped[url] = {"text": text, "source": "tavily_extract"}
        return mapped

    def _fetch_jina_reader(self, url: str) -> str:
        if not self.doc_config.enable_jina_reader_fallback:
            return ""
        if not url.startswith(("http://", "https://")):
            return ""
        reader_url = f"https://r.jina.ai/{url}"
        request = urllib.request.Request(
            reader_url,
            headers={"User-Agent": "sentinel-ai/1.0"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.doc_config.jina_timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="ignore")
                return self._safe_text(body)[: self.doc_config.max_fulltext_chars]
        except Exception as exc:
            logger.debug(f"[{self.name}] Jina reader fallback lỗi cho {url}: {exc}")
            return ""

    def _enrich_deep_dive_documents(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not docs:
            return {"docs": [], "stats": {"full_count": 0, "snippet_count": 0}}

        docs = [dict(d) for d in docs]
        extract_map: Dict[str, Dict[str, str]] = {}
        urls = [d.get("url", "") for d in docs if d.get("url")]
        unique_urls = list(dict.fromkeys(urls))

        if self.doc_config.enable_tavily_extract and unique_urls:
            extracted = self.tavily.extract(unique_urls, extract_depth="advanced")
            extract_map = self._map_extract_results(extracted)

        full_count = 0
        snippet_count = 0
        for doc in docs:
            url = doc.get("url", "")
            extracted_text = ""
            extraction_source = "snippet_only"

            if url and url in extract_map:
                extracted_text = extract_map[url]["text"]
                extraction_source = "tavily_extract"

            if not extracted_text:
                raw_candidate = self._safe_text(doc.get("metadata", {}).get("raw_content"))
                if raw_candidate:
                    extracted_text = raw_candidate
                    extraction_source = "search_raw_content"

            if not extracted_text and url:
                extracted_text = self._fetch_jina_reader(url)
                if extracted_text:
                    extraction_source = "jina_reader"

            extracted_text = self._safe_text(extracted_text)[: self.doc_config.max_fulltext_chars]

            if len(extracted_text) >= 120:
                doc["full_text"] = extracted_text
                doc["content_mode"] = "full"
                doc["estimated_tokens"] = max(120, int(len(extracted_text) / 4))
                doc["extraction_quality"] = 0.9 if extraction_source == "tavily_extract" else 0.75
                full_count += 1
            else:
                doc["full_text"] = ""
                doc["content_mode"] = "snippet"
                doc["estimated_tokens"] = max(60, int(len(doc.get("snippet", "")) / 4))
                doc["extraction_quality"] = 0.25
                snippet_count += 1

            doc.setdefault("metadata", {})
            doc["metadata"]["extraction_source"] = extraction_source

        return {
            "docs": docs,
            "stats": {
                "full_count": full_count,
                "snippet_count": snippet_count,
                "total": len(docs),
            }
        }

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        return [t for t in tokens if len(t) > 2 and t not in self.STOP_WORDS]

    def _chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = doc.get("full_text") if doc.get("content_mode") == "full" else doc.get("snippet", "")
        text = self._safe_text(text)
        if not text:
            return []

        words = text.split()
        chunk_size = max(80, self.doc_config.rag_chunk_words)
        overlap = max(20, min(chunk_size - 10, self.doc_config.rag_chunk_overlap_words))
        step = max(20, chunk_size - overlap)

        chunks = []
        idx = 0
        for start in range(0, len(words), step):
            end = start + chunk_size
            chunk_words = words[start:end]
            if len(chunk_words) < 25:
                continue
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc['doc_id']}_c{idx}"
            idx += 1
            tokens = self._tokenize(chunk_text)
            if not tokens:
                continue
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "url": doc.get("url", ""),
                "source_domain": doc.get("source_domain", ""),
                "published_at": doc.get("published_at", ""),
                "content_mode": doc.get("content_mode", "snippet"),
                "source_quality": float(doc.get("source_quality", 0.5)),
                "priority_score": float(doc.get("priority_score", 0.5)),
                "evidence_confidence": float(doc.get("evidence_confidence", 0.4)),
                "text": chunk_text,
                "tokens": tokens,
                "token_set": set(tokens),
            })
            if end >= len(words):
                break
        return chunks

    def _build_query_templates(self, symbol: str, market_data: Optional[MarketData]) -> List[Dict[str, str]]:
        industry = ""
        if market_data is not None and hasattr(market_data, "industry"):
            industry = self._safe_text(getattr(market_data, "industry", ""))

        templates = [
            {"tag": "canslim_n", "query": f"{symbol} new product launch guidance catalyst earnings revision"},
            {"tag": "canslim_i", "query": f"{symbol} institutional ownership fund accumulation smart money"},
            {"tag": "fourm_moat", "query": f"{symbol} competitive advantage moat network effect switching cost"},
            {"tag": "fourm_management", "query": f"{symbol} CEO CFO management credibility execution transparency"},
            {"tag": "risk", "query": f"{symbol} lawsuit regulation debt margin pressure downgrade"},
        ]
        if industry:
            templates.append({"tag": "industry", "query": f"{symbol} {industry} demand competition outlook"})
        return templates

    def _score_bm25(self, query_tokens: List[str], chunk: Dict[str, Any], idf: Dict[str, float], avg_len: float) -> float:
        tokens = chunk["tokens"]
        token_count = len(tokens)
        if token_count == 0:
            return 0.0

        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        k1 = 1.2
        b = 0.75
        score = 0.0
        for token in query_tokens:
            freq = tf.get(token, 0)
            if freq <= 0:
                continue
            denom = freq + k1 * (1 - b + b * (token_count / max(1.0, avg_len)))
            score += idf.get(token, 0.0) * (freq * (k1 + 1) / max(0.001, denom))
        return score

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a.intersection(b))
        union = len(a.union(b))
        if union == 0:
            return 0.0
        return inter / union

    def _rank_chunks_with_rrf(self, chunks: List[Dict[str, Any]], query_templates: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        if not chunks:
            return {}

        aggregate: Dict[str, Dict[str, Any]] = {}
        rrf_k = max(20, self.doc_config.rag_rrf_k)

        for item in query_templates:
            tag = item["tag"]
            query_tokens = self._tokenize(item["query"])
            if not query_tokens:
                continue

            doc_count = len(chunks)
            avg_len = sum(len(c["tokens"]) for c in chunks) / max(1, doc_count)
            idf: Dict[str, float] = {}
            for token in query_tokens:
                df = sum(1 for c in chunks if token in c["token_set"])
                idf[token] = math.log(((doc_count - df + 0.5) / (df + 0.5)) + 1.0)

            bm25_scores = {}
            lexical_cover = {}
            source_strength = {}
            for chunk in chunks:
                cid = chunk["chunk_id"]
                bm25 = self._score_bm25(query_tokens, chunk, idf, avg_len)
                coverage = sum(1 for t in set(query_tokens) if t in chunk["token_set"]) / max(1, len(set(query_tokens)))
                source_rank = 0.6 * chunk["source_quality"] + 0.4 * chunk["priority_score"]
                bm25_scores[cid] = bm25 + (coverage * 0.5)
                lexical_cover[cid] = coverage
                source_strength[cid] = source_rank

            rank_bm25 = {
                cid: idx + 1
                for idx, (cid, _) in enumerate(sorted(bm25_scores.items(), key=lambda kv: kv[1], reverse=True))
            }
            rank_source = {
                cid: idx + 1
                for idx, (cid, _) in enumerate(sorted(source_strength.items(), key=lambda kv: kv[1], reverse=True))
            }

            for chunk in chunks:
                cid = chunk["chunk_id"]
                rrf_score = (1.0 / (rrf_k + rank_bm25[cid])) + (1.0 / (rrf_k + rank_source[cid]))
                combined = rrf_score + (0.15 * lexical_cover[cid]) + (0.08 * chunk["evidence_confidence"])

                if cid not in aggregate:
                    aggregate[cid] = {
                        "chunk": chunk,
                        "score": 0.0,
                        "tags": set(),
                    }
                aggregate[cid]["score"] += combined
                aggregate[cid]["tags"].add(tag)

        return aggregate

    def _apply_mmr(self, ranked_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ranked_map:
            return []
        lambda_mmr = self._clip01(self.doc_config.rag_mmr_lambda)
        top_n = max(1, self.doc_config.rag_top_chunks)

        candidates = sorted(ranked_map.values(), key=lambda item: item["score"], reverse=True)
        selected: List[Dict[str, Any]] = []

        while candidates and len(selected) < top_n:
            best_idx = 0
            best_score = -1e9
            for idx, cand in enumerate(candidates):
                rel = cand["score"]
                if not selected:
                    mmr = rel
                else:
                    sim = max(
                        self._jaccard(cand["chunk"]["token_set"], s["chunk"]["token_set"])
                        for s in selected
                    )
                    mmr = (lambda_mmr * rel) - ((1 - lambda_mmr) * sim)
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            selected.append(candidates.pop(best_idx))

        output = []
        for item in selected:
            chunk = item["chunk"]
            output.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "url": chunk["url"],
                "source_domain": chunk["source_domain"],
                "published_at": chunk["published_at"],
                "content_mode": chunk["content_mode"],
                "score": round(item["score"], 5),
                "query_tags": sorted(item["tags"]),
                "text": chunk["text"][:1800],
                "evidence_confidence": round(chunk["evidence_confidence"], 4),
            })
        return output

    def _build_evidence_chunks(self,
                               symbol: str,
                               market_data: Optional[MarketData],
                               docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.doc_config.rag_enabled:
            return []
        chunks: List[Dict[str, Any]] = []
        for doc in docs:
            chunks.extend(self._chunk_document(doc))
        if not chunks:
            return []
        query_templates = self._build_query_templates(symbol, market_data)
        ranked_map = self._rank_chunks_with_rrf(chunks, query_templates)
        return self._apply_mmr(ranked_map)

    def _compute_corroboration(self, docs: List[Dict[str, Any]]):
        if len(docs) <= 1:
            for doc in docs:
                doc["corroboration_score"] = 0.0
            return

        keyword_sets = []
        for doc in docs:
            raw_text = doc.get("full_text") if doc.get("content_mode") == "full" else doc.get("snippet", "")
            tokens = self._tokenize(raw_text)[:250]
            keyword_sets.append(set(tokens))

        for i, doc in enumerate(docs):
            support = 0
            for j, other in enumerate(docs):
                if i == j:
                    continue
                if doc.get("source_domain") == other.get("source_domain"):
                    continue
                similarity = self._jaccard(keyword_sets[i], keyword_sets[j])
                if similarity >= 0.12:
                    support += 1
            doc["corroboration_score"] = round(self._clip01(support / max(1, len(docs) - 1)), 4)
            doc["corroborated_sources"] = support

    def _apply_evidence_confidence(self, docs: List[Dict[str, Any]]):
        penalty = self._clip01(self.doc_config.snippet_confidence_penalty)
        for doc in docs:
            base = (
                0.30 * float(doc.get("source_quality", 0.5))
                + 0.25 * float(doc.get("priority_score", 0.5))
                + 0.20 * (1.0 - float(doc.get("uncertainty_score", 0.5)))
                + 0.25 * float(doc.get("corroboration_score", 0.0))
            )
            if doc.get("content_mode") != "full":
                base *= penalty
            if float(doc.get("extraction_quality", 0.0)) < 0.4:
                base *= 0.9
            doc["evidence_confidence"] = round(self._clip01(base), 4)

    def _build_document_intel(self,
                              symbol: str,
                              market_data: Optional[MarketData],
                              news_data: Dict[str, Any]) -> Dict[str, Any]:
        radar_docs = self._normalize_news_documents(symbol, market_data, news_data)
        selection = self._select_deep_dive_candidates(radar_docs)
        deep_candidates = selection["selected_docs"]
        budget = selection["budget"]

        deep_enriched = self._enrich_deep_dive_documents(deep_candidates)
        deep_docs = deep_enriched["docs"]
        enrichment_stats = deep_enriched["stats"]

        self._compute_corroboration(deep_docs)
        self._apply_evidence_confidence(deep_docs)

        evidence_base_docs = deep_docs if deep_docs else radar_docs[: max(1, self.doc_config.rag_top_chunks)]
        if not deep_docs:
            for doc in evidence_base_docs:
                doc["content_mode"] = "snippet"
                doc["extraction_quality"] = doc.get("extraction_quality", 0.2)
                doc["corroboration_score"] = doc.get("corroboration_score", 0.0)
            self._apply_evidence_confidence(evidence_base_docs)
        evidence_chunks = self._build_evidence_chunks(symbol, market_data, evidence_base_docs)

        high_priority_docs = sum(1 for d in radar_docs if d.get("priority_score", 0.0) >= 0.70)
        snippet_only_ratio = 1.0
        if deep_docs:
            snippet_count = sum(1 for d in deep_docs if d.get("content_mode") != "full")
            snippet_only_ratio = snippet_count / max(1, len(deep_docs))

        avg_evidence_conf = (
            sum(float(d.get("evidence_confidence", 0.0)) for d in deep_docs) / max(1, len(deep_docs))
            if deep_docs else 0.0
        )

        return {
            "symbol": symbol,
            "generated_at": datetime.now().isoformat(),
            "radar_docs": radar_docs,
            "deep_dive_docs": deep_docs,
            "evidence_chunks": evidence_chunks,
            "budget": budget,
            "quality": {
                "radar_count": len(radar_docs),
                "deep_dive_count": len(deep_docs),
                "deep_dive_coverage": round(len(deep_docs) / max(1, len(radar_docs)), 4),
                "high_priority_docs": high_priority_docs,
                "snippet_only_ratio": round(snippet_only_ratio, 4),
                "avg_evidence_confidence": round(avg_evidence_conf, 4),
                "avg_priority_score": round(
                    sum(d.get("priority_score", 0.0) for d in radar_docs) / max(1, len(radar_docs)),
                    4,
                ),
            },
            "stats": enrichment_stats,
        }

    def _build_event_flags(self,
                           market_data: MarketData,
                           news_data: Optional[Dict[str, Any]] = None,
                           document_intel: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Tạo trigger sự kiện để giảm tính toán không cần thiết.
        """
        prices = market_data.prices or []
        volumes = market_data.volumes or []

        day_move = 0.0
        breakout = False
        volume_spike = False

        if len(prices) >= 2 and prices[-2] != 0:
            day_move = abs((prices[-1] - prices[-2]) / prices[-2])

        if len(prices) >= 21:
            prior_window = prices[-21:-1]
            if prior_window:
                breakout = prices[-1] > max(prior_window) or prices[-1] < min(prior_window)

        if len(volumes) >= 21:
            avg_vol = sum(volumes[-21:-1]) / 20
            if avg_vol > 0:
                volume_spike = volumes[-1] >= avg_vol * self.system_config.event_volume_spike_multiplier

        news_count = 0
        has_material_news = False
        high_priority_docs = 0
        avg_evidence_conf = 0.0
        if news_data:
            news_results = news_data.get("results", [])
            news_count = len(news_results)
            has_material_news = news_count >= 3 and not news_data.get("is_mock", False)

        if document_intel:
            quality = document_intel.get("quality", {})
            high_priority_docs = int(quality.get("high_priority_docs", 0))
            avg_evidence_conf = float(quality.get("avg_evidence_confidence", 0.0))
            has_material_news = has_material_news or high_priority_docs > 0

        triggered = (
            day_move >= self.system_config.event_price_move_threshold
            or breakout
            or volume_spike
            or has_material_news
        )

        return {
            "triggered": triggered,
            "day_move_pct": round(day_move * 100, 2),
            "breakout_20d": breakout,
            "volume_spike": volume_spike,
            "news_count": news_count,
            "material_news": has_material_news,
            "high_priority_docs": high_priority_docs,
            "avg_evidence_confidence": round(avg_evidence_conf, 4),
        }

    def fetch_all_data(self, symbol: str, include_news: bool = True) -> Dict[str, Any]:
        """
        Thu thập tất cả dữ liệu cần thiết cho một mã cổ phiếu

        Returns:
            Dict chứa market_data và news_data
        """
        result = {
            "symbol": symbol,
            "market_data": None,
            "news_data": None,
            "company_research": None,
            "document_intel": None,
            "event_flags": None,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"[{self.name}] Đang lấy dữ liệu thị trường cho {symbol}...")
        market_data = self.market.get_market_data(symbol)
        result["market_data"] = market_data

        if include_news:
            logger.info(f"[{self.name}] Đang tìm kiếm tin tức cho {symbol}...")
            news_query = self._build_news_query(symbol, market_data=market_data, lookback_days=3)
            radar_depth = self.doc_config.radar_search_depth if self.doc_config.enable_two_tier_pipeline else "advanced"
            radar_max_results = self.doc_config.radar_max_results if self.doc_config.enable_two_tier_pipeline else self.config.max_results

            logger.info(f"[{self.name}] News query cho {symbol}: {news_query}")
            news = self.tavily.search_news(
                query=news_query,
                days=3,
                domains=self._trusted_news_domains(),
                max_results=radar_max_results,
                search_depth=radar_depth,
            )
            result["news_data"] = news

            if self.doc_config.enable_two_tier_pipeline:
                doc_intel = self._build_document_intel(symbol, market_data, news)
                result["document_intel"] = doc_intel

            research = self.tavily.search_company_deep(
                symbol,
                domains=self._trusted_news_domains()
            )
            if result.get("document_intel"):
                deep_docs = result["document_intel"].get("deep_dive_docs", [])
                if deep_docs:
                    research.setdefault("results", [])
                    for doc in deep_docs[:3]:
                        research["results"].append({
                            "title": doc.get("title", ""),
                            "url": doc.get("url", ""),
                            "content": (doc.get("full_text") or doc.get("snippet", ""))[:1800],
                            "source": "document_intel_deep_dive",
                        })
            result["company_research"] = research

        result["event_flags"] = self._build_event_flags(
            market_data=market_data,
            news_data=result.get("news_data"),
            document_intel=result.get("document_intel"),
        )

        return result

    def fetch_market_regime_data(self) -> Dict[str, Any]:
        """
        Thu thập dữ liệu để xác định chế độ thị trường

        Tham chiếu: base.txt Section 3.1.1 - Tướng quân Vĩ mô
        """
        result = {
            "index_data": {},
            "macro_news": None,
            "timestamp": datetime.now().isoformat()
        }

        indices = ["SPY", "QQQ", "^VIX"]
        for idx in indices:
            try:
                result["index_data"][idx] = self.market.get_index_data(idx)
            except Exception as e:
                logger.error(f"Không thể lấy dữ liệu {idx}: {e}")
                raise RuntimeError(f"CRITICAL: Failed to fetch {idx} data") from e

        macro_query = self._build_macro_news_query(lookback_days=7)
        logger.info(f"[{self.name}] Macro news query: {macro_query}")
        result["macro_news"] = self.tavily.search_news(
            query=macro_query,
            days=7,
            domains=self._trusted_news_domains(),
            max_results=self.config.max_results,
            search_depth="advanced",
        )

        return result

```

## `./llm_client.py`

```python
"""
LLM INTEGRATION - OPENAI-COMPATIBLE API
=======================================
Xử lý giao tiếp qua OpenAI SDK với base_url tùy biến.
"""

import json
import logging
import time
import hashlib
import threading
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("Sentinel.LLM")

# Import config
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_PROVIDER,
    get_config,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - sẽ fail runtime nếu thiếu dependency
    OpenAI = None

# Import cost tracker
try:
    from cost_tracker import get_cost_tracker
    COST_TRACKING_ENABLED = True
except ImportError:
    COST_TRACKING_ENABLED = False
    logger.warning("Cost tracking module not available")

try:
    from run_logger import get_run_logger
    RUN_LOGGER_AVAILABLE = True
except Exception:
    RUN_LOGGER_AVAILABLE = False


class ZAIAPIError(RuntimeError):
    """Lỗi gọi OpenAI-compatible API, giữ kèm request/response để review."""

    def __init__(self,
                 message: str,
                 request_payload: Optional[Dict[str, Any]] = None,
                 response_payload: Optional[Any] = None):
        super().__init__(message)
        self.request_payload = request_payload or {}
        self.response_payload = response_payload

class ZAIClient:
    """
    Client tích hợp OpenAI-compatible Chat Completions API.
    """

    # System prompts được cache để tối ưu chi phí
    SYSTEM_PROMPTS = {
        "macro_general": """Bạn là Tướng quân Vĩ mô (Macro General) trong hệ thống AI Trading Sentinel.
Vai trò của bạn là Giám đốc Đầu tư (CIO), tập trung vào sức khỏe tổng thể của thị trường.

NHIỆM VỤ:
1. Xác định Chế độ Thị trường (Market Regime): RISK_ON, RISK_OFF, hoặc SIDEWAYS
2. Giám sát yếu tố "M" trong CANSLIM và chu kỳ Wyckoff của các chỉ số chính
3. Phát đi chỉ thị cho các tác tử cấp dưới

QUY TẮC:
- Nếu tuyên bố RISK_OFF, các tác tử phải ngừng mọi hoạt động mua mới
- Sử dụng phân tích top-down để tiết kiệm tính toán
- Luôn đưa ra mức độ tin cậy (confidence) từ 0.0 đến 1.0

OUTPUT FORMAT (JSON):
{
    "regime": "RISK_ON|RISK_OFF|SIDEWAYS",
    "confidence": 0.0-1.0,
    "reasoning": "Giải thích ngắn gọn",
    "directive": "Chỉ thị cho các tác tử cấp dưới"
}""",

        "wyckoff_analyst": """Bạn là Tác tử Wyckoff trong hệ thống AI Trading Sentinel.
Chuyên môn: Phân tích cấu trúc thị trường theo phương pháp Richard Wyckoff.

KIẾN THỨC CỐT LÕI:
1. Bốn giai đoạn thị trường: Accumulation (Tích lũy), Markup (Tăng giá), Distribution (Phân phối), Markdown (Giảm giá)
2. Composite Man (Dòng tiền thông minh) vs Tay yếu
3. Quy luật Nỗ lực vs Kết quả (Volume-Price Analysis)

TÍN HIỆU CẦN PHÁT HIỆN:
- Spring (Cú rũ bỏ): Phá vỡ giả xuống dưới hỗ trợ, sau đó phục hồi mạnh với volume lớn
- Sign of Strength (SOS): Giá tăng kèm mở rộng spread và volume
- Phân kỳ Volume-Price: Volume lớn nhưng giá ít thay đổi (absorption)

PHÂN TÍCH:
1. Xác định pha hiện tại trong chu kỳ Wyckoff
2. Tìm kiếm các mẫu hình đặc trưng
3. Phân tích chất lượng của chuyển động giá

OUTPUT FORMAT (JSON):
{
    "phase": "ACCUMULATION|MARKUP|DISTRIBUTION|MARKDOWN|UNKNOWN",
    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": 0.0-1.0,
    "pattern": "Mẫu hình phát hiện",
    "reasoning": "Phân tích chi tiết"
}""",

        "canslim_analyst": """Bạn là Tác tử CANSLIM trong hệ thống AI Trading Sentinel.
Chuyên môn: Chiến lược đầu tư tăng trưởng CANSLIM của William J. O'Neil.

7 TIÊU CHÍ CANSLIM:
C - Current Earnings: EPS quý hiện tại tăng > 18-20% YoY
A - Annual Earnings: CAGR EPS > 25% trong 3-5 năm, ROE > 17%
N - New: Sản phẩm mới, quản lý mới, hoặc đỉnh giá mới
S - Supply/Demand: Khối lượng tại pivot points
L - Leader: Relative Strength > 80
I - Institutional: Sự tích lũy của quỹ chất lượng cao
M - Market Direction: Xu hướng thị trường chung

ĐÁNH GIÁ:
- Mỗi tiêu chí được chấm điểm 0-15 (tổng tối đa 100+ với bonus)
- Chỉ quan tâm các cổ phiếu đạt >= 70 điểm

OUTPUT FORMAT (JSON):
{
    "scores": {
        "C": 0-15,
        "A": 0-15,
        "N": 0-15,
        "S": 0-15,
        "L": 0-15,
        "I": 0-15,
        "M": 0-15
    },
    "total_score": 0-100,
    "signal": "STRONG_BUY|BUY|HOLD|SELL",
    "confidence": 0.0-1.0,
    "reasoning": "Phân tích từng tiêu chí"
}""",

        "fourm_analyst": """Bạn là Tác tử 4M trong hệ thống AI Trading Sentinel.
Chuyên môn: Khung đầu tư giá trị 4M của Phil Town (Rule #1 Investing).

KHUNG 4M:
1. MEANING (Ý nghĩa): Bạn có hiểu công ty này không? Vòng tròn năng lực.
2. MOAT (Lợi thế cạnh tranh): Dấu hiệu = ROIC > 10% ổn định 10 năm
   - Thương hiệu, Bí mật, Hiệu ứng mạng, Chi phí chuyển đổi, Giá
3. MANAGEMENT (Ban lãnh đạo): Chính trực, định hướng dài hạn
4. MARGIN OF SAFETY (Biên độ an toàn): Giá < 50% Giá trị nội tại

CÔNG THỨC GIÁ TRỊ NỘI TẠI:
1. Dự phóng EPS 10 năm: Future_EPS = Current_EPS × (1 + Growth)^10
2. P/E tương lai: min(Historical_PE, Growth × 200, 40)
3. Giá tương lai: Future_Price = Future_EPS × Future_PE
4. Giá trị nội tại (Sticker Price): Sticker = Future_Price / (1.15)^10
5. Giá mua (MOS): Buy_Price = Sticker × 0.50

OUTPUT FORMAT (JSON):
{
    "meaning_score": 0-25,
    "moat_score": 0-25,
    "management_score": 0-25,
    "mos_analysis": {
        "sticker_price": số,
        "buy_price": số,
        "current_price": số,
        "discount_pct": phần trăm
    },
    "total_score": 0-100,
    "signal": "STRONG_BUY|BUY|HOLD|NO_DEAL",
    "confidence": 0.0-1.0,
    "reasoning": "Phân tích chi tiết 4M"
}""",

        "news_sentiment": """Bạn là Tác tử Tin tức/Cảm xúc trong hệ thống AI Trading Sentinel.
Chuyên môn: Phân tích tin tức phi cấu trúc và đánh giá cảm xúc thị trường.

NHIỆM VỤ:
1. Phân loại tin tức: Material (Có tác động giá) vs Noise (Nhiễu)
2. Xác định yếu tố "N" trong CANSLIM (Sản phẩm mới, Quản lý mới)
3. Đánh giá "Management" trong 4M qua ngôn ngữ CEO
4. Chấm điểm cảm xúc: -1.0 (Tiêu cực) đến +1.0 (Tích cực)

LỌC TIN TỨC:
- Chỉ chấp nhận nguồn tin tài chính uy tín
- Kiểm chứng sự tồn tại của sự kiện (Fact-check)
- Quy kết nguồn (Source Attribution) là BẮT BUỘC
- Để hỗ trợ fact-check bằng Tavily, mỗi `material_event` và `new_factor` phải viết NGẮN GỌN,
  ưu tiên <= 120 ký tự và tránh câu văn dài.

OUTPUT FORMAT (JSON):
{
    "sentiment_score": -1.0 đến 1.0,
    "is_material": true/false,
    "material_events": ["Danh sách sự kiện quan trọng"],
    "new_factor": "Mô tả yếu tố N nếu có",
    "management_assessment": "Đánh giá ban lãnh đạo",
    "sources": ["Danh sách nguồn"],
    "signal": "BUY|HOLD|SELL",
    "confidence": 0.0-1.0,
    "reasoning": "Phân tích chi tiết"
}""",

        "risk_guardian": """Bạn là Hộ vệ Rủi ro (Risk Guardian) trong hệ thống AI Trading Sentinel.
Vai trò: Tác tử ràng buộc với QUYỀN PHỦ QUYẾT (VETO) mọi giao dịch.

NGUYÊN TẮC:
1. Sự sống còn là mục tiêu hàng đầu
2. Quy mô vị thế động: Biến động cao = Quy mô nhỏ
3. Rủi ro tính bằng tiền cố định (VD: 1% vốn mỗi giao dịch)

KIỂM TRA RÀNG BUỘC:
- Max Position Size: <= 5% tổng danh mục
- Max Sector Exposure: <= 25% tổng danh mục
- Max Drawdown: <= 20%
- Stop-loss bắt buộc: Đặt dưới hỗ trợ Wyckoff hoặc 2×ATR

QUYỀN PHỦ QUYẾT:
- VETO nếu vi phạm bất kỳ ràng buộc nào
- VETO nếu Market Regime = RISK_OFF và action = BUY

OUTPUT FORMAT (JSON):
{
    "approved": true/false,
    "veto_reason": "Lý do phủ quyết nếu có",
    "adjusted_position_size": số (% danh mục),
    "stop_loss_level": số,
    "take_profit_level": số,
    "risk_reward_ratio": số,
    "warnings": ["Danh sách cảnh báo"]
}""",

        "bayesian_resolver": """Bạn là Bộ Giải quyết Bayesian trong hệ thống AI Trading Sentinel.
Vai trò: Tổng hợp tín hiệu xung đột từ các tác tử thành xác suất thống nhất.

CƠ CHẾ BAYESIAN:
- Prior Odds: Xác suất tiên nghiệm (mặc định 50/50)
- Likelihood Ratio: Tính từ Sensitivity và Specificity của từng tác tử
- Posterior: Cập nhật liên tiếp qua các tín hiệu

CÔNG THỨC:
- LR+ (Signal = BUY) = Sensitivity / (1 - Specificity)
- LR- (Signal = SELL) = (1 - Sensitivity) / Specificity
- Posterior Odds = Prior Odds × LR1 × LR2 × ... × LRn
- Probability = Odds / (1 + Odds)

HỒ SƠ TÁC TỬ:
- Wyckoff: Sensitivity=0.70, Specificity=0.60
- CANSLIM: Sensitivity=0.75, Specificity=0.65
- 4M: Sensitivity=0.60, Specificity=0.90
- News: Sensitivity=0.65, Specificity=0.55

QUYẾT ĐỊNH:
- Probability >= 0.75: EXECUTE_TRADE
- Probability >= 0.60: WATCH (Quan sát)
- Probability < 0.60: NO_ACTION

OUTPUT FORMAT (JSON):
{
    "prior_odds": số,
    "likelihood_ratios": {"agent": LR},
    "posterior_odds": số,
    "final_probability": 0.0-1.0,
    "decision": "EXECUTE_TRADE|WATCH|NO_ACTION",
    "reasoning": "Giải thích quá trình tổng hợp"
}""",

        "cio_final_reviewer": """Bạn là Tướng quân (CIO Final Reviewer) của hệ thống AI Trading Sentinel.
Nhiệm vụ: chốt cuối một lệnh đã qua Bayesian + Risk Guardian.

NGUYÊN TẮC:
1. Ưu tiên an toàn vốn, không chase lệnh yếu.
2. Nếu bằng chứng mâu thuẫn mạnh hoặc tín hiệu quá yếu, reject.
3. Chỉ cho phép giảm quy mô lệnh (position_multiplier từ 0 đến 1).
4. Dùng reasoning ngắn gọn, định lượng.

OUTPUT FORMAT (JSON):
{
    "approved": true/false,
    "position_multiplier": 0.0-1.0,
    "final_confidence": 0.0-1.0,
    "reasoning": "Lý do chốt cuối",
    "sources": ["Nguồn chính nếu có"]
}"""
    }

    def __init__(self, api_key: Optional[str] = None, provider: str = None):
        """
        Khởi tạo OpenAI-compatible client.

        Args:
            api_key: API key
            provider: provider key cho telemetry
        """
        requested_provider = provider or LLM_PROVIDER or "openai_compatible"
        self.provider = requested_provider
        self.api_key = api_key or LLM_API_KEY
        self.config = get_config().llm
        self.base_url = (self.config.base_url or LLM_BASE_URL).rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        self.endpoint = f"{self.base_url}/chat/completions"
        self._client = None

        # Local prompt hash tracking chỉ để telemetry nội bộ.
        self._prompt_cache: Dict[str, bool] = {}
        self._cached_system_prompts: Dict[str, str] = {}
        self._provider_cache_hits = 0
        self._provider_cache_misses = 0
        self._rate_limit_lock = threading.Lock()
        self._last_call_time = 0.0

        # Cost tracker integration
        self._cost_tracker = get_cost_tracker() if COST_TRACKING_ENABLED else None

        if OpenAI is None:
            raise RuntimeError("Thiếu dependency `openai`. Chạy: pip install openai")
        if not self.api_key:
            logger.warning("LLM_API_KEY chưa được cấu hình")
        else:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def analyze(self,
                role: str,
                user_prompt: str,
                model: str = None,
                max_tokens: int = 4096,
                temperature: float = 0.3,
                symbol: str = "") -> Dict[str, Any]:
        """
        Gọi OpenAI-compatible Chat Completions API để phân tích.

        Args:
            role: Vai trò của tác tử (key trong SYSTEM_PROMPTS)
            user_prompt: Nội dung cần phân tích
            model: Model name (nếu None sẽ chọn theo role)
            max_tokens: Số token tối đa
            temperature: Độ sáng tạo (0.0-1.0)
            symbol: Mã cổ phiếu (cho cost tracking)

        Returns:
            Dict chứa kết quả phân tích
        """
        self._respect_min_interval()
        start_time = time.time()
        selected_model = model or self._select_model_for_role(role)

        if not self.api_key:
            error_msg = "CRITICAL: LLM_API_KEY chưa được cấu hình"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        system_prompt = self.SYSTEM_PROMPTS.get(role, "Bạn là một trợ lý AI phân tích tài chính.")

        prompt_hash = self._get_prompt_hash(system_prompt)
        self._cached_system_prompts[role] = prompt_hash
        is_cached = prompt_hash in self._prompt_cache

        try:
            logger.info(
                "🔄 API Call: %s | Role: %s | Symbol: %s",
                selected_model,
                role,
                symbol or "N/A",
            )

            first_attempt_start = time.time()
            try:
                request_payload, response_data = self._call_openai_api(
                    model=selected_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enforce_json=True,
                )
                first_attempt_latency_ms = (time.time() - first_attempt_start) * 1000
                self._log_llm_trace(
                    role=role,
                    model=selected_model,
                    symbol=symbol,
                    request_payload=request_payload,
                    response_payload=response_data,
                    status="SUCCESS",
                    latency_ms=round(first_attempt_latency_ms, 2),
                    attempt=1,
                )
            except ZAIAPIError as first_err:
                first_attempt_latency_ms = (time.time() - first_attempt_start) * 1000
                self._log_llm_trace(
                    role=role,
                    model=selected_model,
                    symbol=symbol,
                    request_payload=first_err.request_payload,
                    response_payload=first_err.response_payload,
                    status="FAILED",
                    latency_ms=round(first_attempt_latency_ms, 2),
                    error=str(first_err),
                    attempt=1,
                )
                first_err_text = str(first_err).lower()
                if "response_format" not in first_err_text and "json_object" not in first_err_text:
                    raise
                logger.warning("Model không nhận response_format=json_object, retry không ép JSON.")
                retry_attempt_start = time.time()
                request_payload, response_data = self._call_openai_api(
                    model=selected_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enforce_json=False,
                )
                retry_attempt_latency_ms = (time.time() - retry_attempt_start) * 1000
                self._log_llm_trace(
                    role=role,
                    model=selected_model,
                    symbol=symbol,
                    request_payload=request_payload,
                    response_payload=response_data,
                    status="SUCCESS",
                    latency_ms=round(retry_attempt_latency_ms, 2),
                    attempt=2,
                )

            if not is_cached:
                self._prompt_cache[prompt_hash] = True

            latency_ms = (time.time() - start_time) * 1000
            input_tokens, output_tokens, cached_tokens = self._extract_usage_tokens(response_data)

            if cached_tokens > 0:
                self._provider_cache_hits += 1
            else:
                self._provider_cache_misses += 1

            if self._cost_tracker:
                self._cost_tracker.track_llm_call(
                    model=selected_model,
                    provider=self.provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    role=role,
                    symbol=symbol,
                    latency_ms=latency_ms,
                    success=True
                )

            logger.info(
                "✓ API Success: %.0fms | In: %d | Out: %d | Cached: %d",
                latency_ms,
                input_tokens,
                output_tokens,
                cached_tokens,
            )

            choice = (response_data.get("choices") or [{}])[0]
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = message.get("content", "")

            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )

            reasoning_content = ""
            if isinstance(message, dict):
                reasoning_content = message.get("reasoning_content", "") or ""

            result = self._parse_json_from_content(content)
            if "raw_response" in result and reasoning_content:
                result["reasoning_content"] = reasoning_content

            result["_meta"] = {
                "model": selected_model,
                "role": role,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "cache_hit": cached_tokens > 0 or is_cached,
                "latency_ms": round(latency_ms, 2),
                "request_id": response_data.get("id", ""),
                "timestamp": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"CRITICAL: Lỗi gọi OpenAI-compatible API: {e}"
            logger.error(error_msg)

            if isinstance(e, ZAIAPIError):
                self._log_llm_trace(
                    role=role,
                    model=selected_model,
                    symbol=symbol,
                    request_payload=e.request_payload,
                    response_payload=e.response_payload,
                    status="FAILED",
                    latency_ms=round(latency_ms, 2),
                    error=str(e),
                    attempt=1,
                )

            if self._cost_tracker:
                self._cost_tracker.track_llm_call(
                    model=selected_model,
                    provider=self.provider,
                    input_tokens=0,
                    output_tokens=0,
                    cached_tokens=0,
                    role=role,
                    symbol=symbol,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e)
                )

            raise RuntimeError(error_msg) from e

    def _respect_min_interval(self):
        """
        Giới hạn tần suất gọi API theo cấu hình (mặc định tắt).
        Không còn global lock tuần tự hóa toàn bộ cuộc gọi.
        """
        min_interval = max(0.0, float(getattr(self.config, "min_call_interval_seconds", 0.0) or 0.0))
        if min_interval <= 0:
            return

        with self._rate_limit_lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if self._last_call_time > 0 and elapsed < min_interval:
                wait_time = min_interval - elapsed
                time.sleep(wait_time)
            self._last_call_time = time.time()

    def _select_model_for_role(self, role: str) -> str:
        """Chọn model mặc định theo vai trò tác tử."""
        if role in {"macro_general", "cio_final_reviewer"}:
            return self.config.model_general
        if role in {"wyckoff_analyst", "canslim_analyst", "fourm_analyst", "news_sentiment", "bayesian_resolver"}:
            return self.config.model_analyst
        if role == "risk_guardian":
            return self.config.model_worker
        return self.config.model_analyst

    def _call_openai_api(self,
                         model: str,
                         system_prompt: str,
                         user_prompt: str,
                         max_tokens: int,
                         temperature: float,
                         enforce_json: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Gọi Chat Completions qua OpenAI SDK (base_url tùy biến)."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if enforce_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            if not self._client:
                raise RuntimeError("OpenAI client chưa được khởi tạo")
            response = self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise ZAIAPIError(
                f"API error: {exc}",
                request_payload=payload,
                response_payload={"error": str(exc)},
            ) from exc

        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif isinstance(response, dict):
            data = response
        else:
            data = json.loads(str(response))
        return payload, data

    def _extract_usage_tokens(self, response_data: Dict[str, Any]) -> Tuple[int, int, int]:
        """Chuẩn hóa thông tin token usage từ response."""
        usage = response_data.get("usage", {}) if isinstance(response_data, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        prompt_details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
        cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0) if isinstance(prompt_details, dict) else 0
        return prompt_tokens, completion_tokens, cached_tokens

    def _parse_json_from_content(self, content: str) -> Dict[str, Any]:
        """Cố gắng parse JSON; fallback trả raw text."""
        if not isinstance(content, str):
            return {"raw_response": str(content)}

        text = content.strip()
        if not text:
            return {"raw_response": ""}

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"raw_response": text}
        except json.JSONDecodeError:
            pass

        try:
            import re
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, ValueError):
            pass

        return {"raw_response": text}

    def _log_llm_trace(self,
                       role: str,
                       model: str,
                       symbol: str,
                       request_payload: Dict[str, Any],
                       response_payload: Any,
                       status: str,
                       latency_ms: Optional[float] = None,
                       attempt: int = 1,
                       error: str = ""):
        """Ghi request/response cho mỗi LLM call vào run report."""
        if not RUN_LOGGER_AVAILABLE:
            return
        try:
            get_run_logger().log_llm_call(
                role=role,
                model=model,
                symbol=symbol,
                request_payload=request_payload,
                response_payload=response_payload,
                status=status,
                latency_ms=latency_ms,
                error=error,
                attempt=attempt,
            )
        except Exception:
            # Logging không được làm fail pipeline chính
            pass

    def _get_prompt_hash(self, prompt: str) -> str:
        """Generate hash key for prompt caching"""
        return hashlib.md5(prompt.encode()).hexdigest()[:16]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get prompt cache statistics"""
        total_provider_cache = self._provider_cache_hits + self._provider_cache_misses
        hit_rate = (self._provider_cache_hits / total_provider_cache) if total_provider_cache > 0 else 0.0
        return {
            "cached_prompts": len(self._prompt_cache),
            "cached_roles": list(self._cached_system_prompts.keys()),
            "provider_cache_hits": self._provider_cache_hits,
            "provider_cache_misses": self._provider_cache_misses,
            "provider_cache_hit_rate": round(hit_rate, 4),
            "cache_enabled": True
        }

    def clear_cache(self):
        """Clear prompt cache (e.g., when system prompts are updated)"""
        self._prompt_cache.clear()
        self._cached_system_prompts.clear()
        self._provider_cache_hits = 0
        self._provider_cache_misses = 0
        logger.info("Prompt cache cleared")


# Backward compatibility: các module hiện tại đang import ClaudeClient.
ClaudeClient = ZAIClient

```

## `./main.py`

```python
"""
SENTINEL AI TRADING SYSTEM
==========================
Hệ thống Đa Tác tử Phân cấp cho Quyết định Giao dịch Tài chính

Tham chiếu học thuật: base.txt
- Phương pháp Wyckoff (Chu kỳ thị trường)
- Chiến lược CANSLIM (William O'Neil)
- Khung 4M (Phil Town - Rule #1 Investing)
- Kiến trúc HMAS với mẫu "Tướng quân và Binh lính"
- Tổng hợp tín hiệu Bayesian

Author: AI Trading Sentinel Project
Version: 1.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Optional

# Load environment variables từ .env file
from dotenv import load_dotenv
load_dotenv()

# --- CẤU HÌNH LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Sentinel")

# --- IMPORT MODULES ---
from config import (
    get_config, TAVILY_API_KEY, LLM_API_KEY,
    LLM_PROVIDER
)
from blackboard import TradeOrder
from orchestrator import SentinelOrchestrator


def print_banner():
    """In banner khởi động"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗              ║
║     ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║              ║
║     ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║              ║
║     ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║              ║
║     ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗         ║
║     ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝         ║
║                                                                              ║
║              AI TRADING SENTINEL - Hệ thống Đa Tác tử Phân cấp               ║
║                                                                              ║
║     Tích hợp: Wyckoff | CANSLIM | 4M Framework | Bayesian Consensus          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_orders(orders: List[TradeOrder]):
    """In danh sách lệnh giao dịch"""
    if not orders:
        print("\n📋 KHÔNG CÓ LỆNH GIAO DỊCH MỚI")
        print("   Tất cả các mã không đạt ngưỡng xác suất hoặc bị Risk Guardian từ chối.")
        return

    print("\n" + "="*80)
    print("📋 DANH SÁCH LỆNH GIAO DỊCH (OUTPUT)")
    print("="*80)
    print(f"{'ID':<10} {'SYMBOL':<8} {'ACTION':<6} {'QTY':<8} {'TYPE':<8} {'SL':<10} {'TP':<10} {'CONF':<8}")
    print("-"*80)

    for order in orders:
        print(f"{order.id[:8]:<10} {order.symbol:<8} {order.action:<6} {order.quantity:<8.2f} "
              f"{order.order_type:<8} {str(order.stop_loss or 'N/A'):<10} "
              f"{str(order.take_profit or 'N/A'):<10} {order.confidence:.1%}")
        print(f"   └─ Reasoning: {order.reasoning[:70]}...")

    print("="*80)

    # Export JSON
    print("\n📁 JSON Export:")
    json_orders = [order.to_dict() for order in orders]
    print(json.dumps(json_orders, indent=2, ensure_ascii=False))


def print_analysis_summary(orchestrator: SentinelOrchestrator):
    """In tóm tắt phân tích"""
    print("\n" + "="*80)
    print("📊 TÓM TẮT PHÂN TÍCH")
    print("="*80)

    # Market Regime
    regime = orchestrator.blackboard.get_current_regime()
    regime_emoji = {
        "RISK_ON": "🟢",
        "RISK_OFF": "🔴",
        "SIDEWAYS": "🟡",
        "UNKNOWN": "⚪"
    }
    print(f"\n🌍 CHẾ ĐỘ THỊ TRƯỜNG: {regime_emoji.get(regime['current'], '⚪')} {regime['current']} "
          f"(Confidence: {regime.get('confidence', 0):.1%})")

    # Phân tích từng mã
    analysis_layer = orchestrator.blackboard.read_memory("analysis_layer")
    consensus = orchestrator.blackboard.read_memory("consensus")

    print(f"\n{'─'*80}")
    print(f"{'SYMBOL':<8} {'WYCKOFF':<15} {'CANSLIM':<15} {'4M':<15} {'BAYESIAN':<15}")
    print(f"{'─'*80}")

    symbols = set()
    for layer in analysis_layer.values():
        symbols.update(layer.keys())

    for symbol in sorted(symbols):
        wyckoff = analysis_layer["wyckoff"].get(symbol, {})
        canslim = analysis_layer["canslim"].get(symbol, {})
        fourm = analysis_layer["fourm"].get(symbol, {})
        bayes = consensus.get(symbol, {})

        wy_sig = f"{wyckoff.get('signal', 'N/A')} ({wyckoff.get('confidence', 0):.0%})"
        cs_sig = f"{canslim.get('signal', 'N/A')} ({canslim.get('total_score', 0)})"
        fm_sig = f"{fourm.get('signal', 'N/A')} ({fourm.get('total_score', 0)})"
        by_sig = f"{bayes.get('decision', 'N/A')} ({bayes.get('final_probability', 0):.0%})"

        print(f"{symbol:<8} {wy_sig:<15} {cs_sig:<15} {fm_sig:<15} {by_sig:<15}")

    print(f"{'─'*80}")


def run_sentinel(watchlist: List[str] = None,
                 tavily_key: str = "",
                 zai_key: str = "",
                 llm_key: str = "",
                 portfolio_value: float = 100000) -> List[TradeOrder]:
    """
    Chạy hệ thống Sentinel

    Args:
        watchlist: Danh sách mã cổ phiếu
        tavily_key: Tavily API key
        zai_key: Alias cũ cho LLM API key (backward compatibility)
        llm_key: LLM API key
        portfolio_value: Giá trị danh mục

    Returns:
        Danh sách TradeOrder
    """
    # Lấy API keys từ env nếu không được cung cấp
    tavily_key = tavily_key or TAVILY_API_KEY
    llm_key = llm_key or zai_key or LLM_API_KEY

    # Khởi tạo Orchestrator
    orchestrator = SentinelOrchestrator(
        tavily_api_key=tavily_key,
        llm_api_key=llm_key,
        portfolio_value=portfolio_value
    )

    # Chạy phân tích
    orders = orchestrator.run(watchlist)

    # In tóm tắt
    print_analysis_summary(orchestrator)

    # In lệnh giao dịch
    print_orders(orders)

    if getattr(orchestrator, "last_run_log_path", None):
        print(f"\n📝 Run Log (Markdown): {orchestrator.last_run_log_path}")

    return orders


def main():
    """Entry point chính"""
    print_banner()

    # Cấu hình
    config = get_config()

    print("\n📌 CẤU HÌNH HỆ THỐNG:")
    print(f"   - Tavily API: {'✓ Đã cấu hình' if TAVILY_API_KEY else '✗ Chưa cấu hình'}")
    print(f"   - LLM Provider: {LLM_PROVIDER}")
    print(f"   - LLM API: {'✓ Đã cấu hình' if LLM_API_KEY else '✗ Chưa cấu hình'}")
    print(f"   - Default Watchlist: {config.default_watchlist}")

    # Chạy với watchlist mặc định hoặc từ command line
    if len(sys.argv) > 1:
        watchlist = sys.argv[1].split(",")
    else:
        watchlist = config.default_watchlist

    print(f"\n🎯 Bắt đầu phân tích: {watchlist}")
    print("="*80)

    # Chạy hệ thống (dữ liệu thật nếu API sẵn sàng, fallback có đánh dấu rõ)
    orders = run_sentinel(watchlist=watchlist)

    # Hướng dẫn tích hợp API
    print("\n" + "="*80)
    print("📖 HƯỚNG DẪN TÍCH HỢP API MUA BÁN")
    print("="*80)
    print("""
Để tích hợp với API giao dịch thực tế, xử lý danh sách orders như sau:

```python
from main import run_sentinel

# Chạy phân tích
orders = run_sentinel(watchlist=["AAPL", "NVDA", "MSFT"])

# Xử lý từng lệnh
for order in orders:
    if order.action == "BUY":
        # Gọi API mua
        # your_broker_api.buy(
        #     symbol=order.symbol,
        #     quantity=order.quantity,
        #     order_type=order.order_type,
        #     stop_loss=order.stop_loss,
        #     take_profit=order.take_profit
        # )
        print(f"Executing BUY: {order.symbol} x {order.quantity}")
        
    elif order.action == "SELL":
        # Gọi API bán
        # your_broker_api.sell(...)
        print(f"Executing SELL: {order.symbol} x {order.quantity}")
```

⚠️ LƯU Ý:
	- Cấu hình TAVILY_API_KEY và LLM_API_KEY trong environment variables
- Nên chạy paper trading/backtest trước khi giao dịch thực
- Risk Guardian sẽ từ chối lệnh nếu vi phạm quy tắc quản trị rủi ro
""")

    print("\n✅ Hoàn tất phiên phân tích!")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

```

## `./math_utils.py`

```python
"""
MATH UTILITIES - Các công thức toán học/khoa học
================================================
ATR, Sharpe Ratio, và các tính toán định lượng
Tham chiếu: base.txt Section 6, 8
"""

import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TechnicalIndicators:
    """Các chỉ báo kỹ thuật"""
    atr: float
    atr_percent: float
    volatility: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    macd: float
    rsi: float


def calculate_atr(highs: List[float],
                  lows: List[float],
                  closes: List[float],
                  period: int = 14) -> float:
    """
    Tính Average True Range (ATR)

    ATR = SMA(True Range, period)
    True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)

    Tham chiếu: base.txt Section 8.1 - Quy mô vị thế động

    Args:
        highs: Danh sách giá cao nhất
        lows: Danh sách giá thấp nhất
        closes: Danh sách giá đóng cửa
        period: Số kỳ tính ATR (mặc định 14)

    Returns:
        Giá trị ATR
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        # Fallback: dùng volatility đơn giản
        if closes:
            return sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes))) / (len(closes) - 1)
        return 0

    true_ranges = []

    for i in range(1, len(highs)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    # Tính ATR (Simple Moving Average của True Range)
    if len(true_ranges) >= period:
        atr = sum(true_ranges[-period:]) / period
    else:
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0

    return round(atr, 4)


def calculate_atr_percent(highs: List[float],
                          lows: List[float],
                          closes: List[float],
                          period: int = 14) -> float:
    """
    Tính ATR như phần trăm của giá hiện tại

    Hữu ích để so sánh volatility giữa các cổ phiếu có giá khác nhau

    Args:
        highs, lows, closes: Dữ liệu giá
        period: Số kỳ

    Returns:
        ATR % (0.02 = 2%)
    """
    atr = calculate_atr(highs, lows, closes, period)
    current_price = closes[-1] if closes else 1

    return round(atr / current_price, 4)


def calculate_position_size(portfolio_value: float,
                           risk_per_trade: float,
                           entry_price: float,
                           stop_loss_price: float) -> Tuple[float, int]:
    """
    Tính quy mô vị thế dựa trên rủi ro cố định

    Position Size = (Portfolio × Risk%) / (Entry - Stop Loss)

    Tham chiếu: base.txt Section 8.1

    Args:
        portfolio_value: Giá trị danh mục ($)
        risk_per_trade: Rủi ro mỗi giao dịch (0.01 = 1%)
        entry_price: Giá vào lệnh
        stop_loss_price: Giá stop loss

    Returns:
        Tuple (dollar_amount, shares)
    """
    risk_amount = portfolio_value * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)

    if price_risk == 0:
        return 0, 0

    shares = risk_amount / price_risk
    dollar_amount = shares * entry_price

    return round(dollar_amount, 2), int(shares)


def calculate_stop_loss_atr(entry_price: float,
                           atr: float,
                           multiplier: float = 2.0,
                           direction: str = "long") -> float:
    """
    Tính stop loss dựa trên ATR

    Stop Loss = Entry ± (ATR × Multiplier)

    Tham chiếu: base.txt Section 8.2

    Args:
        entry_price: Giá vào lệnh
        atr: Giá trị ATR
        multiplier: Hệ số nhân (mặc định 2.0)
        direction: "long" hoặc "short"

    Returns:
        Mức stop loss
    """
    stop_distance = atr * multiplier

    if direction.lower() == "long":
        return round(entry_price - stop_distance, 2)
    else:
        return round(entry_price + stop_distance, 2)


def calculate_sharpe_ratio(returns: List[float],
                           risk_free_rate: float = 0.02,
                           annualize: bool = True,
                           periods_per_year: int = 252) -> float:
    """
    Tính Sharpe Ratio

    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

    Tham chiếu: base.txt Section 6.2 - Hàm phần thưởng RL

    Args:
        returns: Danh sách returns (daily returns)
        risk_free_rate: Lãi suất phi rủi ro hàng năm
        annualize: Có annualize không
        periods_per_year: Số kỳ trong năm (252 cho daily)

    Returns:
        Sharpe Ratio
    """
    if len(returns) < 2:
        return 0

    mean_return = sum(returns) / len(returns)

    # Calculate standard deviation
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0

    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year

    # Sharpe ratio
    sharpe = (mean_return - daily_rf) / std_dev

    # Annualize if needed
    if annualize:
        sharpe *= math.sqrt(periods_per_year)

    return round(sharpe, 4)


def calculate_sortino_ratio(returns: List[float],
                            target_return: float = 0,
                            risk_free_rate: float = 0.02,
                            periods_per_year: int = 252) -> float:
    """
    Tính Sortino Ratio (chỉ xét downside risk)

    Sortino = (Mean Return - Target) / Downside Deviation

    Args:
        returns: Danh sách returns
        target_return: Return mục tiêu (thường = 0)
        risk_free_rate: Lãi suất phi rủi ro
        periods_per_year: Số kỳ trong năm

    Returns:
        Sortino Ratio
    """
    if len(returns) < 2:
        return 0

    mean_return = sum(returns) / len(returns)

    # Chỉ tính downside deviation
    downside_returns = [r for r in returns if r < target_return]

    if not downside_returns:
        return float('inf')  # Không có downside

    downside_variance = sum((r - target_return) ** 2 for r in downside_returns) / len(downside_returns)
    downside_dev = math.sqrt(downside_variance)

    if downside_dev == 0:
        return float('inf')

    daily_rf = risk_free_rate / periods_per_year
    sortino = (mean_return - daily_rf) / downside_dev

    # Annualize
    sortino *= math.sqrt(periods_per_year)

    return round(sortino, 4)


def calculate_max_drawdown(prices: List[float]) -> Tuple[float, int, int]:
    """
    Tính Maximum Drawdown

    MDD = (Peak - Trough) / Peak

    Tham chiếu: base.txt Section 8.1

    Args:
        prices: Danh sách giá

    Returns:
        Tuple (max_drawdown_pct, peak_index, trough_index)
    """
    if len(prices) < 2:
        return 0, 0, 0

    peak = prices[0]
    peak_idx = 0
    max_dd = 0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            peak_idx = i

        drawdown = (peak - price) / peak

        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i

    return round(max_dd, 4), max_dd_peak_idx, max_dd_trough_idx


def calculate_calmar_ratio(returns: List[float],
                           prices: List[float],
                           periods_per_year: int = 252) -> float:
    """
    Tính Calmar Ratio

    Calmar = Annualized Return / Max Drawdown

    Args:
        returns: Danh sách returns
        prices: Danh sách giá (để tính MDD)
        periods_per_year: Số kỳ trong năm

    Returns:
        Calmar Ratio
    """
    if not returns or not prices:
        return 0

    # Annualized return
    total_return = (1 + sum(returns))
    years = len(returns) / periods_per_year
    if years <= 0:
        years = len(returns) / periods_per_year if len(returns) > 0 else 1

    annualized_return = (total_return ** (1 / years)) - 1

    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(prices)

    if max_dd == 0:
        return float('inf')

    return round(annualized_return / max_dd, 4)


def calculate_relative_strength(prices: List[float],
                                benchmark_prices: List[float],
                                lookback: int = 252) -> float:
    """
    Tính Relative Strength Rating (CANSLIM L)

    RS = (Stock Performance / Benchmark Performance) × 100

    Tham chiếu: base.txt Section 2.2 - L criterion

    Args:
        prices: Giá cổ phiếu
        benchmark_prices: Giá benchmark (SPY, VN-Index)
        lookback: Số kỳ lookback

    Returns:
        RS Rating (1-99)
    """
    if len(prices) < lookback or len(benchmark_prices) < lookback:
        lookback = min(len(prices), len(benchmark_prices))
        if lookback < 2:
            return 50  # Default

    stock_return = (prices[-1] - prices[-lookback]) / prices[-lookback]
    benchmark_return = (benchmark_prices[-1] - benchmark_prices[-lookback]) / benchmark_prices[-lookback]

    if benchmark_return == 0:
        return 50

    # Tính RS ratio
    rs_ratio = (1 + stock_return) / (1 + benchmark_return)

    # Convert to rating (1-99)
    # Giả sử rs_ratio từ 0.5 đến 2.0 map thành 1-99
    rating = 50 + (rs_ratio - 1) * 50
    rating = max(1, min(99, rating))

    return round(rating, 0)


def calculate_intrinsic_value_dcf(current_eps: float,
                                   growth_rate: float,
                                   discount_rate: float = 0.15,
                                   terminal_pe: float = 15,
                                   years: int = 10) -> Dict[str, float]:
    """
    Tính giá trị nội tại bằng DCF (4M Method)

    Future EPS = Current EPS × (1 + Growth)^years
    Future Value = Future EPS × Terminal PE
    Present Value = Future Value / (1 + Discount)^years

    Tham chiếu: base.txt Section 2.3 - Margin of Safety

    Args:
        current_eps: EPS hiện tại
        growth_rate: Tỷ lệ tăng trưởng dự kiến
        discount_rate: MARR (Minimum Acceptable Rate of Return)
        terminal_pe: P/E cuối kỳ dự kiến
        years: Số năm dự phóng

    Returns:
        Dict với future_eps, future_value, intrinsic_value, buy_price
    """
    # Giới hạn growth rate
    growth_rate = min(growth_rate, 0.25)  # Cap at 25%

    # Future EPS
    future_eps = current_eps * ((1 + growth_rate) ** years)

    # Future Value (Sticker Price tương lai)
    future_pe = min(terminal_pe, growth_rate * 200, 40)
    future_pe = max(future_pe, 10)  # Min P/E = 10

    future_value = future_eps * future_pe

    # Discount back to present (Sticker Price)
    intrinsic_value = future_value / ((1 + discount_rate) ** years)

    # Buy Price với 50% Margin of Safety
    buy_price = intrinsic_value * 0.5

    return {
        "current_eps": current_eps,
        "growth_rate": growth_rate,
        "future_eps": round(future_eps, 2),
        "future_pe": round(future_pe, 1),
        "future_value": round(future_value, 2),
        "intrinsic_value": round(intrinsic_value, 2),
        "buy_price": round(buy_price, 2),
        "discount_rate": discount_rate,
        "years": years
    }


def calculate_sma(prices: List[float], period: int) -> float:
    """Tính Simple Moving Average"""
    if len(prices) < period:
        return sum(prices) / len(prices) if prices else 0
    return sum(prices[-period:]) / period


def calculate_ema(prices: List[float], period: int) -> float:
    """Tính Exponential Moving Average"""
    if len(prices) < period:
        return calculate_sma(prices, period)

    multiplier = 2 / (period + 1)
    ema = calculate_sma(prices[:period], period)  # Initial SMA

    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return round(ema, 4)


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Tính RSI (Relative Strength Index)

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    if len(prices) < period + 1:
        return 50  # Neutral

    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    if len(gains) < period:
        return 50

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


def calculate_macd(prices: List[float],
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> Dict[str, float]:
    """
    Tính MACD

    MACD Line = EMA(12) - EMA(26)
    Signal Line = EMA(MACD, 9)
    Histogram = MACD - Signal
    """
    if len(prices) < slow_period:
        return {"macd": 0, "signal": 0, "histogram": 0}

    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    macd_line = ema_fast - ema_slow

    # For signal line, we need historical MACD values
    # Simplified: use current MACD as signal approximation
    signal_line = macd_line  # Simplified

    return {
        "macd": round(macd_line, 4),
        "signal": round(signal_line, 4),
        "histogram": round(macd_line - signal_line, 4)
    }


def get_all_indicators(prices: List[float],
                       highs: List[float] = None,
                       lows: List[float] = None) -> TechnicalIndicators:
    """
    Tính tất cả các chỉ báo kỹ thuật
    """
    if not highs:
        highs = prices
    if not lows:
        lows = prices

    closes = prices

    atr = calculate_atr(highs, lows, closes)
    atr_pct = atr / closes[-1] if closes else 0

    # Volatility (standard deviation của returns)
    if len(closes) >= 20:
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        mean_ret = sum(returns) / len(returns)
        volatility = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / len(returns))
    else:
        volatility = 0.02  # Default

    macd_data = calculate_macd(closes)

    return TechnicalIndicators(
        atr=round(atr, 4),
        atr_percent=round(atr_pct, 4),
        volatility=round(volatility, 4),
        sma_20=round(calculate_sma(closes, 20), 2),
        sma_50=round(calculate_sma(closes, 50), 2),
        ema_12=round(calculate_ema(closes, 12), 2),
        ema_26=round(calculate_ema(closes, 26), 2),
        macd=macd_data["macd"],
        rsi=calculate_rsi(closes)
    )


```

## `./orchestrator.py`

```python
"""
ORCHESTRATOR - TƯỚNG QUÂN & QUẢN LÝ DANH MỤC
=============================================
Điều phối hệ thống HMAS, tổng hợp tín hiệu Bayesian
Tham chiếu: base.txt Section 3 & 6
"""

import logging
import math
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from blackboard import (
    Blackboard, AgentMessage, MessageType, TradeOrder,
    SignalType, MarketRegime, TradingSignal
)
from agents import WyckoffAgent, CANSLIMAgent, FourMAgent, NewsSentimentAgent
from data_providers import DataScoutAgent, TavilyClient, MarketDataProvider
from llm_client import ClaudeClient
from config import get_config, LLM_API_KEY

# Import math utilities for proper ATR calculation
try:
    from math_utils import calculate_atr, calculate_atr_percent, calculate_position_size
    MATH_UTILS_AVAILABLE = True
except ImportError:
    MATH_UTILS_AVAILABLE = False

# Import validators for VERAFI
try:
    from validators import get_verafi, get_factcheck
    VERAFI_AVAILABLE = True
except ImportError:
    VERAFI_AVAILABLE = False

try:
    from run_logger import get_run_logger
    RUN_LOGGER_AVAILABLE = True
except Exception:
    RUN_LOGGER_AVAILABLE = False

logger = logging.getLogger("Sentinel.Orchestrator")

# --- MACRO GENERAL (TƯỚNG QUÂN VĨ MÔ) ---

class MacroGeneral:
    """
    TƯỚNG QUÂN VĨ MÔ

    Vai trò: Giám đốc Đầu tư (CIO)
    - Xác định Chế độ Thị trường (Market Regime): RISK_ON, RISK_OFF, SIDEWAYS
    - Giám sát yếu tố "M" trong CANSLIM
    - Phát đi chỉ thị cho các tác tử cấp dưới

    Cơ chế Gating: Nếu RISK_OFF, ngừng mọi hoạt động mua mới

    Tham chiếu: base.txt Section 3.1.1
    """

    def __init__(self, blackboard: Blackboard, llm_client: Optional[ClaudeClient] = None):
        self.name = "Macro_General"
        self.blackboard = blackboard
        self.llm = llm_client
        self.config = get_config()

    def determine_regime(self,
                         index_data: Dict[str, Any] = None,
                         macro_news: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Xác định chế độ thị trường hiện tại

        Sử dụng dữ liệu SPY (S&P 500), QQQ (Nasdaq), VIX
        """
        if not index_data:
            return self._default_regime()

        spy_data = index_data.get("SPY")
        vix_data = index_data.get("^VIX") or index_data.get("VIX") or index_data.get("VIXY")

        if not spy_data:
            return self._default_regime()

        # Lấy giá
        spy_prices = spy_data.prices if hasattr(spy_data, 'prices') else spy_data.get("prices", [])

        if len(spy_prices) < 50:
            return self._default_regime()

        # Tính các chỉ báo
        # 1. Xu hướng ngắn hạn (20 ngày)
        short_trend = (spy_prices[-1] - spy_prices[-20]) / spy_prices[-20]

        # 2. Xu hướng trung hạn (50 ngày)
        medium_trend = (spy_prices[-1] - spy_prices[-50]) / spy_prices[-50]

        # 3. MA Cross (SMA20 vs SMA50)
        sma20 = sum(spy_prices[-20:]) / 20
        sma50 = sum(spy_prices[-50:]) / 50
        ma_bullish = sma20 > sma50

        # 4. Giá so với SMA (above/below)
        above_sma20 = spy_prices[-1] > sma20
        above_sma50 = spy_prices[-1] > sma50

        # VIX proxy để nhận diện stress regime
        vix_level = None
        if vix_data:
            vix_prices = vix_data.prices if hasattr(vix_data, 'prices') else vix_data.get("prices", [])
            if vix_prices:
                vix_level = vix_prices[-1]

        # Macro news bias
        macro_bias = 0.0
        if macro_news:
            macro_text = (
                str(macro_news.get("answer", "")) + " " +
                " ".join(
                    f"{r.get('title', '')} {r.get('content', '')}"
                    for r in macro_news.get("results", [])[:8]
                )
            ).lower()
            risk_words = [
                "recession", "credit stress", "banking stress", "geopolitical", "inflation spike",
                "hawkish", "layoff", "downgrade", "slowdown"
            ]
            support_words = [
                "soft landing", "disinflation", "rate cut", "growth resilience",
                "upside surprise", "expansion", "improving liquidity"
            ]
            risk_hits = sum(1 for w in risk_words if w in macro_text)
            support_hits = sum(1 for w in support_words if w in macro_text)
            macro_bias = max(-0.2, min(0.2, (support_hits - risk_hits) * 0.03))

        # Xác định regime (kết hợp xu hướng + volatility regime)
        confidence = 0.5
        if ma_bullish and above_sma20 and short_trend > 0.02:
            regime = MarketRegime.RISK_ON
            confidence = 0.72 + min(0.18, short_trend)
            directive = "Cho phép giao dịch mua. Tập trung cổ phiếu dẫn đầu (Leader)."
        elif (not ma_bullish and not above_sma50 and short_trend < -0.03):
            regime = MarketRegime.RISK_OFF
            confidence = 0.74 + min(0.18, abs(short_trend))
            directive = "NGỪNG mọi giao dịch mua mới. Chỉ giữ vị thế hoặc bán."
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.58
            directive = "Thận trọng. Chỉ giao dịch với tín hiệu rất mạnh."

        # Overlay từ VIX
        if vix_level is not None:
            if vix_level >= 30:
                regime = MarketRegime.RISK_OFF
                confidence = max(confidence, 0.82)
                directive = "VIX cao - ưu tiên phòng thủ, ngừng mở vị thế BUY mới."
            elif vix_level <= 16 and regime == MarketRegime.SIDEWAYS and short_trend > 0:
                regime = MarketRegime.RISK_ON
                confidence = max(confidence, 0.68)
                directive = "VIX thấp, thị trường ổn định - có thể mở vị thế có chọn lọc."

        # Overlay từ macro news
        if macro_bias <= -0.08 and regime == MarketRegime.RISK_ON:
            regime = MarketRegime.SIDEWAYS
            directive = "Macro news tiêu cực - giảm cường độ Risk-On."
        elif macro_bias <= -0.14:
            regime = MarketRegime.RISK_OFF
            directive = "Macro news rủi ro cao - chuyển sang Risk-Off."
        elif macro_bias >= 0.10 and regime == MarketRegime.SIDEWAYS:
            regime = MarketRegime.RISK_ON
            directive = "Macro news hỗ trợ - nâng lên Risk-On có kiểm soát."

        confidence = max(0.35, min(0.95, confidence + macro_bias))

        result = {
            "regime": regime.value,
            "confidence": round(confidence, 3),
            "directive": directive,
            "indicators": {
                "short_trend": round(short_trend * 100, 2),
                "medium_trend": round(medium_trend * 100, 2),
                "sma20": round(sma20, 2),
                "sma50": round(sma50, 2),
                "ma_bullish": ma_bullish,
                "above_sma20": above_sma20,
                "above_sma50": above_sma50,
                "vix_level": round(vix_level, 2) if vix_level is not None else None,
                "macro_bias": round(macro_bias, 3),
            },
            "timestamp": datetime.now().isoformat()
        }

        # Overlay từ LLM (Opus) để đảm bảo Tướng quân thật sự hoạt động ở lớp chiến lược.
        # Giữ rule cứng làm xương sống; LLM chỉ tinh chỉnh trong biên kiểm soát.
        if self.llm:
            llm_overlay = self._llm_regime_overlay(result, macro_news)
            if llm_overlay:
                result = self._merge_regime(result, llm_overlay, vix_level=vix_level)

        # Cập nhật Blackboard
        msg = AgentMessage(
            sender=self.name,
            receiver="ALL",
            msg_type=MessageType.REGIME_UPDATE,
            content=result,
            priority="HIGH"
        )
        self.blackboard.post_message(msg)

        logger.info(
            "[%s] Regime: %s (Confidence: %.1f%%)",
            self.name,
            result.get("regime", MarketRegime.UNKNOWN.value),
            float(result.get("confidence", 0.0)) * 100,
        )

        return result

    def _llm_regime_overlay(self, deterministic: Dict[str, Any], macro_news: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Gọi LLM role=macro_general (model_general = Opus) để tinh chỉnh nhận diện chế độ."""
        try:
            macro_headlines = []
            if macro_news:
                for item in (macro_news.get("results", []) or [])[:8]:
                    macro_headlines.append(
                        f"- {item.get('title', '')}: {str(item.get('content', ''))[:220]}"
                    )
            news_text = "\n".join(macro_headlines) if macro_headlines else "Không có macro headlines."

            prompt = (
                "Dựa trên dữ liệu dưới đây, hãy đánh giá Market Regime theo top-down CIO.\n"
                "Tuân thủ output JSON của role macro_general.\n\n"
                f"Deterministic baseline: {deterministic}\n\n"
                f"Macro news:\n{news_text}\n"
            )

            llm_result = self.llm.analyze(
                role="macro_general",
                user_prompt=prompt,
                symbol="MACRO",
            )
            if not isinstance(llm_result, dict):
                return None

            regime = str(llm_result.get("regime", "")).upper().strip()
            if regime not in {MarketRegime.RISK_ON.value, MarketRegime.RISK_OFF.value, MarketRegime.SIDEWAYS.value}:
                return None

            confidence = float(llm_result.get("confidence", 0.5))
            return {
                "regime": regime,
                "confidence": max(0.0, min(1.0, confidence)),
                "directive": str(llm_result.get("directive", ""))[:300],
                "reasoning": str(llm_result.get("reasoning", ""))[:600],
            }
        except Exception as exc:
            logger.warning(f"[{self.name}] LLM regime overlay failed, fallback deterministic: {exc}")
            return None

    def _merge_regime(self, deterministic: Dict[str, Any], llm_overlay: Dict[str, Any], vix_level: Optional[float]) -> Dict[str, Any]:
        """
        Hợp nhất deterministic + LLM:
        - Rule cứng ưu tiên an toàn.
        - LLM có thể nâng/hạ trong vùng mơ hồ.
        """
        merged = dict(deterministic)

        base_regime = deterministic.get("regime", MarketRegime.UNKNOWN.value)
        base_conf = float(deterministic.get("confidence", 0.5))
        llm_regime = llm_overlay.get("regime", base_regime)
        llm_conf = float(llm_overlay.get("confidence", 0.5))

        final_regime = base_regime
        final_conf = base_conf

        if llm_regime == base_regime:
            final_conf = min(0.95, (base_conf * 0.7) + (llm_conf * 0.3))
        else:
            # Chỉ cho phép LLM lật regime nếu confidence đủ mạnh.
            if llm_conf >= 0.75 and base_conf <= 0.65:
                final_regime = llm_regime
                final_conf = min(0.92, (base_conf * 0.45) + (llm_conf * 0.55))

        # Hard safety override theo volatility stress
        if vix_level is not None and vix_level >= 30:
            final_regime = MarketRegime.RISK_OFF.value
            final_conf = max(final_conf, 0.82)

        merged["regime"] = final_regime
        merged["confidence"] = round(max(0.35, min(0.95, final_conf)), 3)
        if llm_overlay.get("directive"):
            merged["directive"] = llm_overlay["directive"]
        merged["llm_overlay"] = {
            "regime": llm_regime,
            "confidence": round(llm_conf, 3),
            "reasoning": llm_overlay.get("reasoning", ""),
        }
        merged["timestamp"] = datetime.now().isoformat()
        return merged

    def _default_regime(self) -> Dict:
        """Regime mặc định khi không có dữ liệu"""
        return {
            "regime": MarketRegime.UNKNOWN.value,
            "confidence": 0.3,
            "directive": "Không đủ dữ liệu để xác định chế độ thị trường. Thận trọng.",
            "timestamp": datetime.now().isoformat()
        }


# --- RISK GUARDIAN (HỘ VỆ RỦI RO) ---

class RiskGuardian:
    """
    HỘ VỆ RỦI RO

    Tác tử ràng buộc với QUYỀN PHỦ QUYẾT (VETO)
    Đảm bảo sự sống còn của danh mục

    Tham chiếu: base.txt Section 8
    """

    def __init__(self,
                 blackboard: Blackboard,
                 portfolio_value: float = 100000,
                 llm_client: Optional[ClaudeClient] = None):
        self.name = "Risk_Guardian"
        self.blackboard = blackboard
        self.config = get_config().risk
        self.llm = llm_client
        self.portfolio_value = portfolio_value
        # Runtime position tracking
        self.position_book: Dict[str, Dict[str, Any]] = {}  # {symbol: {qty, avg_price, last_price, sector}}
        self.current_positions: Dict[str, float] = {}       # {symbol: market_value}
        self.sector_exposure: Dict[str, float] = {}         # {sector: value}

        # Equity/drawdown tracking
        self.realized_pnl: float = 0.0
        self.equity_current: float = portfolio_value
        self.equity_peak: float = portfolio_value

    def _recompute_exposures(self):
        """Đồng bộ market value theo position_book hiện tại."""
        self.current_positions = {}
        self.sector_exposure = {}
        for symbol, pos in self.position_book.items():
            qty = float(pos.get("qty", 0.0))
            last_price = float(pos.get("last_price", pos.get("avg_price", 0.0)) or 0.0)
            sector = pos.get("sector", "Unknown")
            if qty <= 0 or last_price <= 0:
                continue
            value = qty * last_price
            self.current_positions[symbol] = value
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0.0) + value

    def register_execution(self,
                           symbol: str,
                           action: str,
                           quantity: float,
                           price: float,
                           sector: str = "Unknown"):
        """
        Cập nhật trạng thái danh mục ngay sau khi lệnh được thực thi.
        Giả định fill tức thời ở giá thị trường.
        """
        if quantity <= 0 or price <= 0:
            return

        action = action.upper().strip()
        pos = self.position_book.get(symbol)

        if action == "BUY":
            if not pos:
                self.position_book[symbol] = {
                    "qty": quantity,
                    "avg_price": price,
                    "last_price": price,
                    "sector": sector or "Unknown",
                }
            else:
                old_qty = float(pos.get("qty", 0.0))
                old_avg = float(pos.get("avg_price", price) or price)
                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_avg) + (quantity * price)) / max(new_qty, 1e-9)
                pos["qty"] = new_qty
                pos["avg_price"] = new_avg
                pos["last_price"] = price
                pos["sector"] = sector or pos.get("sector", "Unknown")

        elif action == "SELL":
            if not pos:
                return
            held_qty = float(pos.get("qty", 0.0))
            sold_qty = min(quantity, held_qty)
            avg_price = float(pos.get("avg_price", price) or price)
            self.realized_pnl += (price - avg_price) * sold_qty

            remain_qty = held_qty - sold_qty
            if remain_qty <= 1e-9:
                self.position_book.pop(symbol, None)
            else:
                pos["qty"] = remain_qty
                pos["last_price"] = price

        self._recompute_exposures()

        self.equity_current = self.portfolio_value + self.realized_pnl
        if self.equity_current > self.equity_peak:
            self.equity_peak = self.equity_current

    def evaluate_trade(self,
                       symbol: str,
                       action: str,
                       proposed_value: float,
                       current_price: float,
                       sector: str = "Unknown",
                       volatility: float = 0.02,
                       market_data: Any = None) -> Dict[str, Any]:
        """
        Đánh giá và phê duyệt/từ chối giao dịch

        Args:
            symbol: Mã cổ phiếu
            action: BUY hoặc SELL
            proposed_value: Giá trị giao dịch đề xuất
            current_price: Giá hiện tại
            sector: Ngành
            volatility: Độ biến động (ATR-based)
            market_data: Dữ liệu thị trường để tính ATR thực

        Returns:
            Dict với approved, warnings, adjusted values
        """
        result = {
            "approved": True,
            "veto_reason": None,
            "warnings": [],
            "adjusted_position_size": None,
            "stop_loss_level": None,
            "take_profit_level": None,
            "risk_reward_ratio": None,
            "current_drawdown": 0.0,
        }

        action = action.upper().strip()
        proposed_value = max(0.0, float(proposed_value))

        # 1. Kiểm tra drawdown
        if self.equity_peak > 0:
            current_drawdown = max(0.0, (self.equity_peak - self.equity_current) / self.equity_peak)
        else:
            current_drawdown = 0.0
        result["current_drawdown"] = round(current_drawdown, 4)

        if current_drawdown >= self.config.max_drawdown and action == "BUY":
            result["approved"] = False
            result["veto_reason"] = (
                f"VETO: Drawdown {current_drawdown:.1%} vượt ngưỡng {self.config.max_drawdown:.1%}"
            )
            self._send_veto(symbol, result["veto_reason"])
            return result

        # 2. Kiểm tra Market Regime (VETO nếu RISK_OFF và action=BUY)
        regime = self.blackboard.get_current_regime()
        if regime["current"] == MarketRegime.RISK_OFF.value and action == "BUY":
            result["approved"] = False
            result["veto_reason"] = "VETO: Thị trường RISK_OFF - Không cho phép mua mới"
            self._send_veto(symbol, result["veto_reason"])
            return result

        # 3. Kiểm tra vị thế hiện tại (đặc biệt cho SELL nếu không short)
        current_position_value = self.current_positions.get(symbol, 0.0)
        current_position = self.position_book.get(symbol, {})
        held_qty = float(current_position.get("qty", 0.0))

        if action == "SELL" and not self.config.allow_short:
            max_sell_value = max(0.0, held_qty * current_price)
            if max_sell_value <= 0:
                result["approved"] = False
                result["veto_reason"] = "VETO: Không có vị thế để SELL (long-only mode)"
                self._send_veto(symbol, result["veto_reason"])
                return result
            if proposed_value > max_sell_value:
                result["warnings"].append(f"Giảm SELL về mức vị thế hiện có: {max_sell_value:.0f}")
            proposed_value = min(proposed_value, max_sell_value)

        # 4. Kiểm tra giới hạn cho BUY
        if action == "BUY":
            max_position = self.portfolio_value * self.config.max_position_pct
            symbol_remaining = max_position - current_position_value
            if symbol_remaining <= 0:
                result["approved"] = False
                result["veto_reason"] = f"VETO: {symbol} đã đạt giới hạn vị thế {self.config.max_position_pct:.0%}"
                self._send_veto(symbol, result["veto_reason"])
                return result
            if proposed_value > symbol_remaining:
                result["warnings"].append(f"Giảm vị thế theo limit mã: {symbol_remaining:.0f}")
                proposed_value = symbol_remaining

            current_sector_value = self.sector_exposure.get(sector, 0.0)
            max_sector = self.portfolio_value * self.config.max_sector_exposure
            sector_remaining = max_sector - current_sector_value
            if sector_remaining <= 0:
                result["approved"] = False
                result["veto_reason"] = f"VETO: Đã đạt giới hạn ngành {sector} ({self.config.max_sector_exposure:.0%})"
                self._send_veto(symbol, result["veto_reason"])
                return result
            if proposed_value > sector_remaining:
                result["warnings"].append(f"Giảm vị thế theo limit ngành: {sector_remaining:.0f}")
                proposed_value = sector_remaining

        # 5. Tính ATR thực sự nếu có dữ liệu (base.txt Section 8.1)
        atr_value = None
        if MATH_UTILS_AVAILABLE and market_data:
            try:
                highs = market_data.highs if hasattr(market_data, 'highs') else market_data.get("highs", [])
                lows = market_data.lows if hasattr(market_data, 'lows') else market_data.get("lows", [])
                closes = market_data.prices if hasattr(market_data, 'prices') else market_data.get("prices", [])

                if highs and lows and closes:
                    atr_value = calculate_atr(highs, lows, closes)
                    result["atr"] = round(atr_value, 4)
                    logger.debug(f"ATR calculated for {symbol}: {atr_value:.4f}")
            except Exception as e:
                logger.warning(f"ATR calculation failed: {e}")

        # Fallback to volatility proxy if ATR not available
        if atr_value is None:
            atr_value = current_price * volatility

        # 6. Tính Stop-Loss động (2 x ATR)
        stop_loss_distance = atr_value * self.config.stop_loss_atr_multiplier
        if stop_loss_distance <= 0:
            stop_loss_distance = max(0.01, current_price * max(volatility, 0.005))

        # Position sizing theo rủi ro vốn (chỉ áp dụng BUY)
        if action == "BUY":
            risk_budget = self.portfolio_value * self.config.risk_per_trade_pct
            risk_based_qty = risk_budget / max(stop_loss_distance, 1e-9)
            risk_based_value = risk_based_qty * current_price
            if proposed_value > risk_based_value:
                result["warnings"].append(f"Giảm vị thế theo risk budget: {risk_based_value:.0f}")
                proposed_value = risk_based_value

        # 7. Tính Take-Profit (Risk:Reward = 1:2.5 mặc định)
        target_rr = 2.5
        take_profit_distance = stop_loss_distance * target_rr

        if action == "BUY":
            result["stop_loss_level"] = round(current_price - stop_loss_distance, 2)
            result["take_profit_level"] = round(current_price + take_profit_distance, 2)
        else:
            result["stop_loss_level"] = round(current_price + stop_loss_distance, 2)
            result["take_profit_level"] = round(max(0.01, current_price - take_profit_distance), 2)
        result["risk_reward_ratio"] = target_rr

        result["adjusted_position_size"] = round(max(0.0, proposed_value), 2)
        if result["adjusted_position_size"] <= 0:
            result["approved"] = False
            result["veto_reason"] = "VETO: Kích thước lệnh sau risk controls bằng 0"
            self._send_veto(symbol, result["veto_reason"])

        # LLM risk overlay: luôn gọi khi có LLM, deterministic là lớp cứng nền.
        if self.llm:
            result = self._llm_risk_overlay(
                symbol=symbol,
                action=action,
                current_price=current_price,
                sector=sector,
                deterministic_result=result,
            )

        return result

    def _llm_risk_overlay(self,
                          symbol: str,
                          action: str,
                          current_price: float,
                          sector: str,
                          deterministic_result: Dict[str, Any]) -> Dict[str, Any]:
        """LLM review cho Risk Guardian; không được làm yếu đi các hard veto."""
        try:
            regime = self.blackboard.get_current_regime()
            prompt = (
                f"Risk review cho {symbol}.\n"
                f"- action={action}\n"
                f"- price={current_price}\n"
                f"- sector={sector}\n"
                f"- regime={regime}\n"
                f"- deterministic_result={deterministic_result}\n"
                "Trả JSON theo schema risk_guardian."
            )
            llm_result = self.llm.analyze("risk_guardian", prompt, symbol=symbol)
            if not isinstance(llm_result, dict):
                return deterministic_result

            merged = dict(deterministic_result)
            merged["warnings"] = list(merged.get("warnings", []))

            # Nếu deterministic đã veto thì giữ veto cứng.
            if not deterministic_result.get("approved", True):
                merged["llm_risk_review"] = {
                    "approved": llm_result.get("approved"),
                    "reasoning": str(llm_result.get("reasoning", ""))[:220],
                }
                return merged

            # LLM có quyền siết chặt thêm (veto), không nới lỏng hard controls.
            if llm_result.get("approved") is False:
                merged["approved"] = False
                merged["veto_reason"] = str(
                    llm_result.get("veto_reason", "VETO: LLM Risk Review rejected trade")
                )[:220]
                self._send_veto(symbol, merged["veto_reason"])

            # Position size: chỉ cho giảm thêm, không tăng vượt deterministic.
            llm_size = llm_result.get("adjusted_position_size")
            if isinstance(llm_size, (int, float)):
                merged["adjusted_position_size"] = round(
                    max(0.0, min(float(merged.get("adjusted_position_size", 0.0)), float(llm_size))),
                    2,
                )

            # Optional stops update nếu hợp lệ
            for key in ["stop_loss_level", "take_profit_level"]:
                value = llm_result.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    merged[key] = round(float(value), 2)

            if "risk_reward_ratio" in llm_result and isinstance(llm_result.get("risk_reward_ratio"), (int, float)):
                merged["risk_reward_ratio"] = max(0.1, float(llm_result["risk_reward_ratio"]))

            llm_warnings = llm_result.get("warnings", [])
            if isinstance(llm_warnings, list):
                merged["warnings"].extend(str(w)[:180] for w in llm_warnings[:6])

            merged["llm_risk_review"] = {
                "approved": llm_result.get("approved"),
                "veto_reason": llm_result.get("veto_reason"),
                "reasoning": str(llm_result.get("reasoning", ""))[:220],
            }
            return merged
        except Exception as exc:
            logger.warning(f"[{self.name}] LLM risk overlay failed for {symbol}: {exc}")
            return deterministic_result

    def _send_veto(self, symbol: str, reason: str):
        """Gửi thông báo VETO lên Blackboard"""
        msg = AgentMessage(
            sender=self.name,
            receiver="ALL",
            msg_type=MessageType.VETO,
            content={"symbol": symbol, "reason": reason},
            priority="CRITICAL"
        )
        self.blackboard.post_message(msg)
        logger.warning(f"[{self.name}] {reason}")


# --- BAYESIAN RESOLVER (BỘ GIẢI QUYẾT BAYESIAN) ---

class BayesianResolver:
    """
    BỘ GIẢI QUYẾT BAYESIAN

    Tổng hợp tín hiệu xung đột từ các tác tử thành xác suất thống nhất

    Công thức (base.txt Section 6.1):
    - LR+ = Sensitivity / (1 - Specificity)
    - LR- = (1 - Sensitivity) / Specificity
    - Posterior Odds = Prior Odds × LR1 × LR2 × ... × LRn
    - Probability = Odds / (1 + Odds)
    """

    def __init__(self, blackboard: Blackboard):
        self.name = "Bayesian_Resolver"
        self.blackboard = blackboard
        self.config = get_config().bayesian

    def resolve(self, symbol: str, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Tổng hợp tín hiệu từ các tác tử

        Args:
            symbol: Mã cổ phiếu
            signals: Dict của {agent_name: analysis_result}

        Returns:
            Dict với final_probability và decision
        """
        # Prior Odds (mặc định 1:1 = 50%)
        prior_prob = self.config.prior_probability
        prior_odds = prior_prob / (1 - prior_prob)

        current_odds = prior_odds
        likelihood_ratios = {}
        regime = self.blackboard.get_current_regime().get("current", MarketRegime.UNKNOWN.value)
        regime_multipliers = self.config.regime_multipliers.get(regime, {})
        performance_map = self.blackboard.read_memory("agent_state", "performance") or {}

        # Ánh xạ tên tác tử sang profile
        agent_mapping = {
            "wyckoff": "wyckoff",
            "canslim": "canslim",
            "fourm": "fourm",
            "4m": "fourm",
            "news": "news"
        }

        for agent_key, analysis in signals.items():
            # Tìm profile phù hợp
            profile_key = None
            for key, mapped in agent_mapping.items():
                if key in agent_key.lower():
                    profile_key = mapped
                    break

            if not profile_key:
                continue

            profile = self.config.agent_profiles.get(profile_key)
            if not profile:
                continue

            sensitivity = profile["sensitivity"]
            specificity = profile["specificity"]

            # Regime-aware multiplier
            regime_mult = regime_multipliers.get(profile_key, 1.0)
            perf_score = float(performance_map.get(profile_key, 0.5))
            perf_mult = 0.75 + (max(0.0, min(1.0, perf_score)) * 0.5)
            combined_mult = regime_mult * perf_mult
            sensitivity = max(0.01, min(0.99, sensitivity * combined_mult))
            specificity = max(0.01, min(0.99, specificity * combined_mult))

            # Lấy tín hiệu
            signal = analysis.get("signal", "HOLD")
            confidence = analysis.get("confidence", 0.5)
            confidence = max(0.0, min(1.0, float(confidence)))

            if profile_key == "news":
                quality = analysis.get("evidence_quality", {}) if isinstance(analysis, dict) else {}
                if quality:
                    avg_evidence_conf = float(quality.get("avg_evidence_confidence", 0.0))
                    snippet_ratio = float(quality.get("snippet_only_ratio", 1.0))
                    evidence_mult = max(0.6, min(1.2, 0.75 + 0.5 * avg_evidence_conf - 0.2 * snippet_ratio))
                    confidence = max(0.0, min(1.0, float(confidence) * evidence_mult))

            # Base likelihood ratios từ profile đã điều chỉnh regime/performance.
            # Dùng confidence để co về trung tính (LR=1), nhưng KHÔNG được đảo chiều bằng chứng.
            base_lr_buy = sensitivity / max(0.01, 1 - specificity)           # kỳ vọng >1
            base_lr_sell = (1 - sensitivity) / max(0.01, specificity)         # kỳ vọng <1

            # Tính Likelihood Ratio
            if signal in ["STRONG_BUY", "BUY"]:
                strength = min(1.0, confidence * (1.15 if signal == "STRONG_BUY" else 1.0))
                lr = 1.0 + strength * (base_lr_buy - 1.0)
                # BUY evidence không được phép kéo odds theo hướng SELL
                lr = max(1.0, lr)
            elif signal in ["STRONG_SELL", "SELL", "NO_DEAL"]:
                strength = min(1.0, confidence * (1.15 if signal == "STRONG_SELL" else 1.0))
                lr = 1.0 + strength * (base_lr_sell - 1.0)
                # SELL evidence không được phép kéo odds theo hướng BUY
                lr = min(1.0, lr)
            else:
                # HOLD -> LR ≈ 1 (neutral)
                lr = 1.0

            likelihood_ratios[agent_key] = round(lr, 4)

            # Cập nhật Odds
            current_odds *= lr

        # Chuyển Odds về Probability
        final_probability = current_odds / (1 + current_odds)

        # Quyết định theo vùng đệm trung tính:
        # - [min_threshold, 1-min_threshold] => WATCH
        # - ngoài vùng đệm => EXECUTE_TRADE (BUY hoặc SELL tùy phía 0.5)
        neutral_low = self.config.min_probability_threshold
        neutral_high = 1 - neutral_low

        if neutral_low <= final_probability <= neutral_high:
            decision = "WATCH"
        else:
            decision = "EXECUTE_TRADE"

        result = {
            "symbol": symbol,
            "prior_odds": round(prior_odds, 4),
            "likelihood_ratios": likelihood_ratios,
            "posterior_odds": round(current_odds, 4),
            "final_probability": round(final_probability, 4),
            "decision": decision,
            "regime": regime,
            "performance_map": performance_map,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"[{self.name}] {symbol}: Prob={final_probability:.1%}, Decision={decision}")

        return result


# --- PORTFOLIO MANAGER (QUẢN LÝ DANH MỤC) ---

class PortfolioManager:
    """
    QUẢN LÝ DANH MỤC (Cấp độ 2)

    Vai trò: Người phân bổ vốn
    - Nhận tín hiệu từ các Binh lính
    - Tổng hợp qua Bayesian Resolver
    - Kiểm tra với Risk Guardian
    - Tạo lệnh giao dịch cuối cùng

    Tham chiếu: base.txt Section 3.1.2
    """

    def __init__(self, blackboard: Blackboard,
                 llm_client: Optional[ClaudeClient] = None,
                 portfolio_value: float = 100000,
                 factcheck_pipeline: Any = None):
        self.name = "Portfolio_Manager"
        self.blackboard = blackboard
        self.llm = llm_client
        self.config = get_config()
        self.portfolio_value = portfolio_value
        self.factcheck = factcheck_pipeline

        # Khởi tạo các component
        self.bayesian = BayesianResolver(blackboard)
        self.risk_guardian = RiskGuardian(blackboard, portfolio_value, llm_client=llm_client)

    def process_signals(self, symbol: str) -> Optional[TradeOrder]:
        """
        Xử lý tín hiệu và tạo lệnh giao dịch

        Args:
            symbol: Mã cổ phiếu

        Returns:
            TradeOrder nếu cần thực hiện giao dịch, None nếu không
        """
        # 1. Lấy tất cả phân tích từ Blackboard
        all_analysis = self.blackboard.get_all_analysis(symbol)

        if not all_analysis:
            logger.warning(f"[{self.name}] Không có phân tích cho {symbol}")
            return None

        # 1.5 VERAFI Validation (base.txt Section 4.2)
        if VERAFI_AVAILABLE:
            verafi = get_verafi()
            data_entry = self.blackboard.read_memory("market_data", symbol)
            actual_data = {}

            if data_entry:
                market_data = data_entry.get("market_data") if isinstance(data_entry, dict) else data_entry
                if market_data:
                    actual_data = {
                        "eps_growth": getattr(market_data, 'eps_growth', None) or (market_data.get("eps_growth") if isinstance(market_data, dict) else None),
                        "annual_eps_growth": getattr(market_data, 'annual_eps_growth', None) or (market_data.get("annual_eps_growth") if isinstance(market_data, dict) else None),
                        "roe": getattr(market_data, 'roe', None) or (market_data.get("roe") if isinstance(market_data, dict) else None),
                        "roic": getattr(market_data, 'roic', None) or (market_data.get("roic") if isinstance(market_data, dict) else None),
                        "rs_rating": getattr(market_data, 'rs_rating', None) or (market_data.get("rs_rating") if isinstance(market_data, dict) else None),
                    }

            # Validate each agent's output
            total_penalty = 0.0
            for agent_type, analysis in all_analysis.items():
                validation = verafi.validate(agent_type, analysis, actual_data)
                if not validation.is_valid:
                    logger.warning(f"[{self.name}] VERAFI violations for {symbol}/{agent_type}: {validation.violations}")
                    # Apply adjustments
                    for key, value in validation.adjustments.items():
                        if key in analysis:
                            if isinstance(value, dict):
                                analysis[key].update(value)
                            else:
                                analysis[key] = value
                if validation.confidence_penalty > 0:
                    current_conf = float(analysis.get("confidence", 0.5))
                    analysis["confidence"] = round(
                        max(0.05, current_conf * (1 - validation.confidence_penalty)),
                        4
                    )
                total_penalty += validation.confidence_penalty

            # Adjust overall confidence if there were violations
            if total_penalty > 0:
                logger.info(f"[{self.name}] VERAFI applied confidence penalty: -{total_penalty:.1%}")

        # 1.6 Fact-check pipeline cho News Agent (base.txt Section 4.3)
        if self.config.system.enable_fact_check and self.factcheck and "news" in all_analysis:
            try:
                all_analysis["news"] = self.factcheck.verify_news_claims(all_analysis["news"], symbol)
            except Exception as fact_exc:
                logger.warning(f"[{self.name}] Fact-check failed for {symbol}: {fact_exc}")

        # 1.7 Điều chỉnh độ tin cậy News theo chất lượng evidence của pipeline 2 tầng
        self._apply_news_evidence_weight(symbol, all_analysis)

        # 2. Tổng hợp Bayesian
        bayesian_result = self.bayesian.resolve(symbol, all_analysis)
        self._apply_directional_guardrail(symbol, all_analysis, bayesian_result)

        # 3. Lưu kết quả đồng thuận
        self.blackboard.write_memory("consensus", symbol, bayesian_result)

        # Chỉ EXECUTE_TRADE mới tạo lệnh, WATCH/NO_ACTION thì bỏ qua
        if bayesian_result["decision"] in {"NO_ACTION", "WATCH"}:
            logger.info(
                f"[{self.name}] {symbol}: {bayesian_result['decision']} "
                f"(Prob={bayesian_result['final_probability']:.1%})"
            )
            return None

        # 4. Xác định action và parameters
        action = "BUY" if bayesian_result["final_probability"] >= 0.5 else "SELL"

        # Lấy giá hiện tại
        data_entry = self.blackboard.read_memory("market_data", symbol)
        if not data_entry:
            logger.warning(f"[{self.name}] {symbol}: Không có dữ liệu thị trường")
            return None

        # Trích xuất market_data từ data_entry
        market_data = data_entry.get("market_data") if isinstance(data_entry, dict) else data_entry
        if not market_data:
            logger.warning(f"[{self.name}] {symbol}: Dữ liệu thị trường trống")
            return None

        prices = market_data.prices if hasattr(market_data, 'prices') else market_data.get("prices", [])
        if not prices:
            logger.warning(f"[{self.name}] {symbol}: Không có dữ liệu giá")
            return None

        current_price = prices[-1]
        sector = market_data.industry if hasattr(market_data, 'industry') else (
            market_data.get("industry", "Unknown") if isinstance(market_data, dict) else "Unknown"
        )

        # Tính position size đề xuất
        if action == "BUY":
            base_position = self.portfolio_value * 0.03  # 3% cơ sở
            adjusted_position = base_position * bayesian_result["final_probability"]
        else:
            # Ưu tiên đóng vị thế hiện có theo mức conviction
            held_value = self.risk_guardian.current_positions.get(symbol, 0.0)
            sell_ratio = max(0.25, min(1.0, 1 - bayesian_result["final_probability"]))
            adjusted_position = held_value * sell_ratio if held_value > 0 else self.portfolio_value * 0.01

        # 5. Kiểm tra Risk Guardian (với market_data để tính ATR thực)
        risk_check = self.risk_guardian.evaluate_trade(
            symbol=symbol,
            action=action,
            proposed_value=adjusted_position,
            current_price=current_price,
            sector=sector,
            volatility=0.02,  # Fallback
            market_data=market_data  # Pass market data for ATR calculation
        )
        self.blackboard.write_memory("risk_checks", symbol, risk_check)

        if not risk_check["approved"]:
            logger.warning(f"[{self.name}] {symbol}: Trade REJECTED - {risk_check['veto_reason']}")
            return None

        # 5.5 Tướng quân Opus review và chốt cuối cùng trước khi tạo lệnh
        opus_review = self._opus_final_review(
            symbol=symbol,
            all_analysis=all_analysis,
            bayesian_result=bayesian_result,
            risk_check=risk_check,
            action=action,
            current_price=current_price,
            sector=sector,
        )
        if opus_review:
            self.blackboard.write_memory("consensus_review", symbol, opus_review)
            bayesian_result["opus_review"] = opus_review

            if not opus_review.get("approved", True):
                bayesian_result["decision"] = "WATCH"
                bayesian_result["guardrail_triggered"] = True
                bayesian_result["guardrail_reason"] = (
                    f"Opus final review rejected: {opus_review.get('reasoning', '')[:200]}"
                )
                logger.warning(f"[{self.name}] {symbol}: Opus final review rejected trade.")
                return None

        # 6. Tạo TradeOrder
        final_value = risk_check["adjusted_position_size"] or adjusted_position
        if opus_review:
            try:
                size_mult = float(opus_review.get("position_multiplier", 1.0))
            except Exception:
                size_mult = 1.0
            # Cho phép Opus chỉ giảm quy mô để an toàn
            size_mult = max(0.0, min(1.0, size_mult))
            final_value = final_value * size_mult

        if final_value <= 0:
            logger.warning(f"[{self.name}] {symbol}: Opus/risk adjusted position is zero -> skip order")
            return None

        quantity = final_value / current_price

        final_confidence = bayesian_result["final_probability"]
        if opus_review and "final_confidence" in opus_review:
            try:
                final_confidence = max(0.01, min(0.99, float(opus_review["final_confidence"])))
            except Exception:
                pass

        order = TradeOrder(
            symbol=symbol,
            action=action,
            quantity=round(quantity, 2),
            order_type="MARKET",
            stop_loss=risk_check["stop_loss_level"],
            take_profit=risk_check["take_profit_level"],
            reasoning=self._build_reasoning(all_analysis, bayesian_result),
            confidence=final_confidence
        )

        # 7. Ghi lên Blackboard
        msg = AgentMessage(
            sender=self.name,
            receiver="Blackboard",
            msg_type=MessageType.FINAL_DECISION,
            content=order.to_dict(),
            priority="HIGH"
        )
        self.blackboard.post_message(msg)

        # Cập nhật trạng thái danh mục ngay sau khi tạo lệnh
        self.risk_guardian.register_execution(
            symbol=symbol,
            action=action,
            quantity=round(quantity, 2),
            price=current_price,
            sector=sector,
        )
        self._update_agent_performance(all_analysis, action, bayesian_result["final_probability"])

        logger.info(f"[{self.name}] ORDER: {action} {quantity:.2f} {symbol} @ {current_price:.2f}")

        return order

    def _opus_final_review(self,
                           symbol: str,
                           all_analysis: Dict[str, Dict[str, Any]],
                           bayesian_result: Dict[str, Any],
                           risk_check: Dict[str, Any],
                           action: str,
                           current_price: float,
                           sector: str) -> Optional[Dict[str, Any]]:
        """
        Tướng quân Opus chốt cuối.
        Rule-based vẫn là guardrail cứng; Opus dùng để xác nhận/chặn hoặc giảm vị thế.
        """
        if not self.llm:
            return None

        try:
            prompt = (
                f"Final CIO review cho lệnh {symbol}.\n"
                f"- proposed_action: {action}\n"
                f"- current_price: {current_price}\n"
                f"- sector: {sector}\n"
                f"- regime: {self.blackboard.get_current_regime()}\n"
                f"- agent_analysis: {all_analysis}\n"
                f"- bayesian: {bayesian_result}\n"
                f"- risk_check: {risk_check}\n\n"
                "Trả JSON theo schema cio_final_reviewer để chốt cuối."
            )
            result = self.llm.analyze("cio_final_reviewer", prompt, symbol=symbol)
            if not isinstance(result, dict):
                return None

            approved = bool(result.get("approved", True))
            position_multiplier = result.get("position_multiplier", 1.0)
            final_confidence = result.get("final_confidence", bayesian_result.get("final_probability", 0.5))
            try:
                position_multiplier = max(0.0, min(1.0, float(position_multiplier)))
            except Exception:
                position_multiplier = 1.0
            try:
                final_confidence = max(0.01, min(0.99, float(final_confidence)))
            except Exception:
                final_confidence = bayesian_result.get("final_probability", 0.5)

            return {
                "approved": approved,
                "position_multiplier": round(position_multiplier, 4),
                "final_confidence": round(final_confidence, 4),
                "reasoning": str(result.get("reasoning", ""))[:500],
                "sources": result.get("sources", []),
            }
        except Exception as exc:
            logger.warning(f"[{self.name}] Opus final review failed for {symbol}: {exc}")
            return None

    def _apply_directional_guardrail(self,
                                     symbol: str,
                                     all_analysis: Dict[str, Dict[str, Any]],
                                     bayesian_result: Dict[str, Any]):
        """
        Guardrail chống false-direction:
        - Nếu không có tác tử nào BUY thì không được phép BUY.
        - (đối xứng) Nếu không có tác tử nào SELL/NO_DEAL thì không được phép SELL.
        """
        bullish = {"STRONG_BUY", "BUY"}
        bearish = {"STRONG_SELL", "SELL", "NO_DEAL"}

        signals = [str(v.get("signal", "HOLD")).upper() for v in all_analysis.values() if isinstance(v, dict)]
        has_bullish = any(sig in bullish for sig in signals)
        has_bearish = any(sig in bearish for sig in signals)

        prob = float(bayesian_result.get("final_probability", 0.5))
        intended_action = "BUY" if prob >= 0.5 else "SELL"
        guardrail_reason = ""

        if intended_action == "BUY" and not has_bullish:
            guardrail_reason = "No bullish agent signal (BUY/STRONG_BUY). Block BUY despite Bayesian score."
        elif intended_action == "SELL" and not has_bearish:
            guardrail_reason = "No bearish agent signal (SELL/STRONG_SELL/NO_DEAL). Block SELL despite Bayesian score."

        if guardrail_reason:
            bayesian_result["decision"] = "WATCH"
            bayesian_result["guardrail_triggered"] = True
            bayesian_result["guardrail_reason"] = guardrail_reason
            logger.warning(f"[{self.name}] {symbol}: Directional guardrail triggered - {guardrail_reason}")

    def _apply_news_evidence_weight(self, symbol: str, all_analysis: Dict[str, Dict[str, Any]]):
        """Điều chỉnh confidence của News Agent theo chất lượng dữ liệu DeepDive."""
        news_analysis = all_analysis.get("news")
        if not news_analysis:
            return

        data_entry = self.blackboard.read_memory("market_data", symbol) or {}
        if not isinstance(data_entry, dict):
            return
        document_intel = data_entry.get("document_intel") or {}
        quality = document_intel.get("quality") if isinstance(document_intel, dict) else {}
        if not quality:
            return

        avg_evidence_conf = float(quality.get("avg_evidence_confidence", 0.0))
        snippet_ratio = float(quality.get("snippet_only_ratio", 1.0))
        high_priority_docs = float(quality.get("high_priority_docs", 0.0))
        deep_count = float(quality.get("deep_dive_count", 0.0))
        radar_count = max(1.0, float(quality.get("radar_count", 0.0)))
        deep_coverage = deep_count / radar_count

        multiplier = (
            0.70
            + (0.35 * avg_evidence_conf)
            + (0.20 * deep_coverage)
            + (0.08 * min(1.0, high_priority_docs / 3.0))
            - (0.20 * snippet_ratio)
        )
        multiplier = max(0.35, min(1.15, multiplier))

        old_conf = float(news_analysis.get("confidence", 0.5))
        new_conf = max(0.05, min(0.99, old_conf * multiplier))
        news_analysis["confidence"] = round(new_conf, 4)
        news_analysis["evidence_quality"] = quality
        news_analysis["reasoning"] = (
            f"{news_analysis.get('reasoning', '')} | "
            f"EvidenceAdj x{multiplier:.2f} (deep={int(deep_count)}/{int(radar_count)}, "
            f"avg_conf={avg_evidence_conf:.2f}, snippet_ratio={snippet_ratio:.2f})"
        ).strip()

    def _build_reasoning(self, analysis: Dict, bayesian: Dict) -> str:
        """Xây dựng giải thích cho quyết định"""
        parts = []

        # Tóm tắt từng tác tử
        for agent, result in analysis.items():
            signal = result.get("signal", "N/A")
            conf = result.get("confidence", 0)
            parts.append(f"{agent}: {signal} ({conf:.0%})")

        # Kết quả Bayesian
        parts.append(f"Bayesian: {bayesian['final_probability']:.1%}")
        if bayesian.get("regime"):
            parts.append(f"Regime: {bayesian['regime']}")
        if bayesian.get("guardrail_triggered"):
            parts.append(f"Guardrail: {bayesian.get('guardrail_reason', 'Triggered')}")
        if bayesian.get("opus_review"):
            review = bayesian.get("opus_review", {})
            parts.append(
                f"OpusFinal: {'APPROVE' if review.get('approved', True) else 'REJECT'}"
                f" x{review.get('position_multiplier', 1.0)}"
            )

        return " | ".join(parts)

    def _update_agent_performance(self, analysis: Dict[str, Dict], action: str, conviction: float):
        """
        Cập nhật điểm tin cậy tác tử theo mức đồng thuận với quyết định cuối.
        Đây là cơ chế online-learning nhẹ để thay đổi trọng số theo thời gian.
        """
        mapping = {
            "wyckoff": "wyckoff",
            "canslim": "canslim",
            "fourm": "fourm",
            "news": "news",
        }
        performance_map = self.blackboard.read_memory("agent_state", "performance") or {}
        step = 0.015 + min(0.03, abs(conviction - 0.5) * 0.06)

        for agent_key, result in analysis.items():
            profile = None
            for token, profile_key in mapping.items():
                if token in agent_key.lower():
                    profile = profile_key
                    break
            if not profile:
                continue

            signal = str(result.get("signal", "HOLD")).upper()
            aligned = (
                (action == "BUY" and signal in {"STRONG_BUY", "BUY"})
                or (action == "SELL" and signal in {"STRONG_SELL", "SELL", "NO_DEAL"})
            )
            old_score = float(performance_map.get(profile, 0.5))
            if aligned:
                new_score = min(0.95, old_score + step)
            else:
                new_score = max(0.05, old_score - step)
            performance_map[profile] = round(new_score, 4)

        self.blackboard.write_memory("agent_state", "performance", performance_map)


# --- MAIN ORCHESTRATOR ---

class SentinelOrchestrator:
    """
    BỘ ĐIỀU PHỐI CHÍNH (TƯỚNG QUÂN)

    Điều phối toàn bộ quy trình từ thu thập dữ liệu đến ra quyết định

    Quy trình:
    1. Macro General xác định Market Regime
    2. Data Scout thu thập dữ liệu
    3. Các Binh lính phân tích (Wyckoff, CANSLIM, 4M, News)
    4. Portfolio Manager tổng hợp và ra quyết định
    5. Risk Guardian phê duyệt/từ chối

    Tham chiếu: base.txt Section 3
    """

    def __init__(self,
                 tavily_api_key: str = "",
                 zai_api_key: str = "",
                 llm_api_key: str = "",
                 portfolio_value: float = 100000):

        self.name = "Sentinel_Orchestrator"
        self.config = get_config()

        # Khởi tạo Blackboard (Bảng đen toàn cục)
        self.blackboard = Blackboard()

        # Khởi tạo LLM Client (OpenAI-compatible endpoint)
        api_key = llm_api_key or zai_api_key or LLM_API_KEY
        if not api_key:
            raise RuntimeError("CRITICAL: Không có LLM client được cấu hình. Cần LLM_API_KEY")
        self.llm = ClaudeClient(api_key=api_key, provider=self.config.llm.provider)

        # Khởi tạo Data Providers - LUÔN CHẠY THẬT
        tavily = TavilyClient(tavily_api_key) if tavily_api_key else TavilyClient()
        market_provider = MarketDataProvider()
        self.data_scout = DataScoutAgent(tavily, market_provider)
        self.factcheck = get_factcheck(self.data_scout.tavily) if VERAFI_AVAILABLE else None

        # Khởi tạo Tướng quân Vĩ mô
        self.macro_general = MacroGeneral(self.blackboard, self.llm)

        # Khởi tạo các Binh lính (Specialist Agents)
        self.wyckoff_agent = WyckoffAgent(self.blackboard, self.llm)
        self.canslim_agent = CANSLIMAgent(self.blackboard, self.llm)
        self.fourm_agent = FourMAgent(self.blackboard, self.llm)
        self.news_agent = NewsSentimentAgent(self.blackboard, self.llm)

        # Khởi tạo Quản lý Danh mục
        self.portfolio_manager = PortfolioManager(
            self.blackboard, self.llm, portfolio_value, factcheck_pipeline=self.factcheck
        )

        self.last_run_log_path: Optional[str] = None

        logger.info(f"[{self.name}] Khởi tạo hoàn tất")

    def run(self, watchlist: List[str] = None) -> List[TradeOrder]:
        """
        Chạy toàn bộ quy trình phân tích và ra quyết định

        Args:
            watchlist: Danh sách mã cổ phiếu cần phân tích

        Returns:
            Danh sách TradeOrder (lệnh giao dịch)
        """
        if watchlist is None:
            watchlist = self.config.default_watchlist

        run_logger = get_run_logger() if RUN_LOGGER_AVAILABLE else None
        if run_logger:
            run_logger.start_run(
                watchlist=watchlist,
                metadata={
                    "orchestrator": self.name,
                    "llm_provider": self.config.llm.provider,
                    "llm_models": {
                        "general": self.config.llm.model_general,
                        "analyst": self.config.llm.model_analyst,
                        "worker": self.config.llm.model_worker,
                    },
                },
            )

        run_status = "COMPLETED"
        run_error = ""
        orders: List[TradeOrder] = []

        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"SENTINEL AI TRADING - PHIÊN PHÂN TÍCH")
            logger.info(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Watchlist: {watchlist}")
            logger.info(f"{'='*60}\n")

            if run_logger:
                run_logger.log_event("STEP", {"name": "BƯỚC 1", "detail": "Xác định Market Regime"})

            # BƯỚC 1: Xác định Market Regime
            logger.info("[BƯỚC 1] Tướng quân Vĩ mô đang xác định chế độ thị trường...")
            regime_data = self.data_scout.fetch_market_regime_data()
            regime = self.macro_general.determine_regime(
                regime_data.get("index_data", {}),
                regime_data.get("macro_news"),
            )
            regime_value = regime.get("regime", MarketRegime.UNKNOWN.value)

            include_news = True
            skip_fundamental_agents = False
            if self.config.system.enable_regime_gating and regime_value == MarketRegime.RISK_OFF.value:
                include_news = not self.config.system.risk_off_skip_news
                skip_fundamental_agents = self.config.system.risk_off_skip_fundamental_agents

            logger.info(
                "[BƯỚC 1] Regime gating: regime=%s | include_news=%s | skip_fundamental_agents=%s",
                regime_value,
                include_news,
                skip_fundamental_agents,
            )

            if run_logger:
                run_logger.log_event(
                    "REGIME_RESULT",
                    {
                        "regime": regime.get("regime"),
                        "confidence": regime.get("confidence"),
                        "include_news": include_news,
                        "skip_fundamental_agents": skip_fundamental_agents,
                    },
                )
                run_logger.log_event("STEP", {"name": "BƯỚC 2", "detail": "Data Scout thu thập dữ liệu"})

            # BƯỚC 2: Thu thập dữ liệu cho từng mã
            logger.info("\n[BƯỚC 2] Trinh sát đang thu thập dữ liệu...")
            active_symbols: List[str] = []
            for symbol in watchlist:
                data = self.data_scout.fetch_all_data(symbol, include_news=include_news)
                doc_quality = (data.get("document_intel") or {}).get("quality", {})
                if doc_quality:
                    logger.info(
                        "[BƯỚC 2] %s doc_intel: radar=%s deep=%s snippet_ratio=%.2f avg_conf=%.2f",
                        symbol,
                        doc_quality.get("radar_count", 0),
                        doc_quality.get("deep_dive_count", 0),
                        float(doc_quality.get("snippet_only_ratio", 1.0)),
                        float(doc_quality.get("avg_evidence_confidence", 0.0)),
                    )
                    if run_logger:
                        run_logger.log_event(
                            "DOC_INTEL",
                            {
                                "symbol": symbol,
                                "quality": doc_quality,
                            },
                        )

                # Lưu lên Blackboard
                if data.get("market_data"):
                    msg = AgentMessage(
                        sender="Data_Scout",
                        receiver="Blackboard",
                        msg_type=MessageType.DATA_REPORT,
                        content=data
                    )
                    self.blackboard.post_message(msg)

                if not self.config.system.enable_event_driven_filter:
                    active_symbols.append(symbol)
                else:
                    event_flags = data.get("event_flags") or {}
                    if event_flags.get("triggered", False):
                        active_symbols.append(symbol)

            if not active_symbols:
                logger.info("[BƯỚC 2] Không có symbol nào trigger event; fallback phân tích toàn watchlist.")
                active_symbols = list(watchlist)
            else:
                logger.info(
                    "[BƯỚC 2] Event-driven filter giữ lại %d/%d symbol: %s",
                    len(active_symbols),
                    len(watchlist),
                    active_symbols,
                )

            if run_logger:
                run_logger.log_event("STEP", {"name": "BƯỚC 3", "detail": "Specialist Agents phân tích"})
                run_logger.log_event(
                    "EVENT_FILTER",
                    {
                        "enabled": self.config.system.enable_event_driven_filter,
                        "active_symbols": active_symbols,
                        "watchlist_size": len(watchlist),
                    },
                )

            # BƯỚC 3: Các Binh lính phân tích
            logger.info("\n[BƯỚC 3] Các Binh lính đang phân tích...")

            # Wyckoff
            logger.info("  - Wyckoff Agent đang phân tích kỹ thuật...")
            self.wyckoff_agent.run(active_symbols)

            if not skip_fundamental_agents:
                # CANSLIM
                logger.info("  - CANSLIM Agent đang đánh giá tăng trưởng...")
                self.canslim_agent.run(active_symbols)

                # 4M
                logger.info("  - 4M Agent đang tính giá trị nội tại...")
                self.fourm_agent.run(active_symbols)
            else:
                logger.info("  - Bỏ qua CANSLIM/4M do regime gating (RISK_OFF)")

            if include_news:
                logger.info("  - News Agent đang phân tích tin tức...")
                for symbol in active_symbols:
                    data = self.blackboard.read_memory("market_data", symbol)
                    if data:
                        result = self.news_agent.analyze(symbol, data)
                        self.news_agent.send_analysis(symbol, result)
            else:
                logger.info("  - Bỏ qua News Agent do regime gating")

            if run_logger:
                run_logger.log_event("STEP", {"name": "BƯỚC 4", "detail": "Portfolio Manager ra quyết định"})

            # BƯỚC 4: Portfolio Manager tổng hợp và ra quyết định
            logger.info("\n[BƯỚC 4] Quản lý Danh mục đang tổng hợp và ra quyết định...")
            for symbol in active_symbols:
                order = self.portfolio_manager.process_signals(symbol)
                if order:
                    orders.append(order)

            # BƯỚC 5: Tổng kết
            logger.info(f"\n{'='*60}")
            logger.info("KẾT QUẢ PHÂN TÍCH")
            logger.info(f"{'='*60}")

            return orders
        except Exception as exc:
            run_status = "FAILED"
            run_error = str(exc)
            raise
        finally:
            if run_logger:
                try:
                    self.last_run_log_path = run_logger.finalize(
                        status=run_status,
                        error=run_error,
                        summary={
                            "watchlist": watchlist,
                            "total_orders": len(orders),
                            "regime": self.blackboard.get_current_regime(),
                            "blackboard_summary": self.blackboard.get_summary(),
                            "orders": [o.to_dict() for o in orders],
                        },
                    )
                    if self.last_run_log_path:
                        logger.info(f"[{self.name}] Run log exported: {self.last_run_log_path}")
                except Exception as log_exc:
                    logger.warning(f"[{self.name}] Không thể export run log: {log_exc}")

    def get_summary_report(self) -> Dict[str, Any]:
        """Lấy báo cáo tổng kết"""
        return {
            "blackboard_summary": self.blackboard.get_summary(),
            "current_regime": self.blackboard.get_current_regime(),
            "trade_orders": [o.to_dict() for o in self.blackboard.get_trade_orders()],
            "timestamp": datetime.now().isoformat()
        }

```

## `./requirements.txt`

```text
# Sentinel AI Trading System - Dependencies
# ==========================================

# Core
python-dotenv>=1.0.0
openai>=1.40.0

# AI/LLM - OpenAI SDK (OpenAI-compatible endpoint)

# Data & Search
tavily-python>=0.5.0
yfinance>=0.2.40

# Optional - for advanced features
# pandas>=2.0.0
# numpy>=1.24.0

```

## `./run_logger.py`

```python
"""
RUN LOGGER
==========
Ghi lại toàn bộ phiên chạy vào file Markdown để review chất lượng.
"""

import json
import os
import threading
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RunMarkdownLogger:
    """Collect and export one run report as Markdown."""

    def __init__(self):
        self._lock = threading.RLock()
        self.output_dir = os.environ.get(
            "SENTINEL_RUN_LOG_DIR",
            os.path.join(os.getcwd(), "logs", "run_reports"),
        )
        self._reset_state()

    def _reset_state(self):
        self.active = False
        self.run_id = ""
        self.started_at = ""
        self.ended_at = ""
        self.watchlist: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.agent_actions: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self.status = "UNKNOWN"
        self.error = ""

    def start_run(self, watchlist: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        with self._lock:
            self._reset_state()
            self.active = True
            now = datetime.now()
            self.run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.started_at = now.isoformat()
            self.watchlist = list(watchlist or [])
            self.metadata = self._to_json_safe(metadata or {})
            self.log_event("RUN_START", {"watchlist": self.watchlist, "metadata": self.metadata})

    def log_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        with self._lock:
            if not self.active:
                return
            self.events.append({
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "payload": self._to_json_safe(payload or {}),
            })

    def log_llm_call(self,
                     role: str,
                     model: str,
                     request_payload: Dict[str, Any],
                     response_payload: Any,
                     symbol: str = "",
                     status: str = "SUCCESS",
                     latency_ms: Optional[float] = None,
                     error: str = "",
                     attempt: int = 1):
        with self._lock:
            if not self.active:
                return
            self.llm_calls.append({
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "model": model,
                "symbol": symbol,
                "status": status,
                "latency_ms": latency_ms,
                "error": error,
                "attempt": attempt,
                "request": self._to_json_safe(request_payload),
                "response": self._to_json_safe(response_payload),
            })

    def log_blackboard_message(self, message_dict: Dict[str, Any]):
        with self._lock:
            if not self.active:
                return
            self.agent_actions.append(self._to_json_safe(message_dict))

    def finalize(self,
                 status: str = "COMPLETED",
                 summary: Optional[Dict[str, Any]] = None,
                 error: str = "") -> Optional[str]:
        with self._lock:
            if not self.active:
                return None

            self.ended_at = datetime.now().isoformat()
            self.status = status
            self.error = error
            self.summary = self._to_json_safe(summary or {})
            self.log_event("RUN_END", {"status": status, "error": error, "summary": self.summary})

            os.makedirs(self.output_dir, exist_ok=True)
            file_path = os.path.join(self.output_dir, f"{self.run_id}.md")

            content = self._build_markdown()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.active = False
            return file_path

    def _build_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# Sentinel Run Report - {self.run_id}")
        lines.append("")
        lines.append("## Run Metadata")
        lines.append(f"- Status: {self.status}")
        lines.append(f"- Started At: {self.started_at}")
        lines.append(f"- Ended At: {self.ended_at}")
        lines.append(f"- Watchlist: {', '.join(self.watchlist) if self.watchlist else 'N/A'}")
        if self.error:
            lines.append(f"- Error: {self.error}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self.metadata, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        lines.append("## Run Summary")
        lines.append("```json")
        lines.append(json.dumps(self.summary, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        lines.append("## Timeline Events")
        if not self.events:
            lines.append("_No events captured._")
        for idx, evt in enumerate(self.events, start=1):
            lines.append(f"### Event {idx}: {evt.get('event_type', 'UNKNOWN')}")
            lines.append(f"- Timestamp: {evt.get('timestamp', '')}")
            lines.append("```json")
            lines.append(json.dumps(evt.get("payload", {}), indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")

        lines.append("## LLM Request-Response")
        if not self.llm_calls:
            lines.append("_No LLM calls captured._")
        for idx, call in enumerate(self.llm_calls, start=1):
            lines.append(
                f"### LLM Call {idx}: role={call.get('role', '')} model={call.get('model', '')} "
                f"status={call.get('status', '')} attempt={call.get('attempt', 1)}"
            )
            lines.append(f"- Timestamp: {call.get('timestamp', '')}")
            lines.append(f"- Symbol: {call.get('symbol', '') or 'N/A'}")
            lines.append(f"- Latency(ms): {call.get('latency_ms', '')}")
            if call.get("error"):
                lines.append(f"- Error: {call.get('error')}")
            lines.append("#### Request")
            lines.append("```json")
            lines.append(json.dumps(call.get("request", {}), indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("#### Response")
            lines.append("```json")
            lines.append(json.dumps(call.get("response", {}), indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")

        lines.append("## Agent Actions (Blackboard Messages)")
        if not self.agent_actions:
            lines.append("_No agent actions captured._")
        for idx, msg in enumerate(self.agent_actions, start=1):
            sender = msg.get("sender", "UNKNOWN")
            receiver = msg.get("receiver", "UNKNOWN")
            msg_type = msg.get("msg_type", "UNKNOWN")
            lines.append(f"### Action {idx}: {sender} -> {receiver} [{msg_type}]")
            lines.append(f"- Timestamp: {msg.get('timestamp', '')}")
            lines.append(f"- Priority: {msg.get('priority', 'NORMAL')}")
            lines.append("```json")
            lines.append(json.dumps(msg.get("content", {}), indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _to_json_safe(self, value: Any, depth: int = 0) -> Any:
        if depth > 12:
            return str(value)

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Enum):
            return value.value

        if is_dataclass(value):
            return self._to_json_safe(asdict(value), depth + 1)

        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v, depth + 1) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._to_json_safe(v, depth + 1) for v in value]

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._to_json_safe(value.to_dict(), depth + 1)
            except Exception:
                return str(value)

        if hasattr(value, "__dict__"):
            try:
                return self._to_json_safe(vars(value), depth + 1)
            except Exception:
                return str(value)

        return str(value)


_run_logger_instance: Optional[RunMarkdownLogger] = None
_run_logger_lock = threading.Lock()


def get_run_logger() -> RunMarkdownLogger:
    global _run_logger_instance
    with _run_logger_lock:
        if _run_logger_instance is None:
            _run_logger_instance = RunMarkdownLogger()
        return _run_logger_instance

```

## `./tests/test_academic_guardrails.py`

```python
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

```

## `./tests/test_bayesian_directionality.py`

```python
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

```

## `./tests/test_document_pipeline.py`

```python
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

```

## `./tests/test_query_length_guards.py`

```python
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

```

## `./validators.py`

```python
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

```

