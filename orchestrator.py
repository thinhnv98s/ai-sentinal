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
