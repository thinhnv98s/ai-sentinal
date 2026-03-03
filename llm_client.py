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
Chiến lược giao dịch hiện tại là: SHORT-TERM SWING TRADING (Tìm kiếm lợi nhuận ngắn hạn từ động lượng giá).

NGUYÊN TẮC:
1. KHÔNG đòi hỏi các yếu tố cơ bản (CANSLIM, 4M) phải đẹp.
2. Ưu tiên dòng tiền (Tín hiệu từ Wyckoff) và chất xúc tác tin tức (News Agent). Nếu Wyckoff hoặc News báo BUY/STRONG_BUY, hãy tích cực phê duyệt lệnh.
3. Chỉ Reject (từ chối) nếu Risk Guardian cảnh báo vi phạm rủi ro cứng (drawdown, position limit) hoặc kỹ thuật thực sự quá xấu (thủng đáy, VIX quá cao).
4. Chấp nhận giao dịch trong thị trường SIDEWAYS nếu có tín hiệu bắt đáy (Spring) hoặc tích lũy.

OUTPUT FORMAT (JSON):
{
    "approved": true/false,
    "position_multiplier": 0.0-1.0 (Thường giữ 1.0 trừ khi tin tức cực kỳ mơ hồ),
    "final_confidence": 0.0-1.0,
    "reasoning": "Lý do chốt cuối tập trung vào động lượng và tỷ lệ rủi ro/lợi nhuận",
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
