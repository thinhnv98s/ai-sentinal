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
