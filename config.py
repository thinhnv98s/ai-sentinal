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

# --- CẤU HÌNH CHIẾN LƯỢC (ĐÃ TỐI ƯU CHO SWING TRADING) ---
@dataclass
class WyckoffConfig:
    """Cấu hình cho phân tích Wyckoff (Giữ nguyên)"""
    support_lookback: int = 50
    volume_multiplier: float = 1.3  # Hạ nhẹ để nhạy với dòng tiền hơn (từ 1.5)
    spring_tolerance: float = 0.02

@dataclass
class CANSLIMConfig:
    """Cấu hình CANSLIM (Nới lỏng hoàn toàn để không cản đường lệnh ngắn hạn)"""
    min_eps_growth: float = 0.05     # Chỉ cần > 5% (thay vì 18%)
    min_annual_growth: float = 0.10  # Chỉ cần > 10% (thay vì 25%)
    min_roe: float = 0.05            # Chỉ cần > 5% (thay vì 17%)
    min_rs_rating: int = 65          # Giá khỏe hơn 65% thị trường là ổn
    high_52w_tolerance: float = 0.20 # Cho phép mua ở vùng giá thấp hơn đỉnh 20%

@dataclass
class FourMConfig:
    """Cấu hình 4M (Bỏ qua định giá gắt gao)"""
    min_roic: float = 0.05           # ROIC > 5% (thay vì 10%)
    marr: float = 0.15
    mos_discount: float = 0.85       # Chỉ cần rẻ hơn 15% so với giá trị thực (thay vì 50%)
    max_pe: int = 60                 # Chấp nhận PE cao (cổ phiếu công nghệ thường có PE cao)

@dataclass
class RiskConfig:
    """Cấu hình quản trị rủi ro (Quỹ High-Risk nhưng cắt lỗ chặt)"""
    max_position_pct: float = 0.15   # Max 15% vốn mỗi lệnh (Tăng từ 5%)
    max_sector_exposure: float = 0.35 # Max 35% vốn mỗi ngành
    max_drawdown: float = 0.35       # Chịu đựng sụt giảm 35% (Tăng từ 20%)
    stop_loss_atr_multiplier: float = 1.5  # CẮT LỖ CỰC NHANH: Kéo SL sát giá mua (Giảm từ 2.0 hoặc 1.8)
    risk_per_trade_pct: float = 0.02 # Rủi ro tối đa 2% tổng vốn mỗi lệnh (Chuẩn Day/Swing trader)
    allow_short: bool = os.environ.get("ALLOW_SHORT", "false").lower() == "true"

@dataclass
class BayesianConfig:
    """Cấu hình Bayesian (Chuyển quyền lực cho Wyckoff và News)"""
    prior_probability: float = 0.5
    min_probability_threshold: float = 0.48 # Vùng kích hoạt lệnh nhạy hơn (từ >0.52 là MUA)

    # Bóp nghẹt trọng số của CANSLIM/4M (gần 0.5) để chúng không làm hỏng lệnh kỹ thuật
    # Bơm tối đa sức mạnh cho Wyckoff (kỹ thuật) và News (dòng tiền/tin tức)
    agent_profiles: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "wyckoff": {"sensitivity": 0.85, "specificity": 0.75}, # Trọng số CỰC CAO
        "canslim": {"sensitivity": 0.52, "specificity": 0.52}, # Vô hiệu hóa ảnh hưởng
        "fourm":   {"sensitivity": 0.52, "specificity": 0.52}, # Vô hiệu hóa ảnh hưởng
        "news":    {"sensitivity": 0.75, "specificity": 0.65}  # Trọng số CAO
    })

    regime_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "RISK_ON": {"wyckoff": 1.10, "canslim": 1.0, "fourm": 1.0, "news": 1.10},
        "RISK_OFF": {"wyckoff": 0.80, "canslim": 1.0, "fourm": 1.0, "news": 0.90},
        "SIDEWAYS": {"wyckoff": 1.15, "canslim": 1.0, "fourm": 1.0, "news": 1.05}, # SIDEWAYS ưu tiên đánh kỹ thuật
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
