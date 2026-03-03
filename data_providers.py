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
