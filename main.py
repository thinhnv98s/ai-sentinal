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
