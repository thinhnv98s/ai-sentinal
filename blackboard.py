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
