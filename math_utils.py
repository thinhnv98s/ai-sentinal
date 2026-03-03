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

