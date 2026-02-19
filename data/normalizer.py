"""
Centralized timestamp and rate normalization across all exchanges.
Each exchange has different time formats and funding periods.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Annualization multipliers
ANN_MULT = {
    "hyperliquid": 3 * 365,    # 8h funding -> 3/day
    "aevo":        24 * 365,   # 1h funding
    "dydx":        24 * 365,   # 1h funding
    "drift":       24 * 365,   # 1h funding
}

MAX_PLAUSIBLE_ANNUALIZED = 5000.0


def normalize_timestamp_ms(raw, exchange: str) -> int:
    """
    Convert exchange-specific timestamp to Unix milliseconds UTC.

    - Hyperliquid: already milliseconds (int)
    - Aevo: nanoseconds (int) -> divide by 1_000_000
    - dYdX: ISO 8601 string -> parse
    - Drift: Unix seconds (int) -> multiply by 1000
    """
    if exchange == "hyperliquid":
        return int(raw)
    elif exchange == "aevo":
        return int(raw) // 1_000_000
    elif exchange == "dydx":
        return int(pd.to_datetime(raw, utc=True).timestamp() * 1000)
    elif exchange == "drift":
        return int(raw) * 1000
    raise ValueError(f"Unknown exchange: {exchange}")


def annualize_rate(raw_rate: float, exchange: str) -> tuple[float, bool]:
    """
    Returns (annualized_pct, is_valid).
    is_valid=False if result exceeds plausibility threshold.
    """
    mult = ANN_MULT.get(exchange, 24 * 365)
    annualized = raw_rate * mult * 100

    if abs(annualized) > MAX_PLAUSIBLE_ANNUALIZED:
        logger.warning(
            f"{exchange}: implausible rate {annualized:.1f}% (raw={raw_rate})"
        )
        return annualized, False

    return annualized, True
