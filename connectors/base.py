"""
BaseExchange abstract class and shared dataclasses for all exchange connectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class FundingRate:
    coin: str
    exchange: str
    rate: float                        # Raw rate per period (e.g., 0.0001)
    rate_annualized: Optional[float]   # Annualized % (e.g., 10.95)
    timestamp: int                     # Unix milliseconds UTC
    mark_price: Optional[float] = None
    index_price: Optional[float] = None


@dataclass
class OrderBook:
    coin: str
    exchange: str
    bids: list[tuple[float, float]]   # [(price, size), ...] best first
    asks: list[tuple[float, float]]   # [(price, size), ...] best first
    timestamp: int                     # Unix milliseconds UTC


@dataclass
class Position:
    coin: str
    exchange: str
    side: str             # "long" or "short"
    size: float           # Base asset units
    entry_price: float
    unrealized_pnl: float
    leverage: float


@dataclass
class OrderResult:
    order_id: str
    coin: str
    exchange: str
    side: str            # "buy" or "sell"
    size: float
    price: float
    status: str          # "open", "filled", "rejected"
    timestamp: int       # Unix milliseconds UTC


class BaseExchange(ABC):
    """Abstract base class for all exchange connectors."""

    MAX_PLAUSIBLE_ANNUALIZED = 5000.0  # Anything above = bad data

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._rate_limit = asyncio.Semaphore(
            config.get("max_concurrent_requests", 3)
        )

    # --- READ-ONLY METHODS ---

    @abstractmethod
    async def get_funding_rate(self, coin: str) -> FundingRate:
        """Fetch the most recent funding rate for a coin."""

    @abstractmethod
    async def get_funding_history(self, coin: str, days: int) -> list[FundingRate]:
        """Fetch historical funding rates."""

    @abstractmethod
    async def get_orderbook(self, coin: str) -> OrderBook:
        """Fetch current orderbook depth."""

    @abstractmethod
    async def get_mark_price(self, coin: str) -> float:
        """Current mark price."""

    @abstractmethod
    async def get_index_price(self, coin: str) -> float:
        """Current index/oracle price."""

    # --- EXECUTION METHODS ---

    @abstractmethod
    async def place_order(
        self, coin: str, side: str, size: float, price: float,
        order_type: str = "limit",
    ) -> OrderResult:
        """Place an order. side = 'buy' or 'sell'."""

    @abstractmethod
    async def cancel_order(self, coin: str, order_id: str) -> bool:
        """Cancel an open order."""

    @abstractmethod
    async def get_position(self, coin: str) -> Optional[Position]:
        """Get current open position for a coin."""

    @abstractmethod
    async def get_balance(self) -> dict:
        """Return {'total_usd': float, 'available_usd': float}."""

    # --- SHARED UTILITIES ---

    def annualize_rate(self, raw_rate: float, ann_mult: int) -> Optional[float]:
        """
        Convert raw per-period rate to annualized %.
        Returns None if result exceeds plausibility threshold.
        """
        annualized = raw_rate * ann_mult * 100
        if abs(annualized) > self.MAX_PLAUSIBLE_ANNUALIZED:
            logger.warning(
                f"{self.name}: implausible annualized rate {annualized:.1f}% "
                f"(raw={raw_rate}), discarding"
            )
            return None
        return annualized
