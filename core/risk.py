"""
Risk manager: exposure limits, leverage caps, margin health, circuit breakers.

v3c improvements:
- Min 48h holding period (no early exit except emergency)
- Rolling average spread for exit decisions
- Soft max age (14d) — only exit if unprofitable
- Hard max age (21d) — always exit
- 72h patience for funding flips
"""

import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional

from ..connectors.base import Position
from .strategy import ArbitrageSignal

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_position_usd: float = 15000.0
    max_exposure_per_coin_usd: float = 15000.0
    max_exposure_per_exchange_usd: float = 50000.0
    max_leverage: float = 2.0
    min_margin_ratio: float = 0.15
    max_daily_loss_pct: float = 5.0
    # v3c exit params
    min_hold_hours: float = 48.0
    soft_max_age_hours: float = 336.0    # 14 days
    hard_max_age_hours: float = 504.0    # 21 days
    flip_patience_hours: float = 72.0
    spread_collapse_threshold_pct: float = 15.0  # % of entry spread


class RiskManager:

    def __init__(self, config: dict):
        self.limits = RiskLimits(
            max_position_usd=config.get("max_position_usd", 15000.0),
            max_exposure_per_coin_usd=config.get("max_exposure_per_coin_usd", 15000.0),
            max_exposure_per_exchange_usd=config.get("max_exposure_per_exchange_usd", 50000.0),
            max_leverage=config.get("max_leverage", 2.0),
            min_margin_ratio=config.get("min_margin_ratio", 0.15),
            max_daily_loss_pct=config.get("max_daily_loss_pct", 5.0),
            min_hold_hours=config.get("min_hold_hours", 48.0),
            soft_max_age_hours=config.get("soft_max_age_hours", 336.0),
            hard_max_age_hours=config.get("hard_max_age_hours", 504.0),
            flip_patience_hours=config.get("flip_patience_hours", 72.0),
            spread_collapse_threshold_pct=config.get("spread_collapse_threshold_pct", 15.0),
        )

        # Tracked positions: coin -> exchange -> (Position, open_time, entry_spread)
        self._positions: dict[str, dict[str, tuple]] = defaultdict(dict)
        # Rolling spread history per coin: coin -> deque of last 3 spread readings
        self._spread_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=3))
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        self._daily_pnl_usd = 0.0
        self._day_start_capital = 0.0
        self._day_start_time = time.time()

    def update_spread(self, coin: str, current_spread: float):
        """Record current spread for rolling average exit decisions."""
        self._spread_history[coin].append(current_spread)

    def _avg_spread(self, coin: str, fallback: float) -> float:
        """Get rolling average spread for a coin."""
        sh = self._spread_history.get(coin)
        if sh and len(sh) > 0:
            return sum(sh) / len(sh)
        return fallback

    def can_open_position(
        self, signal: ArbitrageSignal, size_usd: float
    ) -> tuple[bool, str]:
        """Check all risk limits before allowing a new position."""
        if self._circuit_breaker_active:
            return False, f"Circuit breaker: {self._circuit_breaker_reason}"

        if size_usd <= 0:
            return False, "Position size is zero"

        # Per-coin exposure
        coin_exp = self._get_coin_exposure(signal.coin)
        if coin_exp + size_usd > self.limits.max_exposure_per_coin_usd:
            return False, (
                f"Coin exposure limit: {signal.coin} "
                f"({coin_exp:.0f} + {size_usd:.0f} > "
                f"{self.limits.max_exposure_per_coin_usd:.0f})"
            )

        # Per-exchange exposure (both legs)
        for ex in [signal.long_exchange, signal.short_exchange]:
            ex_exp = self._get_exchange_exposure(ex)
            if ex_exp + size_usd > self.limits.max_exposure_per_exchange_usd:
                return False, f"Exchange exposure limit: {ex}"

        return True, "OK"

    def check_exit_conditions(
        self, signal: ArbitrageSignal, current_spread: float,
        estimated_pnl: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Check if an open position should be closed.
        Returns (should_exit, reason).

        v3c logic:
        1. Never exit before min_hold_hours (except emergency spread=0)
        2. Use rolling average spread (not point-in-time)
        3. Soft max age: only exit if unprofitable or spread < 5%
        4. Hard max age: always exit
        5. 72h patience for funding flips
        """
        # Get position age
        pos_data = self._positions.get(signal.coin, {}).get(signal.long_exchange)
        age_hours = 0.0
        if pos_data:
            age_hours = (time.time() - pos_data[1]) / 3600

        avg_spread = self._avg_spread(signal.coin, current_spread)

        # [1] Min holding period — only emergency exit allowed
        if age_hours < self.limits.min_hold_hours:
            if current_spread < 0.5:
                return True, (
                    f"Emergency exit: spread={current_spread:.1f}% "
                    f"(age {age_hours:.0f}h < min {self.limits.min_hold_hours:.0f}h)"
                )
            return False, ""

        # [2] Spread completely gone (rolling avg < 1%)
        if avg_spread < 1.0:
            return True, (
                f"Spread gone: avg={avg_spread:.1f}% "
                f"(entry: {signal.spread:.1f}%)"
            )

        # [2] Spread collapsed below threshold (using rolling avg)
        collapse_limit = (
            signal.spread * self.limits.spread_collapse_threshold_pct / 100
        )
        if avg_spread < collapse_limit:
            return True, (
                f"Spread collapsed: avg={avg_spread:.1f}% "
                f"(entry: {signal.spread:.1f}%, "
                f"threshold: {collapse_limit:.1f}%)"
            )

        # [5] Funding flipped — only after flip_patience_hours
        if (age_hours >= self.limits.flip_patience_hours
                and signal.long_rate > 0 and signal.short_rate < 0):
            return True, (
                f"Funding flipped (sustained {age_hours:.0f}h, "
                f"long={signal.long_rate:+.1f}% short={signal.short_rate:+.1f}%)"
            )

        # [3] Soft max age: exit only if trade is unprofitable or spread weak
        if age_hours > self.limits.soft_max_age_hours:
            if estimated_pnl <= 0 or avg_spread < 5.0:
                return True, (
                    f"Soft max age: {age_hours:.0f}h, "
                    f"pnl=${estimated_pnl:.1f}, spread={avg_spread:.1f}%"
                )

        # [4] Hard max age: always exit
        if age_hours > self.limits.hard_max_age_hours:
            return True, f"Hard max age: {age_hours:.0f}h"

        return False, ""

    def check_daily_loss(self, current_capital: float) -> tuple[bool, str]:
        """Check if daily loss limit has been breached."""
        if time.time() - self._day_start_time > 86400:
            self._day_start_capital = current_capital
            self._daily_pnl_usd = 0.0
            self._day_start_time = time.time()

        if self._day_start_capital > 0:
            loss_pct = (
                (self._day_start_capital - current_capital)
                / self._day_start_capital * 100
            )
            if loss_pct > self.limits.max_daily_loss_pct:
                self._circuit_breaker_active = True
                self._circuit_breaker_reason = (
                    f"Daily loss {loss_pct:.1f}% > {self.limits.max_daily_loss_pct}%"
                )
                return True, self._circuit_breaker_reason

        return False, ""

    def register_position(
        self, coin: str, exchange: str, position: Position,
        entry_spread: float,
    ):
        """Track an opened position."""
        self._positions[coin][exchange] = (position, time.time(), entry_spread)
        logger.info(
            f"Position registered: {coin}/{exchange} "
            f"side={position.side} size={position.size}"
        )

    def remove_position(self, coin: str, exchange: str):
        """Remove a closed position from tracking."""
        if coin in self._positions and exchange in self._positions[coin]:
            del self._positions[coin][exchange]

    def reset_circuit_breaker(self):
        """Manual reset of circuit breaker."""
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        logger.info("Circuit breaker reset")

    def _get_coin_exposure(self, coin: str) -> float:
        total = 0.0
        for exchange, (pos, _, _) in self._positions.get(coin, {}).items():
            total += pos.size * pos.entry_price
        return total

    def _get_exchange_exposure(self, exchange: str) -> float:
        total = 0.0
        for coin_positions in self._positions.values():
            if exchange in coin_positions:
                pos, _, _ = coin_positions[exchange]
                total += pos.size * pos.entry_price
        return total

    def get_status(self) -> dict:
        """Return current risk status summary."""
        total_positions = sum(
            len(exs) for exs in self._positions.values()
        )
        return {
            "circuit_breaker": self._circuit_breaker_active,
            "circuit_breaker_reason": self._circuit_breaker_reason,
            "total_positions": total_positions,
            "daily_pnl_usd": self._daily_pnl_usd,
        }
