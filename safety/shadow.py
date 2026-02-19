"""
Shadow mode: intercepts all order execution, logs virtual trades.
All read-only methods pass through to the real connector.
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional

from ..connectors.base import (
    BaseExchange, FundingRate, OrderBook, Position, OrderResult,
)

logger = logging.getLogger(__name__)


class ShadowExchange:
    """
    Wraps any BaseExchange connector.
    - Read methods (funding, orderbook, prices): pass through to real connector.
    - Write methods (place_order, cancel): simulate locally and log to JSONL.
    """

    def __init__(self, real_connector: BaseExchange, log_dir: str = "shadow_trades"):
        self._real = real_connector
        self.name = real_connector.name
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log = self._log_dir / f"{self.name}_shadow_trades.jsonl"
        self._virtual_positions: dict[str, dict] = {}
        self._trade_count = 0

    # --- PASS-THROUGH READ METHODS ---

    async def get_funding_rate(self, coin: str) -> FundingRate:
        return await self._real.get_funding_rate(coin)

    async def get_funding_history(self, coin: str, days: int) -> list[FundingRate]:
        return await self._real.get_funding_history(coin, days)

    async def get_orderbook(self, coin: str) -> OrderBook:
        return await self._real.get_orderbook(coin)

    async def get_mark_price(self, coin: str) -> float:
        return await self._real.get_mark_price(coin)

    async def get_index_price(self, coin: str) -> float:
        return await self._real.get_index_price(coin)

    async def get_balance(self) -> dict:
        return await self._real.get_balance()

    # --- INTERCEPTED EXECUTION METHODS ---

    async def place_order(
        self, coin: str, side: str, size: float, price: float,
        order_type: str = "limit",
    ) -> OrderResult:
        """Simulate order fill at price with slippage."""
        slippage = 0.0005  # 0.05%
        if side == "buy":
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)

        self._trade_count += 1
        order_id = f"shadow_{self.name}_{self._trade_count}"
        now_ms = int(time.time() * 1000)

        result = OrderResult(
            order_id=order_id,
            coin=coin,
            exchange=self.name,
            side=side,
            size=size,
            price=fill_price,
            status="filled",
            timestamp=now_ms,
        )

        # Update virtual position
        if coin not in self._virtual_positions:
            self._virtual_positions[coin] = {
                "side": None, "size": 0.0, "avg_price": 0.0,
            }
        pos = self._virtual_positions[coin]

        if side == "buy":
            if pos["size"] > 0:  # Adding to long
                total_cost = pos["avg_price"] * pos["size"] + fill_price * size
                pos["size"] += size
                pos["avg_price"] = total_cost / pos["size"]
            elif pos["size"] < 0:  # Closing short
                pos["size"] += size
            else:  # New long
                pos["size"] = size
                pos["avg_price"] = fill_price
            pos["side"] = "long" if pos["size"] > 0 else ("short" if pos["size"] < 0 else None)
        else:  # sell
            if pos["size"] < 0:  # Adding to short
                total_cost = pos["avg_price"] * abs(pos["size"]) + fill_price * size
                pos["size"] -= size
                pos["avg_price"] = total_cost / abs(pos["size"])
            elif pos["size"] > 0:  # Closing long
                pos["size"] -= size
            else:  # New short
                pos["size"] = -size
                pos["avg_price"] = fill_price
            pos["side"] = "long" if pos["size"] > 0 else ("short" if pos["size"] < 0 else None)

        # Log trade
        log_entry = {
            "timestamp": now_ms,
            "exchange": self.name,
            "coin": coin,
            "side": side,
            "size": size,
            "fill_price": fill_price,
            "order_id": order_id,
            "mode": "SHADOW",
            "position_after": {
                "side": pos["side"],
                "size": pos["size"],
                "avg_price": pos["avg_price"],
            },
        }
        with open(self._trade_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(
            f"[SHADOW] {self.name} | {side.upper()} {size:.6f} {coin} "
            f"@ {fill_price:.4f} | pos: {pos['side']} {abs(pos['size']):.6f}"
        )
        return result

    async def cancel_order(self, coin: str, order_id: str) -> bool:
        logger.info(f"[SHADOW] {self.name} | CANCEL {order_id} (no-op)")
        return True

    async def get_position(self, coin: str) -> Optional[Position]:
        """Return virtual position instead of real."""
        pos = self._virtual_positions.get(coin)
        if not pos or pos["size"] == 0:
            return None
        return Position(
            coin=coin,
            exchange=self.name,
            side=pos["side"],
            size=abs(pos["size"]),
            entry_price=pos["avg_price"],
            unrealized_pnl=0.0,
            leverage=1.0,
        )

    def get_summary(self) -> dict:
        """Return shadow trading summary."""
        active = {
            coin: {"side": p["side"], "size": p["size"], "avg_price": p["avg_price"]}
            for coin, p in self._virtual_positions.items()
            if p["size"] != 0
        }
        return {
            "exchange": self.name,
            "total_trades": self._trade_count,
            "active_positions": active,
            "log_file": str(self._trade_log),
        }
