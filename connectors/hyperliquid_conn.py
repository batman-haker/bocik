"""
Hyperliquid connector using hyperliquid-python-sdk.
Read-only via Info; trading via Exchange (requires private key).
All SDK calls are synchronous — wrapped with asyncio.to_thread.
"""

import asyncio
import time
import logging
from typing import Optional

from hyperliquid.info import Info

from .base import (
    BaseExchange, FundingRate, OrderBook, Position, OrderResult,
)
from ..data.normalizer import annualize_rate, ANN_MULT
from ..utils.coins import get_exchange_symbol

logger = logging.getLogger(__name__)


class HyperliquidConnector(BaseExchange):

    def __init__(self, config: dict):
        super().__init__("hyperliquid", config)
        self._info = Info(skip_ws=True)
        self._exchange = None

        # Initialize Exchange for trading if private key is configured
        if config.get("private_key"):
            try:
                import eth_account
                from hyperliquid.exchange import Exchange
                account = eth_account.Account.from_key(config["private_key"])
                self._exchange = Exchange(account)
                logger.info("Hyperliquid: Exchange initialized with private key")
            except Exception as e:
                logger.warning(f"Hyperliquid: Could not init Exchange: {e}")

    def _is_xyz(self, coin: str) -> bool:
        """Check if this coin is on the XYZ DEX (TradFi)."""
        sym = get_exchange_symbol(coin, "hyperliquid")
        return sym is not None and sym.startswith("xyz:")

    def _hl_symbol(self, coin: str) -> str:
        """Get the HL-specific symbol (may include xyz: prefix)."""
        sym = get_exchange_symbol(coin, "hyperliquid")
        return sym or coin

    async def _fetch_funding_raw(self, coin: str, start_ms: int, end_ms: int) -> list:
        """Fetch funding history — uses raw API for xyz: coins, SDK for others."""
        hl_sym = self._hl_symbol(coin)

        if self._is_xyz(coin):
            # TradFi on XYZ DEX — SDK doesn't support xyz: names
            return await asyncio.to_thread(
                self._info.post, "/info",
                {"type": "fundingHistory", "coin": hl_sym,
                 "startTime": start_ms, "endTime": end_ms},
            )
        else:
            return await asyncio.to_thread(
                self._info.funding_history,
                name=hl_sym, startTime=start_ms, endTime=end_ms,
            )

    async def get_funding_rate(self, coin: str) -> FundingRate:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (3 * 3600 * 1000)  # last 3h to catch 1 period

        history = await self._fetch_funding_raw(coin, start_ms, now_ms)
        if not history:
            raise ValueError(f"No funding data for {coin} on Hyperliquid")

        latest = history[-1]
        raw_rate = float(latest["fundingRate"])
        ann, valid = annualize_rate(raw_rate, "hyperliquid")

        return FundingRate(
            coin=coin,
            exchange="hyperliquid",
            rate=raw_rate,
            rate_annualized=ann if valid else None,
            timestamp=int(latest["time"]),
        )

    async def get_funding_history(self, coin: str, days: int) -> list[FundingRate]:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days * 24 * 3600 * 1000)

        history = await self._fetch_funding_raw(coin, start_ms, now_ms)

        results = []
        for e in (history or []):
            raw = float(e["fundingRate"])
            ann, valid = annualize_rate(raw, "hyperliquid")
            results.append(FundingRate(
                coin=coin, exchange="hyperliquid",
                rate=raw,
                rate_annualized=ann if valid else None,
                timestamp=int(e["time"]),
            ))
        return results

    async def get_orderbook(self, coin: str) -> OrderBook:
        snap = await asyncio.to_thread(self._info.l2_snapshot, coin)
        bids = [(float(b["px"]), float(b["sz"])) for b in snap["levels"][0]]
        asks = [(float(a["px"]), float(a["sz"])) for a in snap["levels"][1]]
        return OrderBook(
            coin=coin, exchange="hyperliquid",
            bids=bids, asks=asks,
            timestamp=int(time.time() * 1000),
        )

    async def get_mark_price(self, coin: str) -> float:
        hl_sym = self._hl_symbol(coin)

        if self._is_xyz(coin):
            # TradFi — need separate allMids call with dex='xyz'
            mids = await asyncio.to_thread(
                self._info.post, "/info", {"type": "allMids", "dex": "xyz"},
            )
        else:
            mids = await asyncio.to_thread(self._info.all_mids)

        price = mids.get(hl_sym) or mids.get(coin)
        if price is None:
            raise ValueError(f"No mark price for {coin} on Hyperliquid")
        return float(price)

    async def get_index_price(self, coin: str) -> float:
        # HL doesn't expose index price separately via SDK; use mark as proxy
        return await self.get_mark_price(coin)

    async def place_order(
        self, coin: str, side: str, size: float, price: float,
        order_type: str = "limit",
    ) -> OrderResult:
        if self._exchange is None:
            raise RuntimeError("No private key configured for Hyperliquid trading")

        is_buy = (side == "buy")
        result = await asyncio.to_thread(
            self._exchange.order,
            coin, is_buy, size, price,
            {"limit": {"tif": "Gtc"}},
        )

        statuses = result.get("response", {}).get("data", {}).get("statuses", [{}])
        status_data = statuses[0] if statuses else {}

        oid = ""
        status = "rejected"
        if "resting" in status_data:
            oid = str(status_data["resting"]["oid"])
            status = "open"
        elif "filled" in status_data:
            oid = str(status_data["filled"]["oid"])
            status = "filled"

        return OrderResult(
            order_id=oid, coin=coin, exchange="hyperliquid",
            side=side, size=size, price=price,
            status=status,
            timestamp=int(time.time() * 1000),
        )

    async def cancel_order(self, coin: str, order_id: str) -> bool:
        if self._exchange is None:
            raise RuntimeError("No private key configured")
        result = await asyncio.to_thread(
            self._exchange.cancel, coin, int(order_id),
        )
        return result.get("status") == "ok"

    async def get_position(self, coin: str) -> Optional[Position]:
        wallet = self.config.get("wallet_address")
        if not wallet:
            return None

        state = await asyncio.to_thread(self._info.user_state, wallet)
        for pos in state.get("assetPositions", []):
            p = pos.get("position", {})
            if p.get("coin") == coin and float(p.get("szi", 0)) != 0:
                sz = float(p["szi"])
                return Position(
                    coin=coin, exchange="hyperliquid",
                    side="long" if sz > 0 else "short",
                    size=abs(sz),
                    entry_price=float(p.get("entryPx", 0)),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                    leverage=float(
                        p.get("leverage", {}).get("value", 1)
                    ),
                )
        return None

    async def get_balance(self) -> dict:
        wallet = self.config.get("wallet_address")
        if not wallet:
            return {"total_usd": 0.0, "available_usd": 0.0}

        state = await asyncio.to_thread(self._info.user_state, wallet)
        margin = state.get("marginSummary", {})
        return {
            "total_usd": float(margin.get("accountValue", 0)),
            "available_usd": float(margin.get("withdrawable", 0)),
        }
