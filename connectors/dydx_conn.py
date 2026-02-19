"""
dYdX v4 connector using REST API (indexer.dydx.trade/v4).
Read-only via Indexer; order execution requires v4-client-py (not implemented yet).
Reuses pagination pattern from dydx_api.py.
"""

import asyncio
import time
import logging
from typing import Optional
from datetime import datetime, timezone, timedelta

import aiohttp
import pandas as pd

from .base import (
    BaseExchange, FundingRate, OrderBook, Position, OrderResult,
)
from ..data.normalizer import annualize_rate

logger = logging.getLogger(__name__)

BASE_URL = "https://indexer.dydx.trade/v4"


class DydxConnector(BaseExchange):

    def __init__(self, config: dict):
        super().__init__("dydx", config)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get(self, path: str, params: dict = None) -> dict:
        session = await self._get_session()
        async with self._rate_limit:
            async with session.get(f"{BASE_URL}{path}", params=params) as resp:
                resp.raise_for_status()
                return await resp.json()

    def _symbol(self, coin: str) -> str:
        return f"{coin}-USD"

    async def get_funding_rate(self, coin: str) -> FundingRate:
        data = await self._get(
            f"/historicalFunding/{self._symbol(coin)}",
            params={"limit": 1},
        )
        recs = data.get("historicalFunding", [])
        if not recs:
            raise ValueError(f"No funding data for {coin} on dYdX")

        e = recs[0]
        raw_rate = float(e["rate"])
        ann, valid = annualize_rate(raw_rate, "dydx")
        ts = int(pd.to_datetime(e["effectiveAt"], utc=True).timestamp() * 1000)

        return FundingRate(
            coin=coin, exchange="dydx",
            rate=raw_rate,
            rate_annualized=ann if valid else None,
            timestamp=ts,
            mark_price=float(e.get("price", 0)),
        )

    async def get_funding_history(self, coin: str, days: int) -> list[FundingRate]:
        """Paginated fetch using effectiveBeforeOrAt cursor."""
        start_time = datetime.now(timezone.utc) - timedelta(days=days)
        results = []
        effective_before = None

        while True:
            params = {"limit": 100}
            if effective_before:
                params["effectiveBeforeOrAt"] = effective_before

            data = await self._get(
                f"/historicalFunding/{self._symbol(coin)}", params=params,
            )
            recs = data.get("historicalFunding", [])
            if not recs:
                break

            for e in recs:
                t = pd.to_datetime(e["effectiveAt"], utc=True)
                if t >= start_time.replace(tzinfo=timezone.utc):
                    raw = float(e["rate"])
                    ann, valid = annualize_rate(raw, "dydx")
                    results.append(FundingRate(
                        coin=coin, exchange="dydx",
                        rate=raw,
                        rate_annualized=ann if valid else None,
                        timestamp=int(t.timestamp() * 1000),
                        mark_price=float(e.get("price", 0)),
                    ))

            oldest = pd.to_datetime(recs[-1]["effectiveAt"])
            if oldest.replace(tzinfo=None) <= start_time.replace(tzinfo=None):
                break

            effective_before = recs[-1]["effectiveAt"]

        return results

    async def get_orderbook(self, coin: str) -> OrderBook:
        data = await self._get(
            f"/orderbooks/perpetualMarket/{self._symbol(coin)}",
        )
        bids = [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
        asks = [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]
        return OrderBook(
            coin=coin, exchange="dydx",
            bids=bids, asks=asks,
            timestamp=int(time.time() * 1000),
        )

    async def get_mark_price(self, coin: str) -> float:
        data = await self._get("/perpetualMarkets")
        market = data.get("markets", {}).get(self._symbol(coin), {})
        return float(market.get("oraclePrice", 0))

    async def get_index_price(self, coin: str) -> float:
        return await self.get_mark_price(coin)

    # --- EXECUTION (placeholder — requires v4-client-py) ---

    async def place_order(
        self, coin: str, side: str, size: float, price: float,
        order_type: str = "limit",
    ) -> OrderResult:
        raise NotImplementedError(
            "dYdX v4 order execution requires v4-client-py setup. "
            "Use shadow mode for now."
        )

    async def cancel_order(self, coin: str, order_id: str) -> bool:
        raise NotImplementedError("dYdX v4 cancel not implemented yet")

    async def get_position(self, coin: str) -> Optional[Position]:
        address = self.config.get("address")
        if not address:
            return None
        try:
            data = await self._get(
                f"/addresses/{address}/perpetualPositions",
                params={"status": "OPEN"},
            )
            for pos in data.get("positions", []):
                if pos.get("market") == self._symbol(coin):
                    size = float(pos.get("size", 0))
                    if size == 0:
                        continue
                    return Position(
                        coin=coin, exchange="dydx",
                        side="long" if size > 0 else "short",
                        size=abs(size),
                        entry_price=float(pos.get("entryPrice", 0)),
                        unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                        leverage=1.0,
                    )
        except Exception as e:
            logger.warning(f"dYdX get_position error: {e}")
        return None

    async def get_balance(self) -> dict:
        address = self.config.get("address")
        if not address:
            return {"total_usd": 0.0, "available_usd": 0.0}
        try:
            data = await self._get(f"/addresses/{address}/subaccountNumber/0")
            equity = float(data.get("equity", 0))
            free = float(data.get("freeCollateral", 0))
            return {"total_usd": equity, "available_usd": free}
        except Exception:
            return {"total_usd": 0.0, "available_usd": 0.0}

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
