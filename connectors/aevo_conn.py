"""
Aevo connector using REST API (api.aevo.xyz).
Key quirks: nanosecond timestamps, windowed time pagination for history.
Reuses patterns from aevo_meme_funding.py and tradfi_multi_funding.py.
"""

import asyncio
import time
import logging
from typing import Optional

import aiohttp

from .base import (
    BaseExchange, FundingRate, OrderBook, Position, OrderResult,
)
from ..data.normalizer import annualize_rate
from ..utils.coins import get_exchange_symbol

logger = logging.getLogger(__name__)

BASE_URL = "https://api.aevo.xyz"
WINDOW_DAYS = 20  # Max window per request (~500 hourly records)


class AevoConnector(BaseExchange):

    def __init__(self, config: dict):
        super().__init__("aevo", config)
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
        sym = get_exchange_symbol(coin, "aevo")
        return sym or f"{coin}-PERP"

    async def get_funding_rate(self, coin: str) -> FundingRate:
        now_ns = int(time.time() * 1e9)
        start_ns = now_ns - int(2 * 3600 * 1e9)  # last 2h

        data = await self._get("/funding-history", params={
            "instrument_name": self._symbol(coin),
            "start_time": str(start_ns),
            "end_time": str(now_ns),
            "resolution": 3600,
            "limit": 5,
        })

        records = data.get("funding_history", [])
        if not records:
            raise ValueError(f"No funding data for {coin} on Aevo")

        latest = records[-1]
        raw_rate = float(latest[2])
        ann, valid = annualize_rate(raw_rate, "aevo")
        ts_ms = int(latest[1]) // 1_000_000

        return FundingRate(
            coin=coin, exchange="aevo",
            rate=raw_rate,
            rate_annualized=ann if valid else None,
            timestamp=ts_ms,
            mark_price=float(latest[3]) if len(latest) > 3 else None,
        )

    async def get_funding_history(self, coin: str, days: int) -> list[FundingRate]:
        """Windowed time pagination — split into WINDOW_DAYS chunks."""
        now_ns = int(time.time() * 1e9)
        global_start = now_ns - int(days * 24 * 3600 * 1e9)
        results = []

        w_end = now_ns
        while w_end > global_start:
            w_start = max(global_start, w_end - int(WINDOW_DAYS * 24 * 3600 * 1e9))

            try:
                data = await self._get("/funding-history", params={
                    "instrument_name": self._symbol(coin),
                    "start_time": str(w_start),
                    "end_time": str(w_end),
                    "resolution": 3600,
                    "limit": 500,
                })

                for e in data.get("funding_history", []):
                    raw = float(e[2])
                    ann, valid = annualize_rate(raw, "aevo")
                    results.append(FundingRate(
                        coin=coin, exchange="aevo",
                        rate=raw,
                        rate_annualized=ann if valid else None,
                        timestamp=int(e[1]) // 1_000_000,
                        mark_price=float(e[3]) if len(e) > 3 else None,
                    ))
            except Exception as e:
                logger.warning(f"Aevo history fetch error for {coin}: {e}")

            w_end = w_start
            await asyncio.sleep(0.15)

        return results

    async def get_orderbook(self, coin: str) -> OrderBook:
        # Aevo doesn't have a simple public orderbook endpoint without auth
        # Return empty orderbook as placeholder
        logger.debug(f"Aevo orderbook not available for {coin} (needs auth)")
        return OrderBook(
            coin=coin, exchange="aevo",
            bids=[], asks=[],
            timestamp=int(time.time() * 1000),
        )

    async def get_mark_price(self, coin: str) -> float:
        # Use the latest funding history entry's mark price
        fr = await self.get_funding_rate(coin)
        if fr.mark_price:
            return fr.mark_price
        raise ValueError(f"No mark price for {coin} on Aevo")

    async def get_index_price(self, coin: str) -> float:
        return await self.get_mark_price(coin)

    # --- EXECUTION (placeholder — requires API key + secret) ---

    async def place_order(
        self, coin: str, side: str, size: float, price: float,
        order_type: str = "limit",
    ) -> OrderResult:
        raise NotImplementedError("Aevo order execution requires API key setup")

    async def cancel_order(self, coin: str, order_id: str) -> bool:
        raise NotImplementedError("Aevo cancel not implemented yet")

    async def get_position(self, coin: str) -> Optional[Position]:
        return None  # Requires authenticated API

    async def get_balance(self) -> dict:
        return {"total_usd": 0.0, "available_usd": 0.0}

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
