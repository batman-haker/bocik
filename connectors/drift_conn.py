"""
Drift connector using REST API (data.api.drift.trade).
Key quirks: cursor-based pagination via meta.nextPage, Unix seconds timestamps,
anomalously large funding values for some assets (LOW trust level).
Reuses patterns from drift_meme_funding.py.
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

BASE_URL = "https://data.api.drift.trade"


class DriftConnector(BaseExchange):

    def __init__(self, config: dict):
        super().__init__("drift", config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._scaling_factor = config.get("scaling_factor", 1.0)

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
        sym = get_exchange_symbol(coin, "drift")
        return sym or f"{coin}-PERP"

    async def get_funding_rate(self, coin: str) -> FundingRate:
        data = await self._get(f"/market/{self._symbol(coin)}/fundingRates")
        records = data.get("records", [])
        if not records:
            raise ValueError(f"No funding data for {coin} on Drift")

        latest = records[0]
        raw_rate = float(latest["fundingRate"]) * self._scaling_factor
        ann, valid = annualize_rate(raw_rate, "drift")
        ts_ms = int(latest["ts"]) * 1000

        return FundingRate(
            coin=coin, exchange="drift",
            rate=raw_rate,
            rate_annualized=ann if valid else None,
            timestamp=ts_ms,
            mark_price=float(latest.get("markPriceTwap", 0)) or None,
            index_price=float(latest.get("oraclePriceTwap", 0)) or None,
        )

    async def get_funding_history(self, coin: str, days: int) -> list[FundingRate]:
        """Cursor-based pagination using meta.nextPage."""
        cutoff_ts = int(time.time()) - (days * 24 * 3600)
        results = []
        next_page = None

        while True:
            params = {}
            if next_page:
                params["page"] = next_page

            try:
                data = await self._get(
                    f"/market/{self._symbol(coin)}/fundingRates",
                    params=params,
                )
            except Exception as e:
                logger.warning(f"Drift history fetch error for {coin}: {e}")
                break

            records = data.get("records", [])
            if not records:
                break

            done = False
            for e in records:
                ts = int(e["ts"])
                if ts < cutoff_ts:
                    done = True
                    break

                raw = float(e["fundingRate"]) * self._scaling_factor
                ann, valid = annualize_rate(raw, "drift")
                results.append(FundingRate(
                    coin=coin, exchange="drift",
                    rate=raw,
                    rate_annualized=ann if valid else None,
                    timestamp=ts * 1000,
                    mark_price=float(e.get("markPriceTwap", 0)) or None,
                    index_price=float(e.get("oraclePriceTwap", 0)) or None,
                ))

            if done:
                break

            next_page = data.get("meta", {}).get("nextPage")
            if not next_page:
                break

            await asyncio.sleep(0.1)

        return results

    async def get_orderbook(self, coin: str) -> OrderBook:
        # Drift data API doesn't have a public orderbook endpoint
        logger.debug(f"Drift orderbook not available for {coin}")
        return OrderBook(
            coin=coin, exchange="drift",
            bids=[], asks=[],
            timestamp=int(time.time() * 1000),
        )

    async def get_mark_price(self, coin: str) -> float:
        fr = await self.get_funding_rate(coin)
        if fr.mark_price:
            return fr.mark_price
        raise ValueError(f"No mark price for {coin} on Drift")

    async def get_index_price(self, coin: str) -> float:
        fr = await self.get_funding_rate(coin)
        if fr.index_price:
            return fr.index_price
        return await self.get_mark_price(coin)

    # --- EXECUTION (placeholder — requires Solana keypair + driftpy) ---

    async def place_order(
        self, coin: str, side: str, size: float, price: float,
        order_type: str = "limit",
    ) -> OrderResult:
        raise NotImplementedError("Drift order execution requires Solana setup")

    async def cancel_order(self, coin: str, order_id: str) -> bool:
        raise NotImplementedError("Drift cancel not implemented yet")

    async def get_position(self, coin: str) -> Optional[Position]:
        return None  # Requires on-chain query

    async def get_balance(self) -> dict:
        return {"total_usd": 0.0, "available_usd": 0.0}

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
