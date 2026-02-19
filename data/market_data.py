"""
Async market data polling layer.
Fetches funding rates from all exchanges concurrently, maintains in-memory cache,
and provides a spread matrix for the strategy engine.
"""

import asyncio
import time
import logging
from collections import defaultdict
from typing import Callable, Optional

from ..connectors.base import BaseExchange, FundingRate
from ..utils.coins import get_exchange_symbol

logger = logging.getLogger(__name__)


class MarketDataManager:
    """
    Concurrent polling across all active exchanges.
    Maintains a cache of latest funding rates per (coin, exchange).
    """

    def __init__(
        self,
        connectors: dict[str, BaseExchange],
        coins: list[str],
        poll_interval_seconds: int = 60,
    ):
        self.connectors = connectors
        self.coins = coins
        self.poll_interval = poll_interval_seconds
        self._cache: dict[tuple[str, str], FundingRate] = {}
        self._cache_ts: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()
        self._running = False

    async def _fetch_one(self, exchange_name: str, connector: BaseExchange, coin: str):
        """Fetch a single (coin, exchange) funding rate."""
        # Check if this coin exists on this exchange
        symbol = get_exchange_symbol(coin, exchange_name)
        if not symbol:
            return None

        try:
            fr = await connector.get_funding_rate(coin)
            return fr
        except Exception as e:
            logger.debug(f"Fetch failed {exchange_name}/{coin}: {e}")
            return None

    async def poll_once(self) -> dict[tuple[str, str], FundingRate]:
        """
        Fetch current funding rates for all coins from all exchanges concurrently.
        Returns the updated cache.
        """
        tasks = []
        keys = []

        for ex_name, connector in self.connectors.items():
            for coin in self.coins:
                tasks.append(self._fetch_one(ex_name, connector, coin))
                keys.append((coin, ex_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        updated = 0
        errors = 0
        async with self._lock:
            for key, result in zip(keys, results):
                if isinstance(result, Exception):
                    errors += 1
                elif result is not None:
                    self._cache[key] = result
                    self._cache_ts[key] = time.time()
                    updated += 1

        logger.info(
            f"Poll complete: {updated} updated, {errors} errors, "
            f"{len(self._cache)} total cached"
        )
        return dict(self._cache)

    async def run_polling_loop(self, callback: Callable):
        """
        Continuously polls and calls callback(cache) on each update.
        callback is an async function receiving the full cache dict.
        """
        self._running = True
        logger.info(
            f"Starting polling loop: {len(self.coins)} coins, "
            f"{len(self.connectors)} exchanges, interval={self.poll_interval}s"
        )

        while self._running:
            try:
                cache = await self.poll_once()
                await callback(cache)
            except Exception as e:
                logger.error(f"Polling loop error: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._running = False

    def get_spread_matrix(self) -> dict[str, dict[str, float]]:
        """
        Returns {coin: {exchange: annualized_rate_pct}}.
        Only includes entries with valid (non-None) annualized rates.
        Mirrors the pivot table logic from top10_crypto_funding.py.
        """
        matrix: dict[str, dict[str, float]] = defaultdict(dict)
        for (coin, exchange), fr in self._cache.items():
            if fr.rate_annualized is not None:
                matrix[coin][exchange] = fr.rate_annualized
        return dict(matrix)

    def get_cache_age(self, coin: str, exchange: str) -> Optional[float]:
        """Returns seconds since last update for (coin, exchange), or None."""
        ts = self._cache_ts.get((coin, exchange))
        if ts is None:
            return None
        return time.time() - ts
