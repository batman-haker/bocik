"""
Strategy engine: spread calculation, z-score anomaly detection, signal generation,
and position sizing.

Features:
- Rolling z-score from in-memory history + CSV bootstrap
- Multi-timeframe spread analysis (current, 8h avg, 24h avg)
- Spread trend detection (expanding/contracting/stable)
"""

import csv
import time
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
from pathlib import Path

import numpy as np

from ..utils.coins import TRUST_LEVEL

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageSignal:
    coin: str
    long_exchange: str      # Go long here (lowest funding rate)
    short_exchange: str     # Go short here (highest funding rate)
    long_rate: float        # Annualized % at long exchange
    short_rate: float       # Annualized % at short exchange
    spread: float           # short_rate - long_rate (annualized %)
    z_score: float          # How anomalous vs history
    entry_cost_pct: float   # Round-trip fees + slippage (flat %)
    net_return_pct: float   # Expected return for holding period (flat %)
    confidence: str         # "HIGH", "MEDIUM", "LOW"
    timestamp: int          # Unix ms
    spread_trend: str = "unknown"     # "expanding", "contracting", "stable"
    avg_spread_8h: float = 0.0        # 8-cycle moving average
    avg_spread_24h: float = 0.0       # 24-cycle moving average

    def __str__(self):
        trend_icon = {"expanding": "^", "contracting": "v", "stable": "="}
        t = trend_icon.get(self.spread_trend, "?")
        return (
            f"{self.coin} | long:{self.long_exchange}({self.long_rate:+.1f}%) "
            f"short:{self.short_exchange}({self.short_rate:+.1f}%) | "
            f"spread:{self.spread:.1f}%{t} | z:{self.z_score:.2f} | "
            f"net:{self.net_return_pct:.3f}% | {self.confidence}"
        )


# Fee structure: maker/taker as % of notional
FEES = {
    "hyperliquid": {"maker": 0.010, "taker": 0.035},
    "dydx":        {"maker": 0.010, "taker": 0.050},
    "aevo":        {"maker": 0.000, "taker": 0.050},
    "drift":       {"maker": 0.010, "taker": 0.060},
}


class StrategyEngine:

    def __init__(self, config: dict, reports_dir: str = None):
        self.min_spread = config.get("min_spread_annualized_pct", 10.0)
        self.holding_days = config.get("expected_holding_days", 14)
        self.slippage = config.get("slippage_pct_per_leg", 0.02)
        self.z_threshold = config.get("z_score_threshold", 1.5)
        self.use_maker = config.get("use_maker_orders", True)

        # Per-coin spread history: coin -> deque of (spread, timestamp_ms)
        self._coin_spreads: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=500)
        )
        # Per-pair history: pair_key -> deque of (spread, timestamp_ms)
        self._pair_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=500)
        )

        # Bootstrap from CSV if available
        if reports_dir:
            self._load_history_from_csv(reports_dir)

    def _load_history_from_csv(self, reports_dir: str):
        """Load historical spreads from signals_log.csv to bootstrap z-scores."""
        csv_path = Path(reports_dir) / "signals_log.csv"
        if not csv_path.exists():
            return
        loaded = 0
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    coin = row.get("coin", "")
                    long_ex = row.get("long_exchange", "")
                    short_ex = row.get("short_exchange", "")
                    spread = float(row.get("spread", 0))
                    ts_str = row.get("timestamp", "")

                    # Parse timestamp to ms
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(ts_str)
                        ts_ms = int(dt.timestamp() * 1000)
                    except Exception:
                        ts_ms = int(time.time() * 1000)

                    pair_key = f"{coin}_{long_ex}_{short_ex}"
                    self._pair_history[pair_key].append((spread, ts_ms))
                    self._coin_spreads[coin].append((spread, ts_ms))
                    loaded += 1
            if loaded:
                logger.info(f"Strategy: loaded {loaded} historical spreads from CSV")
        except Exception as e:
            logger.warning(f"Strategy: failed to load CSV history: {e}")

    def _entry_cost(self, long_ex: str, short_ex: str) -> float:
        """
        Total round-trip cost as flat % of notional.
        4 legs: open long, open short, close long, close short.
        """
        fee_type = "maker" if self.use_maker else "taker"
        long_fee = FEES.get(long_ex, {}).get(fee_type, 0.05)
        short_fee = FEES.get(short_ex, {}).get(fee_type, 0.05)
        total_fees = 2 * (long_fee + short_fee)
        total_slippage = 4 * self.slippage
        return total_fees + total_slippage

    def _z_score(self, pair_key: str, current_spread: float) -> float:
        """Z-score of current spread vs historical for this pair."""
        history = self._pair_history.get(pair_key)
        if not history or len(history) < 3:
            return 0.0
        arr = np.array([h[0] for h in history])
        mean, std = arr.mean(), arr.std()
        if std < 0.1:  # avoid division by near-zero
            return 0.0
        return round(float((current_spread - mean) / std), 2)

    def _spread_trend(self, coin: str, current_spread: float) -> tuple[str, float, float]:
        """
        Analyze spread trend using moving averages.
        Returns (trend, avg_8h, avg_24h).
        """
        history = self._coin_spreads.get(coin)
        if not history:
            return "unknown", current_spread, current_spread

        spreads = [h[0] for h in history]

        # 8-cycle MA (8h at 1h intervals)
        recent_8 = spreads[-8:] if len(spreads) >= 8 else spreads
        avg_8h = float(np.mean(recent_8))

        # 24-cycle MA
        recent_24 = spreads[-24:] if len(spreads) >= 24 else spreads
        avg_24h = float(np.mean(recent_24))

        # Trend: compare current to 8h average
        if len(spreads) < 3:
            trend = "unknown"
        elif current_spread > avg_8h * 1.1:
            trend = "expanding"
        elif current_spread < avg_8h * 0.9:
            trend = "contracting"
        else:
            trend = "stable"

        return trend, round(avg_8h, 1), round(avg_24h, 1)

    def _confidence(
        self, long_ex: str, short_ex: str, z: float, trend: str
    ) -> str:
        """Determine signal confidence based on trust levels, z-score, and trend."""
        trust_scores = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        long_trust = trust_scores[TRUST_LEVEL.get(long_ex, "LOW")]
        short_trust = trust_scores[TRUST_LEVEL.get(short_ex, "LOW")]
        worst = max(long_trust, short_trust)

        if worst == 2:
            return "LOW"

        # Boost confidence if spread is expanding and z-score is high
        if worst == 0 and z >= self.z_threshold and trend in ("expanding", "stable"):
            return "HIGH"

        if worst == 1 or z < self.z_threshold:
            return "MEDIUM"

        return "HIGH"

    def generate_signals(
        self, spread_matrix: dict[str, dict[str, float]]
    ) -> list[ArbitrageSignal]:
        """
        For each coin, find the exchange pair with the largest funding spread.
        Filter by: spread > threshold, positive net return, and confidence.
        """
        signals = []
        now_ms = int(time.time() * 1000)

        for coin, rates in spread_matrix.items():
            if len(rates) < 2:
                continue

            # Best pair: lowest rate = long, highest rate = short
            sorted_rates = sorted(rates.items(), key=lambda x: x[1])
            long_ex, long_rate = sorted_rates[0]
            short_ex, short_rate = sorted_rates[-1]
            spread = short_rate - long_rate

            if spread < self.min_spread:
                continue

            entry_cost = self._entry_cost(long_ex, short_ex)

            # Convert annualized spread to expected period return
            period_return = spread * (self.holding_days / 365)
            net_return = period_return - entry_cost

            if net_return <= 0:
                continue

            pair_key = f"{coin}_{long_ex}_{short_ex}"
            z = self._z_score(pair_key, spread)

            # Trend analysis
            trend, avg_8h, avg_24h = self._spread_trend(coin, spread)

            # Record to history
            self._pair_history[pair_key].append((spread, now_ms))
            self._coin_spreads[coin].append((spread, now_ms))

            confidence = self._confidence(long_ex, short_ex, z, trend)

            signals.append(ArbitrageSignal(
                coin=coin,
                long_exchange=long_ex,
                short_exchange=short_ex,
                long_rate=long_rate,
                short_rate=short_rate,
                spread=spread,
                z_score=z,
                entry_cost_pct=entry_cost,
                net_return_pct=net_return,
                confidence=confidence,
                timestamp=now_ms,
                spread_trend=trend,
                avg_spread_8h=avg_8h,
                avg_spread_24h=avg_24h,
            ))

        return sorted(signals, key=lambda s: s.spread, reverse=True)

    def compute_position_size(
        self, signal: ArbitrageSignal, total_capital_usd: float,
        max_position_usd: float = 15000.0,
    ) -> float:
        """
        v3c position sizing — aggressive capital deployment:
        - 10-30% spread:  8% of capital
        - 30-100% spread: 12% of capital
        - >100% spread:   8% of capital (extreme = systemic risk)
        Expanding trend: +25% boost. Contracting trend: -30% reduce.
        LOW confidence = 0 (never trade).
        """
        if signal.confidence == "LOW":
            return 0.0

        spread = signal.spread
        if spread >= 100:
            pct = 0.08
        elif spread >= 30:
            pct = 0.12
        elif spread >= 10:
            pct = 0.08
        else:
            return 0.0

        if signal.spread_trend == "expanding":
            pct *= 1.25
        elif signal.spread_trend == "contracting":
            pct *= 0.70

        return min(total_capital_usd * pct, max_position_usd)

    def get_stats(self) -> dict:
        """Return strategy statistics for dashboard."""
        return {
            "total_pairs_tracked": len(self._pair_history),
            "total_data_points": sum(len(h) for h in self._pair_history.values()),
            "coins_with_history": len(self._coin_spreads),
            "z_threshold": self.z_threshold,
            "min_spread": self.min_spread,
        }
