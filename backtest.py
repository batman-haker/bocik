"""
12-Month Funding Rate Arbitrage Backtest.

Fetches historical funding rates from Hyperliquid, dYdX, and Aevo,
replays the strategy hour-by-hour, and generates an HTML report.

Usage:
    py arbitrage_system/backtest.py
"""

import asyncio
import csv
import json
import math
import os
import sys
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from arbitrage_system.connectors.hyperliquid_conn import HyperliquidConnector
from arbitrage_system.connectors.dydx_conn import DydxConnector
from arbitrage_system.connectors.aevo_conn import AevoConnector
from arbitrage_system.connectors.base import FundingRate
from arbitrage_system.core.strategy import StrategyEngine, ArbitrageSignal, FEES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "backtest_results"
CACHE_CSV = RESULTS_DIR / "historical_rates.csv"

# ── Config ──────────────────────────────────────────────────────────

START_CAPITAL = 100_000.0
BACKTEST_DAYS = 365
HOUR_MS = 3600 * 1000


# ── Data Fetching ──────────────────────────────────────────────────

def load_config() -> dict:
    config_dir = BASE_DIR / "config"
    with open(config_dir / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_path = config_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    hl = config["exchanges"]["hyperliquid"]
    hl["private_key"] = os.getenv("HL_PRIVATE_KEY")
    hl["wallet_address"] = os.getenv("HL_WALLET_ADDRESS")
    dx = config["exchanges"]["dydx"]
    dx["mnemonic"] = os.getenv("DYDX_MNEMONIC")
    dx["address"] = os.getenv("DYDX_ADDRESS")
    return config


async def fetch_all_history(config: dict, coins: list) -> list[dict]:
    """Fetch historical funding rates from all enabled exchanges."""
    connectors = {}
    ex = config["exchanges"]
    if ex.get("hyperliquid", {}).get("enabled"):
        connectors["hyperliquid"] = HyperliquidConnector(ex["hyperliquid"])
    if ex.get("dydx", {}).get("enabled"):
        connectors["dydx"] = DydxConnector(ex["dydx"])
    if ex.get("aevo", {}).get("enabled"):
        connectors["aevo"] = AevoConnector(ex["aevo"])

    all_rates = []
    total = len(coins) * len(connectors)
    done = 0

    for exchange_name, conn in connectors.items():
        for coin in coins:
            done += 1
            logger.info(f"[{done}/{total}] Fetching {coin} from {exchange_name}...")
            try:
                rates = await conn.get_funding_history(coin, BACKTEST_DAYS)
                for r in rates:
                    if r.rate_annualized is not None:
                        all_rates.append({
                            "timestamp_ms": r.timestamp,
                            "coin": r.coin,
                            "exchange": r.exchange,
                            "rate_annualized": round(r.rate_annualized, 4),
                        })
                logger.info(f"  -> {len(rates)} records")
            except Exception as e:
                logger.warning(f"  -> ERROR: {e}")
            await asyncio.sleep(0.1)

    # Close sessions
    for conn in connectors.values():
        if hasattr(conn, "close"):
            await conn.close()

    return all_rates


def save_cache(rates: list[dict]):
    """Save fetched rates to CSV cache."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_ms", "coin", "exchange", "rate_annualized"])
        w.writeheader()
        w.writerows(rates)
    logger.info(f"Saved {len(rates)} rates to {CACHE_CSV}")


def load_cache() -> list[dict]:
    """Load rates from CSV cache."""
    rates = []
    with open(CACHE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rates.append({
                "timestamp_ms": int(row["timestamp_ms"]),
                "coin": row["coin"],
                "exchange": row["exchange"],
                "rate_annualized": float(row["rate_annualized"]),
            })
    logger.info(f"Loaded {len(rates)} rates from cache")
    return rates


# ── Backtest Engine ────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    timestamp_ms: int
    coin: str
    long_exchange: str
    short_exchange: str
    side: str  # "open" or "close"
    size_usd: float
    spread: float
    fees_usd: float
    reason: str = ""


@dataclass
class BacktestResult:
    equity_curve: list  # [{ts_ms, equity, drawdown_pct}]
    trades: list  # [BacktestTrade]
    per_coin_pnl: dict  # {coin: total_pnl_usd}
    monthly_returns: dict  # {"2025-03": pct_return}
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    avg_holding_hours: float = 0.0
    win_rate: float = 0.0
    total_fees_usd: float = 0.0
    total_funding_pnl: float = 0.0


def _align_to_hourly(rates: list[dict]) -> dict:
    """
    Organize rates into hourly buckets.
    Returns: {hour_ts_ms: {coin: {exchange: rate_annualized}}}
    """
    buckets = defaultdict(lambda: defaultdict(dict))
    for r in rates:
        # Round to nearest hour
        hour_ts = (r["timestamp_ms"] // HOUR_MS) * HOUR_MS
        buckets[hour_ts][r["coin"]][r["exchange"]] = r["rate_annualized"]
    return buckets


def run_backtest(
    rates: list[dict],
    strategy_config: dict,
    risk_config: dict,
    start_capital: float = START_CAPITAL,
) -> BacktestResult:
    """
    Replay strategy hour-by-hour through historical data.

    v3 improvements:
    1. Min 48h holding (no early exit except spread=0)
    2. Rolling avg exit (3-reading average, not point-in-time)
    3. Bigger positions (5-10% of capital)
    4. Up to 3 new positions per 8h cycle
    5. No fixed max_age if position is profitable
    6. Slippage 0.02% (limit orders)
    """

    buckets = _align_to_hourly(rates)
    timestamps = sorted(buckets.keys())

    if not timestamps:
        logger.error("No data to backtest!")
        return BacktestResult([], [], {}, {})

    logger.info(f"Backtest period: {len(timestamps)} hours "
                f"({datetime.fromtimestamp(timestamps[0]/1000, tz=timezone.utc).date()} "
                f"to {datetime.fromtimestamp(timestamps[-1]/1000, tz=timezone.utc).date()})")

    strategy = StrategyEngine(strategy_config)

    equity = start_capital
    peak_equity = equity
    max_drawdown = 0.0

    # Active positions: coin -> {long_ex, short_ex, size_usd, open_ts,
    #                             entry_spread, long_rate, short_rate, trade_pnl,
    #                             spread_history (deque of last 3 readings)}
    positions = {}

    # Simple position tracker
    coin_exposure = defaultdict(float)
    exchange_exposure = defaultdict(float)
    max_coin_exp = risk_config.get("max_exposure_per_coin_usd", 10000.0)
    max_ex_exp = risk_config.get("max_exposure_per_exchange_usd", 20000.0)
    max_pos_usd = risk_config.get("max_position_usd", 10000.0)

    # ── IMPROVEMENT CONFIG ──
    MIN_HOLD_HOURS = 48         # [1] Never exit before 48h
    SPREAD_AVG_WINDOW = 3       # [2] Rolling avg of last 3 readings for exit
    MAX_NEW_PER_CYCLE = 3       # [4] Up to 3 positions per 8h cycle
    SOFT_MAX_AGE = 336          # [5] 14 days soft max — only exit if unprofitable
    HARD_MAX_AGE = 504          # [5] 21 days hard max — always exit
    FLIP_PATIENCE_HOURS = 72    # Wait 72h before exiting on funding flip
    LEVERAGE = risk_config.get("max_leverage", 2.0)  # Funding accrues on full notional

    equity_curve = []
    trades = []
    per_coin_pnl = defaultdict(float)
    monthly_equity = {}
    holding_times = []
    wins = 0
    losses = 0
    total_fees = 0.0
    total_funding = 0.0

    prev_month = None

    for i, ts in enumerate(timestamps):
        spread_matrix = buckets[ts]
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        month_str = dt.strftime("%Y-%m")

        if prev_month and month_str != prev_month:
            monthly_equity[prev_month] = equity
        prev_month = month_str

        # ── 1. Funding accrual on open positions ──
        for coin, pos in list(positions.items()):
            if coin not in spread_matrix:
                continue

            rates_now = spread_matrix[coin]
            long_rate = rates_now.get(pos["long_ex"])
            short_rate = rates_now.get(pos["short_ex"])

            # Funding accrues on NOTIONAL (size × leverage), not just margin
            notional = pos["size_usd"] * LEVERAGE

            if long_rate is not None:
                hourly = long_rate / (365 * 24) / 100
                funding = -notional * hourly
                equity += funding
                per_coin_pnl[coin] += funding
                total_funding += funding
                pos["trade_pnl"] += funding
                pos["long_rate"] = long_rate

            if short_rate is not None:
                hourly = short_rate / (365 * 24) / 100
                funding = notional * hourly
                equity += funding
                per_coin_pnl[coin] += funding
                total_funding += funding
                pos["trade_pnl"] += funding
                pos["short_rate"] = short_rate

            # [2] Track rolling spread for this position
            if coin in spread_matrix and len(spread_matrix[coin]) >= 2:
                vals = sorted(spread_matrix[coin].values())
                cur_sp = vals[-1] - vals[0]
                pos["spread_history"].append(cur_sp)

        # ── 2. Check exit conditions (every 8h) ──
        hour_of_day = dt.hour
        check_exits = (hour_of_day % 8 == 0)

        for coin in list(positions.keys()):
            pos = positions[coin]
            if coin not in spread_matrix or len(spread_matrix[coin]) < 2:
                continue

            rates_now = spread_matrix[coin]
            sorted_r = sorted(rates_now.values())
            current_spread = sorted_r[-1] - sorted_r[0]

            age_h = (ts - pos["open_ts"]) / HOUR_MS

            # [1] Never exit before MIN_HOLD_HOURS (except total spread gone)
            if age_h < MIN_HOLD_HOURS:
                # Only emergency exit: spread completely vanished
                if current_spread < 0.5:
                    pass  # allow exit below
                else:
                    continue  # skip all exit checks

            if not check_exits and age_h < HARD_MAX_AGE:
                continue

            # [2] Use rolling average spread (not point-in-time)
            sh = pos["spread_history"]
            avg_spread = sum(sh) / len(sh) if sh else current_spread

            should_exit = False
            reason = ""

            # Emergency: spread completely gone (avg < 1%)
            if avg_spread < 1.0:
                should_exit = True
                reason = f"spread_gone (avg={avg_spread:.1f}%)"

            # Spread collapsed below 15% of entry (using rolling avg)
            if not should_exit and age_h >= MIN_HOLD_HOURS:
                if avg_spread < pos["entry_spread"] * 0.15:
                    should_exit = True
                    reason = f"spread_collapse (avg={avg_spread:.1f}% vs entry {pos['entry_spread']:.1f}%)"

            # Funding flipped — only after 72h (longer patience)
            if not should_exit:
                lr = pos.get("long_rate", 0)
                sr = pos.get("short_rate", 0)
                if lr > 0 and sr < 0 and age_h >= FLIP_PATIENCE_HOURS:
                    should_exit = True
                    reason = "funding_flipped"

            # [5] Soft max age: exit only if trade is unprofitable
            if not should_exit and age_h > SOFT_MAX_AGE:
                if pos["trade_pnl"] <= 0 or avg_spread < 5.0:
                    should_exit = True
                    reason = f"soft_max_age ({age_h:.0f}h, pnl=${pos['trade_pnl']:.1f})"

            # [5] Hard max age: always exit
            if not should_exit and age_h > HARD_MAX_AGE:
                should_exit = True
                reason = f"hard_max_age ({age_h:.0f}h)"

            if should_exit:
                fee_pct = _exit_fees(pos["long_ex"], pos["short_ex"])
                fee_usd = pos["size_usd"] * LEVERAGE * fee_pct / 100
                equity -= fee_usd
                total_fees += fee_usd
                per_coin_pnl[coin] -= fee_usd
                pos["trade_pnl"] -= fee_usd

                if pos["trade_pnl"] > 0:
                    wins += 1
                else:
                    losses += 1
                holding_times.append(age_h)

                trades.append(BacktestTrade(
                    timestamp_ms=ts, coin=coin,
                    long_exchange=pos["long_ex"], short_exchange=pos["short_ex"],
                    side="close", size_usd=pos["size_usd"],
                    spread=current_spread, fees_usd=fee_usd, reason=reason,
                ))

                coin_exposure[coin] -= pos["size_usd"]
                exchange_exposure[pos["long_ex"]] -= pos["size_usd"]
                exchange_exposure[pos["short_ex"]] -= pos["size_usd"]
                del positions[coin]

        # ── 3. Generate signals every 8h ──
        if hour_of_day % 8 != 0:
            signals = []
        else:
            signals = strategy.generate_signals(dict(spread_matrix))

        # [4] Up to MAX_NEW_PER_CYCLE positions per cycle
        opened_this_step = 0
        for signal in signals:
            if opened_this_step >= MAX_NEW_PER_CYCLE:
                break
            if signal.confidence == "LOW":
                continue
            if signal.coin in positions:
                continue

            # [3] Bigger position sizing
            size_usd = _compute_position_size_v2(signal, equity, max_pos_usd)
            if size_usd <= 0:
                continue

            if coin_exposure[signal.coin] + size_usd > max_coin_exp:
                continue
            if exchange_exposure[signal.long_exchange] + size_usd > max_ex_exp:
                continue
            if exchange_exposure[signal.short_exchange] + size_usd > max_ex_exp:
                continue

            fee_pct = _entry_fees(signal.long_exchange, signal.short_exchange)
            fee_usd = size_usd * LEVERAGE * fee_pct / 100  # Fees on notional
            equity -= fee_usd
            total_fees += fee_usd
            per_coin_pnl[signal.coin] -= fee_usd

            positions[signal.coin] = {
                "long_ex": signal.long_exchange,
                "short_ex": signal.short_exchange,
                "size_usd": size_usd,
                "open_ts": ts,
                "entry_spread": signal.spread,
                "long_rate": signal.long_rate,
                "short_rate": signal.short_rate,
                "trade_pnl": -fee_usd,
                "spread_history": deque([signal.spread], maxlen=SPREAD_AVG_WINDOW),
            }

            coin_exposure[signal.coin] += size_usd
            exchange_exposure[signal.long_exchange] += size_usd
            exchange_exposure[signal.short_exchange] += size_usd

            trades.append(BacktestTrade(
                timestamp_ms=ts, coin=signal.coin,
                long_exchange=signal.long_exchange,
                short_exchange=signal.short_exchange,
                side="open", size_usd=size_usd,
                spread=signal.spread, fees_usd=fee_usd,
            ))
            opened_this_step += 1

        # ── 4. Record equity ──
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)

        if i % 4 == 0:
            equity_curve.append({
                "ts_ms": ts,
                "equity": round(equity, 2),
                "drawdown_pct": round(dd, 2),
                "positions": len(positions),
            })

        if i % 720 == 0:
            logger.info(f"  [{dt.date()}] Equity: ${equity:,.0f} | "
                        f"Positions: {len(positions)} | "
                        f"Trades: {len(trades)} | DD: {dd:.1f}%")

    # Final month
    if prev_month:
        monthly_equity[prev_month] = equity

    # Monthly returns
    monthly_returns = {}
    months_sorted = sorted(monthly_equity.keys())
    for i, m in enumerate(months_sorted):
        if i == 0:
            monthly_returns[m] = (monthly_equity[m] / start_capital - 1) * 100
        else:
            prev_eq = monthly_equity[months_sorted[i - 1]]
            monthly_returns[m] = (monthly_equity[m] / prev_eq - 1) * 100 if prev_eq else 0

    # Sharpe ratio
    if len(equity_curve) > 2:
        equities = [e["equity"] for e in equity_curve]
        returns = np.diff(equities) / equities[:-1]
        periods_per_year = 365 * 24 / 4
        if returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    total_trades = len([t for t in trades if t.side == "open"])
    total_closed = wins + losses

    result = BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        per_coin_pnl=dict(per_coin_pnl),
        monthly_returns=monthly_returns,
        total_return_pct=round((equity / start_capital - 1) * 100, 2),
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=round(sharpe, 2),
        total_trades=total_trades,
        avg_holding_hours=round(np.mean(holding_times), 1) if holding_times else 0.0,
        win_rate=round(wins / total_closed * 100, 1) if total_closed > 0 else 0.0,
        total_fees_usd=round(total_fees, 2),
        total_funding_pnl=round(total_funding, 2),
    )

    return result


def _compute_position_size_v2(signal, equity: float, max_pos_usd: float) -> float:
    """
    [Improvement 3] Aggressive position sizing — maximize capital deployment.
    - 10-30% spread: 8% of capital
    - 30-100% spread: 12% of capital
    - >100% spread: 8% of capital (extreme = systemic risk)
    Trend boost: expanding +25%, contracting -30%
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

    if getattr(signal, "spread_trend", "") == "expanding":
        pct *= 1.25
    elif getattr(signal, "spread_trend", "") == "contracting":
        pct *= 0.7

    return min(equity * pct, max_pos_usd)


def _entry_fees(long_ex: str, short_ex: str) -> float:
    """Entry fees as % of notional (2 legs). Uses maker fees + reduced slippage."""
    lf = FEES.get(long_ex, {}).get("maker", 0.01)
    sf = FEES.get(short_ex, {}).get("maker", 0.01)
    return lf + sf + 0.04  # [6] 2 × 0.02% slippage (was 0.03)


def _exit_fees(long_ex: str, short_ex: str) -> float:
    """Exit fees as % of notional (2 legs)."""
    return _entry_fees(long_ex, short_ex)


# ── HTML Report ────────────────────────────────────────────────────

def generate_report(result: BacktestResult, output_dir: Path):
    """Generate a self-contained HTML report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for charts
    equity_labels = [
        datetime.fromtimestamp(e["ts_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        for e in result.equity_curve
    ]
    equity_values = [e["equity"] for e in result.equity_curve]
    dd_values = [e["drawdown_pct"] for e in result.equity_curve]
    pos_values = [e["positions"] for e in result.equity_curve]

    # Per-coin P&L sorted
    coin_pnl_sorted = sorted(result.per_coin_pnl.items(), key=lambda x: x[1], reverse=True)
    coin_labels = [c for c, _ in coin_pnl_sorted]
    coin_values = [round(v, 2) for _, v in coin_pnl_sorted]
    coin_colors = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in coin_values]

    # Monthly returns
    month_labels = list(result.monthly_returns.keys())
    month_values = [round(v, 2) for v in result.monthly_returns.values()]
    month_colors = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in month_values]

    # Trade log (last 50)
    trade_rows = ""
    for t in result.trades[-100:]:
        dt = datetime.fromtimestamp(t.timestamp_ms / 1000, tz=timezone.utc)
        cls = "open" if t.side == "open" else "close"
        trade_rows += f"""<tr class="{cls}">
            <td>{dt.strftime('%Y-%m-%d %H:%M')}</td>
            <td>{t.coin}</td>
            <td>{t.side.upper()}</td>
            <td>{t.long_exchange}</td>
            <td>{t.short_exchange}</td>
            <td>${t.size_usd:,.0f}</td>
            <td>{t.spread:.1f}%</td>
            <td>${t.fees_usd:.2f}</td>
            <td>{t.reason}</td>
        </tr>"""

    final_equity = equity_values[-1] if equity_values else START_CAPITAL

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report — Funding Rate Arbitrage</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0d1117; color:#c9d1d9; font-family:'Segoe UI',sans-serif; padding:20px; }}
h1 {{ color:#58a6ff; margin-bottom:10px; }}
h2 {{ color:#58a6ff; margin:20px 0 10px; font-size:1.2em; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:20px 0; }}
.stat {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; text-align:center; }}
.stat .value {{ font-size:2em; font-weight:bold; margin:4px 0; }}
.stat .label {{ color:#8b949e; font-size:0.85em; }}
.positive {{ color:#00d4aa; }}
.negative {{ color:#ff6b6b; }}
.chart-container {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin:12px 0; }}
canvas {{ max-height:350px; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.85em; }}
th {{ background:#21262d; color:#58a6ff; padding:8px; text-align:left; position:sticky; top:0; }}
td {{ padding:6px 8px; border-bottom:1px solid #21262d; }}
tr.open td {{ color:#00d4aa; }}
tr.close td {{ color:#ff6b6b; }}
.table-wrap {{ max-height:500px; overflow-y:auto; background:#161b22; border:1px solid #30363d; border-radius:8px; }}
.footer {{ color:#484f58; margin-top:30px; text-align:center; font-size:0.8em; }}
</style>
</head>
<body>
<h1>Funding Rate Arbitrage — 12-Month Backtest</h1>
<p style="color:#8b949e">Period: {equity_labels[0] if equity_labels else 'N/A'} to {equity_labels[-1] if equity_labels else 'N/A'} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="stats">
  <div class="stat">
    <div class="label">Final Equity</div>
    <div class="value {'positive' if final_equity >= START_CAPITAL else 'negative'}">${final_equity:,.0f}</div>
    <div class="label">from ${START_CAPITAL:,.0f}</div>
  </div>
  <div class="stat">
    <div class="label">Total Return</div>
    <div class="value {'positive' if result.total_return_pct >= 0 else 'negative'}">{result.total_return_pct:+.1f}%</div>
  </div>
  <div class="stat">
    <div class="label">Max Drawdown</div>
    <div class="value negative">-{result.max_drawdown_pct:.1f}%</div>
  </div>
  <div class="stat">
    <div class="label">Sharpe Ratio</div>
    <div class="value {'positive' if result.sharpe_ratio > 1 else ''}">{result.sharpe_ratio:.2f}</div>
  </div>
  <div class="stat">
    <div class="label">Total Trades</div>
    <div class="value">{result.total_trades}</div>
  </div>
  <div class="stat">
    <div class="label">Win Rate</div>
    <div class="value">{result.win_rate:.0f}%</div>
  </div>
  <div class="stat">
    <div class="label">Avg Hold Time</div>
    <div class="value">{result.avg_holding_hours:.0f}h</div>
  </div>
  <div class="stat">
    <div class="label">Total Fees</div>
    <div class="value negative">${result.total_fees_usd:,.0f}</div>
  </div>
  <div class="stat">
    <div class="label">Funding P&L</div>
    <div class="value {'positive' if result.total_funding_pnl >= 0 else 'negative'}">${result.total_funding_pnl:,.0f}</div>
  </div>
</div>

<h2>Equity Curve</h2>
<div class="chart-container"><canvas id="equityChart"></canvas></div>

<h2>Drawdown</h2>
<div class="chart-container"><canvas id="ddChart"></canvas></div>

<h2>Per-Coin P&L</h2>
<div class="chart-container"><canvas id="coinChart"></canvas></div>

<h2>Monthly Returns</h2>
<div class="chart-container"><canvas id="monthChart"></canvas></div>

<h2>Trade Log (last 100)</h2>
<div class="table-wrap">
<table>
<thead><tr>
  <th>Date</th><th>Coin</th><th>Action</th><th>Long Ex</th><th>Short Ex</th>
  <th>Size</th><th>Spread</th><th>Fees</th><th>Reason</th>
</tr></thead>
<tbody>{trade_rows}</tbody>
</table>
</div>

<div class="footer">Backtest by Funding Rate Arbitrage Bot</div>

<script>
const chartOpts = {{
  responsive: true,
  plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#484f58', maxTicksLimit: 12 }}, grid: {{ color: '#21262d' }} }},
    y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
  }}
}};

// Equity
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(equity_labels)},
    datasets: [{{
      label: 'Equity ($)',
      data: {json.dumps(equity_values)},
      borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)',
      fill: true, pointRadius: 0, borderWidth: 2,
    }}]
  }},
  options: {{ ...chartOpts, plugins: {{ ...chartOpts.plugins,
    annotation: {{ }} }} }}
}});

// Drawdown
new Chart(document.getElementById('ddChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(equity_labels)},
    datasets: [{{
      label: 'Drawdown (%)',
      data: {json.dumps(dd_values)},
      borderColor: '#ff6b6b', backgroundColor: 'rgba(255,107,107,0.1)',
      fill: true, pointRadius: 0, borderWidth: 2,
    }}]
  }},
  options: chartOpts,
}});

// Per-coin
new Chart(document.getElementById('coinChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(coin_labels)},
    datasets: [{{
      label: 'P&L ($)',
      data: {json.dumps(coin_values)},
      backgroundColor: {json.dumps(coin_colors)},
    }}]
  }},
  options: chartOpts,
}});

// Monthly
new Chart(document.getElementById('monthChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(month_labels)},
    datasets: [{{
      label: 'Return (%)',
      data: {json.dumps(month_values)},
      backgroundColor: {json.dumps(month_colors)},
    }}]
  }},
  options: chartOpts,
}});
</script>
</body>
</html>"""

    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Report saved to {report_path}")

    # Also save trades CSV
    trades_csv = output_dir / "trades.csv"
    with open(trades_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "coin", "side", "long_exchange", "short_exchange",
                     "size_usd", "spread", "fees_usd", "reason"])
        for t in result.trades:
            dt = datetime.fromtimestamp(t.timestamp_ms / 1000, tz=timezone.utc)
            w.writerow([dt.isoformat(), t.coin, t.side, t.long_exchange,
                        t.short_exchange, t.size_usd, t.spread, t.fees_usd, t.reason])

    # Save equity CSV
    eq_csv = output_dir / "equity_curve.csv"
    with open(eq_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "equity", "drawdown_pct", "positions"])
        for e in result.equity_curve:
            dt = datetime.fromtimestamp(e["ts_ms"] / 1000, tz=timezone.utc)
            w.writerow([dt.isoformat(), e["equity"], e["drawdown_pct"], e["positions"]])


# ── Main ───────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 60)
    logger.info("  FUNDING RATE ARBITRAGE — 12-MONTH BACKTEST")
    logger.info("=" * 60)

    config = load_config()
    coins = config["coins"]["active"]

    # Phase 1: Fetch or load cached data
    if CACHE_CSV.exists():
        logger.info(f"Found cached data at {CACHE_CSV}")
        rates = load_cache()
    else:
        logger.info(f"No cache found. Fetching {BACKTEST_DAYS} days of history...")
        rates = await fetch_all_history(config, coins)
        if rates:
            save_cache(rates)
        else:
            logger.error("No data fetched!")
            return

    logger.info(f"Total data points: {len(rates)}")

    # Phase 2: Run backtest with optimized params
    # Override config for aggressive capital deployment
    strategy_config = dict(config["strategy"])
    strategy_config["min_spread_annualized_pct"] = 10.0  # Lower threshold for more trades
    strategy_config["slippage_pct_per_leg"] = 0.02       # Tighter with limit orders

    risk_config = dict(config["risk"])
    risk_config["max_position_usd"] = 15000.0            # Bigger per-position
    risk_config["max_exposure_per_coin_usd"] = 15000.0   # More per coin
    risk_config["max_exposure_per_exchange_usd"] = 50000.0  # Allow heavy exchange usage
    risk_config["max_leverage"] = 2.0                        # 2x leverage — best risk/reward

    logger.info("\nRunning backtest simulation...")
    logger.info(f"  Strategy: min_spread={strategy_config['min_spread_annualized_pct']}%, "
                f"slippage={strategy_config['slippage_pct_per_leg']}%")
    logger.info(f"  Risk: max_pos=${risk_config['max_position_usd']:,.0f}, "
                f"max_coin=${risk_config['max_exposure_per_coin_usd']:,.0f}, "
                f"max_exchange=${risk_config['max_exposure_per_exchange_usd']:,.0f}")
    t0 = time.time()
    result = run_backtest(
        rates,
        strategy_config=strategy_config,
        risk_config=risk_config,
        start_capital=START_CAPITAL,
    )
    elapsed = time.time() - t0
    logger.info(f"Backtest completed in {elapsed:.1f}s")

    # Phase 3: Generate report
    logger.info("\nGenerating report...")
    generate_report(result, RESULTS_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period:          {BACKTEST_DAYS} days")
    print(f"  Start capital:   ${START_CAPITAL:,.0f}")
    eq = result.equity_curve[-1]["equity"] if result.equity_curve else START_CAPITAL
    print(f"  Final equity:    ${eq:,.0f}")
    print(f"  Total return:    {result.total_return_pct:+.1f}%")
    print(f"  Max drawdown:    -{result.max_drawdown_pct:.1f}%")
    print(f"  Sharpe ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Total trades:    {result.total_trades}")
    print(f"  Win rate:        {result.win_rate:.0f}%")
    print(f"  Avg hold time:   {result.avg_holding_hours:.0f}h")
    print(f"  Total fees:      ${result.total_fees_usd:,.0f}")
    print(f"  Funding P&L:     ${result.total_funding_pnl:,.0f}")
    print(f"\n  Top coins:")
    for coin, pnl in sorted(result.per_coin_pnl.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {coin:10s} ${pnl:>+10,.0f}")
    print(f"\n  Report: {RESULTS_DIR / 'report.html'}")
    print("=" * 60)

    # Open in browser
    import webbrowser
    webbrowser.open(str(RESULTS_DIR / "report.html"))


if __name__ == "__main__":
    asyncio.run(main())
