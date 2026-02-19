"""
Funding Rate Arbitrage Bot — Main Entry Point.
Polls funding rates from multiple exchanges, generates arbitrage signals,
and executes trades (shadow or live).

Usage:
    py arbitrage_system/bot.py
    py -m arbitrage_system.bot
"""

import asyncio
import csv
import os
import sys
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from arbitrage_system.connectors.base import Position
from arbitrage_system.connectors.hyperliquid_conn import HyperliquidConnector
from arbitrage_system.connectors.dydx_conn import DydxConnector
from arbitrage_system.connectors.aevo_conn import AevoConnector
from arbitrage_system.connectors.drift_conn import DriftConnector
from arbitrage_system.safety.shadow import ShadowExchange
from arbitrage_system.core.strategy import StrategyEngine, ArbitrageSignal
from arbitrage_system.core.risk import RiskManager
from arbitrage_system.data.market_data import MarketDataManager
from arbitrage_system.web.server import start_web_server, update_state

logger = logging.getLogger("arb_bot")

BASE_DIR = Path(__file__).parent


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


def setup_logging(config: dict):
    log_dir = BASE_DIR / config["system"].get("log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, config["system"].get("log_level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_dir / "bot.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def build_connectors(config: dict) -> dict:
    connectors = {}
    ex = config["exchanges"]
    if ex.get("hyperliquid", {}).get("enabled"):
        connectors["hyperliquid"] = HyperliquidConnector(ex["hyperliquid"])
    if ex.get("dydx", {}).get("enabled"):
        connectors["dydx"] = DydxConnector(ex["dydx"])
    if ex.get("aevo", {}).get("enabled"):
        connectors["aevo"] = AevoConnector(ex["aevo"])
    if ex.get("drift", {}).get("enabled"):
        connectors["drift"] = DriftConnector(ex["drift"])
    return connectors


# ─── Dashboard ───────────────────────────────────────────────────────

def print_banner(mode: str, coins: int, exchanges: list, interval: int):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print(f"  FUNDING RATE ARBITRAGE BOT")
    print(f"  Mode: {mode.upper()}  |  Coins: {coins}  |  "
          f"Exchanges: {', '.join(exchanges)}")
    print(f"  Poll interval: {interval // 60}min  |  Started: {now}")
    print("=" * 70)


def print_dashboard(
    cache: dict,
    signals: list[ArbitrageSignal],
    cycle: int,
    exchanges: list[str],
):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'-' * 70}")
    print(f"  CYCLE #{cycle}  |  {now}  |  {len(cache)} rates cached")
    print(f"{'-' * 70}")

    # Build spread table
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for (coin, ex), fr in cache.items():
        if fr.rate_annualized is not None:
            matrix[coin][ex] = fr.rate_annualized

    # Header
    ex_list = sorted(exchanges)
    header = f"  {'COIN':10s}"
    for ex in ex_list:
        header += f" | {ex:>12s}"
    header += f" | {'SPREAD':>8s}"
    print(header)
    print("  " + "-" * (14 + len(ex_list) * 16 + 12))

    # Sort by spread descending
    spreads = {}
    for coin, rates in matrix.items():
        if len(rates) >= 2:
            spreads[coin] = max(rates.values()) - min(rates.values())
        else:
            spreads[coin] = 0.0

    for coin in sorted(spreads, key=spreads.get, reverse=True):
        rates = matrix[coin]
        row = f"  {coin:10s}"
        vals = []
        for ex in ex_list:
            val = rates.get(ex)
            if val is not None:
                # Color hint: positive=green area, negative=red area
                row += f" | {val:>+11.1f}%"
                vals.append(val)
            else:
                row += f" | {'---':>12s}"

        sp = spreads[coin]
        if sp > 20:
            row += f" | {sp:>7.1f}%!"
        elif sp > 5:
            row += f" | {sp:>7.1f}% "
        else:
            row += f" | {sp:>7.1f}  "
        print(row)

    # Signals
    if signals:
        print(f"\n  SIGNALS ({len(signals)}):")
        for s in signals:
            conf_mark = {"HIGH": "***", "MEDIUM": "** ", "LOW": "*  "}
            print(f"    {conf_mark.get(s.confidence, '   ')} {s.coin:8s} "
                  f"long:{s.long_exchange}({s.long_rate:+.1f}%) "
                  f"short:{s.short_exchange}({s.short_rate:+.1f}%) "
                  f"spread:{s.spread:.1f}% net:{s.net_return_pct:.3f}%")
    else:
        print(f"\n  No signals above threshold")

    next_poll = datetime.now().strftime("%H:%M:%S")
    print(f"\n  Next poll in {config_ref['system']['poll_interval_seconds'] // 60} min...")


def print_positions(risk: "RiskManager"):
    """Show current open positions tracked by the risk manager."""
    if not risk._positions:
        return
    has_any = any(bool(exs) for exs in risk._positions.values())
    if not has_any:
        return
    print(f"\n  OPEN POSITIONS:")
    for coin, ex_positions in sorted(risk._positions.items()):
        for ex, (pos, open_time, entry_spread) in ex_positions.items():
            age_h = (time.time() - open_time) / 3600
            size_usd = pos.size * pos.entry_price
            print(f"    {coin:8s} {pos.side:5s} on {ex:12s} "
                  f"size=${size_usd:,.0f} entry@{pos.entry_price:.4f} "
                  f"age={age_h:.1f}h spread={entry_spread:.1f}%")


# ─── CSV Report ──────────────────────────────────────────────────────

def save_cycle_report(cache: dict, signals: list, cycle: int):
    """Append current cycle data to CSV reports."""
    report_dir = BASE_DIR / "reports"
    report_dir.mkdir(exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Rates CSV
    rates_file = report_dir / "funding_rates_log.csv"
    write_header = not rates_file.exists()
    with open(rates_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "cycle", "coin", "exchange",
                         "rate_raw", "rate_annualized_pct"])
        for (coin, ex), fr in sorted(cache.items()):
            w.writerow([now_iso, cycle, coin, ex, fr.rate,
                         fr.rate_annualized])

    # Signals CSV
    if signals:
        signals_file = report_dir / "signals_log.csv"
        write_header = not signals_file.exists()
        with open(signals_file, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "cycle", "coin", "long_exchange",
                             "short_exchange", "long_rate", "short_rate",
                             "spread", "z_score", "net_return", "confidence"])
            for s in signals:
                w.writerow([now_iso, cycle, s.coin, s.long_exchange,
                             s.short_exchange, s.long_rate, s.short_rate,
                             s.spread, s.z_score, s.net_return_pct,
                             s.confidence])


# ─── Main Loop ───────────────────────────────────────────────────────

config_ref = {}  # global ref for dashboard

async def on_market_update(
    cache: dict,
    strategy: StrategyEngine,
    strategy_old: StrategyEngine,
    risk: RiskManager,
    connectors: dict,
    config: dict,
    cycle_counter: list,
):
    cycle_counter[0] += 1
    cycle = cycle_counter[0]

    # Build spread matrix
    spread_matrix = {}
    for (coin, exchange), fr in cache.items():
        if coin not in spread_matrix:
            spread_matrix[coin] = {}
        if fr.rate_annualized is not None:
            spread_matrix[coin][exchange] = fr.rate_annualized

    signals = strategy.generate_signals(spread_matrix)
    signals_old = strategy_old.generate_signals(spread_matrix)

    # Print dashboard
    print_dashboard(cache, signals, cycle, list(connectors.keys()))
    print_positions(risk)

    # Save CSV report
    save_cycle_report(cache, signals, cycle)

    mode = config["system"]["mode"]

    # --- Update rolling spreads + Check existing positions for exit ---
    for coin, ex_positions in list(risk._positions.items()):
        if coin not in spread_matrix or len(spread_matrix[coin]) < 2:
            continue
        rates = spread_matrix[coin]
        sorted_rates = sorted(rates.items(), key=lambda x: x[1])
        current_spread = sorted_rates[-1][1] - sorted_rates[0][1]

        # Update rolling spread history for this coin
        risk.update_spread(coin, current_spread)

        for ex, (pos, open_time, entry_spread) in list(ex_positions.items()):
            # Build a dummy signal for exit check
            # Include current rates for flip detection
            long_rate = spread_matrix[coin].get(sorted_rates[0][0], 0)
            short_rate = spread_matrix[coin].get(sorted_rates[-1][0], 0)
            dummy = ArbitrageSignal(
                coin=coin, long_exchange=ex, short_exchange="",
                long_rate=long_rate, short_rate=short_rate,
                spread=entry_spread,
                z_score=0, entry_cost_pct=0, net_return_pct=0,
                confidence="MEDIUM", timestamp=0,
            )
            should_exit, reason = risk.check_exit_conditions(dummy, current_spread)
            if should_exit:
                logger.info(f"EXIT | {coin}/{ex}: {reason}")
                # Close the position
                try:
                    conn = connectors.get(ex)
                    if conn:
                        close_side = "sell" if pos.side == "long" else "buy"
                        price = await conn.get_mark_price(coin)
                        await conn.place_order(coin, close_side, pos.size, price)
                        risk.remove_position(coin, ex)
                        logger.info(f"  -> CLOSED {pos.side} {pos.size:.6f} {coin} on {ex}")
                except Exception as e:
                    logger.error(f"  -> Close error {coin}/{ex}: {e}")

    # --- Open new positions from signals ---
    for signal in signals:
        logger.info(f"SIGNAL | {signal}")

        if signal.confidence == "LOW":
            logger.info(f"  -> Skipping LOW confidence")
            continue

        # Skip if we already have a position for this coin
        if signal.coin in risk._positions and risk._positions[signal.coin]:
            logger.info(f"  -> SKIP: already have position in {signal.coin}")
            continue

        max_pos = config["risk"].get("max_position_usd", 10000.0)
        total_capital = 100_000.0 if mode == "shadow" else 0.0

        if mode == "live":
            try:
                balance = await connectors[signal.long_exchange].get_balance()
                total_capital = balance.get("total_usd", 0)
            except Exception as e:
                logger.warning(f"  -> Balance error: {e}")
                continue

        size_usd = strategy.compute_position_size(signal, total_capital, max_pos)
        if size_usd <= 0:
            continue

        allowed, reason = risk.can_open_position(signal, size_usd)
        if not allowed:
            logger.warning(f"  -> BLOCKED: {reason}")
            continue

        try:
            long_conn = connectors[signal.long_exchange]
            short_conn = connectors[signal.short_exchange]

            long_price = await long_conn.get_mark_price(signal.coin)
            short_price = await short_conn.get_mark_price(signal.coin)

            long_size = size_usd / long_price
            short_size = size_usd / short_price

            logger.info(
                f"  -> EXEC: long {long_size:.6f} {signal.coin} "
                f"@ {long_price:.2f} on {signal.long_exchange}, "
                f"short {short_size:.6f} @ {short_price:.2f} "
                f"on {signal.short_exchange} (${size_usd:.0f})"
            )

            long_r = await long_conn.place_order(
                signal.coin, "buy", long_size, long_price)
            short_r = await short_conn.place_order(
                signal.coin, "sell", short_size, short_price)

            logger.info(
                f"  -> FILLED: long={long_r.status} short={short_r.status}")

            # Register positions with risk manager
            if long_r.status == "filled":
                long_pos = Position(
                    coin=signal.coin, exchange=signal.long_exchange,
                    side="long", size=long_size,
                    entry_price=long_r.price, unrealized_pnl=0.0,
                    leverage=1.0,
                )
                risk.register_position(
                    signal.coin, signal.long_exchange, long_pos, signal.spread)

            if short_r.status == "filled":
                short_pos = Position(
                    coin=signal.coin, exchange=signal.short_exchange,
                    side="short", size=short_size,
                    entry_price=short_r.price, unrealized_pnl=0.0,
                    leverage=1.0,
                )
                risk.register_position(
                    signal.coin, signal.short_exchange, short_pos, signal.spread)

        except NotImplementedError as e:
            logger.warning(f"  -> Not implemented: {e}")
        except Exception as e:
            logger.error(f"  -> Exec error: {e}", exc_info=True)

    # Update web dashboard state (after all trades executed)
    positions_data = []
    for coin, ex_positions in risk._positions.items():
        for ex, (pos, open_time, entry_spread) in ex_positions.items():
            positions_data.append({
                "coin": coin,
                "exchange": ex,
                "side": pos.side,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "size_usd": pos.size * pos.entry_price,
                "age_hours": round((time.time() - open_time) / 3600, 1),
                "entry_spread": entry_spread,
            })
    update_state(
        mode=mode, cycle=cycle, cache=cache,
        signals=signals, signals_old=signals_old,
        positions=positions_data,
        exchanges=list(connectors.keys()),
        poll_interval=config["system"]["poll_interval_seconds"],
    )


async def main():
    global config_ref
    config = load_config()
    config_ref = config
    setup_logging(config)

    mode = config["system"]["mode"]
    active_coins = config["coins"]["active"]
    interval = config["system"]["poll_interval_seconds"]

    raw_connectors = build_connectors(config)
    if not raw_connectors:
        print("ERROR: No exchanges enabled!")
        return

    if mode == "shadow":
        shadow_dir = str(BASE_DIR / config["system"].get(
            "shadow_trades_dir", "shadow_trades"))
        connectors = {
            name: ShadowExchange(conn, shadow_dir)
            for name, conn in raw_connectors.items()
        }
    else:
        connectors = raw_connectors

    reports_dir = str(BASE_DIR / "reports")
    strategy = StrategyEngine(config["strategy"], reports_dir=reports_dir)

    # Old strategy for comparison (original params before backtest optimization)
    old_strategy_config = {
        "min_spread_annualized_pct": 5.0,
        "expected_holding_days": 7,
        "slippage_pct_per_leg": 0.05,
        "z_score_threshold": 1.5,
        "use_maker_orders": False,
    }
    strategy_old = StrategyEngine(old_strategy_config, reports_dir=reports_dir)

    risk = RiskManager(config["risk"])
    cycle_counter = [0]

    print_banner(mode, len(active_coins), list(connectors.keys()), interval)

    for name in connectors:
        logger.info(f"Connector: {name} enabled")
    if mode == "shadow":
        logger.info("SHADOW MODE: All trades are virtual")

    # Start web dashboard (PORT env var for mikr.us / Docker)
    web_port = int(os.environ.get("PORT", config["system"].get("web_port", 8080)))
    web_runner = await start_web_server(port=web_port)
    if os.environ.get("NO_BROWSER") != "1":
        import webbrowser
        webbrowser.open(f"http://localhost:{web_port}")

    market_data = MarketDataManager(
        connectors=connectors,
        coins=active_coins,
        poll_interval_seconds=interval,
    )

    try:
        await market_data.run_polling_loop(
            lambda cache: on_market_update(
                cache, strategy, strategy_old, risk, connectors, config, cycle_counter)
        )
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
    finally:
        await web_runner.cleanup()
        for conn in raw_connectors.values():
            if hasattr(conn, "close"):
                await conn.close()

        if mode == "shadow":
            print(f"\n{'=' * 70}")
            print("  SHADOW TRADING SUMMARY")
            print(f"{'=' * 70}")
            for name, conn in connectors.items():
                if isinstance(conn, ShadowExchange):
                    s = conn.get_summary()
                    print(f"  {name}: {s['total_trades']} trades, "
                          f"{len(s['active_positions'])} open, "
                          f"log: {s['log_file']}")

        print(f"\n  Reports saved to: {BASE_DIR / 'reports'}/")
        print(f"  Logs saved to: {BASE_DIR / 'logs'}/")
        print("  Bot stopped.\n")


if __name__ == "__main__":
    asyncio.run(main())
