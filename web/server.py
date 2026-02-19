"""
Web dashboard server — runs alongside the bot on port 8080.
Serves a live HTML dashboard + JSON API endpoints.
"""

import json
import time
import logging
from collections import deque
from pathlib import Path
from aiohttp import web

logger = logging.getLogger(__name__)

# Shared state — populated by bot.py on each cycle
_state = {
    "mode": "shadow",
    "cycle": 0,
    "last_update": None,
    "next_poll_at": None,
    "poll_interval": 3600,
    "exchanges": [],
    "rates": {},       # {coin: {exchange: annualized_pct}}
    "spreads": {},     # {coin: spread_pct}
    "signals": [],     # list of signal dicts (new strategy)
    "signals_old": [],  # list of signal dicts (old strategy for comparison)
    "positions": [],   # list of position dicts
    "started_at": None,
}

# History buffers (kept in memory, max 168 = 7 days at 1h intervals)
MAX_HISTORY = 168
_spread_history = deque(maxlen=MAX_HISTORY)   # [{ts, spreads: {coin: pct}}]
_signal_history = deque(maxlen=MAX_HISTORY)   # [{ts, cycle, signals: [...]}]
_pnl_history = deque(maxlen=MAX_HISTORY)      # [{ts, cycle, positions: [...], total_pnl}]
_rate_history = deque(maxlen=MAX_HISTORY)      # [{ts, rates: {coin: {ex: pct}}}]

STATIC_DIR = Path(__file__).parent / "static"


def _compute_position_pnl(positions: list, rates: dict) -> list:
    """Estimate P&L for shadow positions based on funding accrual."""
    enriched = []
    for p in positions:
        coin = p["coin"]
        exchange = p["exchange"]
        side = p["side"]
        size_usd = p["size_usd"]
        age_hours = p["age_hours"]
        entry_spread = p["entry_spread"]

        # Current rate on this exchange
        current_rate = 0.0
        if coin in rates and exchange in rates[coin]:
            r = rates[coin][exchange]
            if r is not None:
                current_rate = r

        # Funding P&L estimate: if we're short and rate is positive,
        # we receive funding. If long and rate is negative, we receive funding.
        # rate is annualized %, convert to hourly accrual
        hourly_rate = current_rate / (365 * 24) / 100  # as decimal per hour
        if side == "short":
            # Short receives funding when rate > 0
            funding_pnl = size_usd * hourly_rate * age_hours
        else:
            # Long pays funding when rate > 0
            funding_pnl = -size_usd * hourly_rate * age_hours

        enriched.append({
            **p,
            "current_rate": round(current_rate, 2),
            "funding_pnl_usd": round(funding_pnl, 2),
        })
    return enriched


def update_state(
    mode: str,
    cycle: int,
    cache: dict,
    signals: list,
    positions: list,
    exchanges: list,
    poll_interval: int,
    signals_old: list = None,
):
    """Called by bot.py after each cycle to push data to the web layer."""
    now = time.time()
    _state["mode"] = mode
    _state["cycle"] = cycle
    _state["last_update"] = now
    _state["next_poll_at"] = now + poll_interval
    _state["poll_interval"] = poll_interval
    _state["exchanges"] = exchanges
    if _state["started_at"] is None:
        _state["started_at"] = now

    rates = {}
    for (coin, ex), fr in cache.items():
        if coin not in rates:
            rates[coin] = {}
        rates[coin][ex] = fr.rate_annualized

    spreads = {}
    for coin, ex_rates in rates.items():
        vals = [v for v in ex_rates.values() if v is not None]
        if len(vals) >= 2:
            spreads[coin] = round(max(vals) - min(vals), 2)
        else:
            spreads[coin] = 0.0

    _state["rates"] = rates
    _state["spreads"] = spreads

    sig_list = [
        {
            "coin": s.coin,
            "long_exchange": s.long_exchange,
            "short_exchange": s.short_exchange,
            "long_rate": round(s.long_rate, 2),
            "short_rate": round(s.short_rate, 2),
            "spread": round(s.spread, 1),
            "z_score": round(s.z_score, 2),
            "net_return": round(s.net_return_pct, 3),
            "confidence": s.confidence,
            "trend": getattr(s, "spread_trend", "unknown"),
            "avg_8h": round(getattr(s, "avg_spread_8h", 0), 1),
            "avg_24h": round(getattr(s, "avg_spread_24h", 0), 1),
        }
        for s in signals
    ]
    _state["signals"] = sig_list

    # Old strategy signals (for comparison)
    sig_old_list = []
    if signals_old:
        sig_old_list = [
            {
                "coin": s.coin,
                "long_exchange": s.long_exchange,
                "short_exchange": s.short_exchange,
                "long_rate": round(s.long_rate, 2),
                "short_rate": round(s.short_rate, 2),
                "spread": round(s.spread, 1),
                "z_score": round(s.z_score, 2),
                "net_return": round(s.net_return_pct, 3),
                "confidence": s.confidence,
                "trend": getattr(s, "spread_trend", "unknown"),
                "avg_8h": round(getattr(s, "avg_spread_8h", 0), 1),
                "avg_24h": round(getattr(s, "avg_spread_24h", 0), 1),
            }
            for s in signals_old
        ]
    _state["signals_old"] = sig_old_list

    # Enrich positions with P&L
    enriched_positions = _compute_position_pnl(positions, rates)
    _state["positions"] = enriched_positions

    # --- Record history ---
    ts_iso = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))

    _spread_history.append({"ts": ts_iso, "spreads": dict(spreads)})
    _rate_history.append({"ts": ts_iso, "rates": {
        coin: {ex: round(r, 2) if r is not None else None for ex, r in ex_rates.items()}
        for coin, ex_rates in rates.items()
    }})
    _signal_history.append({"ts": ts_iso, "cycle": cycle, "signals": sig_list})

    total_pnl = sum(p.get("funding_pnl_usd", 0) for p in enriched_positions)
    _pnl_history.append({
        "ts": ts_iso, "cycle": cycle,
        "total_pnl": round(total_pnl, 2),
        "position_count": len(enriched_positions),
        "positions": [
            {"coin": p["coin"], "exchange": p["exchange"],
             "side": p["side"], "pnl": p.get("funding_pnl_usd", 0)}
            for p in enriched_positions
        ],
    })


async def handle_index(request):
    html_path = STATIC_DIR / "index.html"
    return web.FileResponse(html_path)


async def handle_api_state(request):
    data = dict(_state)
    data["server_time"] = time.time()
    # Add summary stats
    total_pnl = sum(p.get("funding_pnl_usd", 0) for p in _state.get("positions", []))
    total_exposure = sum(p.get("size_usd", 0) for p in _state.get("positions", []))
    data["total_pnl_usd"] = round(total_pnl, 2)
    data["total_exposure_usd"] = round(total_exposure, 2)
    data["uptime_hours"] = round((time.time() - (_state["started_at"] or time.time())) / 3600, 1)
    return web.json_response(data)


async def handle_api_rates(request):
    return web.json_response(_state["rates"])


async def handle_api_signals(request):
    return web.json_response({
        "new": _state["signals"],
        "old": _state["signals_old"],
    })


async def handle_api_positions(request):
    return web.json_response(_state["positions"])


async def handle_api_history(request):
    """Return full history for charts."""
    return web.json_response({
        "spread_history": list(_spread_history),
        "pnl_history": list(_pnl_history),
        "signal_history": list(_signal_history),
    })


async def handle_api_coin_history(request):
    """Return spread history for a specific coin."""
    coin = request.match_info.get("coin", "").upper()
    history = []
    for entry in _spread_history:
        sp = entry["spreads"].get(coin)
        if sp is not None:
            history.append({"ts": entry["ts"], "spread": sp})
    # Also get per-exchange rates
    rate_history = []
    for entry in _rate_history:
        if coin in entry["rates"]:
            rate_history.append({"ts": entry["ts"], **entry["rates"][coin]})
    return web.json_response({
        "coin": coin,
        "spread_history": history,
        "rate_history": rate_history,
    })


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_api_state)
    app.router.add_get("/api/rates", handle_api_rates)
    app.router.add_get("/api/signals", handle_api_signals)
    app.router.add_get("/api/positions", handle_api_positions)
    app.router.add_get("/api/history", handle_api_history)
    app.router.add_get("/api/coin/{coin}", handle_api_coin_history)
    app.router.add_static("/static/", STATIC_DIR, name="static")
    return app


async def start_web_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the web server as a background task."""
    app = create_app()
    runner = web.AppRunner(app, access_log=None)  # suppress access logs
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"Web dashboard running at http://localhost:{port}")
    return runner
