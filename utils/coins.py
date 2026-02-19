"""
Coin registry and exchange symbol mapping.
Extracted from top10_crypto_funding.py — single source of truth.
"""

# Top 10 crypto by market cap + meme coins
# label -> {exchange_key: exchange_symbol}
COINS = {
    # === Top 10 Crypto ===
    "BTC":  {"hl": "BTC",  "aevo": "BTC-PERP",  "dydx": "BTC-USD",  "drift": "BTC-PERP"},
    "ETH":  {"hl": "ETH",  "aevo": "ETH-PERP",  "dydx": "ETH-USD",  "drift": "ETH-PERP"},
    "XRP":  {"hl": "XRP",  "aevo": "XRP-PERP",  "dydx": "XRP-USD",  "drift": "XRP-PERP"},
    "BNB":  {"hl": "BNB",  "aevo": "BNB-PERP",  "dydx": "BNB-USD",  "drift": "BNB-PERP"},
    "SOL":  {"hl": "SOL",  "aevo": "SOL-PERP",  "dydx": "SOL-USD",  "drift": "SOL-PERP"},
    "ADA":  {"hl": "ADA",  "aevo": "ADA-PERP",  "dydx": "ADA-USD",  "drift": "ADA-PERP"},
    "LINK": {"hl": "LINK", "aevo": "LINK-PERP", "dydx": "LINK-USD", "drift": "LINK-PERP"},
    "AVAX": {"hl": "AVAX", "aevo": "AVAX-PERP", "dydx": "AVAX-USD", "drift": "AVAX-PERP"},
    "SUI":  {"hl": "SUI",  "aevo": "SUI-PERP",  "dydx": "SUI-USD",  "drift": "SUI-PERP"},
    "DOT":  {"hl": "DOT",  "aevo": "DOT-PERP",  "dydx": "DOT-USD",  "drift": None},

    # === Meme Coins ===
    "HYPE":     {"hl": "HYPE",     "aevo": "HYPE-PERP",     "dydx": "HYPE-USD",     "drift": "HYPE-PERP"},
    "FARTCOIN": {"hl": "FARTCOIN", "aevo": "FARTCOIN-PERP", "dydx": None,            "drift": "FARTCOIN-PERP"},
    "DOGE":     {"hl": "DOGE",     "aevo": "DOGE-PERP",     "dydx": "DOGE-USD",      "drift": "DOGE-PERP"},
    "TRUMP":    {"hl": "TRUMP",    "aevo": "TRUMP-PERP",    "dydx": "TRUMP-USD",     "drift": "TRUMP-PERP"},
    "PENGU":    {"hl": "PENGU",    "aevo": "PENGU-PERP",    "dydx": "PENGU-USD",     "drift": "PENGU-PERP"},
    "SPX":      {"hl": "SPX",      "aevo": "SPX-PERP",      "dydx": None,            "drift": None},
    "MOODENG":  {"hl": "MOODENG",  "aevo": "MOODENG-PERP",  "dydx": "MOODENG-USD",   "drift": None},
    "WIF":      {"hl": "WIF",      "aevo": "WIF-PERP",      "dydx": "WIF-USD",       "drift": "WIF-PERP"},
    "POPCAT":   {"hl": "POPCAT",   "aevo": "POPCAT-PERP",   "dydx": "POPCAT-USD",    "drift": "POPCAT-PERP"},
    "PNUT":     {"hl": "PNUT",     "aevo": "PNUT-PERP",     "dydx": "PNUT-USD",      "drift": None},

    # === TradFi (HL uses xyz: prefix, Aevo has some) ===
    "NVDA":     {"hl": "xyz:NVDA",    "aevo": "NVDA-PERP",    "dydx": None, "drift": None},
    "TSLA":     {"hl": "xyz:TSLA",    "aevo": "TSLA-PERP",    "dydx": None, "drift": None},
    "AMD":      {"hl": "xyz:AMD",     "aevo": "AMD-PERP",     "dydx": None, "drift": None},
    "INTC":     {"hl": "xyz:INTC",    "aevo": "INTC-PERP",    "dydx": None, "drift": None},
    "GOLD":     {"hl": "xyz:GOLD",    "aevo": "XAU-PERP",     "dydx": None, "drift": None},
    "SILVER":   {"hl": "xyz:SILVER",  "aevo": "XAG-PERP",     "dydx": None, "drift": None},
    "COPPER":   {"hl": "xyz:COPPER",  "aevo": "COPPER-PERP",  "dydx": None, "drift": None},
    "PLATINUM": {"hl": "xyz:PLATINUM","aevo": "PLATINUM-PERP", "dydx": None, "drift": None},
    "AAPL":     {"hl": "xyz:AAPL",    "aevo": None,           "dydx": None, "drift": None},
    "GOOGL":    {"hl": "xyz:GOOGL",   "aevo": None,           "dydx": None, "drift": None},
    "MSFT":     {"hl": "xyz:MSFT",    "aevo": None,           "dydx": None, "drift": None},
    "META":     {"hl": "xyz:META",    "aevo": None,           "dydx": None, "drift": None},
}

# Annualization multipliers: funding_periods_per_day * 365
# HL: every 8h (3/day); Aevo/dYdX/Drift: every 1h (24/day)
ANN_MULT = {
    "hyperliquid": 3 * 365,
    "aevo":        24 * 365,
    "dydx":        24 * 365,
    "drift":       24 * 365,
}

# Exchange data quality trust levels
# LOW = don't trade signals from this exchange until validated
TRUST_LEVEL = {
    "hyperliquid": "HIGH",
    "dydx":        "HIGH",
    "aevo":        "MEDIUM",
    "drift":       "LOW",
}

# Exchange key mapping (config key -> COINS dict key)
EXCHANGE_KEYS = {
    "hyperliquid": "hl",
    "aevo":        "aevo",
    "dydx":        "dydx",
    "drift":       "drift",
}


def get_exchange_symbol(coin: str, exchange: str) -> str | None:
    """Return the exchange-specific symbol for a coin, or None if not listed."""
    key = EXCHANGE_KEYS.get(exchange)
    if not key:
        return None
    return COINS.get(coin, {}).get(key)
