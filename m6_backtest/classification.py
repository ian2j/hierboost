"""Sector/category classification for all 100 M6 assets -- the relevance-prior input.

Equities use yfinance's own GICS-style sector. ETFs don't carry a "sector" field, so
each is mapped by hand from its category/description to the *same* sector taxonomy
used for equities, so a stock and an ETF can be compared on equal footing (e.g. XLK
and AMAT both resolve to "Technology"). Three tickers (DRE, RE, WRK) fail a live
yfinance lookup today because they're no longer actively quoted under that identifier
(DRE: acquired by Prologis Oct 2022, see project-m6-backtest memory; RE: Everest Group
rebranded; WRK: merged into Smurfit WestRock in 2024) -- classified from general
knowledge instead, clearly marked as such.
"""

EQUITY_SECTOR_OVERRIDE = {
    "DRE": "Real Estate",           # Duke Realty, REIT -- delisted mid-competition
    "RE": "Financial Services",     # Everest Re Group, reinsurance
    "WRK": "Basic Materials",       # WestRock, containers & packaging (GICS Materials)
}

# ETF ticker -> sector-equivalent bucket, assigned from each fund's own category/
# longName (see asset_info.json). Sector SPDRs map 1:1 to the matching GICS sector
# name so they align exactly with equities in that sector.
ETF_SECTOR = {
    "XLB": "Basic Materials", "XLC": "Communication Services", "XLE": "Energy",
    "XLF": "Financial Services", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Consumer Defensive", "XLU": "Utilities", "XLV": "Healthcare",
    "XLY": "Consumer Cyclical",
    "IXN": "Technology", "ICLN": "Energy", "IGF": "Industrials", "REET": "Real Estate",
    "IVV": "Broad Market", "IWM": "Broad Market",
    "TLT": "Fixed Income", "IEF": "Fixed Income", "SHY": "Fixed Income",
    "LQD": "Fixed Income", "HYG": "Fixed Income", "HIGH.L": "Fixed Income",
    "IEAA.L": "Fixed Income", "SEGA.L": "Fixed Income", "JPEA.L": "Fixed Income",
    "GSG": "Commodities", "IAU": "Commodities", "SLV": "Commodities",
    "VXX": "Volatility",
    "EWA": "International", "EWC": "International", "EWG": "International",
    "EWH": "International", "EWJ": "International", "EWL": "International",
    "EWQ": "International", "EWT": "International", "EWU": "International",
    "EWY": "International", "EWZ": "International", "MCHI": "International",
    "INDA": "International", "IEMG": "International", "IEUS": "International",
    "IEFM.L": "International", "IEVL.L": "International", "MVEU.L": "International",
    "IUMO.L": "Broad Market", "IUVL.L": "Broad Market", "SPMV.L": "Broad Market",
}

# Sector adjacency for relevance scoring (mirrors finance_cross_sectional.py's
# SECTOR_TABLE, extended to cover every sector appearing in the M6 universe).
SECTOR_ADJACENCY = {
    "Technology": ["Communication Services"],
    "Communication Services": ["Technology"],
    "Financial Services": ["Real Estate"],
    "Healthcare": [],
    "Consumer Cyclical": ["Consumer Defensive"],
    "Consumer Defensive": ["Consumer Cyclical"],
    "Energy": ["Basic Materials"],
    "Industrials": ["Basic Materials"],
    "Basic Materials": ["Industrials", "Energy"],
    "Utilities": ["Fixed Income"],
    "Real Estate": ["Financial Services"],
    "Broad Market": [],
    "Fixed Income": [],
    "Commodities": [],
    "Volatility": [],
    "International": [],
}


def classify(ticker, asset_info):
    """asset_info: dict as saved in asset_info.json (per-ticker yfinance metadata)."""
    if ticker in EQUITY_SECTOR_OVERRIDE:
        return EQUITY_SECTOR_OVERRIDE[ticker]
    info = asset_info.get(ticker, {})
    if info.get("quoteType") == "EQUITY" and info.get("sector"):
        return info["sector"]
    if ticker in ETF_SECTOR:
        return ETF_SECTOR[ticker]
    return "Broad Market"  # conservative fallback, shouldn't be hit for the M6 100


def relevance_for_pair(target_sector, other_sector):
    """Relevance of an asset in other_sector to a target asset in target_sector."""
    if other_sector == target_sector:
        return 5.0
    if other_sector in SECTOR_ADJACENCY.get(target_sector, []):
        return 3.0
    if other_sector == "Broad Market":
        return 2.0
    if other_sector in ("Fixed Income", "Commodities", "Volatility"):
        return 0.05
    if other_sector == "International":
        return 1.0
    return 0.5


if __name__ == "__main__":
    import json
    asset_info = json.load(open("data/asset_info.json"))
    symbols = json.load(open("data/symbols.json"))
    sectors = {s: classify(s, asset_info) for s in symbols}
    import collections
    by_sector = collections.defaultdict(list)
    for s, sec in sectors.items():
        by_sector[sec].append(s)
    for sec, members in sorted(by_sector.items()):
        print(f"{sec} ({len(members)}): {members}")
    json.dump(sectors, open("data/sectors.json", "w"), indent=1)
