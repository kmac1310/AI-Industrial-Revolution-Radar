#!/usr/bin/env python3
"""
1440TIME Heat Engine
--------------------
Reads data/universe.json, pulls 1 year of daily prices, scores every stock
with the 1440TIME Decision Matrix, and writes data/heat.json for the dashboard.

  Relative Volume ........ 25%
  Price Momentum ......... 20%
  20 / 50 / 200 DMA ...... 20%
  RSI .................... 10%
  MACD ................... 10%
  Sector Relative Strength 10%
  Catalyst (manual) ......  5%

Extension is a separate timing dial (how far price has already run).

Usage:
  python scripts/heat_engine.py          # live data (Yahoo Finance via yfinance)
  python scripts/heat_engine.py --demo   # synthetic data, for testing the page
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "universe.json"
OUTPUT = ROOT / "data" / "heat.json"

WEIGHTS = {
    "relvol": 0.25, "momentum": 0.20, "trend": 0.20,
    "rsi": 0.10, "macd": 0.10, "sector_rs": 0.10, "catalyst": 0.05,
}


def clip01(x):
    return float(max(0.0, min(1.0, x)))


def scale(x, lo, hi):
    """Map x from [lo, hi] to 0-100, clipped."""
    return 100.0 * clip01((x - lo) / (hi - lo))


# ---------------------------------------------------------------- indicators
def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def macd_hist(close):
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def ret(close, days):
    if len(close) <= days:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


# ---------------------------------------------------------------- scoring
def score_stock(df):
    """df: DataFrame with Close and Volume columns, oldest first. Returns metrics + sub-scores."""
    df = df.dropna(subset=["Close"])
    if len(df) < 210:
        raise ValueError(f"only {len(df)} bars — need ~210 for a 200-day average")
    close, vol = df["Close"], df["Volume"].fillna(0)
    price = float(close.iloc[-1])

    # Relative volume: last 5 sessions vs the 50 before them (smooths one-day spikes)
    recent, base = vol.iloc[-5:].mean(), vol.iloc[-55:-5].mean()
    relvol = float(recent / base) if base > 0 else 1.0
    s_relvol = scale(relvol, 0.5, 1.5)  # 0.5x→0, 1.0x→50, 1.5x+→100

    # Momentum: 1D / 5D / 1M blend
    r1, r5, r21 = ret(close, 1), ret(close, 5), ret(close, 21)
    s_mom = 0.2 * scale(r1, -0.03, 0.03) + 0.3 * scale(r5, -0.08, 0.08) + 0.5 * scale(r21, -0.20, 0.20)

    # Trend structure: above 20 / 50 / 200 DMA
    d20, d50, d200 = (float(close.rolling(n).mean().iloc[-1]) for n in (20, 50, 200))
    above = {"20": price > d20, "50": price > d50, "200": price > d200}
    s_trend = 25 * above["20"] + 35 * above["50"] + 40 * above["200"]

    # RSI: reward 50-70 and rising, penalize stretched or weak
    r = rsi(close)
    rsi_now, rsi_prev = float(r.iloc[-1]), float(r.iloc[-6])
    rising = rsi_now > rsi_prev
    if 50 <= rsi_now <= 70:
        s_rsi = 100 if rising else 75
    elif 70 < rsi_now <= 80:
        s_rsi = 55
    elif rsi_now > 80:
        s_rsi = 30
    elif 40 <= rsi_now < 50:
        s_rsi = 50 if rising else 35
    else:
        s_rsi = 15

    # MACD: fresh bullish cross > positive & expanding > positive & fading > turning up > negative
    h = macd_hist(close)
    h_now, h_prev = float(h.iloc[-1]), float(h.iloc[-2])
    fresh_cross = h_now > 0 and (h.iloc[-6:-1] <= 0).any()
    if fresh_cross:
        s_macd, macd_state = 100, "Fresh bullish cross"
    elif h_now > 0 and h_now >= h_prev:
        s_macd, macd_state = 85, "Bullish, expanding"
    elif h_now > 0:
        s_macd, macd_state = 60, "Bullish, fading"
    elif h_now >= h_prev:
        s_macd, macd_state = 35, "Bearish, improving"
    else:
        s_macd, macd_state = 10, "Bearish"

    # Extension (timing dial, not part of heat): distance above 200 DMA + RSI stretch
    pct200 = price / d200 - 1
    extension = 0.7 * scale(pct200, 0.0, 0.50) + 0.3 * scale(rsi_now, 50, 80)

    return {
        "price": round(price, 2),
        "chg1d": round(r1 * 100, 2), "chg5d": round(r5 * 100, 2), "chg1m": round(r21 * 100, 2),
        "relvol": round(relvol, 2), "rsi": round(rsi_now, 1), "rsi_rising": bool(rising),
        "macd": macd_state, "above": above, "pct200": round(pct200 * 100, 1),
        "r21": r21,
        "sub": {"relvol": s_relvol, "momentum": s_mom, "trend": s_trend, "rsi": s_rsi, "macd": s_macd},
        "extension": round(extension),
    }


def status(heat, ext):
    # Extension is checked first: a stretched stock is "extended" even if heat is middling.
    if ext >= 70 and heat >= 50:
        return "extended"
    if heat >= 70:
        return "hot"
    if heat >= 50 and ext < 50:
        return "heating"
    if heat < 40:
        return "cooling"
    return "watch"


def build(universe, prices):
    """prices: dict ticker -> DataFrame(Close, Volume). Returns heat.json payload."""
    bench_t = universe.get("benchmark", "QQQ")
    bench_r21 = ret(prices[bench_t]["Close"].dropna(), 21) if bench_t in prices else 0.0

    out_sectors, errors, as_of = {}, {}, None
    for key, sec in universe["sectors"].items():
        rows = []
        for st in sec["stocks"]:
            t = st["t"]
            try:
                if t not in prices:
                    raise ValueError("no price data returned")
                m = score_stock(prices[t])
                m["t"] = t
                m["catalyst"] = st.get("catalyst", 50)
                rows.append(m)
                last = prices[t]["Close"].dropna().index[-1]
                as_of = max(as_of, last) if as_of is not None else last
            except Exception as e:  # keep going; report the miss
                errors[t] = str(e)

        if not rows:
            continue
        # Sector relative strength: sector's avg 1M return vs benchmark
        sec_r21 = float(np.mean([r["r21"] for r in rows]))
        rs_excess = sec_r21 - bench_r21
        s_rs = scale(rs_excess, -0.10, 0.10)

        stocks_out = {}
        for r in rows:
            sub = dict(r["sub"], sector_rs=s_rs, catalyst=float(r["catalyst"]))
            heat = round(sum(WEIGHTS[k] * sub[k] for k in WEIGHTS))
            stocks_out[r["t"]] = {
                k: r[k] for k in ("price", "chg1d", "chg5d", "chg1m", "relvol", "rsi",
                                  "rsi_rising", "macd", "above", "pct200", "extension")
            }
            stocks_out[r["t"]].update(
                heat=heat,
                status=status(heat, r["extension"]),
                breakdown={k: round(v) for k, v in sub.items()},
            )

        heats = [v["heat"] for v in stocks_out.values()]
        exts = [v["extension"] for v in stocks_out.values()]
        sh, se = round(float(np.mean(heats))), round(float(np.mean(exts)))
        out_sectors[key] = {
            "heat": sh, "ext": se, "status": status(sh, se),
            "rs_vs_bench": round(rs_excess * 100, 1),
            "stocks": stocks_out,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(as_of.date()) if as_of is not None else None,
        "benchmark": bench_t,
        "benchmark_1m": round(bench_r21 * 100, 2),
        "weights": WEIGHTS,
        "sectors": out_sectors,
        "errors": errors,
    }


# ---------------------------------------------------------------- data
def fetch_live(tickers):
    import yfinance as yf
    raw = yf.download(tickers, period="14mo", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    prices = {}
    for t in tickers:
        try:
            df = raw[t][["Close", "Volume"]].dropna(subset=["Close"])
            if len(df):
                prices[t] = df
        except KeyError:
            pass
    return prices


def fetch_demo(tickers, seed=1440):
    """Synthetic random-walk prices so the page can be tested without market data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    prices = {}
    for t in tickers:
        drift = rng.normal(0.0008, 0.0012)
        rets = rng.normal(drift, 0.022, len(idx))
        close = 100 * np.exp(np.cumsum(rets))
        vol = rng.lognormal(15, 0.35, len(idx))
        vol[-5:] *= rng.uniform(0.7, 2.0)
        prices[t] = pd.DataFrame({"Close": close, "Volume": vol}, index=idx)
    return prices


def main():
    demo = "--demo" in sys.argv
    universe = json.loads(UNIVERSE.read_text())
    tickers = sorted({s["t"] for sec in universe["sectors"].values() for s in sec["stocks"]})
    tickers.append(universe.get("benchmark", "QQQ"))

    prices = fetch_demo(tickers) if demo else fetch_live(tickers)
    payload = build(universe, prices)
    payload["demo"] = demo
    n = sum(len(s["stocks"]) for s in payload["sectors"].values())
    if not demo and n == 0:
        sys.exit(f"No stocks scored — data source failed. Keeping previous heat.json. {payload['errors']}")
    OUTPUT.write_text(json.dumps(payload, indent=1))

    print(f"Scored {n} stocks across {len(payload['sectors'])} sectors (as of {payload['as_of']}).")
    for k, s in sorted(payload["sectors"].items(), key=lambda kv: -kv[1]["heat"]):
        print(f"  {k:<11} heat {s['heat']:>3}  ext {s['ext']:>3}  {s['status']}")
    if payload["errors"]:
        print("Skipped:", payload["errors"])


if __name__ == "__main__":
    main()
