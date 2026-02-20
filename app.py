#!/usr/bin/env python3
"""
Geomet Recycling - Copper Intelligence Dashboard v4.1
Fix Window, Fed Funds, Warehouse Stocks, Price Context, DXY, S/R
"""

import json, os, csv, glob, re, time
from datetime import datetime, timedelta
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

# Load .env file into environment before config
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

DATA_DIR = Path(__file__).parent / "data"
POSITION_CSV = DATA_DIR / "geomet_position.csv"
SPREAD_HISTORY = DATA_DIR / "spread_history.json"
STATIC_DIR = Path(__file__).parent / "static"
PORT = 8777

import importlib.util
def load_config():
    cfg_path = Path(__file__).parent / "config.py"
    defaults = {
        "METALS_DEV_API_KEY": "", "LME_MANUAL_USD_MT": 0,
        "FIX_TARGET": 5.90, "GTC_LEVELS": [5.90, 5.95, 6.00, 6.05],
        "TRUCKLOAD_LBS": 42000, "ATTENTION_MOVE": 0.10, "BIG_MOVE": 0.20,
        "COMEX_WAREHOUSE_MT": 0, "COMEX_WAREHOUSE_DATE": "",
        "COMEX_WAREHOUSE_TREND": "", "FRED_API_KEY": "",
        "FED_FUNDS_RATE": "4.25-4.50", "FED_FUNDS_MIDPOINT": 4.375,
    }
    if cfg_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("config", cfg_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for k, v in defaults.items():
                defaults[k] = getattr(mod, k, v)
        except Exception as e:
            print(f"[WARN] config.py error: {e}")
    return defaults

CFG = load_config()
MT_TO_LB = 2204.62


# ---------------------------------------------------------------------------
# LME CACHE — 30 min cache + market hours only
# ---------------------------------------------------------------------------
_lme_cache = {"price_mt": None, "price_lb": None, "timestamp": 0, "source": "none"}

def fetch_lme_price():
    global _lme_cache
    now = time.time()
    _now = datetime.now()
    _hour = _now.hour; _wd = _now.weekday()
    _lme_open = (
        (_wd == 6 and _hour >= 19) or
        (_wd in (0, 1, 2, 3)) or
        (_wd == 4 and _hour < 13)
    )
    if not _lme_open and _lme_cache["price_lb"]:
        return _lme_cache
    if _lme_cache["price_lb"] and (now - _lme_cache["timestamp"]) < 1800:
        return _lme_cache

    api_key = CFG["METALS_DEV_API_KEY"]
    if api_key:
        try:
            import urllib.request
            url = f"https://api.metals.dev/v1/metal/spot?api_key={api_key}&metal=copper&currency=USD"
            req = urllib.request.Request(url, headers={"User-Agent": "GeometDashboard/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if "rate" in data:
                    rate = data["rate"]
                    price_mt = rate["price"] if isinstance(rate, dict) else rate
                    price_lb = round(price_mt / MT_TO_LB, 4)
                    _lme_cache = {"price_mt": round(price_mt, 2), "price_lb": price_lb, "timestamp": now, "source": "metals.dev"}
                    return _lme_cache
        except Exception as e:
            print(f"[WARN] metals.dev API error: {e}")

    manual = CFG["LME_MANUAL_USD_MT"]
    if manual and manual > 0:
        price_lb = round(manual / MT_TO_LB, 4)
        _lme_cache = {"price_mt": manual, "price_lb": price_lb, "timestamp": now, "source": "manual"}
        return _lme_cache
    return {"price_mt": None, "price_lb": None, "timestamp": now, "source": "none"}


# ---------------------------------------------------------------------------
# DXY
# ---------------------------------------------------------------------------
_dxy_cache = {"price": None, "change": None, "change_pct": None, "timestamp": 0}

def fetch_dxy():
    global _dxy_cache
    now = time.time()
    if _dxy_cache["price"] and (now - _dxy_cache["timestamp"]) < 300:
        return _dxy_cache
    try:
        import yfinance as yf
        ticker = yf.Ticker("DX-Y.NYB")
        hist = ticker.history(period="5d", interval="1d")
        if not hist.empty:
            hist = hist.reset_index()
            hist.columns = [c if isinstance(c, str) else c[0] for c in hist.columns]
            latest = hist.iloc[-1]; prev = hist.iloc[-2] if len(hist) > 1 else latest
            p = float(latest["Close"]); pc = float(prev["Close"]); ch = p - pc
            _dxy_cache = {"price": round(p, 2), "change": round(ch, 2),
                          "change_pct": round((ch / pc) * 100, 2) if pc else 0, "timestamp": now}
    except Exception as e:
        print(f"[WARN] DXY fetch error: {e}")
    return _dxy_cache


# ---------------------------------------------------------------------------
# FRED API — Fed Funds target rate (24h cache)
# ---------------------------------------------------------------------------
_fred_cache = {"upper": None, "lower": None, "midpoint": None, "rate_str": None, "timestamp": 0}

def fetch_fred_fed_rate():
    """Fetch current Fed Funds target rate from FRED (DFEDTARU/DFEDTARL). 24h cache."""
    global _fred_cache
    now = time.time()
    if _fred_cache["upper"] is not None and (now - _fred_cache["timestamp"]) < 86400:
        return _fred_cache

    api_key = CFG["FRED_API_KEY"]
    if not api_key:
        return _fred_cache

    import urllib.request
    upper = None; lower = None
    for series_id in ("DFEDTARU", "DFEDTARL"):
        try:
            url = (f"https://api.stlouisfed.org/fred/series/observations"
                   f"?series_id={series_id}&api_key={api_key}"
                   f"&file_type=json&sort_order=desc&limit=5")
            req = urllib.request.Request(url, headers={"User-Agent": "GeometDashboard/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                for obs in data.get("observations", []):
                    if obs.get("value") and obs["value"] != ".":
                        val = float(obs["value"])
                        if series_id == "DFEDTARU":
                            upper = val
                        else:
                            lower = val
                        break
        except Exception as e:
            print(f"[WARN] FRED {series_id} error: {e}")

    if upper is not None and lower is not None:
        midpoint = round((upper + lower) / 2, 3)
        rate_str = f"{lower:.2f}-{upper:.2f}"
        _fred_cache = {"upper": upper, "lower": lower, "midpoint": midpoint,
                       "rate_str": rate_str, "timestamp": now}
        print(f"[INFO] FRED fed funds: {rate_str}% (midpoint {midpoint}%)")
    return _fred_cache


# ---------------------------------------------------------------------------
# FED FUNDS / RATE EXPECTATIONS
# ---------------------------------------------------------------------------
_fed_cache = {"data": None, "timestamp": 0}

def fetch_fed_data():
    global _fed_cache
    now = time.time()
    if _fed_cache["data"] and (now - _fed_cache["timestamp"]) < 900:
        return _fed_cache["data"]

    # Use FRED data if available, otherwise fall back to static config
    fred = fetch_fred_fed_rate()
    if fred.get("rate_str"):
        current_rate = fred["rate_str"]
        midpoint = fred["midpoint"]
    else:
        current_rate = CFG["FED_FUNDS_RATE"]
        midpoint = CFG["FED_FUNDS_MIDPOINT"]

    result = {
        "current_rate": current_rate,
        "midpoint": midpoint,
    }
    try:
        import yfinance as yf
        # Try fed funds futures for upcoming months
        # ZQ contracts: price = 100 - implied rate
        implied_rates = {}
        month_labels = {
            "ZQH26.CBT": "Mar 26", "ZQJ26.CBT": "Apr 26", "ZQK26.CBT": "May 26",
            "ZQM26.CBT": "Jun 26", "ZQN26.CBT": "Jul 26", "ZQQ26.CBT": "Aug 26",
            "ZQU26.CBT": "Sep 26", "ZQV26.CBT": "Oct 26", "ZQX26.CBT": "Nov 26",
            "ZQZ26.CBT": "Dec 26",
        }
        found_any = False
        for sym, label in month_labels.items():
            try:
                t = yf.Ticker(sym)
                h = t.history(period="5d")
                if not h.empty:
                    price = float(h.iloc[-1]["Close"])
                    implied = round(100 - price, 3)
                    implied_rates[label] = implied
                    found_any = True
            except: continue

        if found_any:
            result["implied_rates"] = implied_rates
            # Find first month where rate drops by 25bp+
            first_cut = None; total_cuts = 0
            for label in ["Mar 26","Apr 26","May 26","Jun 26","Jul 26","Aug 26","Sep 26","Oct 26","Nov 26","Dec 26"]:
                if label in implied_rates:
                    cuts = max(0, round((midpoint - implied_rates[label]) / 0.25))
                    if cuts > 0 and not first_cut:
                        first_cut = label
                    total_cuts = max(total_cuts, cuts)
            result["first_cut"] = first_cut if first_cut else "None priced"
            result["total_cuts_2026"] = total_cuts
        else:
            # Fallback: 10Y yield as macro context
            try:
                t10 = yf.Ticker("^TNX")
                h10 = t10.history(period="5d")
                if not h10.empty:
                    h10 = h10.reset_index()
                    h10.columns = [c if isinstance(c, str) else c[0] for c in h10.columns]
                    y10 = float(h10.iloc[-1]["Close"])
                    prev10 = float(h10.iloc[-2]["Close"]) if len(h10) > 1 else y10
                    result["yield_10y"] = round(y10, 2)
                    result["yield_10y_change"] = round(y10 - prev10, 2)
            except: pass
    except Exception as e:
        print(f"[WARN] Fed data error: {e}")

    _fed_cache = {"data": result, "timestamp": now}
    return result


# ---------------------------------------------------------------------------
# CHINA / SHFE STATUS
# ---------------------------------------------------------------------------
def get_china_status():
    now = datetime.now()
    lny_start = datetime(2026, 2, 15)
    lny_end = datetime(2026, 2, 23, 23, 59, 59)
    if lny_start <= now <= lny_end:
        days_left = (lny_end - now).days
        return {"status": "CLOSED", "reason": "Lunar New Year",
                "detail": f"SHFE closed \u2014 returns Feb 24 ({days_left}d)", "color": "red", "thin_liquidity": True}
    wd = now.weekday(); hr = now.hour
    if wd >= 5:
        return {"status": "CLOSED", "reason": "Weekend", "detail": "SHFE closed \u2014 weekend",
                "color": "yellow", "thin_liquidity": wd == 6 and hr >= 17}
    if hr >= 19 or hr < 2:
        return {"status": "OPEN", "reason": "Night session", "detail": "SHFE night session active",
                "color": "green", "thin_liquidity": False}
    if 7 <= hr <= 14:
        return {"status": "CLOSED", "reason": "Between sessions", "detail": "SHFE between sessions",
                "color": "yellow", "thin_liquidity": False}
    return {"status": "CLOSED", "reason": "Off hours", "detail": "SHFE closed",
            "color": "yellow", "thin_liquidity": False}


# ---------------------------------------------------------------------------
# SPREAD HISTORY
# ---------------------------------------------------------------------------
def load_spread_history():
    if SPREAD_HISTORY.exists():
        try:
            with open(SPREAD_HISTORY) as f: return json.load(f)
        except: pass
    return []

def save_spread_entry(comex, lme, spread):
    history = load_spread_history()
    today = datetime.now().strftime("%Y-%m-%d")
    if history and history[-1].get("date") == today:
        history[-1] = {"date": today, "comex": comex, "lme": lme, "spread": spread}
    else:
        history.append({"date": today, "comex": comex, "lme": lme, "spread": spread})
    history = history[-180:]
    try:
        with open(SPREAD_HISTORY, "w") as f: json.dump(history, f)
    except: pass
    return history

def compute_spread_intelligence(history, current_spread):
    if not history or current_spread is None: return None
    spreads = [h["spread"] for h in history if h.get("spread") is not None]
    if len(spreads) < 3:
        return {"history_days": len(spreads)}
    s30 = spreads[-30:] if len(spreads) >= 30 else spreads
    below30 = sum(1 for s in s30 if s < current_spread)
    pct30 = round(below30 / len(s30) * 100, 1)
    streak = 0; direction = None
    for i in range(len(spreads)-1, 0, -1):
        diff = spreads[i] - spreads[i-1]
        if diff > 0.001:
            if direction == "widening" or direction is None: streak += 1; direction = "widening"
            else: break
        elif diff < -0.001:
            if direction == "narrowing" or direction is None: streak += 1; direction = "narrowing"
            else: break
        else: break
    return {"pct_30d": pct30, "streak": streak, "streak_direction": direction,
            "range_30d_min": round(min(s30), 4), "range_30d_max": round(max(s30), 4), "history_days": len(spreads)}


# ---------------------------------------------------------------------------
# AUTO SUPPORT/RESISTANCE
# ---------------------------------------------------------------------------
def calc_support_resistance(closes, highs, lows):
    if len(closes) < 20: return {"support": [], "resistance": []}
    current = closes[-1]; swing_highs = []; swing_lows = []; lookback = 5
    for i in range(lookback, len(highs) - lookback):
        if highs[i] == max(highs[i-lookback:i+lookback+1]): swing_highs.append(round(highs[i], 4))
        if lows[i] == min(lows[i-lookback:i+lookback+1]): swing_lows.append(round(lows[i], 4))

    def cluster(levels, threshold=0.03):
        if not levels: return []
        levels = sorted(levels); clusters = []; cc = [levels[0]]
        for i in range(1, len(levels)):
            if levels[i] - cc[-1] < threshold: cc.append(levels[i])
            else: clusters.append(round(sum(cc)/len(cc), 4)); cc = [levels[i]]
        clusters.append(round(sum(cc)/len(cc), 4)); return clusters

    support = cluster([s for s in swing_lows if s < current])
    resistance = cluster([r for r in swing_highs if r > current])
    support = sorted(support, reverse=True)[:3]
    resistance = sorted(resistance)[:3]
    return {
        "support": [{"level": s, "distance": round(current-s, 4), "distance_pct": round((current-s)/current*100, 2)} for s in support],
        "resistance": [{"level": r, "distance": round(r-current, 4), "distance_pct": round((r-current)/current*100, 2)} for r in resistance],
    }


# ---------------------------------------------------------------------------
# FIX WINDOW SCORING
# ---------------------------------------------------------------------------
def calc_fix_window(md, sig):
    """Composite score: should you fix/price against unpriced longs right now?"""
    if not md or not sig: return None
    score = 50  # neutral baseline

    # 1. Percentile position (higher = better for fixing)
    pct90 = md.get("pct_90d", 50)
    if pct90 >= 85: score += 18
    elif pct90 >= 70: score += 12
    elif pct90 >= 55: score += 5
    elif pct90 <= 15: score -= 18
    elif pct90 <= 30: score -= 12
    elif pct90 <= 45: score -= 5

    # 2. Momentum (rising = better for fixing)
    roc = md.get("roc", {})
    r5 = roc.get("5d", {}).get("pct", 0)
    if r5 > 3: score += 12
    elif r5 > 1: score += 6
    elif r5 < -3: score -= 12
    elif r5 < -1: score -= 6

    # 3. DXY (dollar weak = bullish copper = better for fixing)
    dxy = md.get("dxy", {})
    dch = dxy.get("change_pct", 0)
    if dch < -0.5: score += 10
    elif dch < -0.2: score += 5
    elif dch > 0.5: score -= 10
    elif dch > 0.2: score -= 5

    # 4. Trend
    trend = sig.get("trend", "")
    ts = sig.get("trend_strength", "")
    if trend == "UPTREND" and ts == "strong": score += 10
    elif trend == "UPTREND": score += 5
    elif trend == "DOWNTREND" and ts == "strong": score -= 10
    elif trend == "DOWNTREND": score -= 5

    # 5. Price vs fix target
    ft = CFG["FIX_TARGET"]; p = md["price"]
    if p >= ft + 0.10: score += 15
    elif p >= ft: score += 12
    elif p >= ft - 0.05: score += 5
    elif p < ft - 0.20: score -= 5

    # 6. Spread (COMEX premium = favorable for COMEX fixing)
    spread = md.get("comex_lme_spread")
    if spread is not None:
        if spread > 0.10: score += 5
        elif spread < -0.10: score -= 3

    # 7. China status (thin liquidity = less reliable signals)
    china = md.get("china", {})
    if china.get("thin_liquidity"): score -= 5

    score = max(0, min(100, score))

    if score >= 80: label, color = "STRONG FIX", "green"
    elif score >= 65: label, color = "FAVORABLE", "green"
    elif score >= 45: label, color = "NEUTRAL", "yellow"
    elif score >= 25: label, color = "UNFAVORABLE", "orange"
    else: label, color = "HOLD / BUY", "red"

    # Build factors list
    factors = []
    if pct90 >= 75: factors.append(f"Near 90d highs ({pct90}th pctl)")
    elif pct90 <= 25: factors.append(f"Near 90d lows ({pct90}th pctl)")
    if r5 > 1: factors.append(f"Momentum up ({r5:+.1f}%)")
    elif r5 < -1: factors.append(f"Momentum down ({r5:+.1f}%)")
    if dch < -0.3: factors.append("Dollar weakening")
    elif dch > 0.3: factors.append("Dollar strengthening")
    if p >= ft: factors.append(f"Above ${ft:.2f} target")
    elif p >= ft - 0.10: factors.append(f"Near ${ft:.2f} target")
    if china.get("thin_liquidity"): factors.append("Thin liquidity")
    if trend == "UPTREND": factors.append("Uptrend")
    elif trend == "DOWNTREND": factors.append("Downtrend")

    return {"score": score, "label": label, "color": color, "factors": factors}


# ---------------------------------------------------------------------------
# WAREHOUSE STOCKS — CME scraper with 24h cache
# ---------------------------------------------------------------------------
SHORT_TON_TO_MT = 0.907185
_cme_wh_cache = {"data": None, "timestamp": 0}

def fetch_cme_warehouse():
    """Download and parse CME Copper_Stocks.xls for live warehouse data."""
    global _cme_wh_cache
    now = time.time()
    if _cme_wh_cache["data"] and (now - _cme_wh_cache["timestamp"]) < 86400:
        return _cme_wh_cache["data"]

    try:
        import urllib.request, tempfile, xlrd
        url = "https://www.cmegroup.com/delivery_reports/Copper_Stocks.xls"
        req = urllib.request.Request(url, headers={"User-Agent": "GeometDashboard/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xls_data = resp.read()

        tmp = tempfile.NamedTemporaryFile(suffix=".xls", delete=False)
        tmp.write(xls_data); tmp.close()

        wb = xlrd.open_workbook(tmp.name)
        ws = wb.sheet_by_index(0)

        # Extract activity date from row 8 (e.g. "Activity Date: 2/18/2026")
        activity_date = ""
        for r in range(min(10, ws.nrows)):
            val = str(ws.cell_value(r, 6)).strip()
            if "Activity Date" in val:
                activity_date = val.replace("Activity Date:", "").strip()
                break

        # Find TOTAL COPPER row for totals
        total_today_st = 0; prev_total_st = 0
        for r in range(ws.nrows):
            label = str(ws.cell_value(r, 0)).strip()
            if label == "TOTAL COPPER":
                prev_total_st = float(ws.cell_value(r, 2)) if ws.cell_value(r, 2) else 0
                total_today_st = float(ws.cell_value(r, 7)) if ws.cell_value(r, 7) else 0
                break

        os.unlink(tmp.name)

        if total_today_st <= 0:
            return None

        total_mt = int(round(total_today_st * SHORT_TON_TO_MT))
        prev_mt = int(round(prev_total_st * SHORT_TON_TO_MT))
        net_change_mt = total_mt - prev_mt
        if net_change_mt > 0:
            trend = "building"
        elif net_change_mt < 0:
            trend = "drawing"
        else:
            trend = "stable"

        result = {
            "mt": total_mt, "lbs": int(total_mt * MT_TO_LB),
            "date": activity_date, "trend": trend,
            "short_tons": int(total_today_st),
            "net_change_mt": net_change_mt,
            "source": "cme",
        }
        _cme_wh_cache = {"data": result, "timestamp": now}
        print(f"[INFO] CME warehouse: {total_mt:,} MT ({trend}, {activity_date})")
        return result
    except Exception as e:
        print(f"[WARN] CME warehouse scrape error: {e}")
        return None

def get_warehouse_data():
    # Try live CME data first, fall back to static config
    live = fetch_cme_warehouse()
    if live:
        return live
    wh = CFG.get("COMEX_WAREHOUSE_MT", 0)
    if not wh: return None
    return {
        "mt": wh, "lbs": int(wh * MT_TO_LB),
        "date": CFG.get("COMEX_WAREHOUSE_DATE", ""),
        "trend": CFG.get("COMEX_WAREHOUSE_TREND", ""),
        "source": "config",
    }


# ---------------------------------------------------------------------------
# HEDGE SPREADSHEET READER
# ---------------------------------------------------------------------------
def is_real_file(filepath):
    """Check if file is actually downloaded (not a OneDrive placeholder)."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
        return header == b'PK\x03\x04'  # Valid xlsx/zip header
    except:
        return False

def trigger_onedrive_download(filepath):
    """Ask OneDrive to download a cloud-only placeholder file."""
    import subprocess
    try:
        subprocess.run(["brctl", "download", filepath], timeout=5, capture_output=True)
        print(f"[INFO] Triggered OneDrive download: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"[WARN] brctl download failed: {e}")

def find_latest_hedge_file():
    # Scan OneDrive sync folder + local data dir
    onedrive_dirs = [
        os.path.expanduser(
            "~/Library/CloudStorage/OneDrive-GeometRecycle/"
            "Pricing_Hedge - Hedge Worksheet"
        ),
        # Legacy path (older OneDrive versions)
        os.path.expanduser(
            "~/Library/Group Containers/UBF8T346G9.OneDriveSyncClientSuite/"
            "OneDrive - Geomet Recycle.noindex/OneDrive - Geomet Recycle/"
            "Pricing_Hedge - Hedge Worksheet"
        ),
    ]
    files = glob.glob(str(DATA_DIR / "Hedge*.xlsx"))
    for onedrive_dir in onedrive_dirs:
        try:
            files += glob.glob(os.path.join(onedrive_dir, "Hedge*.xlsx"))
        except PermissionError:
            print(f"[WARN] Permission denied: {onedrive_dir} — grant Full Disk Access to Python")
    if not files: return None
    def extract_date(f):
        base = os.path.basename(f)
        match = re.search(r'Hedge(\d{8})', base)
        if match:
            try: return datetime.strptime(match.group(1), "%m%d%Y")
            except: pass
        return datetime.fromtimestamp(os.path.getmtime(f))
    files.sort(key=extract_date, reverse=True)

    # Try latest file first — trigger download if placeholder
    for filepath in files:
        if is_real_file(filepath):
            return filepath
        # It's a placeholder — try to trigger OneDrive download
        trigger_onedrive_download(filepath)
        # Wait up to 15 seconds for download
        for i in range(15):
            time.sleep(1)
            if is_real_file(filepath):
                print(f"[INFO] OneDrive download complete: {os.path.basename(filepath)}")
                return filepath
        print(f"[WARN] Skipping placeholder: {os.path.basename(filepath)}")
    return None

def read_hedge_spreadsheet(filepath):
    try:
        import openpyxl
        wb = openpyxl.load_workbook(filepath, data_only=True)
        result = {
            "net_lbs": 0, "avg_cost": 0, "hedge_lbs": 0,
            "priced_sales_lbs": 0, "priced_sales_avg": 0, "unpriced_sales_lbs": 0,
            "total_inv_po": 0, "comex_futures": 0, "lme_futures": 0,
            "updated": "", "source_file": os.path.basename(filepath),
            "sales_priced_unshipped": [], "sales_unpriced_shipped": [], "sales_unpriced_unshipped": [],
            "sales_priced_unshipped_lbs": 0, "sales_unpriced_shipped_lbs": 0, "sales_unpriced_unshipped_lbs": 0,
            "inv_by_commodity": {"BB": 0, "#1": 0, "#2": 0, "Chops": 0},
            "sales_by_commodity": {"BB": 0, "#1": 0, "#2": 0, "Chops": 0},
        }

        if "POSITION" in wb.sheetnames:
            ws = wb["POSITION"]
            grid = [[c if c is not None else "" for c in row] for row in ws.iter_rows(values_only=True)]
            for i, row in enumerate(grid):
                for j, cell in enumerate(row):
                    s = str(cell).strip()
                    if i == 0 and j == 1 and isinstance(cell, (int, float)): result["net_lbs"] = float(cell)
                    # avg_cost calculated below from per-commodity costs
                    if s == "COMEX FUTURES":
                        try:
                            v = row[1] if j == 0 else (grid[i][j+1] if j+1<len(row) else 0)
                            if isinstance(v, (int, float)): result["comex_futures"] = float(v)
                        except: pass
                    if s == "LME FUTURES":
                        try:
                            v = row[1] if j == 0 else (grid[i][j+1] if j+1<len(row) else 0)
                            if isinstance(v, (int, float)): result["lme_futures"] = float(v)
                        except: pass
                    if s == "PRICED SO OS" and j == 0:
                        try: result["priced_sales_lbs"] = float(row[1]) if row[1] else 0
                        except: pass
            # Per-commodity avg costs from Power BI (until Jorge adds cost column to spreadsheet)
            _COST_PER_LB = {
                "CU1": 5.280171, "CU2": 5.134602, "CU2DIRTY": 5.134602, "CUBB": 5.455980,
                "CAT5": 2.024765, "CUINS1": 3.041050, "CUINS2": 2.093361,
                "MCM": 4.199354, "THHN": 3.914800, "WAVEOPENCU": 2.355245,
                "CUCHOP CUBB": 4.616107, "CUCHOP1A_M": 4.616107, "CUCHOPS2": 4.616107,
            }
            total_cost = 0; total_cu_lbs = 0
            # Extract per-commodity inventory from solid inventory columns (col 2=item, col 3=weight, col 5=CuUnits)
            for row in grid[7:25]:
                if len(row) > 5 and isinstance(row[2], str) and isinstance(row[5], (int, float)):
                    item = row[2].strip().upper()
                    cu = float(row[5])
                    wt = float(row[3]) if isinstance(row[3], (int, float)) else cu
                    if item in _COST_PER_LB:
                        total_cost += wt * _COST_PER_LB[item]; total_cu_lbs += cu
                    if item.startswith("CUBB") and "CHOP" not in item:
                        result["inv_by_commodity"]["BB"] += cu
                    elif item.startswith("CU1"):
                        result["inv_by_commodity"]["#1"] += cu
                    elif item.startswith("CU2"):
                        result["inv_by_commodity"]["#2"] += cu
                    elif item.startswith("CUCHOP"):
                        result["inv_by_commodity"]["Chops"] += cu
            # Add ICW inventory (at projected recovery) to Chops — insulated wire becomes chops when processed
            # Also include ICW in weighted avg cost
            for i, row in enumerate(grid):
                for j, cell in enumerate(row):
                    if str(cell).strip() == "ICW INV" and j + 1 < len(row) and isinstance(row[j + 1], (int, float)):
                        icw_cu = float(row[j + 1])
                        result["inv_by_commodity"]["Chops"] += icw_cu
                        result["icw_cu_lbs"] = icw_cu
                        result["chops_solid_lbs"] = result["inv_by_commodity"]["Chops"] - icw_cu
            # ICW per-item costs (col 9=item, col 10=weight, col 12=CuUnits)
            for row in grid[7:25]:
                if len(row) > 12 and isinstance(row[9], str) and isinstance(row[10], (int, float)):
                    item = row[9].strip().upper()
                    wt = float(row[10])
                    cu = float(row[12]) if isinstance(row[12], (int, float)) else 0
                    if item in _COST_PER_LB:
                        total_cost += wt * _COST_PER_LB[item]; total_cu_lbs += cu
            # Avg cost per lb of recovered copper (total $ paid / total Cu lbs out)
            result["avg_cost"] = round(total_cost / total_cu_lbs, 6) if total_cu_lbs > 0 else 0

        shipped_orders = {}
        if "O SOLID SALE" in wb.sheetnames:
            ws = wb["O SOLID SALE"]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not row or not row[0]: continue
                try:
                    so_num = str(row[2]).strip() if row[2] else ""
                    shipped_lbs = float(row[8]) if row[8] else 0
                    if so_num and shipped_lbs > 0:
                        shipped_orders[so_num] = shipped_orders.get(so_num, 0) + shipped_lbs
                except: continue

        if "SOSOLIDS" in wb.sheetnames:
            ws = wb["SOSOLIDS"]
            priced_total = 0; priced_value = 0; unpriced_total = 0
            for row in ws.iter_rows(min_row=3, values_only=True):
                if not row or not row[0]: continue
                try:
                    order = str(row[1]).strip() if row[1] else ""
                    consumer = str(row[2]) if row[2] else ""
                    poref = str(row[3]) if row[3] else ""
                    option = str(row[4]) if row[4] else ""
                    commodity = str(row[5]) if row[5] else ""
                    total_tons = float(row[6]) if row[6] else 0
                    open_priced = float(row[7]) if row[7] else 0
                    priced_tons = float(row[8]) if row[8] else 0
                    final_price = float(row[15]) if row[15] else 0
                    fix_month = str(row[12]) if row[12] else ""
                    spread = float(row[14]) if row[14] else 0
                    basis = "COMEX"
                    if "LME" in poref.upper() or "LME" in option.upper() or "LME" in fix_month.upper():
                        basis = "LME"
                    sale = {"order": order, "consumer": consumer, "commodity": commodity,
                            "lbs": total_tons, "option": option, "basis": basis, "spread": spread, "poref": poref}
                    if priced_tons > 0:
                        priced_total += priced_tons; priced_value += priced_tons * final_price
                        sale["price"] = round(final_price, 4); sale["status"] = "PRICED"
                        if order not in shipped_orders:
                            result["sales_priced_unshipped"].append(sale)
                            result["sales_priced_unshipped_lbs"] += priced_tons
                    elif open_priced > 0:
                        unpriced_total += open_priced; sale["status"] = "UNPRICED"
                        if order in shipped_orders:
                            sale["shipped_lbs"] = shipped_orders[order]
                            result["sales_unpriced_shipped"].append(sale)
                            result["sales_unpriced_shipped_lbs"] += open_priced
                        else:
                            result["sales_unpriced_unshipped"].append(sale)
                            result["sales_unpriced_unshipped_lbs"] += open_priced
                    sale_lbs = priced_tons + open_priced
                    cu = commodity.upper()
                    if cu.startswith("CUBB") and "CHOP" not in cu:
                        result["sales_by_commodity"]["BB"] += sale_lbs
                    elif cu.startswith("CU1"):
                        result["sales_by_commodity"]["#1"] += sale_lbs
                    elif cu.startswith("CU2"):
                        result["sales_by_commodity"]["#2"] += sale_lbs
                    elif "CHOP" in cu:
                        result["sales_by_commodity"]["Chops"] += sale_lbs
                except (TypeError, ValueError, IndexError): continue
            if priced_total > 0: result["priced_sales_avg"] = round(priced_value / priced_total, 4)
            result["priced_sales_lbs"] = priced_total; result["unpriced_sales_lbs"] = unpriced_total

        if "Report" in wb.sheetnames:
            ws = wb["Report"]
            for row in ws.iter_rows(values_only=True):
                if row and row[0]:
                    label = str(row[0]).strip(); val = row[1] if len(row) > 1 else None
                    if label == "Total Inv/PO" and isinstance(val, (int, float)):
                        result["total_inv_po"] = float(val)
                    if label == "Inventory Copper" and isinstance(val, (int, float)):
                        result["inventory_cu_lbs"] = float(val)
                    if label.startswith("PO Waiting") and isinstance(val, (int, float)):
                        result["po_lbs"] = float(val)

        result["hedge_lbs"] = abs(result["comex_futures"]) + abs(result["lme_futures"])
        match = re.search(r'Hedge(\d{8})', os.path.basename(filepath))
        if match:
            try: result["updated"] = datetime.strptime(match.group(1), "%m%d%Y").strftime("%Y-%m-%d")
            except: result["updated"] = match.group(1)
        wb.close()
        return result
    except Exception as e:
        print(f"[ERROR] read_hedge: {e}")
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# COMEX MARKET DATA + PRICE CONTEXT
# ---------------------------------------------------------------------------
_copper_cache = {"data": None, "timestamp": 0}
COPPER_CACHE_TTL = 300  # 5 minutes

def _fetch_ohlc_investiny():
    """Fetch ~1 year COMEX copper OHLC from investing.com via investiny."""
    from investiny import historical_data
    now = datetime.now()
    one_year_ago = now - timedelta(days=365)
    data = historical_data(
        investing_id=8831,
        from_date=one_year_ago.strftime("%m/%d/%Y"),
        to_date=now.strftime("%m/%d/%Y"),
    )
    if not data or not data.get("close") or len(data["close"]) < 20:
        return None
    n = len(data["close"])
    dates = [datetime.strptime(d, "%m/%d/%Y") for d in data["date"]]
    closes = [float(c) for c in data["close"]]
    highs = [float(h) for h in data["high"]]
    lows = [float(lo) for lo in data["low"]]
    print(f"[INFO] Copper from investing.com ({n} days)")
    return {"dates": dates, "closes": closes, "highs": highs, "lows": lows, "volumes": None, "source": "investing.com"}

def _fetch_ohlc_yfinance():
    """Fallback: fetch COMEX copper OHLCV from yfinance."""
    import yfinance as yf
    ticker = yf.Ticker("HG=F")
    hist = ticker.history(period="1y", interval="1d")
    if hist.empty:
        return None
    hist = hist.reset_index()
    hist.columns = [c if isinstance(c, str) else c[0] for c in hist.columns]
    dates = [r["Date"].to_pydatetime() if hasattr(r["Date"], "to_pydatetime") else r["Date"] for _, r in hist.iterrows()]
    closes = hist["Close"].astype(float).tolist()
    highs = hist["High"].astype(float).tolist()
    lows = hist["Low"].astype(float).tolist()
    volumes = hist["Volume"].astype(float).tolist()
    print(f"[INFO] Copper from yfinance ({len(closes)} days)")
    return {"dates": dates, "closes": closes, "highs": highs, "lows": lows, "volumes": volumes, "source": "yfinance"}

def fetch_copper_data():
    global _copper_cache
    now = time.time()
    if _copper_cache["data"] and (now - _copper_cache["timestamp"]) < COPPER_CACHE_TTL:
        return _copper_cache["data"]

    try:
        # Try investing.com first, fall back to yfinance
        ohlc = None
        try:
            ohlc = _fetch_ohlc_investiny()
        except Exception as e:
            print(f"[WARN] investiny error: {e}")
        if not ohlc:
            ohlc = _fetch_ohlc_yfinance()
        if not ohlc:
            return None

        dates = ohlc["dates"]; closes = ohlc["closes"]; highs = ohlc["highs"]; lows = ohlc["lows"]
        copper_source = ohlc["source"]
        n_closes = len(closes)

        price = closes[-1]; prev_close = closes[-2] if n_closes > 1 else price
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0

        ma50 = sum(closes[-50:]) / min(n_closes, 50)
        ma100 = sum(closes[-100:]) / min(n_closes, 100)
        ma200 = sum(closes[-200:]) / min(n_closes, 200)

        # Volume (only available from yfinance)
        volumes = ohlc.get("volumes")
        if volumes:
            avg_vol = sum(volumes[-20:]) / min(len(volumes), 20)
            vol = volumes[-1]
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        recent = closes[-5:] if n_closes >= 5 else closes
        spark = [{"date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10],
                  "close": round(c, 4)} for d, c in zip(dates[-30:], closes[-30:])]

        today_high = highs[-1]; today_low = lows[-1]
        today_range = today_high - today_low

        c30 = closes[-30:] if n_closes >= 30 else closes
        c90 = closes[-90:] if n_closes >= 90 else closes
        pct_30d = round(sum(1 for c in c30 if c < price) / len(c30) * 100, 0)
        pct_90d = round(sum(1 for c in c90 if c < price) / len(c90) * 100, 0)
        range_30d_low = min(c30); range_30d_high = max(c30)
        range_90d_low = min(c90); range_90d_high = max(c90)

        roc = {}
        for n in [1, 3, 5, 10]:
            if n_closes > n:
                rc = price - closes[-(n+1)]
                roc[f"{n}d"] = {"change": round(rc, 4), "pct": round(rc / closes[-(n+1)] * 100, 2)}

        streak = 0; streak_dir = None
        for i in range(len(closes)-1, 0, -1):
            if closes[i] > closes[i-1]:
                if streak_dir == "up" or streak_dir is None: streak += 1; streak_dir = "up"
                else: break
            elif closes[i] < closes[i-1]:
                if streak_dir == "down" or streak_dir is None: streak += 1; streak_dir = "down"
                else: break
            else: break

        last10_ranges = [highs[i] - lows[i] for i in range(-min(10, n_closes), 0)]
        avg_daily_range = round(sum(last10_ranges) / len(last10_ranges), 4) if last10_ranges else 0
        vol_vs_avg = round(today_range / avg_daily_range, 2) if avg_daily_range > 0 else 1.0

        sr = calc_support_resistance(closes, highs, lows)
        lme = fetch_lme_price()
        lme_price = lme["price_lb"]; lme_mt = lme["price_mt"]; lme_source = lme["source"]
        spread = round(price - lme_price, 4) if lme_price else None
        spread_pct = round((spread / lme_price) * 100, 2) if lme_price and spread else None
        spread_intel = None
        if spread is not None and lme_price:
            history = save_spread_entry(round(price, 4), round(lme_price, 4), round(spread, 4))
            spread_intel = compute_spread_intelligence(history, spread)

        dxy = fetch_dxy()
        china = get_china_status()
        fed = fetch_fed_data()
        warehouse = get_warehouse_data()

        result = {
            "price": round(price, 4), "prev_close": round(prev_close, 4),
            "change": round(change, 4), "change_pct": round(change_pct, 2),
            "ma50": round(ma50, 4), "ma100": round(ma100, 4), "ma200": round(ma200, 4),
            "vol_ratio": round(vol_ratio, 2), "recent_closes": [round(c, 4) for c in recent],
            "sparkline": spark, "copper_source": copper_source,
            "lme_price_lb": lme_price, "lme_price_mt": lme_mt, "lme_source": lme_source,
            "comex_lme_spread": spread, "comex_lme_spread_pct": spread_pct, "spread_intel": spread_intel,
            "today_high": round(today_high, 4), "today_low": round(today_low, 4),
            "today_range": round(today_range, 4),
            "pct_30d": pct_30d, "pct_90d": pct_90d,
            "range_30d": [round(range_30d_low, 4), round(range_30d_high, 4)],
            "range_90d": [round(range_90d_low, 4), round(range_90d_high, 4)],
            "roc": roc, "streak": streak, "streak_dir": streak_dir,
            "avg_daily_range": avg_daily_range, "vol_vs_avg": vol_vs_avg,
            "support_resistance": sr, "dxy": dxy, "china": china,
            "fed": fed, "warehouse": warehouse,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _copper_cache = {"data": result, "timestamp": time.time()}
        return result
    except Exception as e:
        print(f"[ERROR] fetch_copper: {e}")
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# SIGNALS
# ---------------------------------------------------------------------------
def compute_signals(md):
    if not md: return None
    p = md["price"]; ma50 = md["ma50"]; ma100 = md["ma100"]; ma200 = md["ma200"]
    vr = md["vol_ratio"]; ch = md["change"]; recent = md["recent_closes"]
    above50 = p > ma50; above100 = p > ma100; above200 = p > ma200

    if above50 and above100 and above200: trend, ts = "UPTREND", "strong"
    elif above200 and (above50 or above100): trend, ts = "UPTREND", "moderate"
    elif above200: trend, ts = "NEUTRAL", "weakening"
    elif not above200 and not above100: trend, ts = "DOWNTREND", "strong"
    else: trend, ts = "NEUTRAL", "mixed"

    sd = ch < -CFG["BIG_MOVE"]; md_drop = ch < -CFG["ATTENTION_MOVE"]
    sr = ch > CFG["BIG_MOVE"]; mr = ch > CFG["ATTENTION_MOVE"]
    hv = vr > 2.0; lv = vr < 0.7

    if sd and hv: mt, mdesc = "LIQUIDATION", "High-volume selloff"
    elif sd and lv: mt, mdesc = "FLASH_CRASH", "Sharp drop on thin volume"
    elif sd: mt, mdesc = "BIG_DROP", f"Down {abs(ch):.2f}"
    elif md_drop: mt, mdesc = "DIP", f"Down {abs(ch):.2f}"
    elif sr and hv: mt, mdesc = "BREAKOUT", "Strong rally on high volume"
    elif sr: mt, mdesc = "BIG_RALLY", f"Up {ch:.2f}"
    elif mr: mt, mdesc = "RALLY", f"Up {ch:.2f}"
    else: mt, mdesc = "NORMAL", "Normal trading range"

    cb100 = sum(1 for c in recent if c < ma100)
    cb200 = sum(1 for c in recent if c < ma200)
    tbw = None
    if cb200 >= 3: tbw = "3+ closes below 200DMA \u2014 mills will likely cut bids"
    elif cb100 >= 2: tbw = "2+ closes below 100DMA \u2014 mills may shade bids"

    ft = CFG["FIX_TARGET"]
    if mt in ("LIQUIDATION", "FLASH_CRASH"):
        sig, sc, sd_txt = "BUY OPP", "blue", "Competitors scared \u2014 strong buying opportunity"
    elif mt == "BIG_DROP":
        sig, sc, sd_txt = "BUY OPP", "blue", "Big drop \u2014 buying opportunity"
    elif mt == "DIP":
        sig, sc, sd_txt = "OPPORTUNISTIC", "blue", "Dip day \u2014 lean into buys"
    elif p >= ft:
        sig, sc, sd_txt = "FIX ALERT", "green", f"Above ${ft:.2f} target \u2014 price against unpriced longs"
    elif p >= ft - 0.10:
        sig, sc, sd_txt = "APPROACHING", "yellow", f"${p:.4f} nearing ${ft:.2f} \u2014 get GTCs in place"
    elif trend == "DOWNTREND" and ts == "strong":
        sig, sc, sd_txt = "CAUTION", "orange", "Sustained downtrend \u2014 mills shading bids"
    elif tbw and cb200 >= 3:
        sig, sc, sd_txt = "CAUTION", "orange", tbw
    elif trend == "UPTREND" and ts == "strong":
        sig, sc, sd_txt = "NORMAL", "green", "Strong uptrend \u2014 normal operations"
    else:
        sig, sc, sd_txt = "NORMAL", "green", "Markets stable \u2014 normal operations"

    momentum_note = None
    roc = md.get("roc", {}); stk = md.get("streak", 0); stk_dir = md.get("streak_dir")
    if stk >= 5:
        momentum_note = f"{stk} consecutive {'up' if stk_dir=='up' else 'down'} days"
    elif roc.get("5d") and abs(roc["5d"]["pct"]) > 2:
        d = roc["5d"]
        momentum_note = f"5d: {d['change']:+.2f} ({d['pct']:+.1f}%)"

    spread = md.get("comex_lme_spread"); spread_signal = None
    if spread is not None:
        if spread > 0.15: spread_signal = {"direction": "COMEX PREMIUM", "msg": f"COMEX +{spread:.2f}/lb over LME \u2014 domestic COMEX sales favorable", "color": "green"}
        elif spread > 0.05: spread_signal = {"direction": "COMEX SLIGHT", "msg": f"COMEX +{spread:.2f}/lb \u2014 slight premium", "color": "green"}
        elif spread > -0.05: spread_signal = {"direction": "PARITY", "msg": f"COMEX-LME near parity ({spread:+.2f})", "color": "yellow"}
        elif spread > -0.15: spread_signal = {"direction": "LME PREMIUM", "msg": f"LME +{abs(spread):.2f}/lb \u2014 LME export slightly favorable", "color": "blue"}
        else: spread_signal = {"direction": "LME PREMIUM", "msg": f"LME +{abs(spread):.2f}/lb \u2014 lean into LME exports", "color": "blue"}

    si = md.get("spread_intel")
    if si and spread_signal:
        if si.get("streak") and si["streak"] >= 3:
            spread_signal["msg"] += f" ({si['streak_direction']} {si['streak']}d)"
        if si.get("pct_30d") is not None:
            if si["pct_30d"] > 85: spread_signal["msg"] += " \u2014 at 30d highs"
            elif si["pct_30d"] < 15: spread_signal["msg"] += " \u2014 at 30d lows"

    dxy = md.get("dxy", {}); dxy_signal = None
    if dxy.get("price"):
        dch = dxy.get("change_pct", 0)
        if dch > 0.3: dxy_signal = {"direction": "up", "msg": f"Dollar +{dch:.1f}% \u2014 bearish copper", "color": "red"}
        elif dch < -0.3: dxy_signal = {"direction": "down", "msg": f"Dollar {dch:.1f}% \u2014 tailwind for copper", "color": "green"}
        else: dxy_signal = {"direction": "flat", "msg": f"Dollar flat ({dch:+.1f}%)", "color": "yellow"}

    return {
        "trend": trend, "trend_strength": ts,
        "above_50": above50, "above_100": above100, "above_200": above200,
        "move_type": mt, "move_desc": mdesc,
        "signal": sig, "signal_color": sc, "signal_detail": sd_txt,
        "trend_break_warning": tbw, "momentum_note": momentum_note,
        "spread_signal": spread_signal, "dxy_signal": dxy_signal,
    }


# ---------------------------------------------------------------------------
# GTC SUGGESTIONS
# ---------------------------------------------------------------------------
def gen_gtc(position, md):
    if not position or not md: return []
    p = md["price"]; net = position.get("net_lbs", 0)
    if net <= 0: return []
    tl = CFG["TRUCKLOAD_LBS"]; loads = int(net / tl)
    levels = [l for l in CFG["GTC_LEVELS"] if l > p - 0.05]
    suggestions = []
    if not levels:
        suggestions.append({"action": "PRICE NOW", "detail": f"Above all targets \u2014 fix {loads} loads ({int(net):,} lbs) at ${p:.4f}", "urgency": "high"})
        return suggestions
    per = max(1, loads // len(levels)); rem = loads
    for lvl in levels:
        if rem <= 0: break
        n = min(per, rem); lbs = n * tl
        if lvl <= p:
            suggestions.append({"action": "FIX NOW", "detail": f"Fix {n} load{'s' if n>1 else ''} ({lbs:,} lbs) at ${lvl:.2f} \u2014 price is here", "urgency": "high", "level": lvl})
        elif lvl <= p + 0.10:
            suggestions.append({"action": "GTC", "detail": f"GTC {n} load{'s' if n>1 else ''} ({lbs:,} lbs) at ${lvl:.2f} \u2014 {(lvl-p)*100:.1f}c away", "urgency": "medium", "level": lvl})
        else:
            suggestions.append({"action": "GTC", "detail": f"GTC {n} load{'s' if n>1 else ''} ({lbs:,} lbs) at ${lvl:.2f} \u2014 {(lvl-p)*100:.1f}c away", "urgency": "low", "level": lvl})
        rem -= n
    return suggestions


def calc_risk(pos, md):
    if not pos or not md: return None
    p = md["price"]; net = pos["net_lbs"]; ac = pos["avg_cost"]; hl = pos.get("hedge_lbs", 0)
    uh = net - hl; mtm = (p - ac) * net
    return {
        "net_lbs": net, "unhedged_lbs": uh, "avg_cost": round(ac, 4),
        "mtm_pl": round(mtm, 2), "risk_per_cent": round(uh * 0.01, 2),
        "risk_per_dime": round(uh * 0.10, 2),
        "hedge_pct": round((hl / net) * 100, 1) if net else 0,
        "priced_sales_lbs": pos.get("priced_sales_lbs", 0),
        "unpriced_sales_lbs": pos.get("unpriced_sales_lbs", 0),
        "priced_sales_avg": pos.get("priced_sales_avg", 0),
        "loads_unpriced": int(net / CFG["TRUCKLOAD_LBS"]) if net > 0 else 0,
        "sales_priced_unshipped_lbs": pos.get("sales_priced_unshipped_lbs", 0),
        "sales_unpriced_shipped_lbs": pos.get("sales_unpriced_shipped_lbs", 0),
        "sales_unpriced_unshipped_lbs": pos.get("sales_unpriced_unshipped_lbs", 0),
        "total_inv_po": pos.get("total_inv_po", 0),
        "inventory_cu_lbs": pos.get("inventory_cu_lbs", 0),
        "po_lbs": pos.get("po_lbs", 0),
        "total_sales_lbs": pos.get("priced_sales_lbs", 0) + pos.get("unpriced_sales_lbs", 0),
        "coverage_pct": round((pos.get("total_inv_po", 0) / (pos.get("priced_sales_lbs", 0) + pos.get("unpriced_sales_lbs", 0))) * 100, 1) if (pos.get("priced_sales_lbs", 0) + pos.get("unpriced_sales_lbs", 0)) > 0 else 0,
        "surplus_deficit_lbs": pos.get("total_inv_po", 0) - (pos.get("priced_sales_lbs", 0) + pos.get("unpriced_sales_lbs", 0)),
        "inv_by_commodity": pos.get("inv_by_commodity", {}),
        "sales_by_commodity": pos.get("sales_by_commodity", {}),
        "icw_cu_lbs": pos.get("icw_cu_lbs", 0),
        "chops_solid_lbs": pos.get("chops_solid_lbs", 0),
    }


def gen_decisions(sig, risk, md, fix_window):
    dec = []
    if not sig: return ["Unable to fetch market data"]
    p = md["price"] if md else 0; ft = CFG["FIX_TARGET"]

    # Fix window headline
    if fix_window:
        fw = fix_window
        if fw["score"] >= 65:
            factors_str = " + ".join(fw["factors"][:3]) if fw["factors"] else ""
            dec.append(f"\U0001F7E2 FIX WINDOW: {fw['label']} ({fw['score']}/100) \u2014 {factors_str}")
        elif fw["score"] <= 30:
            factors_str = " + ".join(fw["factors"][:3]) if fw["factors"] else ""
            dec.append(f"\U0001F534 FIX WINDOW: {fw['label']} ({fw['score']}/100) \u2014 {factors_str}")

    china = md.get("china", {}) if md else {}
    if china.get("thin_liquidity"):
        dec.append(f"\u26A0 {china.get('detail', 'SHFE closed')} \u2014 thin liquidity, watch for flash moves")
    elif china.get("status") == "CLOSED" and china.get("reason") == "Lunar New Year":
        dec.append(china.get("detail", "SHFE closed"))

    if risk and risk.get("sales_unpriced_shipped_lbs", 0) > 0:
        dec.append(f"\u26A0 {risk['sales_unpriced_shipped_lbs']:,.0f} lbs shipped but UNPRICED \u2014 fix these first")

    if sig["move_type"] == "LIQUIDATION":
        dec.append("Liquidation event \u2014 competitors scared. Lean into buys.")
    elif sig["move_type"] == "FLASH_CRASH":
        dec.append("Flash crash on thin volume \u2014 buying window.")
    elif sig["move_type"] == "BIG_DROP":
        dec.append("Big drop \u2014 buying opportunity.")
    elif sig["move_type"] == "DIP":
        dec.append("Dip day \u2014 outbid competitors, build relationships.")

    if p >= ft and risk and risk["net_lbs"] > 0:
        dec.append(f"ABOVE ${ft:.2f} TARGET \u2014 {risk.get('loads_unpriced',0)} loads unpriced. Fix some.")
    elif p >= ft - 0.10:
        dec.append(f"${p:.4f} \u2014 approaching ${ft:.2f}. GTCs in place.")

    sr = md.get("support_resistance", {}) if md else {}
    if sr.get("support"):
        s = sr["support"][0]
        dec.append(f"Support at ${s['level']:.2f} ({s['distance']*100:.0f}c below)")
    if sr.get("resistance"):
        r = sr["resistance"][0]
        dec.append(f"Resistance at ${r['level']:.2f} ({r['distance']*100:.0f}c above)")

    ss = sig.get("spread_signal")
    if ss: dec.append(ss["msg"])
    ds = sig.get("dxy_signal")
    if ds and ds["direction"] != "flat": dec.append(ds["msg"])

    # Fed context
    fed = md.get("fed", {}) if md else {}
    if fed.get("first_cut") and fed["first_cut"] != "None priced":
        dec.append(f"Fed: {fed.get('total_cuts_2026', 0)} cut(s) priced by year-end, first in {fed['first_cut']}")
    elif fed.get("yield_10y"):
        dec.append(f"10Y yield: {fed['yield_10y']}% ({fed.get('yield_10y_change', 0):+.2f})")

    # Warehouse
    wh = md.get("warehouse") if md else None
    if wh and wh.get("mt"):
        arrow = "\u2191" if wh["trend"] == "building" else "\u2193" if wh["trend"] == "drawing" else "\u2192"
        dec.append(f"COMEX warehouse: {wh['mt']:,} MT {arrow} ({wh.get('date','')}) \u2014 {'bearish overhang' if wh['trend']=='building' else 'supply tightening' if wh['trend']=='drawing' else 'stable'}")

    if sig.get("momentum_note"): dec.append(sig["momentum_note"])

    if risk:
        uh = risk["unhedged_lbs"]; rd = risk["risk_per_dime"]
        if uh > 0: dec.append(f"Long {uh:,.0f} lbs unpriced \u2014 ${abs(rd):,.0f} per 10c move")

    if not dec: dec.append("Markets stable \u2014 normal operations")
    return dec


def load_position():
    hf = find_latest_hedge_file()
    if hf:
        print(f"[INFO] Reading: {os.path.basename(hf)}")
        pos = read_hedge_spreadsheet(hf)
        if pos: return pos
    if POSITION_CSV.exists():
        try:
            with open(POSITION_CSV, "r") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    r = rows[0]
                    return {"net_lbs": float(r.get("net_copper_lbs", 0)), "avg_cost": float(r.get("avg_cost_per_lb", 0)),
                            "hedge_lbs": float(r.get("hedge_lbs", 0)), "updated": r.get("date", "unknown"),
                            "source_file": "geomet_position.csv"}
        except Exception as e: print(f"[ERROR] CSV: {e}")
    return None


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            md = fetch_copper_data()
            sig = compute_signals(md)
            pos = load_position()
            risk = calc_risk(pos, md)
            fix_window = calc_fix_window(md, sig)
            dec = gen_decisions(sig, risk, md, fix_window)
            gtc = gen_gtc(pos, md)
            payload = {
                "market": md, "signals": sig, "position": pos, "position_risk": risk,
                "decisions": dec, "gtc_suggestions": gtc, "fix_window": fix_window,
                "config": {"fix_target": CFG["FIX_TARGET"], "truckload_lbs": CFG["TRUCKLOAD_LBS"], "gtc_levels": CFG["GTC_LEVELS"]},
                "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.wfile.write(json.dumps(payload).encode())
            return
        if self.path in ("/", ""): self.path = "/index.html"
        fp = STATIC_DIR / self.path.lstrip("/")
        if fp.exists() and fp.is_file():
            self.send_response(200)
            ct = "text/html" if str(fp).endswith(".html") else "text/css" if str(fp).endswith(".css") else "application/javascript"
            self.send_header("Content-Type", ct); self.end_headers()
            self.wfile.write(fp.read_bytes())
        else: self.send_error(404)
    def log_message(self, *a): pass

def main():
    DATA_DIR.mkdir(exist_ok=True); STATIC_DIR.mkdir(exist_ok=True)
    hf = find_latest_hedge_file()
    lme_src = "metals.dev API" if CFG["METALS_DEV_API_KEY"] else ("manual" if CFG["LME_MANUAL_USD_MT"] else "NOT CONFIGURED")
    print()
    print("=" * 55)
    print("  GEOMET COPPER INTELLIGENCE DASHBOARD v4.1")
    print("=" * 55)
    print(f"  Dashboard:  http://localhost:{PORT}")
    print(f"  Fix target: ${CFG['FIX_TARGET']}")
    print(f"  LME source: {lme_src}")
    print(f"  LME cache:  30 min (market hours only)")
    fed_src = "FRED API" if CFG["FRED_API_KEY"] else "static config"
    print(f"  Fed rate:   {CFG['FED_FUNDS_RATE']}% ({fed_src})")
    wh = CFG.get("COMEX_WAREHOUSE_MT", 0)
    if wh: print(f"  Warehouse:  {wh:,} MT ({CFG.get('COMEX_WAREHOUSE_DATE','')})")
    if hf: print(f"  Hedge file: {os.path.basename(hf)}")
    print("=" * 55)
    print()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    main()
