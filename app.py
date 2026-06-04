#!/usr/bin/env python3
"""
Geomet Recycling - Copper Intelligence Dashboard v5.0
Multi-timeframe charts, LME hours, fix window, warehouse (COMEX+LME),
sales pipeline, actionable orders, Tailscale remote access
"""

import json, os, csv, glob, re, time, hashlib, secrets, calendar, threading, string, random
from datetime import datetime, timedelta
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from http.cookies import SimpleCookie
import socketserver

# Load .env files into environment before config
for _env_name in (".env", ".env.rom"):
    _env_path = Path(__file__).parent / _env_name
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
BROKER_INTEL_FILE = DATA_DIR / "broker_intel.json"
SHIP_SCHEDULE_FILE = DATA_DIR / "ship_schedule.json"
DAILY_INSIGHT_FILE = DATA_DIR / "daily_insight.json"
OPTIONS_OI_FILE = DATA_DIR / "options_oi.json"
MARKET_RATES_FILE = DATA_DIR / "market_rates.json"
STATIC_DIR = Path(__file__).parent / "static"
PORT = 8777

import importlib.util
def load_config():
    cfg_path = Path(__file__).parent / "config.py"
    defaults = {
        "METALS_DEV_API_KEY": "", "LME_MANUAL_USD_MT": 0,
        "FIX_TARGET": 5.90, "GTC_LEVELS": [5.90, 5.95, 6.00, 6.05],
        "TRUCKLOAD_LBS": 42000, "BASELINE_LBS": 200000,
        "POSITION_RANGE_MIN": 80000, "POSITION_RANGE_MAX": 400000,
        "ATTENTION_MOVE": 0.10, "BIG_MOVE": 0.20,
        "COMEX_WAREHOUSE_MT": 0, "COMEX_WAREHOUSE_DATE": "",
        "COMEX_WAREHOUSE_TREND": "", "FRED_API_KEY": "",
        "FED_FUNDS_RATE": "4.25-4.50", "FED_FUNDS_MIDPOINT": 4.375,
        "MONTHLY_FLOW": {"Chops": 171800, "BB": 162700, "#2": 106600, "#1": 82700},
        "CUSTOMER_HOURS": {},
        "MARKET_RATES": {"BB": {"type": "flat", "discount": 0.15, "basis": "comex_front"}, "#1": {"pct": 0.94, "basis": "cash"}, "#2": {"pct": 0.91, "basis": "cash"}, "Chops": {"pct": 0.94, "basis": "cash"}},
        "MARKET_RATES_LME_AT_UPDATE": 0,
        "MARKET_RATES_DATE": "",
        "MARKET_RATES_STALE_THRESHOLD": 0.05,
        "ICW_RECOVERY": {},
        "CUSTOM_LEVELS": [
            {"price": 6.00, "label": "$6.00 Resistance (Bloomberg 4/13)"},
            {"price": 6.50, "label": "Jan highs (Bloomberg 4/13)"},
        ],
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
_insight_cache = {"data": None, "timestamp": 0}
LME_PRICE_FILE = DATA_DIR / "lme_last_price.json"
VOL_SNAPSHOT_FILE = DATA_DIR / "volume_snapshots.json"

def _is_lme_open():
    """Check if LME is currently open using London time."""
    import zoneinfo
    try:
        london = zoneinfo.ZoneInfo("Europe/London")
    except Exception:
        from datetime import timezone
        london = timezone.utc
    now_london = datetime.now(london)
    wd = now_london.weekday()
    t = now_london.hour * 60 + now_london.minute
    if wd >= 5:
        return False
    # LME Select: 01:00 (60) - 19:00 (1140) London
    return 60 <= t < 1140


# ---------------------------------------------------------------------------
# VOLUME SNAPSHOTS — parallel time-of-week comparison
# ---------------------------------------------------------------------------
def _record_volume_snapshot(volume):
    """Record current session volume with dow/hour for parallel comparison."""
    if not volume or volume <= 0:
        return
    now = datetime.now()
    dow = now.weekday()       # 0=Mon … 6=Sun
    hour = now.hour
    date_str = now.strftime("%Y-%m-%d")
    key = f"{date_str}-{hour}"
    try:
        snapshots = json.load(open(VOL_SNAPSHOT_FILE)) if VOL_SNAPSHOT_FILE.exists() else []
    except Exception:
        snapshots = []
    found = False
    for s in snapshots:
        if s.get("key") == key:
            s["vol"] = int(volume)
            found = True
            break
    if not found:
        snapshots.append({"key": key, "dow": dow, "hour": hour, "vol": int(volume), "date": date_str})
    # Keep last 8 weeks
    cutoff = (now - timedelta(days=56)).strftime("%Y-%m-%d")
    snapshots = [s for s in snapshots if s["date"] >= cutoff]
    try:
        with open(VOL_SNAPSHOT_FILE, "w") as f:
            json.dump(snapshots, f)
    except Exception:
        pass


def _get_parallel_avg_volume():
    """Average volume at this same day-of-week + hour from history."""
    now = datetime.now()
    dow = now.weekday()
    hour = now.hour
    today = now.strftime("%Y-%m-%d")
    try:
        snapshots = json.load(open(VOL_SNAPSHOT_FILE)) if VOL_SNAPSHOT_FILE.exists() else []
    except Exception:
        return None
    matching = [s["vol"] for s in snapshots
                if s["dow"] == dow and s["hour"] == hour and s["date"] != today]
    if len(matching) < 2:
        return None
    return int(sum(matching) / len(matching))


def fetch_lme_price():
    global _lme_cache
    now = time.time()
    lme_open = _is_lme_open()

    # When LME is closed, use cached/persisted price
    if not lme_open:
        if _lme_cache["price_lb"]:
            _lme_cache["source"] = "cached"
            return _lme_cache
        if LME_PRICE_FILE.exists():
            try:
                with open(LME_PRICE_FILE) as f:
                    saved = json.load(f)
                pm = saved.get("price_mt") or saved.get("official_mt")
                if pm:
                    pl = round(pm / MT_TO_LB, 4)
                    _lme_cache = {"price_mt": pm, "price_lb": pl,
                                  "timestamp": now, "source": "cached"}
                    print(f"[INFO] LME closed — using last known price ${pm}/MT from file")
                    return _lme_cache
            except Exception:
                pass
        return {"price_mt": None, "price_lb": None, "timestamp": now, "source": "none"}

    # --- Priority 1: Live WebSocket (CAPITALCOM:MCU3) ---
    with _tv_lock:
        live_mt = _tv_state_lme["price_mt"]
        live_ts = _tv_state_lme["timestamp"]
        live_ch = _tv_state_lme["change_mt"]
        live_chp = _tv_state_lme["change_pct"]
    if live_mt is not None and (now - live_ts) < 300:
        price_lb = round(live_mt / MT_TO_LB, 4)
        change_lb = round(live_ch / MT_TO_LB, 4) if live_ch is not None else None
        _lme_cache = {"price_mt": live_mt, "price_lb": price_lb,
                      "timestamp": now, "source": "live",
                      "change_lb": change_lb, "change_pct": live_chp}
        try:
            with open(LME_PRICE_FILE, "w") as f:
                json.dump({"price_mt": live_mt, "price_lb": price_lb, "official_mt": live_mt}, f)
        except Exception:
            pass
        return _lme_cache

    # --- Priority 2+3: Projected / Official via metals.dev API ---
    # Use cache if fresh (5 min for projected, 30 min for API)
    cache_ttl = 300 if _lme_cache.get("source") == "projected" else 1800
    if _lme_cache["price_lb"] and (now - _lme_cache["timestamp"]) < cache_ttl:
        return _lme_cache

    lme_official_mt = None
    api_key = CFG["METALS_DEV_API_KEY"]
    if api_key:
        try:
            import urllib.request
            url = f"https://api.metals.dev/v1/latest?api_key={api_key}&currency=USD&unit=mt"
            req = urllib.request.Request(url, headers={"User-Agent": "GeometDashboard/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "success" and "metals" in data:
                    lme_official_mt = data["metals"].get("lme_copper")
        except Exception as e:
            print(f"[WARN] metals.dev API error: {e}")

    # Fallback: recover official from saved file
    if not lme_official_mt and LME_PRICE_FILE.exists():
        try:
            with open(LME_PRICE_FILE) as f:
                saved = json.load(f)
            lme_official_mt = saved.get("official_mt") or saved.get("price_mt")
        except Exception:
            pass

    if lme_official_mt:
        # Project live LME 3M by applying COMEX intraday % change
        price_mt = lme_official_mt
        source = "lme_official"
        with _tv_lock:
            comex_live = _tv_state["price"]
            comex_prev = _tv_state["prev_close"]
        if comex_live and comex_prev and comex_prev > 0:
            factor = comex_live / comex_prev
            price_mt = round(lme_official_mt * factor, 2)
            source = "projected"
            print(f"[INFO] LME 3M projected: ${lme_official_mt} official × {factor:.4f} COMEX factor = ${price_mt}/MT")
        else:
            print(f"[INFO] LME 3M using official settlement: ${price_mt}/MT (no COMEX data for projection)")

        price_lb = round(price_mt / MT_TO_LB, 4)
        _lme_cache = {"price_mt": price_mt, "price_lb": price_lb,
                      "timestamp": now, "source": source}
        try:
            with open(LME_PRICE_FILE, "w") as f:
                json.dump({"price_mt": price_mt, "price_lb": price_lb, "official_mt": lme_official_mt}, f)
        except Exception:
            pass
        return _lme_cache

    # --- Priority 4: Manual config ---
    manual = CFG["LME_MANUAL_USD_MT"]
    if manual and manual > 0:
        price_lb = round(manual / MT_TO_LB, 4)
        _lme_cache = {"price_mt": manual, "price_lb": price_lb, "timestamp": now, "source": "manual"}
        return _lme_cache
    return {"price_mt": None, "price_lb": None, "timestamp": now, "source": "none"}


# ---------------------------------------------------------------------------
# TRADINGVIEW WEBSOCKET — near-real-time COMEX copper
# ---------------------------------------------------------------------------
_tv_state = {
    "price": None, "prev_close": None, "open": None,
    "high": None, "low": None, "volume": None,
    "change": None, "change_pct": None,
    "timestamp": 0, "connected": False,
}
_tv_state_2 = {
    "price": None, "prev_close": None,
    "change": None, "change_pct": None,
    "timestamp": 0,
}
_tv_state_lme = {
    "price_mt": None, "prev_close_mt": None,
    "change_mt": None, "change_pct": None,
    "timestamp": 0,
}
_tv_lock = threading.Lock()


def _tv_frame(msg):
    """Wrap a message in TradingView's ~m~LENGTH~m~ framing."""
    return f"~m~{len(msg)}~m~{msg}"


def _tv_send(ws, msg):
    """Send a framed JSON message."""
    ws.send(_tv_frame(json.dumps(msg)))


def _tv_parse_frames(raw):
    """Parse one or more ~m~LENGTH~m~PAYLOAD frames from raw data."""
    frames = []
    i = 0
    while i < len(raw):
        if not raw[i:].startswith("~m~"):
            break
        i += 3
        j = raw.index("~m~", i)
        length = int(raw[i:j])
        j += 3
        frames.append(raw[j:j + length])
        i = j + length
    return frames


def _tv_worker():
    """Background thread: maintain TradingView WebSocket for COMEX:HG1! quotes."""
    global _tv_state
    try:
        import websocket
    except ImportError:
        print("[WARN] websocket-client not installed — TradingView feed disabled")
        return

    while True:
        session_id = "qs_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
        try:
            ws = websocket.create_connection(
                "wss://data.tradingview.com/socket.io/websocket",
                header={"Origin": "https://www.tradingview.com"},
                timeout=60,
            )
            print(f"[INFO] TradingView WebSocket connected (session {session_id})")

            # Auth + subscribe
            _tv_send(ws, {"m": "set_auth_token", "p": ["unauthorized_user_token"]})
            _tv_send(ws, {"m": "quote_create_session", "p": [session_id]})
            _tv_send(ws, {"m": "quote_set_fields", "p": [
                session_id,
                "lp", "ch", "chp", "open_price", "high_price", "low_price",
                "prev_close_price", "volume", "description", "short_name",
            ]})
            _tv_send(ws, {"m": "quote_add_symbols", "p": [session_id, "COMEX:HG1!"]})
            _tv_send(ws, {"m": "quote_add_symbols", "p": [session_id, "COMEX:HG2!"]})
            _tv_send(ws, {"m": "quote_add_symbols", "p": [session_id, "CAPITALCOM:MCU3"]})

            with _tv_lock:
                _tv_state["connected"] = True

            while True:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    # No data for 60s — send a ping to keep alive
                    try:
                        ws.ping()
                    except Exception:
                        break
                    continue
                if not raw:
                    break
                frames = _tv_parse_frames(raw)
                for frame in frames:
                    # Heartbeat
                    if frame.startswith("~h~"):
                        ws.send(_tv_frame(frame))
                        continue
                    # Data
                    try:
                        msg = json.loads(frame)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if msg.get("m") == "qsd":
                        p_data = msg.get("p", [None, {}])
                        sym = p_data[1].get("n", "") if len(p_data) > 1 else ""
                        vals = p_data[1].get("v", {}) if len(p_data) > 1 else {}
                        lp = vals.get("lp")
                        if lp is not None and "MCU3" in sym:
                            # LME 3M copper (CAPITALCOM:MCU3) — $/MT
                            with _tv_lock:
                                _tv_state_lme["price_mt"] = round(float(lp), 2)
                                _tv_state_lme["timestamp"] = time.time()
                                if "ch" in vals:
                                    _tv_state_lme["change_mt"] = round(float(vals["ch"]), 2)
                                if "chp" in vals:
                                    _tv_state_lme["change_pct"] = round(float(vals["chp"]), 2)
                                if "prev_close_price" in vals:
                                    _tv_state_lme["prev_close_mt"] = round(float(vals["prev_close_price"]), 2)
                            if not hasattr(_tv_worker, '_lme_log_ts') or time.time() - _tv_worker._lme_log_ts > 300:
                                print(f"[INFO] LME 3M live: ${_tv_state_lme['price_mt']}/MT via MCU3")
                                _tv_worker._lme_log_ts = time.time()
                        elif lp is not None and "HG2" in sym:
                            # Next month contract (HG2!)
                            with _tv_lock:
                                _tv_state_2["price"] = round(float(lp), 4)
                                _tv_state_2["timestamp"] = time.time()
                                if "ch" in vals:
                                    _tv_state_2["change"] = round(float(vals["ch"]), 4)
                                if "chp" in vals:
                                    _tv_state_2["change_pct"] = round(float(vals["chp"]), 2)
                                if "prev_close_price" in vals:
                                    _tv_state_2["prev_close"] = round(float(vals["prev_close_price"]), 4)
                        elif lp is not None:
                            # Front month contract (HG1!)
                            with _tv_lock:
                                _tv_state["price"] = round(float(lp), 4)
                                _tv_state["timestamp"] = time.time()
                                if "ch" in vals:
                                    _tv_state["change"] = round(float(vals["ch"]), 4)
                                if "chp" in vals:
                                    _tv_state["change_pct"] = round(float(vals["chp"]), 2)
                                if "prev_close_price" in vals:
                                    _tv_state["prev_close"] = round(float(vals["prev_close_price"]), 4)
                                if "open_price" in vals:
                                    _tv_state["open"] = round(float(vals["open_price"]), 4)
                                if "high_price" in vals:
                                    _tv_state["high"] = round(float(vals["high_price"]), 4)
                                if "low_price" in vals:
                                    _tv_state["low"] = round(float(vals["low_price"]), 4)
                                if "volume" in vals:
                                    _tv_state["volume"] = vals["volume"]

        except Exception as e:
            print(f"[WARN] TradingView WebSocket error: {e}")
            with _tv_lock:
                _tv_state["connected"] = False
        # Reconnect after 5 seconds
        time.sleep(5)


# Start TradingView WebSocket in background daemon thread
_tv_thread = threading.Thread(target=_tv_worker, daemon=True, name="tv-ws")
_tv_thread.start()


# ---------------------------------------------------------------------------
# REAL-TIME PRICE — TradingView WebSocket primary, yfinance fallback
# ---------------------------------------------------------------------------
_rt_cache = {"price": None, "prev_close": None, "timestamp": 0, "source": None}
_prev_settle_cache = {}  # keyed by ticker: {"price": ..., "timestamp": ...}

def _is_new_comex_session():
    """Check if we're in the new COMEX session (after 5PM CT daily break).
    COMEX sessions run 5PM CT to 4PM CT next day, with a 4-5PM CT break.
    After 5PM CT Mon-Thu, the new session has started and today's close
    becomes the previous settlement, not yesterday's."""
    import zoneinfo
    try:
        chicago = zoneinfo.ZoneInfo("America/Chicago")
    except Exception:
        from datetime import timezone, timedelta as td
        chicago = timezone(td(hours=-6))
    now_ct = datetime.now(chicago)
    wd = now_ct.weekday()  # 0=Mon
    t = now_ct.hour * 60 + now_ct.minute
    # After 5PM CT (1020 min) Mon-Thu = new session has started
    # Sunday 5PM+ also starts the week's first session
    if wd <= 3 and t >= 1020:
        return True
    if wd == 6 and t >= 1020:  # Sunday evening open
        return True
    return False


_session_settle_cache = {"price": None, "date": None, "ticker": None}

def _fetch_session_settle(ticker="HG=F"):
    """Get today's 4PM CT settlement from intraday data.
    After the 4-5PM CT break, the last bar before 4PM = the settlement."""
    global _session_settle_cache
    import zoneinfo
    try:
        chicago = zoneinfo.ZoneInfo("America/Chicago")
    except Exception:
        from datetime import timezone, timedelta as td
        chicago = timezone(td(hours=-6))
    today_ct = datetime.now(chicago).date()
    # Return cached if we already found today's settlement
    if _session_settle_cache["price"] and _session_settle_cache["date"] == str(today_ct) \
       and _session_settle_cache["ticker"] == ticker:
        return _session_settle_cache["price"]
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        h = t.history(period="2d", interval="5m")
        if h.empty:
            return None
        h = h.reset_index()
        h.columns = [c if isinstance(c, str) else c[0] for c in h.columns]
        settle = None
        for _, row in h.iterrows():
            ts = row.get("Datetime", row.get("Date"))
            if not hasattr(ts, 'astimezone'):
                continue
            ts_ct = ts.astimezone(chicago)
            if ts_ct.date() == today_ct:
                ct_min = ts_ct.hour * 60 + ts_ct.minute
                # 5-min bars before 4PM CT (960 min) — last one is the settlement
                if ct_min < 960:
                    settle = round(float(row["Close"]), 4)
        if settle:
            _session_settle_cache = {"price": settle, "date": str(today_ct), "ticker": ticker}
            print(f"[INFO] Session settle {ticker} from intraday: ${settle:.4f}")
        return settle
    except Exception as e:
        print(f"[WARN] Session settle fetch error: {e}")
    return None


def _fetch_prev_settle_yf(ticker="HG=F"):
    """Get previous session's settlement from yfinance (5-min cache per ticker).
    After 5PM CT (new COMEX session), gets today's 4PM settlement from intraday
    data, since the daily bar close keeps updating with live prices."""
    global _prev_settle_cache
    now = time.time()
    new_session = _is_new_comex_session()
    cached = _prev_settle_cache.get(ticker, {})
    # Invalidate cache if session state changed (crossed 5PM CT boundary)
    if cached.get("price") and (now - cached.get("timestamp", 0)) < 300 \
       and cached.get("new_session") == new_session:
        return cached["price"]

    # After 5PM CT: get today's 4PM settlement from intraday bars
    if new_session:
        settle = _fetch_session_settle(ticker)
        if settle:
            _prev_settle_cache[ticker] = {"price": settle, "timestamp": now, "new_session": new_session}
            return settle
        # Fall through to daily data if intraday failed

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hd = t.history(period="5d", interval="1d")
        if not hd.empty and len(hd) >= 2:
            hd = hd.reset_index()
            hd.columns = [c if isinstance(c, str) else c[0] for c in hd.columns]
            today_date = datetime.now().date()
            last_date = hd.iloc[-1]["Date"]
            if hasattr(last_date, 'date'):
                last_date = last_date.date()
            elif hasattr(last_date, 'to_pydatetime'):
                last_date = last_date.to_pydatetime().date()
            if last_date >= today_date:
                prev = round(float(hd.iloc[-2]["Close"]), 4)
                print(f"[INFO] Prev settle {ticker} (today in data): ${prev:.4f} from {hd.iloc[-2]['Date']}")
            else:
                prev = round(float(hd.iloc[-1]["Close"]), 4)
                print(f"[INFO] Prev settle {ticker} (no today): ${prev:.4f} from {hd.iloc[-1]['Date']}")
            _prev_settle_cache[ticker] = {"price": prev, "timestamp": now, "new_session": new_session}
            return prev
    except Exception as e:
        print(f"[WARN] Prev settle fetch error ({ticker}): {e}")
        import traceback; traceback.print_exc()
    return None


def _get_active_yf_ticker():
    """Determine which COMEX copper contract to show based on FND proximity.
    Returns (ticker, active_contract, label) where active_contract is 'front' or 'next'.
    Near FND (<=5 trading days), show the next month as liquidity migrates.
    """
    today = datetime.now().date()
    MONTHS = [
        ("H", "Mar", 3, 2), ("K", "May", 5, 4), ("N", "Jul", 7, 6),
        ("U", "Sep", 9, 8), ("Z", "Dec", 12, 11),
    ]
    contracts = []
    for year in [today.year, today.year + 1]:
        for code, label, del_mo, notice_mo in MONTHS:
            last = calendar.monthrange(year, notice_mo)[1]
            d = datetime(year, notice_mo, last).date()
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            yy = str(year)[-2:]
            contracts.append({"code": code, "yf": f"HG{code}{yy}.CMX", "fnd": d})
    contracts.sort(key=lambda c: c["fnd"])

    front = None
    next_mo = None
    for i, c in enumerate(contracts):
        if c["fnd"] >= today:
            front = c
            if i + 1 < len(contracts):
                next_mo = contracts[i + 1]
            break

    if not front:
        return "HG=F", "front"

    # Count trading days to FND
    days = 0
    d = today + timedelta(days=1)
    while d <= front["fnd"]:
        if d.weekday() < 5:
            days += 1
        d += timedelta(days=1)

    # Show next month when <=5 trading days to FND (liquidity migrating)
    if days <= 5 and next_mo:
        return next_mo["yf"], "next"
    return front["yf"], "front"


def _fetch_realtime_price():
    """Get most current COMEX copper price.
    Priority: TradingView WebSocket (near-real-time) → yfinance (15-30 min delayed).
    Returns dict with price, prev_close, source, and active_contract.
    """
    global _rt_cache
    now = time.time()

    # Method 1: TradingView WebSocket (near-real-time, front month continuous)
    with _tv_lock:
        tv_price = _tv_state["price"]
        tv_age = now - _tv_state["timestamp"] if _tv_state["timestamp"] else 999
        tv_prev = _tv_state["prev_close"]
        tv_change = _tv_state["change"]
        tv_change_pct = _tv_state["change_pct"]

    if tv_price and tv_age < 120:  # Accept if data is < 2 min old
        # HG1! is a continuous contract — near FND it rolls to the next month
        _, tv_active = _get_active_yf_ticker()
        _rt_cache = {"price": tv_price, "prev_close": tv_prev,
                     "change": tv_change, "change_pct": tv_change_pct,
                     "timestamp": now, "source": "tradingview",
                     "active_contract": tv_active}
        return _rt_cache

    # Method 2: yfinance fallback (15-30 min delayed, 1-min cache)
    if _rt_cache["price"] and (now - _rt_cache["timestamp"]) < 60 and _rt_cache["source"] == "yfinance":
        return _rt_cache

    active_ticker, active_contract = _get_active_yf_ticker()
    try:
        import yfinance as yf
        t = yf.Ticker(active_ticker)
        h = t.history(period="1d", interval="5m")
        if not h.empty:
            h = h.reset_index()
            h.columns = [c if isinstance(c, str) else c[0] for c in h.columns]
            price = float(h.iloc[-1]["Close"])
            if price > 1:
                prev_close = _fetch_prev_settle_yf(active_ticker)
                _rt_cache = {"price": round(price, 4), "prev_close": prev_close,
                             "timestamp": now, "source": "yfinance",
                             "active_contract": active_contract}
                print(f"[INFO] RT from yfinance ({active_ticker}): ${price:.4f} (prev settle: {prev_close})")
                return _rt_cache
    except Exception as e:
        print(f"[WARN] yfinance {active_ticker} intraday error: {e}")

    # Method 3: fallback to HG=F if specific contract failed
    if active_ticker != "HG=F":
        try:
            import yfinance as yf
            t = yf.Ticker("HG=F")
            h = t.history(period="1d", interval="5m")
            if not h.empty:
                h = h.reset_index()
                h.columns = [c if isinstance(c, str) else c[0] for c in h.columns]
                price = float(h.iloc[-1]["Close"])
                if price > 1:
                    prev_close = _fetch_prev_settle_yf("HG=F")
                    _rt_cache = {"price": round(price, 4), "prev_close": prev_close,
                                 "timestamp": now, "source": "yfinance",
                                 "active_contract": "front"}
                    print(f"[INFO] RT fallback from yfinance (HG=F): ${price:.4f}")
                    return _rt_cache
        except Exception as e:
            print(f"[WARN] yfinance HG=F fallback error: {e}")

    return _rt_cache


# ---------------------------------------------------------------------------
# INTRADAY SPARKLINE — 5-min bars for "Today" chart
# ---------------------------------------------------------------------------
_intraday_cache = {"data": None, "timestamp": 0}

def fetch_intraday_spark(ticker=None):
    """Fetch today's 5-min OHLC bars from yfinance for intraday sparkline."""
    global _intraday_cache
    now = time.time()
    if _intraday_cache["data"] and (now - _intraday_cache["timestamp"]) < 120:
        return _intraday_cache["data"]
    if not ticker:
        ticker, _ = _get_active_yf_ticker()
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        h = t.history(period="1d", interval="5m")
        if h.empty:
            return _intraday_cache["data"]
        h = h.reset_index()
        h.columns = [c if isinstance(c, str) else c[0] for c in h.columns]
        points = []
        for _, row in h.iterrows():
            ts = row["Datetime"] if "Datetime" in h.columns else row.get("Date")
            c = float(row["Close"])
            hi = float(row["High"])
            lo = float(row["Low"])
            if c > 1:
                label = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)[-8:-3]
                points.append({"time": label, "close": round(c, 4), "high": round(hi, 4), "low": round(lo, 4)})
        if points:
            _intraday_cache = {"data": points, "timestamp": now}
            return points
    except Exception as e:
        print(f"[WARN] intraday spark error: {e}")
    return _intraday_cache["data"]


# ---------------------------------------------------------------------------
# DXY
# ---------------------------------------------------------------------------
_dxy_cache = {"price": None, "change": None, "change_pct": None, "sparkline": [], "timestamp": 0}

def fetch_dxy():
    global _dxy_cache
    now = time.time()
    if _dxy_cache["price"] and (now - _dxy_cache["timestamp"]) < 300:
        return _dxy_cache
    try:
        import yfinance as yf
        ticker = yf.Ticker("DX-Y.NYB")
        hist = ticker.history(period="1mo", interval="1d")
        if not hist.empty:
            hist = hist.reset_index()
            hist.columns = [c if isinstance(c, str) else c[0] for c in hist.columns]
            latest = hist.iloc[-1]; prev = hist.iloc[-2] if len(hist) > 1 else latest
            p = float(latest["Close"]); pc = float(prev["Close"]); ch = p - pc
            spark = [{"date": r["Date"].strftime("%Y-%m-%d") if hasattr(r["Date"], "strftime") else str(r["Date"])[:10],
                      "close": round(float(r["Close"]), 2)} for _, r in hist.iterrows()]
            _dxy_cache = {"price": round(p, 2), "change": round(ch, 2),
                          "change_pct": round((ch / pc) * 100, 2) if pc else 0,
                          "sparkline": spark, "timestamp": now}
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
# COT — CFTC Commitment of Traders (4h cache)
# ---------------------------------------------------------------------------
_cot_cache = {"data": None, "timestamp": 0}

def fetch_cot_data():
    """Fetch CFTC Commitment of Traders data for COMEX copper.
    Uses disaggregated futures-only report via Socrata API (free, no auth).
    Cache: 4h normally, but force-refresh on Fridays after 3:30 PM ET if we
    don't yet have the current week's report (CFTC publishes ~3:30 PM ET Fri).
    """
    global _cot_cache
    now = time.time()
    cache_valid = False
    if _cot_cache["data"] and (now - _cot_cache["timestamp"]) < 14400:
        # Check if it's Friday after 3:30 PM ET and we might have a new report
        from datetime import timezone, timedelta
        import zoneinfo
        try:
            et = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            et = timezone(timedelta(hours=-4))
        now_et = datetime.now(et)
        if now_et.weekday() == 4 and (now_et.hour > 15 or (now_et.hour == 15 and now_et.minute >= 30)):
            # It's Friday after 3:30 PM ET — check if cached report is current week
            # Current week's report date = most recent Tuesday
            days_since_tue = (now_et.weekday() - 1) % 7  # Fri=4, Tue=1 → 3
            this_tuesday = (now_et - timedelta(days=days_since_tue)).strftime("%Y-%m-%d")
            cached_date = _cot_cache["data"].get("report_date", "")
            if cached_date < this_tuesday:
                print(f"[INFO] COT: Friday refresh — cached {cached_date}, expecting {this_tuesday}")
                cache_valid = False
            else:
                cache_valid = True
        else:
            cache_valid = True
    if cache_valid:
        return _cot_cache["data"]
    try:
        import urllib.request, urllib.parse
        base = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
        params = urllib.parse.urlencode({
            "$where": "commodity_name like '%COPPER%' AND open_interest_all > 100000",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": "52",
        })
        url = f"{base}?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "GeometDashboard/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            rows = json.loads(resp.read().decode())
        if not rows:
            print("[WARN] COT: no data returned")
            return None

        latest = rows[0]
        prior = rows[1] if len(rows) > 1 else None

        mm_long = int(latest.get("m_money_positions_long_all", 0))
        mm_short = int(latest.get("m_money_positions_short_all", 0))
        mm_net = mm_long - mm_short

        mm_weekly_change = 0
        if prior:
            prior_net = (int(prior.get("m_money_positions_long_all", 0))
                         - int(prior.get("m_money_positions_short_all", 0)))
            mm_weekly_change = mm_net - prior_net

        nets = []
        for r in rows:
            net = (int(r.get("m_money_positions_long_all", 0))
                   - int(r.get("m_money_positions_short_all", 0)))
            nets.append(net)
        mm_52w_low = min(nets)
        mm_52w_high = max(nets)
        rng = mm_52w_high - mm_52w_low
        mm_pct_52w = round((mm_net - mm_52w_low) / rng * 100, 1) if rng > 0 else 50

        if mm_pct_52w >= 90: crowding = "EXTREMELY_LONG"
        elif mm_pct_52w >= 70: crowding = "LONG"
        elif mm_pct_52w <= 10: crowding = "EXTREMELY_SHORT"
        elif mm_pct_52w <= 30: crowding = "SHORT"
        else: crowding = "NEUTRAL"

        prod_long = int(latest.get("prod_merc_positions_long_all",
                       latest.get("prod_merc_positions_long", 0)))
        prod_short = int(latest.get("prod_merc_positions_short_all",
                        latest.get("prod_merc_positions_short", 0)))
        prod_net = prod_long - prod_short

        swap_long = int(latest.get("swap_positions_long_all",
                        latest.get("swap__positions_long_all", 0)))
        swap_short = int(latest.get("swap__positions_short_all",
                         latest.get("swap_positions_short_all", 0)))
        swap_net = swap_long - swap_short

        traders_long = int(latest.get("traders_m_money_long_all", 0))
        traders_short = int(latest.get("traders_m_money_short_all", 0))
        report_date = latest.get("report_date_as_yyyy_mm_dd", "")[:10]
        oi_at_report = int(latest.get("open_interest_all", 0))

        # One-liner insight — only when positioning is notable
        insight = None
        if mm_pct_52w >= 85:
            insight = "Crowded long \u2014 selloffs will be sharp on bad news"
        elif mm_pct_52w >= 70:
            if mm_weekly_change < -3000:
                insight = "Funds long but trimming \u2014 momentum fading"
            else:
                insight = "Funds well long \u2014 price supported but less fuel to rally"
        elif mm_pct_52w <= 15:
            insight = "Funds heavily short \u2014 squeeze risk if sentiment turns"
        elif mm_pct_52w <= 30:
            if mm_weekly_change > 3000:
                insight = "Funds light but adding \u2014 early buying pressure"
            else:
                insight = "Funds light \u2014 buying fuel available on any catalyst"
        elif mm_pct_52w > 55 and mm_weekly_change < -5000:
            insight = "Funds unwinding longs \u2014 selling pressure"
        elif mm_pct_52w < 45 and mm_weekly_change > 5000:
            insight = "Funds covering shorts \u2014 buying pressure building"

        # Flag staleness: report > 9 days old means we missed a week
        days_old = (datetime.now() - datetime.strptime(report_date, "%Y-%m-%d")).days if report_date else 99
        stale = days_old > 9

        result = {
            "mm_net": mm_net, "mm_long": mm_long, "mm_short": mm_short,
            "mm_weekly_change": mm_weekly_change,
            "mm_pct_52w": mm_pct_52w, "mm_52w_low": mm_52w_low, "mm_52w_high": mm_52w_high,
            "crowding": crowding, "insight": insight,
            "prod_net": prod_net, "swap_net": swap_net,
            "report_date": report_date, "oi_at_report": oi_at_report,
            "traders_long": traders_long, "traders_short": traders_short,
            "stale": stale, "days_old": days_old,
        }
        _cot_cache = {"data": result, "timestamp": now}
        print(f"[INFO] COT: MM net {mm_net:+,} ({crowding}, {mm_pct_52w}th pctl) as of {report_date}")
        return result
    except Exception as e:
        print(f"[WARN] COT fetch error: {e}")
        return None


# ---------------------------------------------------------------------------
# CME COPPER OPTIONS OPEN INTEREST (daily bulletin PDF)
# ---------------------------------------------------------------------------
_options_oi_cache = {"data": None, "timestamp": 0}
OPTIONS_OI_CACHE_TTL = 43200  # 12 hours


def _oi_parse_int(s):
    """Parse integer from CME PDF field, handling commas."""
    try:
        return int(s.replace(",", "").strip())
    except Exception:
        return None


def _parse_cme_options_pdf():
    """Download and parse CME daily bulletin PDF for copper options OI.
    Returns dict with calls/puts per strike for the front month (highest OI)."""
    try:
        import pdfplumber
    except ImportError:
        print("[WARN] pdfplumber not installed — skipping options OI")
        return None

    import urllib.request
    import io
    import re

    url = "https://www.cmegroup.com/daily_bulletin/current/Section64_Metals_Option_Products.pdf"
    try:
        import subprocess
        result = subprocess.run([
            "curl", "-s", "-L", "--compressed", "--max-time", "45",
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "-H", "Accept-Language: en-US,en;q=0.9",
            "-H", "Sec-Fetch-Dest: document",
            "-H", "Sec-Fetch-Mode: navigate",
            "-H", "Sec-Fetch-Site: none",
            "-H", "Sec-Fetch-User: ?1",
            url
        ], capture_output=True, timeout=60)
        pdf_bytes = result.stdout
        if not pdf_bytes or len(pdf_bytes) < 10000:
            print(f"[WARN] Options OI PDF download failed: got {len(pdf_bytes)} bytes")
            return None
    except Exception as e:
        print(f"[WARN] Options OI PDF download failed: {e}")
        return None

    # Regex: data line starts with 3-4 digit strike, ends with OI + change/UNCH
    # strike ... OI +/- change  OR  strike ... OI UNCH
    oi_line_re = re.compile(
        r'^\s*(\d{3,4})\s+'        # strike in cents
        r'.*\s'                     # middle (prices, ranges, etc.)
        r'(\d{1,6})\s+'            # open interest
        r'([+-]\s*\d+|UNCH)\s*$'   # change or UNCH
    )
    # Settlement + delta pattern in middle of line:
    # settlement [+/-/NEW] pt_change delta
    settle_re = re.compile(
        r'\s([\d.]+)\s+'           # settlement price
        r'([+-]\s*[\d.]+|NEW|UNCH)\s+'  # point change
        r'([.\d]{4,6})\s'          # delta (.XXXX)
    )

    calls = {}  # {month: {strike_cents: {"oi": int, "settle": float, "delta": float}}}
    puts = {}
    bulletin_date = ""

    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as e:
        print(f"[WARN] Options OI PDF parse error: {e}")
        return None

    in_hx_call = False
    in_hx_put = False
    current_month = None

    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        # Extract bulletin date from header
        if not bulletin_date:
            dm = re.search(r'BULLETIN\s+#\s*\d+@?\s+.*?\s+((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\w+\s+\d+,\s+\d{4})', text)
            if dm:
                bulletin_date = dm.group(1)

        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()

            # Detect section headers
            if "HX CALL" in upper and "COMEX COPPER" in upper:
                in_hx_call = True
                in_hx_put = False
                current_month = None
                continue
            if "HX PUT" in upper and "COMEX COPPER" in upper:
                in_hx_put = True
                in_hx_call = False
                current_month = None
                continue

            # End of HX sections: next product (HXE, gold, silver, etc.)
            if (in_hx_call or in_hx_put) and re.match(r'^(HXE|HWR|HWT|HWW|OG|SO|SI)\s', upper):
                in_hx_call = False
                in_hx_put = False
                current_month = None
                continue

            if not in_hx_call and not in_hx_put:
                continue

            # Month header (e.g. APR26, MAY26)
            month_m = re.match(r'^([A-Z]{3}\d{2})\s*$', stripped)
            if month_m:
                current_month = month_m.group(1)
                continue

            # TOTAL line — skip
            if stripped.startswith("TOTAL"):
                continue

            if not current_month:
                continue

            # Parse data line
            m = oi_line_re.match(stripped)
            if not m:
                continue

            strike_cents = int(m.group(1))
            oi = _oi_parse_int(m.group(2))
            if oi is None or oi == 0:
                continue

            # Extract settlement and delta
            settle = None
            delta = None
            sm = settle_re.search(stripped)
            if sm:
                try:
                    settle = float(sm.group(1))
                except Exception:
                    pass
                try:
                    delta = float(sm.group(3))
                except Exception:
                    pass

            target = calls if in_hx_call else puts
            if current_month not in target:
                target[current_month] = {}

            # Aggregate OI at same strike (some strikes appear multiple times)
            if strike_cents in target[current_month]:
                target[current_month][strike_cents]["oi"] += oi
            else:
                target[current_month][strike_cents] = {
                    "oi": oi, "settle": settle, "delta": delta
                }

    pdf.close()

    if not calls and not puts:
        print("[WARN] Options OI: no HX data found in PDF")
        return None

    # Identify front month by highest total OI across calls+puts
    month_oi = {}
    for month, strikes in calls.items():
        month_oi[month] = month_oi.get(month, 0) + sum(s["oi"] for s in strikes.values())
    for month, strikes in puts.items():
        month_oi[month] = month_oi.get(month, 0) + sum(s["oi"] for s in strikes.values())

    if not month_oi:
        return None

    front_month = max(month_oi, key=month_oi.get)
    fm_calls = calls.get(front_month, {})
    fm_puts = puts.get(front_month, {})

    # --- Analytics ---

    # Max pain first (needed to anchor near-money filter)
    all_strikes = sorted(set(list(fm_calls.keys()) + list(fm_puts.keys())))
    max_pain = None
    if all_strikes:
        min_pain_cost = float("inf")
        for test_strike in all_strikes:
            total_cost = 0
            for cs, cd in fm_calls.items():
                if test_strike >= cs:
                    total_cost += cd["oi"] * (test_strike - cs)
            for ps, pd in fm_puts.items():
                if test_strike <= ps:
                    total_cost += pd["oi"] * (ps - test_strike)
            if total_cost < min_pain_cost:
                min_pain_cost = total_cost
                max_pain = {"strike": test_strike / 100.0, "strike_cents": test_strike}

    # Near-money filter: ±20% of max pain (or median strike if no max pain)
    anchor = max_pain["strike_cents"] if max_pain else (all_strikes[len(all_strikes)//2] if all_strikes else 550)
    near_lo = int(anchor * 0.80)
    near_hi = int(anchor * 1.20)

    # Put wall: highest put OI at or below anchor, within near-money range
    put_wall = None
    if fm_puts:
        near_puts = {k: v for k, v in fm_puts.items() if near_lo <= k <= anchor}
        if near_puts:
            pw_strike = max(near_puts, key=lambda k: near_puts[k]["oi"])
            put_wall = {"strike": pw_strike / 100.0, "oi": near_puts[pw_strike]["oi"],
                         "strike_cents": pw_strike}

    # Call wall: highest call OI at or above anchor, within near-money range
    call_wall = None
    if fm_calls:
        near_calls = {k: v for k, v in fm_calls.items() if anchor <= k <= near_hi}
        if near_calls:
            cw_strike = max(near_calls, key=lambda k: near_calls[k]["oi"])
            call_wall = {"strike": cw_strike / 100.0, "oi": near_calls[cw_strike]["oi"],
                          "strike_cents": cw_strike}

    # P/C ratio (near-money only for meaningful signal)
    nm_put_oi = sum(v["oi"] for k, v in fm_puts.items() if near_lo <= k <= near_hi)
    nm_call_oi = sum(v["oi"] for k, v in fm_calls.items() if near_lo <= k <= near_hi)
    total_put_oi = sum(s["oi"] for s in fm_puts.values())
    total_call_oi = sum(s["oi"] for s in fm_calls.values())
    pc_ratio = round(nm_put_oi / nm_call_oi, 2) if nm_call_oi > 0 else None

    # Top 10 strikes by combined OI (near-money only)
    combined = {}
    for s, d in fm_calls.items():
        if s < near_lo or s > near_hi:
            continue
        combined[s] = combined.get(s, {"call_oi": 0, "put_oi": 0})
        combined[s]["call_oi"] += d["oi"]
    for s, d in fm_puts.items():
        if s < near_lo or s > near_hi:
            continue
        combined[s] = combined.get(s, {"call_oi": 0, "put_oi": 0})
        combined[s]["put_oi"] += d["oi"]

    top10 = sorted(combined.items(), key=lambda x: x[1]["call_oi"] + x[1]["put_oi"], reverse=True)[:10]
    top10_list = [{"strike": s / 100.0, "call_oi": d["call_oi"], "put_oi": d["put_oi"],
                   "total_oi": d["call_oi"] + d["put_oi"]} for s, d in top10]
    # Sort by strike for display
    top10_list.sort(key=lambda x: x["strike"])

    # All near-money strikes bucketed to 10¢ increments for zoomed chart
    buckets = {}
    for s, d in fm_calls.items():
        if s < near_lo or s > near_hi:
            continue
        bucket = (s // 10) * 10  # round down to nearest 10¢
        buckets[bucket] = buckets.get(bucket, {"call_oi": 0, "put_oi": 0})
        buckets[bucket]["call_oi"] += d["oi"]
    for s, d in fm_puts.items():
        if s < near_lo or s > near_hi:
            continue
        bucket = (s // 10) * 10
        buckets[bucket] = buckets.get(bucket, {"call_oi": 0, "put_oi": 0})
        buckets[bucket]["put_oi"] += d["oi"]
    all_buckets = [{"strike": b / 100.0, "call_oi": d["call_oi"], "put_oi": d["put_oi"],
                    "total_oi": d["call_oi"] + d["put_oi"]}
                   for b, d in sorted(buckets.items())]

    result = {
        "front_month": front_month,
        "bulletin_date": bulletin_date,
        "put_wall": put_wall,
        "call_wall": call_wall,
        "max_pain": max_pain,
        "pc_ratio": pc_ratio,
        "total_put_oi": total_put_oi,
        "total_call_oi": total_call_oi,
        "top10": top10_list,
        "buckets": all_buckets,
    }

    if put_wall and call_wall and max_pain:
        print(f"[INFO] Options OI parsed: {front_month} — put wall ${put_wall['strike']:.2f} "
              f"({put_wall['oi']:,} contracts) / call wall ${call_wall['strike']:.2f} "
              f"({call_wall['oi']:,} contracts) / max pain ${max_pain['strike']:.2f}")
    return result


def fetch_options_oi():
    """Fetch CME copper options OI with 12h memory cache → 36h file cache → PDF parse."""
    global _options_oi_cache
    now = time.time()

    # Memory cache (12h)
    if _options_oi_cache["data"] and (now - _options_oi_cache["timestamp"]) < OPTIONS_OI_CACHE_TTL:
        return _options_oi_cache["data"]

    # File cache (36h)
    if OPTIONS_OI_FILE.exists():
        try:
            age = now - OPTIONS_OI_FILE.stat().st_mtime
            if age < 129600:  # 36 hours
                with open(OPTIONS_OI_FILE) as f:
                    data = json.load(f)
                if data:
                    _options_oi_cache = {"data": data, "timestamp": now}
                    print(f"[INFO] Options OI loaded from file cache ({age / 3600:.0f}h old)")
                    return data
        except Exception:
            pass

    # Fresh parse
    result = _parse_cme_options_pdf()
    if result:
        _options_oi_cache = {"data": result, "timestamp": now}
        try:
            with open(OPTIONS_OI_FILE, "w") as f:
                json.dump(result, f)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# CHINA / SHFE STATUS
# ---------------------------------------------------------------------------
def get_china_status():
    now = datetime.now()
    wd = now.weekday(); hr = now.hour
    if wd >= 5:
        return {"status": "CLOSED", "reason": "Weekend", "detail": "SHFE closed \u2014 weekend",
                "color": "yellow", "thin_liquidity": wd == 6 and hr >= 17, "desc": ""}
    if hr >= 19 or hr < 2:
        return {"status": "OPEN", "reason": "Night session", "detail": "SHFE night session active",
                "color": "green", "thin_liquidity": False,
                "desc": "Night session \u2014 tracks US/LME overnight"}
    if 7 <= hr <= 14:
        return {"status": "CLOSED", "reason": "Between sessions", "detail": "SHFE between sessions",
                "color": "yellow", "thin_liquidity": False, "desc": ""}
    return {"status": "CLOSED", "reason": "Off hours", "detail": "SHFE closed",
            "color": "yellow", "thin_liquidity": False, "desc": ""}


# ---------------------------------------------------------------------------
# LME STATUS — electronic + ring hours (London time)
# ---------------------------------------------------------------------------
def get_lme_status():
    """LME market hours status based on London time.
    LME Select (electronic): 01:00-19:00 London
    Official Ring session: 11:40-17:00 London
    Includes UK bank holiday closures.
    """
    from datetime import timezone
    import zoneinfo
    try:
        london = zoneinfo.ZoneInfo("Europe/London")
    except Exception:
        # Fallback: UTC offset approximation (GMT/BST)
        london = timezone.utc
    now_london = datetime.now(london)
    wd = now_london.weekday()
    hr = now_london.hour
    mn = now_london.minute
    t = hr * 60 + mn  # minutes since midnight

    if wd >= 5:
        return {"status": "CLOSED", "detail": "LME CLOSED", "color": "yellow", "session": "weekend", "desc": ""}

    # UK bank holidays — LME closed
    # Easter-based dates shift yearly; update annually or compute dynamically
    today_str = now_london.strftime("%m-%d")
    year = now_london.year
    uk_holidays = _get_uk_bank_holidays(year)
    if now_london.date() in uk_holidays:
        label = uk_holidays[now_london.date()]
        return {"status": "CLOSED", "detail": "LME HOLIDAY", "color": "yellow", "session": "holiday",
                "desc": label}

    # Ring session: 11:40 (700) - 17:00 (1020) London
    if 700 <= t < 1020:
        return {"status": "RING", "detail": "LME RING", "color": "green", "session": "ring",
                "desc": "Official ring \u2014 benchmark pricing, peak LME liquidity"}

    # LME Select electronic: 01:00 (60) - 19:00 (1140) London
    if 60 <= t < 1140:
        return {"status": "OPEN", "detail": "LME OPEN", "color": "green", "session": "electronic",
                "desc": "Electronic session \u2014 steady liquidity"}

    return {"status": "CLOSED", "detail": "LME CLOSED", "color": "yellow", "session": "closed", "desc": ""}


def _get_uk_bank_holidays(year):
    """Return dict of {date: label} for UK bank holidays that close the LME."""
    from datetime import date, timedelta
    holidays = {}

    # Fixed dates
    holidays[date(year, 1, 1)] = "New Year's Day"
    holidays[date(year, 12, 25)] = "Christmas Day"
    holidays[date(year, 12, 26)] = "Boxing Day"

    # If Christmas/Boxing Day fall on weekend, substitute Monday/Tuesday
    xmas = date(year, 12, 25)
    if xmas.weekday() == 5:  # Saturday
        holidays[date(year, 12, 27)] = "Christmas substitute"
        holidays[date(year, 12, 28)] = "Boxing Day substitute"
    elif xmas.weekday() == 6:  # Sunday
        holidays[date(year, 12, 27)] = "Boxing Day substitute"
        holidays[date(year, 12, 28)] = "Christmas substitute"

    if date(year, 1, 1).weekday() == 5:
        holidays[date(year, 1, 3)] = "New Year substitute"
    elif date(year, 1, 1).weekday() == 6:
        holidays[date(year, 1, 2)] = "New Year substitute"

    # Early May bank holiday (first Monday in May)
    d = date(year, 5, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    holidays[d] = "Early May Bank Holiday"

    # Spring bank holiday (last Monday in May)
    d = date(year, 5, 31)
    while d.weekday() != 0:
        d -= timedelta(days=1)
    holidays[d] = "Spring Bank Holiday"

    # Summer bank holiday (last Monday in August)
    d = date(year, 8, 31)
    while d.weekday() != 0:
        d -= timedelta(days=1)
    holidays[d] = "Summer Bank Holiday"

    # Easter (computed via anonymous Gregorian algorithm)
    a = year % 19
    b, c = divmod(year, 100)
    d_val, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d_val - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m_val = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m_val + 114) // 31
    day = ((h + l - 7 * m_val + 114) % 31) + 1
    easter_sunday = date(year, month, day)

    holidays[easter_sunday - timedelta(days=2)] = "Good Friday"
    holidays[easter_sunday + timedelta(days=1)] = "Easter Monday"

    return holidays


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CUSTOMER AVAILABILITY — fix-pricing windows
# ---------------------------------------------------------------------------
def get_customer_availability():
    """Check which customers are available to fix prices right now."""
    hours_cfg = CFG.get("CUSTOMER_HOURS", {})
    if not hours_cfg:
        return {}
    now = datetime.now()
    hr = now.hour
    result = {}
    for name, info in hours_cfg.items():
        s, e = info["start"], info["end"]
        # Overnight window (e.g. 19-10 means 7PM to 10AM)
        if s > e:
            available = hr >= s or hr < e
        else:
            available = s <= hr < e
        result[name] = {
            "available": available,
            "hours": f"{s % 12 or 12}{'am' if s < 12 else 'pm'}-{e % 12 or 12}{'am' if e < 12 else 'pm'}",
            "basis": info.get("basis", ""),
        }
    return result


# ---------------------------------------------------------------------------
# COMEX STATUS — Globex hours (Chicago/Central time)
# ---------------------------------------------------------------------------
def get_comex_status():
    """COMEX Globex hours: Sun 5PM CT - Fri 4PM CT, daily break 4-5PM CT Mon-Thu."""
    import zoneinfo
    try:
        chicago = zoneinfo.ZoneInfo("America/Chicago")
    except Exception:
        from datetime import timezone, timedelta
        chicago = timezone(timedelta(hours=-6))
    now_ct = datetime.now(chicago)
    wd = now_ct.weekday()  # 0=Mon
    hr = now_ct.hour
    mn = now_ct.minute
    t = hr * 60 + mn

    # Weekend: closed from Fri 4PM CT (wd=4, t>=960) to Sun 5PM CT (wd=6, t>=1020)
    if wd == 5:  # Saturday — always closed
        return {"status": "CLOSED", "detail": "COMEX CLOSED", "color": "red", "reason": "weekend", "window": "", "desc": "", "is_peak": False}
    if wd == 6 and t < 1020:  # Sunday before 5PM CT
        return {"status": "CLOSED", "detail": "COMEX CLOSED", "color": "red", "reason": "weekend", "window": "", "desc": "", "is_peak": False}
    if wd == 4 and t >= 960:  # Friday 4PM+ CT
        return {"status": "CLOSED", "detail": "COMEX CLOSED", "color": "red", "reason": "weekend", "window": "", "desc": "", "is_peak": False}

    # Daily maintenance break: 4PM-5PM CT (960-1020) Mon-Thu
    if 960 <= t < 1020 and wd <= 3:
        return {"status": "CLOSED", "detail": "COMEX MAINT", "color": "yellow", "reason": "maintenance", "window": "", "desc": "", "is_peak": False}

    # OPEN — determine sub-window
    # 7:30-10:00 AM CT (450-600): peak liquidity
    if 450 <= t < 600:
        window, desc, is_peak = "peak", "Peak liquidity \u2014 biggest volume, tightest spreads, most price action", True
    # 5:00-7:30 AM CT (300-450): early session
    elif 300 <= t < 450:
        window, desc, is_peak = "early", "Early session \u2014 London overlap, volume building", False
    # 10:00 AM-12:00 PM CT (600-720): midday
    elif 600 <= t < 720:
        window, desc, is_peak = "midday", "Midday \u2014 volume tapering, less reactive", False
    # 12:00-4:00 PM CT (720-960): afternoon
    elif 720 <= t < 960:
        window, desc, is_peak = "afternoon", "Afternoon \u2014 thinner liquidity, wider spreads", False
    # 5:00-8:00 PM CT (1020-1200): evening open
    elif t >= 1020:
        window, desc, is_peak = "evening", "Evening open \u2014 Asia overlap, moderate flow", False
    # 8:00 PM-5:00 AM CT (1200-0 + 0-300): overnight
    else:
        window, desc, is_peak = "overnight", "Overnight \u2014 follows Shanghai/London moves", False

    return {"status": "OPEN", "detail": "COMEX OPEN", "color": "green", "reason": "globex",
            "window": window, "desc": desc, "is_peak": is_peak}


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
    if len(closes) < 20: return {"support": [], "resistance": [], "near_support": [], "near_resistance": [], "context": []}
    current = closes[-1]

    def cluster(levels, threshold=0.03):
        if not levels: return []
        levels = sorted(levels); clusters = []; cc = [levels[0]]
        for i in range(1, len(levels)):
            if levels[i] - cc[-1] < threshold: cc.append(levels[i])
            else: clusters.append(round(sum(cc)/len(cc), 4)); cc = [levels[i]]
        clusters.append(round(sum(cc)/len(cc), 4)); return clusters

    def find_swings(h, l, lookback):
        sh = []; sl = []
        for i in range(lookback, len(h) - lookback):
            if h[i] == max(h[i-lookback:i+lookback+1]): sh.append(round(h[i], 4))
            if l[i] == min(l[i-lookback:i+lookback+1]): sl.append(round(l[i], 4))
        return sh, sl

    def make_levels(sh, sl, n=3):
        sup = sorted(cluster([s for s in sl if s < current]), reverse=True)[:n]
        res = sorted(cluster([r for r in sh if r > current]))[:n]
        return (
            [{"level": s, "distance": round(current-s, 4), "distance_pct": round((current-s)/current*100, 2)} for s in sup],
            [{"level": r, "distance": round(r-current, 4), "distance_pct": round((r-current)/current*100, 2)} for r in res],
        )

    # Major S/R — full 90-day window, lookback=5
    major_sh, major_sl = find_swings(highs, lows, 5)
    support, resistance = make_levels(major_sh, major_sl)

    # Near-term S/R — last 20 days, lookback=3 (tighter swings)
    near_days = min(20, len(closes))
    near_h, near_l = highs[-near_days:], lows[-near_days:]
    near_sh, near_sl = find_swings(near_h, near_l, 3)
    near_support, near_resistance = make_levels(near_sh, near_sl, 2)

    # Context — plain-English interpretation
    context = []
    if near_support:
        ns = near_support[0]
        # Check if major support confirms near-term (within 5c)
        confirmed = support and abs(support[0]["level"] - ns["level"]) <= 0.05
        if confirmed:
            context.append(f"Near-term floor ${ns['level']:.2f} ({ns['distance']*100:.0f}c below) — confirmed by 90-day support, strong reload zone on a dip")
        else:
            context.append(f"Near-term floor ${ns['level']:.2f} ({ns['distance']*100:.0f}c below) — if a dip holds here, short-term trend intact")
    if support:
        ms = support[0]
        if not near_support or abs(ms["level"] - near_support[0]["level"]) > 0.05:
            context.append(f"Major support ${ms['level']:.2f} ({ms['distance']*100:.0f}c below) — break below this signals bigger trend change")
        # Add deeper major support if available (second level)
        if len(support) >= 2:
            ms2 = support[1]
            context.append(f"Deep support ${ms2['level']:.2f} ({ms2['distance']*100:.0f}c below) — worst case floor, long-term uptrend holds above here")
    if near_resistance:
        nr = near_resistance[0]
        confirmed = resistance and abs(resistance[0]["level"] - nr["level"]) <= 0.05
        if confirmed:
            context.append(f"Near-term ceiling ${nr['level']:.2f} ({nr['distance']*100:.0f}c above) — confirmed by 90-day resistance, strong wall")
        else:
            context.append(f"Near-term ceiling ${nr['level']:.2f} ({nr['distance']*100:.0f}c above) — expect sellers here short-term")
    if resistance:
        mr = resistance[0]
        if not near_resistance or abs(mr["level"] - near_resistance[0]["level"]) > 0.05:
            context.append(f"Major resistance ${mr['level']:.2f} ({mr['distance']*100:.0f}c above) — breakout above means new leg higher")

    return {
        "support": support, "resistance": resistance,
        "near_support": near_support, "near_resistance": near_resistance,
        "context": context,
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


def calc_price_outlook(sig, md, cot, roll, options_oi=None):
    """Directional price outlook for 3 timeframes: today, this week, this month.
    Each returns score (-100 to +100), label, confidence, and top 2 reasons."""
    if not sig or not md:
        return None

    roc = md.get("roc", {})
    streak = md.get("streak", 0)
    streak_dir = md.get("streak_dir")
    vol_ratio = md.get("vol_ratio", 1.0)
    dxy = md.get("dxy", {})
    trend = sig.get("trend", "")
    ts = sig.get("trend_strength", "")
    price = md.get("price", 0)

    # ── TODAY (intraday bias) ──
    today_score = 0
    today_reasons = []

    # Momentum/ROC 1d-3d (heavy weight)
    r1 = roc.get("1d", {}).get("pct", 0)
    r3 = roc.get("3d", {}).get("pct", 0)
    mom_avg = (r1 * 2 + r3) / 3
    if mom_avg > 1.5:
        today_score += 30; today_reasons.append(f"Strong momentum +{mom_avg:.1f}%")
    elif mom_avg > 0.5:
        today_score += 18; today_reasons.append(f"Positive momentum +{mom_avg:.1f}%")
    elif mom_avg < -1.5:
        today_score -= 30; today_reasons.append(f"Negative momentum {mom_avg:.1f}%")
    elif mom_avg < -0.5:
        today_score -= 18; today_reasons.append(f"Weak momentum {mom_avg:.1f}%")

    # Streak direction
    if streak >= 3 and streak_dir == "up":
        today_score += 15; today_reasons.append(f"{streak}-day winning streak")
    elif streak >= 3 and streak_dir == "down":
        today_score -= 15; today_reasons.append(f"{streak}-day losing streak")
    elif streak >= 2 and streak_dir == "up":
        today_score += 8
    elif streak >= 2 and streak_dir == "down":
        today_score -= 8

    # Volume conviction
    if vol_ratio > 1.5 and r1 > 0:
        today_score += 12; today_reasons.append("High volume confirms move")
    elif vol_ratio > 1.5 and r1 < 0:
        today_score -= 12; today_reasons.append("High volume selling")
    elif vol_ratio < 0.5:
        today_score -= 5  # thin volume = less reliable

    # DXY intraday
    dch = dxy.get("change_pct", 0)
    if dch < -0.3:
        today_score += 12; today_reasons.append("Dollar weakening")
    elif dch > 0.3:
        today_score -= 12; today_reasons.append("Dollar strengthening")

    # China/LME session (more liquidity when open)
    china = md.get("china", {})
    if china.get("thin_liquidity"):
        today_score *= 0.7  # dampen signals in thin markets
        today_score = int(today_score)

    # Calendar adjustments — TODAY
    today_cal_note = None
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon, 4=Fri
    pct90 = md.get("pct_90d", 50)
    days_fnd = roll.get("days_to_fnd", 99) if roll else 99

    if weekday == 4 and pct90 >= 75:
        today_score -= 10; today_reasons.append("Friday profit-taking risk")
    if days_fnd <= 2:
        today_score -= 12; today_reasons.append("FND imminent — roll selling")
    if weekday == 4 and (now.hour > 15 or (now.hour == 15 and now.minute >= 30)):
        today_cal_note = "COT report dropping — positioning shift possible"

    today_score = max(-100, min(100, today_score))

    # ── THIS WEEK ──
    week_score = 0
    week_reasons = []

    # Trend + strength (primary)
    if trend == "UPTREND" and ts == "strong":
        week_score += 30; week_reasons.append("Strong uptrend")
    elif trend == "UPTREND":
        week_score += 18; week_reasons.append("Uptrend intact")
    elif trend == "DOWNTREND" and ts == "strong":
        week_score -= 30; week_reasons.append("Strong downtrend")
    elif trend == "DOWNTREND":
        week_score -= 18; week_reasons.append("Downtrend pressure")

    # ROC 5d
    r5 = roc.get("5d", {}).get("pct", 0)
    if r5 > 2:
        week_score += 20; week_reasons.append(f"5d momentum +{r5:.1f}%")
    elif r5 > 0.5:
        week_score += 10; week_reasons.append(f"5d momentum +{r5:.1f}%")
    elif r5 < -2:
        week_score -= 20; week_reasons.append(f"5d momentum {r5:.1f}%")
    elif r5 < -0.5:
        week_score -= 10; week_reasons.append(f"5d momentum {r5:.1f}%")

    # COT weekly change
    if cot:
        wc = cot.get("mm_weekly_change", 0)
        if wc > 5000:
            week_score += 12; week_reasons.append("Funds adding longs")
        elif wc < -5000:
            week_score -= 12; week_reasons.append("Funds cutting longs")

    # OI trend
    if roll and roll.get("open_interest"):
        oi = roll["open_interest"]
        oi_trend = oi.get("trend", "")
        if oi_trend == "building":
            week_score += 8; week_reasons.append("Open interest building")
        elif oi_trend == "declining":
            week_score -= 8; week_reasons.append("Open interest declining")

    # DXY trend
    if dch < -0.3:
        week_score += 8; week_reasons.append("Dollar weak")
    elif dch > 0.3:
        week_score -= 8; week_reasons.append("Dollar firm")

    # S/R proximity
    sr = md.get("support_resistance", {})
    if sr:
        supports = sr.get("support", [])
        resistances = sr.get("resistance", [])
        if supports and price:
            nearest_sup = max(s["level"] for s in supports) if supports else 0
            if nearest_sup and (price - nearest_sup) / price < 0.01:
                week_score += 8; week_reasons.append("Near support")
        if resistances and price:
            nearest_res = min(r["level"] for r in resistances) if resistances else 999
            if nearest_res and (nearest_res - price) / price < 0.01:
                week_score -= 8; week_reasons.append("Near resistance")

    # Options OI wall proximity
    if options_oi and price:
        pw = options_oi.get("put_wall")
        cw = options_oi.get("call_wall")
        mp = options_oi.get("max_pain")
        if pw and price:
            pw_dist = (price - pw["strike"]) / price
            if pw_dist < 0.01 and pw_dist >= 0:
                week_score += 12; week_reasons.append(f"Near put wall ${pw['strike']:.2f} (institutional support)")
            elif pw_dist < 0:
                week_score -= 8; week_reasons.append(f"Below put wall ${pw['strike']:.2f}")
        if cw and price:
            cw_dist = (cw["strike"] - price) / price
            if cw_dist < 0.01 and cw_dist >= 0:
                week_score -= 10; week_reasons.append(f"Near call wall ${cw['strike']:.2f} (institutional resistance)")
            elif cw_dist < 0:
                week_score += 8; week_reasons.append(f"Above call wall ${cw['strike']:.2f}")
        if mp and price:
            mp_dist = price - mp["strike"]
            if abs(mp_dist) / price < 0.005:
                week_reasons.append(f"At max pain ${mp['strike']:.2f} (expiry magnet)")

    # Calendar adjustments — THIS WEEK
    week_cal_note = None
    if days_fnd <= 5:
        penalty = 15 if days_fnd <= 2 else 8
        week_score -= penalty
        week_reasons.append(f"FND in {days_fnd} days — liquidity migrating")

    # Month-end rebalancing: ≤5 trading days left in month + elevated price
    today_date = now.date()
    last_day = today_date.replace(day=calendar.monthrange(today_date.year, today_date.month)[1])
    td_left = 0
    d = today_date
    while d <= last_day:
        if d.weekday() < 5:
            td_left += 1
        d += timedelta(days=1)
    if td_left <= 5 and pct90 >= 70:
        week_score -= 8; week_reasons.append("Month-end rebalancing window")

    if cot and cot.get("days_old", 0) > 9:
        week_cal_note = "COT data stale — positioning uncertain"

    week_score = max(-100, min(100, week_score))

    # ── THIS MONTH ──
    month_score = 0
    month_reasons = []

    # DMA alignment
    above50 = sig.get("above_50", False)
    above100 = sig.get("above_100", False)
    above200 = sig.get("above_200", False)
    dma_count = sum([above50, above100, above200])
    if dma_count == 3:
        month_score += 25; month_reasons.append("Above all moving averages")
    elif dma_count == 2:
        month_score += 12; month_reasons.append("Above 2 of 3 DMAs")
    elif dma_count == 0:
        month_score -= 25; month_reasons.append("Below all moving averages")
    elif dma_count == 1:
        month_score -= 10; month_reasons.append("Below most moving averages")

    # COT percentile extremes (contrarian)
    if cot:
        pct = cot.get("mm_pct_52w", 50)
        if pct > 90:
            month_score -= 12; month_reasons.append(f"Funds crowded long ({pct:.0f}th pctl)")
        elif pct > 70:
            month_score += 8; month_reasons.append("Funds well positioned long")
        elif pct < 10:
            month_score += 15; month_reasons.append(f"Funds crowded short ({pct:.0f}th pctl)")
        elif pct < 30:
            month_score -= 5; month_reasons.append("Funds lightly positioned")

    # Warehouse trend
    wh = md.get("warehouse", {})
    if wh:
        wh_trend = wh.get("trend", "")
        if wh_trend == "drawing":
            month_score += 15; month_reasons.append("Warehouse stocks drawing")
        elif wh_trend == "building":
            month_score -= 15; month_reasons.append("Warehouse stocks building")

    # Market structure (backwardation/contango)
    if roll:
        structure = roll.get("market_structure", "")
        if structure == "backwardation":
            month_score += 12; month_reasons.append("Backwardation (tight supply)")
        elif structure == "contango":
            month_score -= 8; month_reasons.append("Contango (ample supply)")

    # 90d percentile (mean reversion at extremes)
    if pct90 >= 90:
        month_score -= 10; month_reasons.append(f"Near 90d highs ({pct90:.0f}th pctl)")
    elif pct90 <= 10:
        month_score += 10; month_reasons.append(f"Near 90d lows ({pct90:.0f}th pctl)")

    # Fed rate trajectory
    fed = md.get("fed", {})
    if fed:
        traj = fed.get("trajectory", "")
        if traj == "cutting":
            month_score += 10; month_reasons.append("Fed cutting rates")
        elif traj == "hiking":
            month_score -= 10; month_reasons.append("Fed hiking rates")

    # Calendar adjustments — THIS MONTH
    month_cal_note = None
    if days_fnd <= 10:
        month_score -= 5; month_reasons.append("Roll period approaching")

    first_cut = fed.get("first_cut", "") if fed else ""
    if first_cut and first_cut != "None priced":
        cur_month_label = now.strftime("%b") + " " + now.strftime("%y")
        next_month = (now.replace(day=28) + timedelta(days=4)).replace(day=1)
        next_month_label = next_month.strftime("%b") + " " + next_month.strftime("%y")
        if first_cut in (cur_month_label, next_month_label):
            month_score += 8; month_reasons.append(f"Rate cut expected {first_cut}")

    month_score = max(-100, min(100, month_score))

    # ── Build output ──
    def _build(score, reasons, cal_note=None):
        if score >= 40:
            label = "HIGHER"
        elif score >= 15:
            label = "LEAN HIGHER"
        elif score <= -40:
            label = "LOWER"
        elif score <= -15:
            label = "LEAN LOWER"
        else:
            label = "NEUTRAL"
        # Confidence = how many reasons agree on direction
        pos_r = sum(1 for _ in reasons if score > 0)
        neg_r = sum(1 for _ in reasons if score < 0)
        conf = min(5, max(1, max(pos_r, neg_r)))
        # Top 2 reasons
        top2 = reasons[:2] if reasons else ["No strong signals"]
        out = {"label": label, "score": score, "confidence": conf, "reasons": top2}
        if cal_note:
            out["calendar_note"] = cal_note
        return out

    return {
        "today": _build(today_score, today_reasons, today_cal_note),
        "week": _build(week_score, week_reasons, week_cal_note),
        "month": _build(month_score, month_reasons, month_cal_note),
    }


def calc_fixable_orders(pos, md):
    """Determine which unpriced orders are fixable now based on exchange hours."""
    if not pos or not md:
        return None
    shipped = pos.get("sales_unpriced_shipped", [])
    unshipped = pos.get("sales_unpriced_unshipped", [])
    all_unpriced = shipped + unshipped
    if not all_unpriced:
        return None

    lme = md.get("lme", {})
    lme_open = lme.get("status") in ("OPEN", "RING") if lme else False

    fixable = 0; fixable_lbs = 0; blocked_lme = 0; blocked_lme_lbs = 0
    shipped_fixable = []
    for sale in all_unpriced:
        basis = sale.get("basis", "COMEX")
        lbs = sale.get("open_lbs", sale.get("lbs", 0))
        if basis == "LME" and not lme_open:
            blocked_lme += 1; blocked_lme_lbs += lbs
        else:
            fixable += 1; fixable_lbs += lbs
            if sale in shipped:
                shipped_fixable.append(sale)

    return {
        "fixable_count": fixable, "fixable_lbs": round(fixable_lbs),
        "blocked_lme_count": blocked_lme, "blocked_lme_lbs": round(blocked_lme_lbs),
        "total_unpriced": len(all_unpriced),
        "shipped_fixable": shipped_fixable,
    }


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

_lme_wh_cache = {"data": None, "timestamp": 0}

def fetch_lme_warehouse():
    """Scrape LME copper warehouse stocks + Cash/3M settlements from westmetall.com (24h cache)."""
    global _lme_wh_cache
    now = time.time()
    if _lme_wh_cache["data"] and (now - _lme_wh_cache["timestamp"]) < 86400:
        return _lme_wh_cache["data"]
    try:
        import urllib.request
        url = "https://www.westmetall.com/en/markdaten.php?action=table&field=LME_Cu_cash"
        req = urllib.request.Request(url, headers={"User-Agent": "GeometDashboard/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Parse rows: date | cash-settlement (USD/MT) | 3-month (USD/MT) | stock (MT)
        rows = []
        for line in html.split("</tr>"):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", line, re.DOTALL)
            if len(cells) >= 4:
                date_str = re.sub(r"<[^>]+>", "", cells[0]).strip()
                stock_str = re.sub(r"<[^>]+>", "", cells[3]).strip().replace(",", "").replace(".", "")
                if stock_str.isdigit() and int(stock_str) > 1000:
                    # Parse Cash and 3M prices — format: "12,832.00"
                    cash_str = re.sub(r"<[^>]+>", "", cells[1]).strip().replace(",", "")
                    three_m_str = re.sub(r"<[^>]+>", "", cells[2]).strip().replace(",", "")
                    cash_mt = None
                    three_m_mt = None
                    try:
                        cash_mt = float(cash_str)
                    except ValueError:
                        pass
                    try:
                        three_m_mt = float(three_m_str)
                    except ValueError:
                        pass
                    rows.append({"date": date_str, "stock": int(stock_str),
                                 "cash_mt": cash_mt, "three_m_mt": three_m_mt})
        if not rows:
            return None
        latest = rows[0]
        prev = rows[1] if len(rows) > 1 else latest
        stock_mt = latest["stock"]
        prev_mt = prev["stock"]
        net_change = stock_mt - prev_mt
        trend = "building" if net_change > 0 else "drawing" if net_change < 0 else "stable"
        result = {
            "mt": stock_mt, "lbs": int(stock_mt * MT_TO_LB),
            "date": latest["date"], "trend": trend,
            "net_change_mt": net_change, "source": "westmetall",
        }
        # LME Official Settlement prices (Cash and 3M)
        if latest.get("cash_mt"):
            cash_3m_spread = None
            if latest.get("three_m_mt"):
                cash_3m_spread = round(latest["three_m_mt"] - latest["cash_mt"], 2)
            result["lme_cash_settle_mt"] = latest["cash_mt"]
            result["lme_cash_settle_lb"] = round(latest["cash_mt"] / MT_TO_LB, 4)
            result["lme_3m_settle_mt"] = latest.get("three_m_mt")
            result["lme_cash_3m_spread_mt"] = cash_3m_spread
            print(f"[INFO] LME Officials: Cash ${latest['cash_mt']:,.0f}/MT, "
                  f"3M ${latest.get('three_m_mt', 0):,.0f}/MT, "
                  f"spread ${cash_3m_spread}/MT ({latest['date']})")
        _lme_wh_cache = {"data": result, "timestamp": now}
        print(f"[INFO] LME warehouse: {stock_mt:,} MT ({trend}, {latest['date']})")
        return result
    except Exception as e:
        print(f"[WARN] LME warehouse scrape error: {e}")
        return None


def get_warehouse_data():
    # COMEX: try live CME data first, fall back to static config
    comex = fetch_cme_warehouse()
    if not comex:
        wh = CFG.get("COMEX_WAREHOUSE_MT", 0)
        if wh:
            comex = {
                "mt": wh, "lbs": int(wh * MT_TO_LB),
                "date": CFG.get("COMEX_WAREHOUSE_DATE", ""),
                "trend": CFG.get("COMEX_WAREHOUSE_TREND", ""),
                "source": "config",
            }
    lme_wh = fetch_lme_warehouse()
    # Build combined result
    result = {"comex": comex, "lme": lme_wh}
    if comex and lme_wh:
        global_mt = comex["mt"] + lme_wh["mt"]
        result["global_mt"] = global_mt
        result["global_lbs"] = int(global_mt * MT_TO_LB)
    # Backward compat: keep top-level fields from COMEX
    if comex:
        result["mt"] = comex["mt"]
        result["lbs"] = comex["lbs"]
        result["date"] = comex["date"]
        result["trend"] = comex["trend"]
        result["source"] = comex["source"]
    # Save daily warehouse history for sparkline
    if comex or lme_wh:
        _save_warehouse_history(comex, lme_wh)
        result["sparkline"] = _load_warehouse_history()
    return result if (comex or lme_wh) else None


WAREHOUSE_HISTORY = DATA_DIR / "warehouse_history.json"

def _save_warehouse_history(comex, lme):
    try:
        history = []
        if WAREHOUSE_HISTORY.exists():
            with open(WAREHOUSE_HISTORY) as f: history = json.load(f)
        today = datetime.now().strftime("%Y-%m-%d")
        entry = {"date": today}
        if comex: entry["comex_mt"] = comex["mt"]
        if lme: entry["lme_mt"] = lme["mt"]
        if comex and lme: entry["global_mt"] = comex["mt"] + lme["mt"]
        if history and history[-1].get("date") == today:
            history[-1] = entry
        else:
            history.append(entry)
        history = history[-90:]
        with open(WAREHOUSE_HISTORY, "w") as f: json.dump(history, f)
    except Exception as e:
        print(f"[WARN] warehouse history save: {e}")

def _load_warehouse_history():
    try:
        if WAREHOUSE_HISTORY.exists():
            with open(WAREHOUSE_HISTORY) as f: return json.load(f)
    except: pass
    return []


# ---------------------------------------------------------------------------
# CONTRACT ROLL + OPEN INTEREST
# ---------------------------------------------------------------------------
_roll_cache = {"data": None, "timestamp": 0}
OI_HISTORY = DATA_DIR / "oi_history.json"

def get_contract_roll(copper_price=None):
    """COMEX copper contract roll status, calendar spread, and open interest."""
    global _roll_cache
    now = time.time()
    if _roll_cache["data"] and (now - _roll_cache["timestamp"]) < 300:
        cached = _roll_cache["data"].copy()
        if copper_price is not None:
            cached["front_price"] = round(copper_price, 4)
            if cached.get("next_price"):
                spread = round(cached["next_price"] - copper_price, 4)
                cached["calendar_spread"] = spread
                cached["market_structure"] = "contango" if spread > 0.001 else "backwardation" if spread < -0.001 else "flat"
        return cached

    today = datetime.now().date()

    # COMEX copper active months: H=Mar, K=May, N=Jul, U=Sep, Z=Dec
    MONTHS = [
        ("H", "Mar", 3, 2),
        ("K", "May", 5, 4),
        ("N", "Jul", 7, 6),
        ("U", "Sep", 9, 8),
        ("Z", "Dec", 12, 11),
    ]

    def last_biz_day(year, month):
        last = calendar.monthrange(year, month)[1]
        d = datetime(year, month, last).date()
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    def trading_days_until(target):
        if today >= target:
            return 0
        count = 0
        d = today + timedelta(days=1)
        while d <= target:
            if d.weekday() < 5:
                count += 1
            d += timedelta(days=1)
        return count

    contracts = []
    for year in [today.year, today.year + 1]:
        for code, label, del_mo, notice_mo in MONTHS:
            fnd = last_biz_day(year, notice_mo)
            yy = str(year)[-2:]
            contracts.append({
                "code": code, "label": f"{label} {yy}",
                "ticker": f"HG{code}{yy}",
                "yf_ticker": f"HG{code}{yy}.CMX",
                "fnd": fnd, "fnd_str": fnd.strftime("%b %d"),
                "year": year,
            })
    contracts.sort(key=lambda c: c["fnd"])

    front = None; next_mo = None; third_mo = None
    for i, c in enumerate(contracts):
        if c["fnd"] >= today:
            front = c
            if i + 1 < len(contracts):
                next_mo = contracts[i + 1]
            if i + 2 < len(contracts):
                third_mo = contracts[i + 2]
            break
    if not front:
        return None

    days_to_fnd = trading_days_until(front["fnd"])

    if today == front["fnd"]:
        roll_status = "FIRST NOTICE DAY"
        roll_urgency = "critical"
        roll_color = "red"
    elif days_to_fnd <= 2:
        roll_status = f"{days_to_fnd} trading day{'s' if days_to_fnd != 1 else ''} to FND"
        roll_urgency = "critical"
        roll_color = "red"
    elif days_to_fnd <= 5:
        roll_status = f"{days_to_fnd} trading days to FND"
        roll_urgency = "warning"
        roll_color = "orange"
    elif days_to_fnd <= 10:
        roll_status = f"{days_to_fnd} trading days to FND"
        roll_urgency = "attention"
        roll_color = "yellow"
    else:
        roll_status = f"{days_to_fnd} trading days to FND"
        roll_urgency = "normal"
        roll_color = "green"

    result = {
        "front_month": {"label": front["label"], "ticker": front["ticker"], "fnd": front["fnd_str"]},
        "days_to_fnd": days_to_fnd,
        "roll_status": roll_status,
        "roll_urgency": roll_urgency,
        "roll_color": roll_color,
    }
    if next_mo:
        result["next_month"] = {"label": next_mo["label"], "ticker": next_mo["ticker"], "fnd": next_mo["fnd_str"]}
    if third_mo:
        result["third_month"] = {"label": third_mo["label"], "ticker": third_mo["ticker"], "fnd": third_mo["fnd_str"]}
    if copper_price is not None:
        result["front_price"] = round(copper_price, 4)
    else:
        # Fetch front month price from yfinance (needed when RT source is next month)
        try:
            import yfinance as yf
            t = yf.Ticker(front["yf_ticker"])
            h = t.history(period="1d", interval="5m")
            if h.empty:
                h = t.history(period="5d")
            if not h.empty:
                h = h.reset_index()
                h.columns = [c if isinstance(c, str) else c[0] for c in h.columns]
                result["front_price"] = round(float(h.iloc[-1]["Close"]), 4)
                print(f"[INFO] Front month {front['ticker']} (yf): ${result['front_price']:.4f}")
        except Exception as e:
            print(f"[WARN] Front month price error: {e}")

    # Fetch next month contract price for calendar spread
    # Priority: TradingView HG2! → yfinance fallback
    if next_mo:
        with _tv_lock:
            tv2_price = _tv_state_2["price"]
            tv2_age = time.time() - _tv_state_2["timestamp"] if _tv_state_2["timestamp"] else 999
        if tv2_price and tv2_age < 120:
            result["next_price"] = tv2_price
            result["next_source"] = "tradingview"
            # Pass change data for next month (hero contract)
            with _tv_lock:
                tv2_ch = _tv_state_2.get("change")
                tv2_chp = _tv_state_2.get("change_pct")
            if tv2_ch is not None:
                result["next_change"] = tv2_ch
            if tv2_chp is not None:
                result["next_change_pct"] = tv2_chp
            print(f"[INFO] Next month (HG2! via TV): ${tv2_price:.4f}")
        else:
            try:
                import yfinance as yf
                t = yf.Ticker(next_mo["yf_ticker"])
                # Try intraday first for most current price
                h = t.history(period="1d", interval="5m")
                if h.empty:
                    h = t.history(period="5d")  # fallback to daily
                if not h.empty:
                    h = h.reset_index()
                    h.columns = [c if isinstance(c, str) else c[0] for c in h.columns]
                    result["next_price"] = round(float(h.iloc[-1]["Close"]), 4)
                    result["next_source"] = "yfinance"
                    print(f"[INFO] Next month {next_mo['ticker']} (yf): ${result['next_price']:.4f}")
            except Exception as e:
                print(f"[WARN] Next month price error: {e}")

    # Fetch third month contract price
    if third_mo:
        try:
            import yfinance as yf
            t = yf.Ticker(third_mo["yf_ticker"])
            h = t.history(period="1d", interval="5m")
            if h.empty:
                h = t.history(period="5d")
            if not h.empty:
                h = h.reset_index()
                h.columns = [c if isinstance(c, str) else c[0] for c in h.columns]
                result["third_price"] = round(float(h.iloc[-1]["Close"]), 4)
                print(f"[INFO] Third month {third_mo['ticker']}: ${result['third_price']:.4f}")
        except Exception as e:
            print(f"[WARN] Third month price error: {e}")

    # Calendar spread
    if copper_price and result.get("next_price"):
        spread = round(result["next_price"] - copper_price, 4)
        result["calendar_spread"] = spread
        result["market_structure"] = "contango" if spread > 0.001 else "backwardation" if spread < -0.001 else "flat"

    # Open interest
    try:
        import yfinance as yf
        t = yf.Ticker("HG=F")
        info = t.info or {}
        oi = info.get("openInterest")
        if oi and oi > 0:
            oi_history = _save_oi(oi)
            oi_data = {"total": oi, "source": "yfinance"}
            trend = _compute_oi_trend(oi_history, oi)
            if trend:
                oi_data.update(trend)
            result["open_interest"] = oi_data
            print(f"[INFO] Open interest: {oi:,} contracts")
    except Exception as e:
        print(f"[WARN] Open interest error: {e}")

    _roll_cache = {"data": result, "timestamp": now}
    return result


def _save_oi(oi):
    try:
        history = []
        if OI_HISTORY.exists():
            with open(OI_HISTORY) as f:
                history = json.load(f)
        today_str = datetime.now().strftime("%Y-%m-%d")
        if history and history[-1].get("date") == today_str:
            history[-1] = {"date": today_str, "oi": oi}
        else:
            history.append({"date": today_str, "oi": oi})
        history = history[-30:]
        with open(OI_HISTORY, "w") as f:
            json.dump(history, f)
        return history
    except:
        return []


def _compute_oi_trend(history, current_oi):
    if not history or len(history) < 2:
        return None
    recent = history[-5:] if len(history) >= 5 else history
    first_oi = recent[0]["oi"]
    if first_oi <= 0:
        return None
    change = current_oi - first_oi
    change_pct = round((change / first_oi) * 100, 1)
    trend = "building" if change > 0 else "declining" if change < 0 else "stable"
    return {"trend": trend, "change_5d": change, "change_5d_pct": change_pct, "history_days": len(history)}


def _get_oi_on_date(target_date):
    """Get OI on or before target_date (YYYY-MM-DD string) from history."""
    try:
        if OI_HISTORY.exists():
            with open(OI_HISTORY) as f:
                history = json.load(f)
            for entry in reversed(history):
                if entry["date"] <= target_date:
                    return entry["oi"]
    except:
        pass
    return None


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
            # Fallback costs for spreadsheets without Inv Cost column
            _COST_PER_LB = {
                "CU1": 5.280171, "CU2": 5.134602, "CU2DIRTY": 5.134602, "CUBB": 5.455980,
                "CAT5": 2.024765, "CUINS1": 3.041050, "CUINS2": 2.093361,
                "MCM": 4.199354, "THHN": 3.914800, "WAVEOPENCU": 2.355245,
                "CUCHOP CUBB": 4.616107, "CUCHOP1A_M": 4.616107, "CUCHOPS2": 4.616107,
            }
            total_cost = 0; total_cu_lbs = 0
            # Detect layout: Jorge added col 6 "Inv Cost" which shifts ICW columns by 1
            has_cost_col = len(grid[5]) > 6 and str(grid[5][6]).strip().upper().startswith("INV COST")
            icw_item_col = 10 if has_cost_col else 9
            icw_wt_col = 11 if has_cost_col else 10
            icw_cu_col = 13 if has_cost_col else 12
            icw_cost_col = 14 if has_cost_col else -1
            # Solid inventory: col 2=item, col 3=weight, col 5=CuUnits, col 6=Inv Cost (per raw lb)
            for row in grid[7:25]:
                if len(row) > 5 and isinstance(row[2], str) and isinstance(row[5], (int, float)):
                    item = row[2].strip().upper()
                    cu = float(row[5])
                    wt = float(row[3]) if isinstance(row[3], (int, float)) else cu
                    # Read cost from spreadsheet col 6 if available, fall back to hardcoded
                    cost = 0
                    if has_cost_col and len(row) > 6 and isinstance(row[6], (int, float)) and row[6] > 0:
                        cost = float(row[6])
                    else:
                        cost = _COST_PER_LB.get(item, 0)
                    if cost > 0 and cu > 0:
                        total_cost += wt * cost; total_cu_lbs += cu
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
            # ICW per-item costs (column positions adapt to layout)
            for row in grid[7:25]:
                if len(row) > icw_cu_col and isinstance(row[icw_item_col], str) and isinstance(row[icw_wt_col], (int, float)):
                    item = row[icw_item_col].strip().upper()
                    wt = float(row[icw_wt_col])
                    cu = float(row[icw_cu_col]) if isinstance(row[icw_cu_col], (int, float)) else 0
                    # Read cost from spreadsheet if available, fall back to hardcoded
                    cost = 0
                    if icw_cost_col >= 0 and len(row) > icw_cost_col and isinstance(row[icw_cost_col], (int, float)) and row[icw_cost_col] > 0:
                        cost = float(row[icw_cost_col])
                    else:
                        cost = _COST_PER_LB.get(item, 0)
                    if cost > 0 and cu > 0:
                        # Cost is "as is" (per raw lb) — multiply by gross weight
                        total_cost += wt * cost; total_cu_lbs += cu
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
                        sale["priced_lbs"] = priced_tons
                        if order not in shipped_orders:
                            result["sales_priced_unshipped"].append(sale)
                            result["sales_priced_unshipped_lbs"] += priced_tons
                    elif open_priced > 0:
                        unpriced_total += open_priced; sale["status"] = "UNPRICED"
                        sale["open_lbs"] = open_priced
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

def _lme_cash_mt(lme_3m_mt, warehouse):
    """Get LME Cash price in $/MT — use westmetall Official if available, else estimate from 3M."""
    lme_wh = warehouse.get("lme") if warehouse else None
    if lme_wh and lme_wh.get("lme_cash_settle_mt"):
        # Use live 3M minus official Cash-3M spread (more accurate than day-old Cash settle)
        spread = lme_wh.get("lme_cash_3m_spread_mt")
        if spread is not None and lme_3m_mt:
            return round(lme_3m_mt - spread, 2)
        return lme_wh["lme_cash_settle_mt"]
    # Fallback: estimate $90/MT below 3M
    if lme_3m_mt:
        return round(lme_3m_mt - 90, 2)
    return None

def _lme_cash_lb(lme_3m_mt, warehouse):
    """Get LME Cash price in $/lb."""
    cash_mt = _lme_cash_mt(lme_3m_mt, warehouse)
    if cash_mt:
        return round(cash_mt / MT_TO_LB, 4)
    return None

def _lme_cash_3m_spread(lme_3m_mt, warehouse):
    """Get Cash-3M spread in $/MT from westmetall, or fallback to $90 estimate."""
    lme_wh = warehouse.get("lme") if warehouse else None
    if lme_wh and lme_wh.get("lme_cash_3m_spread_mt") is not None:
        return lme_wh["lme_cash_3m_spread_mt"]
    return 90  # fallback estimate


def fetch_copper_data():
    global _copper_cache
    now = time.time()
    if _copper_cache["data"] and (now - _copper_cache["timestamp"]) < COPPER_CACHE_TTL:
        return _copper_cache["data"]

    try:
        # Try yfinance first (closer to CME settlements), fall back to investing.com
        ohlc = None
        try:
            ohlc = _fetch_ohlc_yfinance()
        except Exception as e:
            print(f"[WARN] yfinance error: {e}")
        if not ohlc:
            try:
                ohlc = _fetch_ohlc_investiny()
            except Exception as e:
                print(f"[WARN] investiny error: {e}")
        if not ohlc:
            return None

        dates = ohlc["dates"]; closes = ohlc["closes"]; highs = ohlc["highs"]; lows = ohlc["lows"]
        copper_source = ohlc["source"]
        n_closes = len(closes)

        price = closes[-1]; prev_close = closes[-2] if n_closes > 1 else price
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0

        # Fix for contract roll: HG=F continuous data has a gap when the front
        # month rolls (e.g. May→Jul).  closes[-2] is old contract, closes[-1] is
        # new contract — the diff is a roll gap, not a real price move.
        # Use the specific front contract's prev settle for an accurate change.
        try:
            _active_yf, _ac = _get_active_yf_ticker()
            if _active_yf and _active_yf != "HG=F":
                _spec_prev = _fetch_prev_settle_yf(_active_yf)
                if _spec_prev and abs(_spec_prev - prev_close) > 0.02:
                    print(f"[INFO] Roll-gap fix: HG=F prev_close ${prev_close:.4f} → {_active_yf} prev ${_spec_prev:.4f}")
                    prev_close = _spec_prev
                    change = round(price - prev_close, 4)
                    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0
        except Exception as e:
            print(f"[WARN] Roll-gap prev_close fix failed: {e}")

        # Determine previous settlement for RT overlay
        # After 5PM CT: use 4PM settlement from intraday data (daily bar keeps updating)
        # During the day: today's bar is partial, use yesterday's close
        today_date = datetime.now().date()
        last_ohlc_date = dates[-1].date() if hasattr(dates[-1], 'date') else dates[-1]
        if isinstance(last_ohlc_date, datetime):
            last_ohlc_date = last_ohlc_date.date()
        new_session = _is_new_comex_session()
        if new_session:
            # Try to get exact 4PM CT settlement from intraday bars
            settle = _fetch_session_settle()
            if settle:
                _daily_prev_settle = settle
            elif last_ohlc_date >= today_date and n_closes > 1:
                _daily_prev_settle = prev_close  # use roll-adjusted prev
            else:
                _daily_prev_settle = closes[-1]
        elif last_ohlc_date >= today_date and n_closes > 1:
            _daily_prev_settle = prev_close  # use roll-adjusted prev
        else:
            _daily_prev_settle = closes[-1]

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
            vol = None
            avg_vol = None
            vol_ratio = 1.0

        # Prefer TradingView live volume; fall back to yfinance
        with _tv_lock:
            tv_vol = _tv_state.get("volume")
        session_vol = int(tv_vol) if tv_vol else (int(vol) if vol else None)
        if session_vol:
            _record_volume_snapshot(session_vol)
        parallel_avg = _get_parallel_avg_volume()

        recent = closes[-5:] if n_closes >= 5 else closes
        spark_30d = [{"date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10],
                      "close": round(c, 4)} for d, c in zip(dates[-30:], closes[-30:])]
        spark_7d = [{"date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10],
                     "close": round(c, 4)} for d, c in zip(dates[-7:], closes[-7:])]
        spark_1d = fetch_intraday_spark() or []
        # Full COMEX history for chart timeframe toggles (up to all available data)
        spark_full = [{"date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10],
                       "close": round(c, 4)} for d, c in zip(dates, closes)]

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
        lme_st = get_lme_status()
        comex_st = get_comex_status()

        # --- Contemporaneous spread: peak-hours snapshot ---
        # Both markets are liquid during COMEX peak (7:30-10 AM CT) + LME Ring.
        # Capture a snapshot during that overlap; once the window passes, lock it.
        spread_timing = None  # "peak" = snapshot, "stale" = no snapshot available
        spread = None; spread_pct = None
        if lme_price:
            lme_ring = lme_st.get("session") == "ring" or lme_st.get("status") == "RING"
            comex_peak = comex_st.get("window") == "peak"
            both_liquid = lme_ring and comex_peak
            today_str = datetime.now().strftime("%Y-%m-%d")
            history = load_spread_history()
            today_entry = history[-1] if history and history[-1].get("date") == today_str else None

            if both_liquid:
                # Both markets open & liquid — capture live contemporaneous spread
                spread = round(price - lme_price, 4)
                spread_timing = "peak"
                # Save as peak snapshot
                entry = {"date": today_str, "comex": round(price, 4), "lme": round(lme_price, 4),
                         "spread": spread, "peak_snapshot": True}
                if today_entry:
                    history[-1] = entry
                else:
                    history.append(entry)
                history = history[-180:]
                try:
                    with open(SPREAD_HISTORY, "w") as f: json.dump(history, f)
                except: pass
            elif today_entry and today_entry.get("peak_snapshot"):
                # Window passed but we have today's snapshot — use it
                spread = today_entry["spread"]
                spread_timing = "peak"
            else:
                # No snapshot yet today (dashboard started late, or weekend)
                # Fall back to raw diff but mark it stale
                spread = round(price - lme_price, 4)
                spread_timing = "stale"
                # Still save for history continuity
                entry = {"date": today_str, "comex": round(price, 4), "lme": round(lme_price, 4),
                         "spread": spread, "peak_snapshot": False}
                if today_entry:
                    if not today_entry.get("peak_snapshot"):
                        history[-1] = entry
                else:
                    history.append(entry)
                history = history[-180:]
                try:
                    with open(SPREAD_HISTORY, "w") as f: json.dump(history, f)
                except: pass

        spread_pct = round((spread / lme_price) * 100, 2) if lme_price and spread else None
        spread_intel = None
        lme_change = None; lme_change_pct = None; lme_prev_lb = None
        # Prefer TradingView MCU3 daily change (tracks actual trading session, like COMEX)
        warehouse = get_warehouse_data()
        with _tv_lock:
            _tv_ch_mt = _tv_state_lme.get("change_mt")
            _tv_chp = _tv_state_lme.get("change_pct")
        if lme_price and _tv_ch_mt is not None and _tv_chp is not None:
            lme_change = round(_tv_ch_mt / MT_TO_LB, 4)
            lme_change_pct = round(_tv_chp, 2)
            lme_prev_lb = round(lme_price - lme_change, 4)
        else:
            # Fallback: compute from official 3M settlement (may be stale over weekends)
            _lme_settle_mt = warehouse.get("lme", {}).get("lme_3m_settle_mt") if warehouse else None
            if lme_price and _lme_settle_mt:
                settle_lb = round(_lme_settle_mt / MT_TO_LB, 4)
                if settle_lb and abs(settle_lb - lme_price) > 0.0001:
                    lme_prev_lb = settle_lb
                    lme_change = round(lme_price - settle_lb, 4)
                    lme_change_pct = round((lme_change / settle_lb) * 100, 2)
        if spread is not None and lme_price:
            history = load_spread_history()
            spread_intel = compute_spread_intelligence(history, spread)

        dxy = fetch_dxy()
        china = get_china_status()
        lme_status = lme_st
        comex_status = comex_st
        fed = fetch_fed_data()

        # LME sparkline from spread history
        lme_spark = []
        try:
            sh = load_spread_history()
            lme_spark = [{"date": e["date"], "close": e["lme"]} for e in sh if e.get("lme")]
        except: pass

        result = {
            "price": round(price, 4), "prev_close": round(prev_close, 4),
            "change": round(change, 4), "change_pct": round(change_pct, 2),
            "ma50": round(ma50, 4), "ma100": round(ma100, 4), "ma200": round(ma200, 4),
            "vol_ratio": round(vol_ratio, 2),
            "volume": int(vol) if vol else None, "avg_volume": int(avg_vol) if avg_vol else None,
            "parallel_avg_volume": parallel_avg,
            "is_peak": comex_status.get("is_peak", False),
            "recent_closes": [round(c, 4) for c in recent],
            "sparkline": spark_30d, "spark_7d": spark_7d, "spark_1d": spark_1d,
            "spark_full": spark_full, "lme_spark": lme_spark, "copper_source": copper_source,
            "lme_price_lb": lme_price, "lme_price_mt": lme_mt, "lme_source": lme_source,
            "lme_cash_mt": _lme_cash_mt(lme_mt, warehouse),
            "lme_cash_lb": _lme_cash_lb(lme_mt, warehouse),
            "lme_cash_3m_spread_mt": _lme_cash_3m_spread(lme_mt, warehouse),
            "lme_change": lme_change, "lme_change_pct": lme_change_pct,
            "comex_lme_spread": spread, "comex_lme_spread_pct": spread_pct,
            "spread_timing": spread_timing, "spread_intel": spread_intel,
            "today_high": round(today_high, 4), "today_low": round(today_low, 4),
            "today_range": round(today_range, 4),
            "pct_30d": pct_30d, "pct_90d": pct_90d,
            "range_30d": [round(range_30d_low, 4), round(range_30d_high, 4)],
            "range_90d": [round(range_90d_low, 4), round(range_90d_high, 4)],
            "roc": roc, "streak": streak, "streak_dir": streak_dir,
            "avg_daily_range": avg_daily_range, "vol_vs_avg": vol_vs_avg,
            "support_resistance": sr, "dxy": dxy, "china": china, "lme": lme_status, "comex": comex_status,
            "customer_availability": get_customer_availability(),
            "fed": fed, "warehouse": warehouse, "daily_prev_settle": round(_daily_prev_settle, 4),
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
    # Auto-adjust fix target if price has moved >15% away
    if ft and p and abs(p - ft) / ft > 0.15:
        ft = round(p + 0.05, 2)  # set target 5c above current
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
        abs_sp = abs(spread)
        if spread > 0.15: spread_signal = {"direction": "ARB: COMEX PREMIUM", "msg": f"COMEX +{spread:.2f}/lb over LME \u2014 sell COMEX basis, buy LME to hedge", "color": "green", "arb": True}
        elif spread > 0.05: spread_signal = {"direction": "COMEX SLIGHT PREMIUM", "msg": f"COMEX +{spread:.2f}/lb \u2014 favor COMEX-basis sales", "color": "green", "arb": False}
        elif spread > -0.05: spread_signal = {"direction": "PARITY", "msg": f"COMEX-LME spread {spread:+.3f}/lb \u2014 no arb, price either basis equally", "color": "yellow", "arb": False}
        elif spread > -0.15: spread_signal = {"direction": "LME SLIGHT PREMIUM", "msg": f"LME +{abs_sp:.2f}/lb \u2014 favor LME-basis export sales", "color": "blue", "arb": False}
        else: spread_signal = {"direction": "ARB: LME PREMIUM", "msg": f"LME +{abs_sp:.2f}/lb over COMEX \u2014 sell LME basis, hedge on COMEX", "color": "blue", "arb": True}

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
    """Generate GTC suggestions: 2 sell orders above + 2 buy orders below current price.
    Above = lock in margin when market is up.
    Below = limit downside / cut losses.
    """
    if not position or not md: return []
    p = md["price"]; net = position.get("net_lbs", 0)
    tl = CFG["TRUCKLOAD_LBS"]
    LBS_PER_MT = 2204.62
    suggestions = []

    def _lme_mt(lb_price):
        return round(lb_price * LBS_PER_MT)

    # --- SELL LIMIT (lock in margin when market is up) ---
    if net > 0:
        loads = max(1, int(net / tl))
        sell_per = max(1, loads // 2)
        # +10c above current
        lvl1 = round(p + 0.10, 2)
        n1 = min(sell_per, loads)
        suggestions.append({
            "action": "SELL LIMIT GTC", "level": lvl1, "urgency": "medium",
            "detail": f"Fix {n1} load{'s' if n1>1 else ''} ({n1*tl:,} lbs) at ${lvl1:.2f}/lb (\u2248${_lme_mt(lvl1):,}/MT) \u2014 +10\u00a2",
            "side": "sell",
        })
        # +20c above current
        lvl2 = round(p + 0.20, 2)
        n2 = min(sell_per, max(1, loads - n1))
        suggestions.append({
            "action": "SELL LIMIT GTC", "level": lvl2, "urgency": "low",
            "detail": f"Fix {n2} load{'s' if n2>1 else ''} ({n2*tl:,} lbs) at ${lvl2:.2f}/lb (\u2248${_lme_mt(lvl2):,}/MT) \u2014 +20\u00a2",
            "side": "sell",
        })

    # --- SELL STOP (limit downside / stop loss) ---
    # -10c below current
    lvl3 = round(p - 0.10, 2)
    suggestions.append({
        "action": "SELL STOP GTC", "level": lvl3, "urgency": "medium",
        "detail": f"Stop loss: fix 1 load ({tl:,} lbs) at ${lvl3:.2f}/lb (\u2248${_lme_mt(lvl3):,}/MT) \u2014 \u221210\u00a2",
        "side": "buy",
    })
    # -20c below current
    lvl4 = round(p - 0.20, 2)
    suggestions.append({
        "action": "SELL STOP GTC", "level": lvl4, "urgency": "low",
        "detail": f"Stop loss: fix 1 load ({tl:,} lbs) at ${lvl4:.2f}/lb (\u2248${_lme_mt(lvl4):,}/MT) \u2014 \u221220\u00a2",
        "side": "buy",
    })

    return suggestions


# ---------------------------------------------------------------------------
# PLACED GTC ORDERS — actual orders placed with customers
# ---------------------------------------------------------------------------
GTC_ORDERS_FILE = DATA_DIR / "gtc_orders.json"

def load_gtc_orders():
    if not GTC_ORDERS_FILE.exists():
        return []
    try:
        with open(GTC_ORDERS_FILE) as f:
            orders = json.load(f)
        return [o for o in orders if o.get("status") == "active"]
    except Exception as e:
        print(f"[WARN] gtc_orders.json error: {e}")
        return []

def save_gtc_orders(orders):
    with open(GTC_ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)


# ---------------------------------------------------------------------------
# MARKET RATES — editable from dashboard, persisted to JSON
# ---------------------------------------------------------------------------

def load_market_rates():
    """Load market rates from JSON file, falling back to config.py defaults."""
    if MARKET_RATES_FILE.exists():
        try:
            with open(MARKET_RATES_FILE) as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"[WARN] market_rates.json error: {e}")
    return None


def save_market_rates(rates, source="", sources=None):
    """Save market rates to JSON, auto-stamping date and current COMEX price.
    Also builds per-grade quote history (last 3 per grade).
    sources: optional dict of per-grade source names (overrides global source)."""
    sources = sources or {}
    now = datetime.now()
    with _tv_lock:
        comex_now = _tv_state["price"] or 0
        lme_mt = _tv_state_lme.get("price_mt") or 0
    lme_lb = round(lme_mt / 2204.62, 4) if lme_mt else 0

    # Contract roll prices for COMEX basis resolution
    _cr = get_contract_roll(comex_now if comex_now else None) or {}
    cr_next = _cr.get("next_price") or 0
    cr_third = _cr.get("third_price") or 0

    # Compare against saved rates to detect which grades actually changed
    existing = load_market_rates()
    old_rates = existing.get("rates", {}) if existing else {}
    old_history = existing.get("history", []) if existing else []

    # Build per-grade history entries — only for grades whose formula changed
    new_history = []
    date_str = now.strftime("%-m/%-d")
    hr = now.hour % 12 or 12
    ampm = "a" if now.hour < 12 else "p"
    time_str = f"{hr}:{now.strftime('%M')}{ampm}"
    for grade, mr in rates.items():
        # Skip if rate is identical to what's already saved
        prev = old_rates.get(grade, {})
        if mr == prev:
            continue
        entry = {
            "grade": grade,
            "source": sources.get(grade, source),
            "date": date_str,
            "time": time_str,
            "comex_stamp": round(comex_now, 4) if comex_now else 0,
            "lme_stamp": lme_lb,
        }
        if mr.get("type") == "flat":
            # COMEX basis: base - discount
            basis = mr.get("basis", "comex_next")
            if basis == "comex_front":
                base = comex_now
                entry["basis"] = _cr.get("front_month", {}).get("label", "Front").split()[0] if _cr.get("front_month") else "Front"
            elif basis == "comex_third":
                base = cr_third
                entry["basis"] = _cr.get("third_month", {}).get("label", "Third").split()[0] if _cr.get("third_month") else "Third"
            else:  # comex_next
                base = cr_next
                entry["basis"] = _cr.get("next_month", {}).get("label", "Next").split()[0] if _cr.get("next_month") else "Next"
            disc = mr.get("discount", 0)
            entry["formula"] = f"{entry['basis']} -${disc:.4f}"
            entry["input"] = disc
            entry["base_price"] = round(base, 4) if base else 0
            entry["result_lb"] = round(base - disc, 4) if base else 0
        else:
            # LME pct basis
            pct = mr.get("pct", 0)
            basis = mr.get("basis", "3m")
            deduct = mr.get("deduct", 0)
            base = lme_lb if basis != "cash" else lme_lb  # both use lme_lb for now
            pct_display = round(pct * 1000) / 10
            entry["basis"] = "3M" if basis != "cash" else "Cash"
            formula = f"{pct_display}% LME {entry['basis']}"
            if deduct > 0:
                formula += f" -${deduct:.2f}"
                entry["deduct"] = deduct
            entry["formula"] = formula
            entry["input"] = pct_display
            entry["base_price"] = round(base, 4) if base else 0
            entry["result_lb"] = round(base * pct - deduct, 4) if base else 0

        new_history.append(entry)

    # Prepend new entries, cap at 5 per grade
    combined = new_history + old_history
    # Keep last 5 per grade
    grade_counts = {}
    trimmed = []
    for h_entry in combined:
        g = h_entry.get("grade", "")
        grade_counts[g] = grade_counts.get(g, 0) + 1
        if grade_counts[g] <= 5:
            trimmed.append(h_entry)

    data = {
        "rates": rates,
        "date": now.strftime("%Y-%m-%d"),
        "comex_stamp": round(comex_now, 4) if comex_now else 0,
        "source": source,
        "history": trimmed,
    }
    DATA_DIR.mkdir(exist_ok=True)
    with open(MARKET_RATES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data


def enrich_gtc_orders(orders, md):
    """Add distance and trigger info to each placed GTC order."""
    if not md:
        return orders
    lme_cash_mt = md.get("lme_cash_mt")
    result = []
    for o in orders:
        o = dict(o)  # copy
        if o.get("basis") == "LME_CASH" and lme_cash_mt:
            target = o["price_mt"]
            dist = round(lme_cash_mt - target, 2)
            o["current_mt"] = lme_cash_mt
            o["distance_mt"] = dist
            o["distance_pct"] = round((dist / target) * 100, 2) if target else 0
            o["triggered"] = lme_cash_mt >= target
        else:
            o["triggered"] = False
            o["distance_mt"] = None
            o["distance_pct"] = None
        result.append(o)
    return result


def calc_risk(pos, md):
    if not pos or not md: return None
    p = md["price"]; net = pos["net_lbs"]; ac = pos["avg_cost"]; hl = pos.get("hedge_lbs", 0)
    uh = net - hl; mtm = (p - ac) * net
    # Sales by basis (COMEX vs LME)
    comex_sales_lbs = 0; lme_sales_lbs = 0
    for sale_list in [pos.get("sales_priced_unshipped", []),
                      pos.get("sales_unpriced_shipped", []),
                      pos.get("sales_unpriced_unshipped", [])]:
        for sale in sale_list:
            lbs = sale.get("priced_lbs", 0) or sale.get("open_lbs", 0) or sale.get("lbs", 0)
            if sale.get("basis") == "LME":
                lme_sales_lbs += lbs
            else:
                comex_sales_lbs += lbs
    # Add priced+shipped (already sold, no longer in lists) from totals minus what's in lists
    # The lists above cover all open sales, which is what matters for basis exposure
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
        "comex_sales_lbs": round(comex_sales_lbs),
        "lme_sales_lbs": round(lme_sales_lbs),
        "comex_hedge_lbs": abs(pos.get("comex_futures", 0)),
        "lme_hedge_lbs": abs(pos.get("lme_futures", 0)),
    }


def _grade_for_commodity(raw):
    """Map ROM commodity name to standard grade. Mirrors JS soGrade()."""
    c = (raw or "").upper()
    if "CHOP" in c: return "Chops"
    if "CUBB" in c: return "BB"
    if "CU1" in c: return "#1"
    if "CU2" in c: return "#2"
    return None  # ICW grades (THHN/MCM/etc.) not modeled in cats4


def _so_eff_price(so, comex, lme):
    """Effective sell price for an SO: locked price if priced, else project at current."""
    if so.get("price"):
        return so["price"]
    spread = so.get("spread", 0)
    basis = so.get("basis", "COMEX")
    if basis == "LME" and lme:
        return (lme * spread) if spread else lme
    return (comex - spread) if spread else comex


def calc_margin_projection(pos, md, risk):
    """Live GM = avg SO sell price − avg ROM grade cost, both sales-weighted per grade.
    Mirrors the blended table's All-row math so the two displays line up exactly."""
    if not pos or not md: return None
    comex = md.get("price", 0)
    lme = md.get("lme_price_lb", 0)
    grade_costs = pos.get("grade_costs") or {}
    if not comex or not grade_costs: return None

    # Walk every open SO (priced + unpriced) and accumulate per-grade lbs/rev/cost.
    # Skip SOs whose grade isn't in BB/#1/#2/Chops or has no cost data (e.g. ICW).
    per_grade = {}  # grade -> {"lbs", "rev", "cost"}
    priced_lbs = 0
    unpriced_lbs = 0

    so_lists = (
        ("sales_priced_unshipped", "priced"),
        ("sales_unpriced_shipped", "unpriced"),
        ("sales_unpriced_unshipped", "unpriced"),
    )
    for key, kind in so_lists:
        for sale in pos.get(key, []):
            g = _grade_for_commodity(sale.get("commodity"))
            if g is None or g not in grade_costs:
                continue
            lbs = (sale.get("priced_lbs") if kind == "priced" else sale.get("open_lbs")) or sale.get("lbs") or 0
            if lbs <= 0:
                continue
            eff = _so_eff_price(sale, comex, lme)
            gd = per_grade.setdefault(g, {"lbs": 0, "rev": 0, "cost": 0})
            gd["lbs"] += lbs
            gd["rev"] += lbs * eff
            gd["cost"] += lbs * grade_costs[g]
            if kind == "priced":
                priced_lbs += lbs
            else:
                unpriced_lbs += lbs

    total_lbs = sum(gd["lbs"] for gd in per_grade.values())
    total_rev = sum(gd["rev"] for gd in per_grade.values())
    total_cost = sum(gd["cost"] for gd in per_grade.values())
    if total_lbs <= 0:
        return None

    avg_sell = total_rev / total_lbs
    avg_cost = total_cost / total_lbs
    gm_per_lb = avg_sell - avg_cost
    return {
        "total_revenue_now": round(total_rev, 2),
        "priced_lbs": round(priced_lbs),
        "unpriced_lbs": round(unpriced_lbs),
        "total_lbs": round(total_lbs),
        "avg_sell_price": round(avg_sell, 4),
        "avg_cost": round(avg_cost, 4),
        "gross_margin_per_lb": round(gm_per_lb, 4),
        "total_gross_margin": round(gm_per_lb * total_lbs, 2),
        "margin_pct": round((gm_per_lb / avg_sell) * 100, 2) if avg_sell else 0,
        "comex_now": round(comex, 4),
        "lme_now": round(lme, 4) if lme else None,
    }


def load_broker_intel():
    """Load today's broker intel from file."""
    if not BROKER_INTEL_FILE.exists():
        return None
    try:
        with open(BROKER_INTEL_FILE) as f:
            data = json.load(f)
        # Only return if from today
        if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
            return data
        return None
    except Exception:
        return None


def _keyword_intel(text):
    """Fallback: extract signals via keyword matching (used if Claude API unavailable)."""
    signals = []
    t = text.lower()

    if any(w in t for w in ["china backing off", "china demand weak", "china slowing", "china retreat"]):
        signals.append({"short": "bear", "long": "watch",
            "headline": "China pulling back",
            "near": "Less physical buying = prices likely to drift lower next few days",
            "far": "Usually temporary — China has been buying every dip this cycle. Watch if it lasts more than a week"})
    if any(w in t for w in ["recession", "slowdown", "contraction", "demand destruct"]):
        signals.append({"short": "bear", "long": "bear",
            "headline": "Economic slowdown concerns",
            "near": "Traders sell copper on recession fears — expect downward pressure",
            "far": "If real, copper demand drops for months. Major risk to being long"})
    if any(w in t for w in ["softer", "eases", "easing", "prices lower", "prices down", "selling pressure"]):
        signals.append({"short": "bear", "long": "watch",
            "headline": "Prices softening",
            "near": "Sellers are in control right now — good time to fix if you need to trim",
            "far": "A pullback after a run-up is normal. Doesn't change the bigger trend unless it breaks support"})
    if any(w in t for w in ["profit.taking", "profit taking", "liquidat"]):
        signals.append({"short": "bear", "long": "watch",
            "headline": "Profit-taking / liquidation",
            "near": "Funds are cashing out gains — prices drop fast during liquidation",
            "far": "Usually creates a buying opportunity once selling exhausts itself"})
    if any(w in t for w in ["rally", "surge", "spike", "breakout", "new high"]):
        signals.append({"short": "bull", "long": "watch",
            "headline": "Prices rallying",
            "near": "Momentum buyers pushing prices up — don't chase, but don't sell into strength either",
            "far": "Could be start of a new leg up, or could fade. Watch if it holds above prior highs"})
    if any(w in t for w in ["call option", "call oi", "calls increase", "bullish option", "bullish bet"]):
        signals.append({"short": "watch", "long": "bull",
            "headline": "Big call option activity",
            "near": "Could go either way short-term — sometimes it's hedging, sometimes it's speculative bets",
            "far": "Someone is paying real money for upside exposure. Usually means smart money expects higher prices in weeks/months"})
    if any(w in t for w in ["short cover", "shorts cover", "net-short cut", "cut net-short", "short squeeze"]):
        signals.append({"short": "watch", "long": "watch",
            "headline": "Shorts reducing positions",
            "near": "Traders closing bets on lower prices — could cause a quick pop but it's not new buying",
            "far": "De-risking ahead of an event (NPC, tariffs). Not a strong signal either way"})
    if any(w in t for w in ["put option", "put oi", "puts increase", "bearish option"]):
        signals.append({"short": "watch", "long": "bear",
            "headline": "Put option activity rising",
            "near": "Could be hedging existing long positions or genuine bearish bets",
            "far": "If sustained, smart money may be positioning for a move lower"})
    if any(w in t for w in ["supply disrupt", "supply threat", "mine shut", "mine strike", "peru", "chile", "congo", "zambia"]):
        signals.append({"short": "watch", "long": "bull",
            "headline": "Mine/supply disruption risk",
            "near": "Threats don't move prices much until they become real disruptions",
            "far": "Copper supply is already tight. Any actual shutdown tightens the market further — bullish for prices"})
    if any(w in t for w in ["stimulus", "npc", "national people", "infrastructure", "green energy", "ev demand"]):
        signals.append({"short": "watch", "long": "bull",
            "headline": "Stimulus / policy catalyst ahead",
            "near": "Markets wait for details — expect choppy trading until announcements",
            "far": "If real spending is announced, copper demand goes up. China stimulus has driven every major copper rally"})
    if any(w in t for w in ["tariff", "duties", "trade war", "sanction"]):
        signals.append({"short": "bear", "long": "watch",
            "headline": "Tariff / trade policy in play",
            "near": "Tariff headlines spook traders — expect volatility and possible dip",
            "far": "Tariffs on China copper could actually tighten non-China supply and push US prices higher"})
    if any(w in t for w in ["fed meet", "fomc", "rate decision", "rate hike", "rate cut"]):
        signals.append({"short": "watch", "long": "watch",
            "headline": "Fed / rate decision upcoming",
            "near": "Markets go sideways ahead of Fed. Expect a move in either direction after the announcement",
            "far": "Rate cuts = weaker dollar = higher copper. Rate hikes = opposite"})
    if any(w in t for w in ["energy policy", "white house", "policy meeting"]):
        signals.append({"short": "watch", "long": "watch",
            "headline": "Policy meeting ahead",
            "near": "Wait for details before acting",
            "far": "Energy policy could boost copper demand (EVs, grid) or hurt it (tariffs, regulation)"})
    if any(w in t for w in ["warehouse draw", "inventory draw", "stocks fall", "stocks decline"]):
        signals.append({"short": "bull", "long": "bull",
            "headline": "Warehouse inventories dropping",
            "near": "Physical copper is being pulled — supports prices now",
            "far": "Falling inventories mean real demand is exceeding supply. Bullish until restocked"})
    if any(w in t for w in ["warehouse build", "inventory build", "stocks rise", "stocks increase", "stocks rose", "rising stocks", "inflows", "highest since"]):
        signals.append({"short": "bear", "long": "watch",
            "headline": "Warehouse inventories rising",
            "near": "More copper sitting in warehouses — less urgency to buy. Prices may soften",
            "far": "Could be seasonal or temporary restocking. Watch the trend over weeks, not days"})
    if any(w in t for w in ["yangshan premium", "china buying", "china import", "stockpiling", "strategic stockpil"]):
        signals.append({"short": "bull", "long": "bull",
            "headline": "China physical buying picking up",
            "near": "Real demand from the biggest buyer — supports prices even during paper selling",
            "far": "When China stockpiles copper, it tightens global supply for months. Bullish signal"})
    if any(w in t for w in ["dollar firm", "dollar strength", "dollar index firm", "dxy rise", "dxy up", "stronger dollar"]):
        signals.append({"short": "bear", "long": "watch",
            "headline": "Dollar strengthening",
            "near": "Stronger dollar makes copper more expensive globally — headwind for prices",
            "far": "Dollar moves are usually short-lived. Fed policy drives the bigger dollar trend"})
    if any(w in t for w in ["dollar weak", "dollar ease", "dollar fell", "dollar drop", "dxy down", "weaker dollar"]):
        signals.append({"short": "bull", "long": "watch",
            "headline": "Dollar weakening",
            "near": "Weaker dollar makes copper cheaper globally — tailwind for prices",
            "far": "If driven by rate cuts, sustained weakness supports higher copper prices"})
    return signals


# ---------------------------------------------------------------------------
# DAILY MARKET INSIGHT — AI-generated morning briefing
# ---------------------------------------------------------------------------

def _fetch_google_news_headlines(max_items=8):
    """Scrape Google News RSS for copper market headlines."""
    import urllib.request
    import xml.etree.ElementTree as ET
    url = "https://news.google.com/rss/search?q=copper+market+price&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "")
            source = item.findtext("source", "")
            pub_date = item.findtext("pubDate", "")
            if title:
                items.append({"title": title, "source": source, "pub_date": pub_date})
            if len(items) >= max_items:
                break
        return items
    except Exception as e:
        print(f"[WARN] Google News fetch failed: {e}")
        return []


def _build_insight_context(md, sig, cot, roll, outlook):
    """Serialize key market data into plain text for Claude prompt."""
    lines = []
    if md:
        lines.append(f"COMEX May (front): ${md.get('price', 0):.4f}/lb  change: {md.get('change', 0):+.4f} ({md.get('change_pct', 0):+.1f}%)")
        if roll and roll.get("next_price"):
            jul_ch = roll.get("next_change", 0) or 0
            jul_chp = roll.get("next_change_pct", 0) or 0
            lines.append(f"COMEX Jul (active month, highest volume): ${roll['next_price']:.4f}/lb  change: {jul_ch:+.4f} ({jul_chp:+.1f}%)")
        if roll and roll.get("third_price"):
            lines.append(f"COMEX Sep: ${roll['third_price']:.4f}/lb")
        if md.get("lme_price_lb"):
            lme_ch = md.get("lme_change", 0) or 0
            lme_chp = md.get("lme_change_pct", 0) or 0
            lines.append(f"LME 3M: ${md['lme_price_lb']:.4f}/lb  change: {lme_ch:+.4f} ({lme_chp:+.1f}%)  spread vs COMEX: {md.get('comex_lme_spread', 0):+.4f}")
        if md.get("lme_cash_lb"):
            lines.append(f"LME Cash: ${md['lme_cash_lb']:.4f}/lb  cash-3M spread: ${md.get('lme_cash_3m_spread_mt', 0)}/MT")
        if md.get("ma50"):
            p = md.get("price", 0)
            lines.append(f"Moving averages: 50d={md['ma50']:.4f} ({('ABOVE' if p>md['ma50'] else 'BELOW')})  100d={md.get('ma100', 0):.4f} ({('ABOVE' if p>md.get('ma100',0) else 'BELOW')})  200d={md['ma200']:.4f} ({('ABOVE' if p>md['ma200'] else 'BELOW')})")
        roc = md.get("roc", {})
        if roc:
            parts = []
            for k in ("1d", "3d", "5d", "10d", "20d"):
                r = roc.get(k, {})
                if r.get("pct") is not None:
                    parts.append(f"{k}: {r['pct']:+.1f}%")
            if parts:
                lines.append(f"Momentum (rate of change): {', '.join(parts)}")
        if md.get("dxy"):
            dxy = md["dxy"]
            lines.append(f"DXY (Dollar Index): {dxy.get('price', 'N/A')} change: {dxy.get('change', 'N/A')}")
        wh = md.get("warehouse", {})
        if wh.get("global_mt"):
            lines.append(f"Global warehouse: {wh['global_mt']:,} MT")
        if wh.get("comex", {}).get("mt"):
            lines.append(f"  COMEX: {wh['comex']['mt']:,} MT  trend: {wh['comex'].get('trend', 'N/A')}")
        if wh.get("lme", {}).get("mt"):
            lines.append(f"  LME: {wh['lme']['mt']:,} MT  trend: {wh['lme'].get('trend', 'N/A')}")
    if sig:
        lines.append(f"Trend: {sig.get('trend', 'N/A')} ({sig.get('trend_strength', '')})")
        if sig.get("move_type"):
            lines.append(f"Move type: {sig['move_type']}")
    if cot:
        mm_net = cot.get("mm_net", cot.get("fund_net", "N/A"))
        mm_chg = cot.get("mm_weekly_change", cot.get("fund_net_change", "N/A"))
        mm_pct = cot.get("mm_pct_52w", cot.get("fund_pctile", "N/A"))
        lines.append(f"COT Managed Money Net: {mm_net} contracts  weekly change: {mm_chg}  52w percentile: {mm_pct}%")
    if roll:
        lines.append(f"Front month: {roll.get('front_month', 'N/A')}  days to FND: {roll.get('days_to_fnd', 'N/A')}  spread: {roll.get('calendar_spread', 'N/A')}  structure: {roll.get('market_structure', 'N/A')}")
        if roll.get("open_interest"):
            oi = roll["open_interest"]
            lines.append(f"Open Interest: {oi.get('total', 'N/A')}  trend: {oi.get('trend', 'N/A')}  5d change: {oi.get('change_5d_pct', 'N/A')}%")
    if outlook:
        for tf in ("today", "this_week", "this_month"):
            o = outlook.get(tf)
            if o:
                reasons = "; ".join(o.get("reasons", []))
                cal = o.get("calendar_note", "")
                lines.append(f"Outlook {tf}: {o.get('label', '')} (score {o.get('score', 0)}, confidence {o.get('confidence', '')}). Reasons: {reasons}" + (f" Calendar: {cal}" if cal else ""))
    return "\n".join(lines)


def _claude_daily_insight(md, sig, cot, roll, outlook, headlines):
    """Call Claude Haiku to generate daily market insight bullets."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "YOUR_KEY_HERE":
        return None

    context = _build_insight_context(md, sig, cot, roll, outlook)
    headline_text = ""
    if headlines:
        headline_text = "\n\nRECENT NEWS HEADLINES:\n" + "\n".join(
            f"- {h['title']}" + (f" ({h['source']})" if h.get("source") else "")
            for h in headlines
        )

    user_msg = (
        f"Today is {datetime.now().strftime('%A, %B %d, %Y')}.\n\n"
        f"CURRENT MARKET DATA:\n{context}"
        f"{headline_text}\n\n"
        "Write exactly 5 bullet points for today's copper market briefing. "
        "Label each bullet with one of: [PRICE], [FUNDS], [MACRO], [SUPPLY], [OUTLOOK]. "
        "Use all 5 labels, one each, in that order.\n\n"
        "Guidelines per bullet:\n"
        "- [PRICE]: Lead with the active month (Jul COMEX). Note where price sits relative to key moving averages and support/resistance. "
        "Mention if momentum is uniformly negative or positive across timeframes. If price is in a dip within an uptrend, say so explicitly.\n"
        "- [FUNDS]: Interpret COT data — are funds adding or cutting? Is positioning crowded or has room to run? "
        "Note whether current positioning supports or threatens the price trend. Mention 52-week percentile context.\n"
        "- [MACRO]: DXY direction and what it means for copper. Any relevant news headlines (tariffs, China, Fed). "
        "Connect macro to copper — don't just state facts, say what they mean for price.\n"
        "- [SUPPLY]: Warehouse stock trends at COMEX and LME. Contango/backwardation structure and what it signals. "
        "Connect supply data to physical market tightness or looseness.\n"
        "- [OUTLOOK]: This is the actionable bullet. Tie it all together: given the price action, fund positioning, macro, and supply picture, "
        "what should a physical copper scrap buyer do today? Be specific about levels to watch (support, resistance, DMA). "
        "Frame buying advice as: dips in uptrends = stay aggressive on purchasing, rallies = fix/lock open sales orders. "
        "Mention specific risk: only unpriced long inventory is at market risk on down moves.\n\n"
        "Start each bullet with '• [LABEL] ' then the text. 2-3 sentences per bullet max. No markdown."
    )

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=900,
        system=(
            "You are a senior copper market analyst writing a daily briefing for Geomet, "
            "a physical scrap copper recycler based in Texas. They are ALWAYS buying — never on the sidelines. "
            "Down days are opportunities to outbid competitors. Up days are for fixing/locking open sales orders.\n\n"
            "Your audience knows copper. Don't explain basics. Be direct, specific, and actionable. "
            "Reference actual price levels, DMAs, and support/resistance. "
            "Never minimize dollar risk — every cent/lb matters in a margin business.\n\n"
            "Key context: 'Unpriced long lbs' (inventory + POs minus priced SOs) is the only real downside risk. "
            "Unpriced sales orders are margin capture opportunities, not risk.\n\n"
            "Return plain text only: one bullet per line starting with '• '. No markdown, no headers."
        ),
        messages=[{"role": "user", "content": user_msg}],
        timeout=25.0,
    )
    raw = resp.content[0].text.strip()
    bullets = [line.strip() for line in raw.split("\n") if line.strip().startswith("•")]
    return bullets if bullets else None


def _fallback_insight(md, sig, cot, roll, outlook):
    """Template-based insight bullets when Claude is unavailable."""
    bullets = []
    if md:
        direction = "higher" if md.get("change", 0) > 0 else "lower" if md.get("change", 0) < 0 else "flat"
        bullets.append(f"• [PRICE] COMEX copper is trading {direction} at ${md.get('price', 0):.4f}/lb ({md.get('change_pct', 0):+.1f}%).")
    mm_net = cot.get("mm_net") if cot else None
    if mm_net is not None:
        stance = "net long" if mm_net > 0 else "net short" if mm_net < 0 else "flat"
        pctile = cot.get("mm_pct_52w", "N/A")
        bullets.append(f"• [FUNDS] Managed money is {stance} {abs(mm_net):,} contracts (52w percentile: {pctile}%).")
    if md and md.get("dxy"):
        dxy = md["dxy"]
        bullets.append(f"• [MACRO] Dollar index at {dxy.get('price', 'N/A')} — {'headwind' if dxy.get('change', 0) > 0 else 'tailwind'} for copper.")
    if md and md.get("warehouse"):
        wh = md["warehouse"]
        comex_wh = wh.get("comex", {})
        lme_wh = wh.get("lme", {})
        global_mt = wh.get("global_mt")
        if global_mt:
            parts = [f"Global warehouse stocks at {global_mt:,} MT"]
            if comex_wh.get("mt"):
                parts.append(f"COMEX {comex_wh['mt']:,} ({comex_wh.get('trend', '?')})")
            if lme_wh.get("mt"):
                parts.append(f"LME {lme_wh['mt']:,} ({lme_wh.get('trend', '?')})")
            bullets.append(f"• [SUPPLY] {', '.join(parts)}.")
        elif comex_wh.get("mt"):
            bullets.append(f"• [SUPPLY] COMEX warehouse stocks at {comex_wh['mt']:,} MT, trend {comex_wh.get('trend', 'unknown')}.")
    if outlook and outlook.get("today"):
        o = outlook["today"]
        bullets.append(f"• [OUTLOOK] Today's bias: {o.get('label', 'neutral')} — physical buyers should {'wait for pullbacks' if o.get('score', 0) > 15 else 'consider covering needs' if o.get('score', 0) < -15 else 'maintain normal buying pace'}.")
    return bullets if bullets else ["• [OUTLOOK] Market data loading — check back shortly."]


def fetch_daily_insight(md, sig, cot, roll, outlook):
    """Orchestrator: check caches, generate if needed, persist."""
    global _insight_cache
    today = datetime.now().strftime("%Y-%m-%d")

    # 1. In-memory cache
    if _insight_cache["data"] and _insight_cache["data"].get("generated_date") == today:
        return _insight_cache["data"]

    # 2. File cache
    if DAILY_INSIGHT_FILE.exists():
        try:
            with open(DAILY_INSIGHT_FILE) as f:
                cached = json.load(f)
            if cached.get("generated_date") == today:
                _insight_cache["data"] = cached
                return cached
        except Exception:
            pass

    # 3. Generate new
    print("[INFO] Generating daily market insight...")
    headlines = _fetch_google_news_headlines()
    bullets = None
    source = "ai"
    try:
        bullets = _claude_daily_insight(md, sig, cot, roll, outlook, headlines)
    except Exception as e:
        print(f"[WARN] Claude daily insight failed ({e}), using fallback")

    if not bullets:
        bullets = _fallback_insight(md, sig, cot, roll, outlook)
        source = "template"

    result = {
        "bullets": bullets,
        "headlines": headlines[:8],
        "generated_date": today,
        "generated_time": datetime.now().strftime("%H:%M"),
        "source": source,
    }

    # Persist
    try:
        with open(DAILY_INSIGHT_FILE, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save daily insight: {e}")

    _insight_cache["data"] = result
    return result


def _claude_intel(text):
    """Use Claude Haiku to extract trading signals from broker notes."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "YOUR_KEY_HERE":
        return None

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=(
            "You are a copper market analyst at a scrap metal recycler. "
            "Extract trading signals from the broker notes provided. "
            "Return ONLY a JSON array (no markdown, no wrapping) of signal objects. "
            "Each signal must have exactly these keys:\n"
            '  "short": one of "bull", "bear", or "watch" (next-few-days outlook)\n'
            '  "long": one of "bull", "bear", or "watch" (weeks/months outlook)\n'
            '  "headline": short title (max 6 words)\n'
            '  "near": one sentence plain-English explanation of near-term impact\n'
            '  "far": one sentence plain-English explanation of bigger-picture impact\n'
            "Focus on copper-relevant signals only. "
            "If the text contains nothing relevant to copper markets, return an empty array [].\n"
            "Return at most 8 signals. Prioritize the most actionable ones."
        ),
        messages=[{"role": "user", "content": text}],
        timeout=15.0,
    )
    raw_json = resp.content[0].text.strip()
    # Strip markdown fences if model wraps them
    if raw_json.startswith("```"):
        raw_json = raw_json.split("\n", 1)[1] if "\n" in raw_json else raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3].strip()
    signals = json.loads(raw_json)
    # Validate structure
    valid = []
    for s in signals:
        if isinstance(s, dict) and all(k in s for k in ("short", "long", "headline", "near", "far")):
            s["short"] = s["short"] if s["short"] in ("bull", "bear", "watch") else "watch"
            s["long"] = s["long"] if s["long"] in ("bull", "bear", "watch") else "watch"
            valid.append(s)
    return valid[:8]


def save_broker_intel(text):
    """Save broker intel and extract signals. Uses Claude AI with keyword fallback."""
    # Try Claude first, fall back to keywords
    try:
        signals = _claude_intel(text)
    except Exception as e:
        print(f"[WARN] Claude intel failed ({e}), using keyword fallback")
        signals = None

    if signals is None:
        signals = _keyword_intel(text)

    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "raw": text,
        "signals": signals,
    }
    DATA_DIR.mkdir(exist_ok=True)
    with open(BROKER_INTEL_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_ship_schedule():
    """Load shipping schedule and return upcoming shipments sorted by date."""
    if not SHIP_SCHEDULE_FILE.exists():
        return []
    try:
        with open(SHIP_SCHEDULE_FILE) as f:
            schedule = json.load(f)
        today = datetime.now().strftime("%Y-%m-%d")
        # Include today and future, sorted by ship date
        upcoming = [s for s in schedule if s.get("ship_date", "") >= today]
        upcoming.sort(key=lambda x: x.get("ship_date", ""))
        return upcoming
    except Exception:
        return []


def get_ship_aware_fix_suggestions(pos, schedule):
    """Cross-reference unpriced orders with ship schedule, return fix priority list."""
    if not pos:
        return []
    unpriced_shipped = pos.get("sales_unpriced_shipped", [])
    unpriced_unshipped = pos.get("sales_unpriced_unshipped", [])
    all_unpriced = {str(s.get("order", "")): s for s in unpriced_shipped + unpriced_unshipped}

    # Build schedule lookup by SO number
    sched_by_so = {}
    for s in schedule:
        so = str(s.get("so", ""))
        if so not in sched_by_so:
            sched_by_so[so] = s

    # Tag unpriced orders with ship dates
    fix_list = []
    for order_num, sale in all_unpriced.items():
        entry = dict(sale)
        sched = sched_by_so.get(order_num)
        if sched:
            entry["ship_date"] = sched["ship_date"]
            entry["delivery_date"] = sched.get("delivery_date", "")
        else:
            entry["ship_date"] = None
        entry["shipped"] = sale in unpriced_shipped
        fix_list.append(entry)

    # Sort: shipped first, then by ship_date (soonest first), then no-date last
    def sort_key(x):
        shipped_priority = 0 if x["shipped"] else 1
        date = x.get("ship_date") or "9999-99-99"
        return (shipped_priority, date)

    fix_list.sort(key=sort_key)
    return fix_list


def gen_decisions(sig, risk, md, fix_window, roll=None, cot=None, pos=None, options_oi=None):
    if not sig: return ["Unable to fetch market data"]
    p = md["price"] if md else 0; ft = CFG["FIX_TARGET"]
    intel = load_broker_intel()
    pct_30d = md.get("pct_30d", 50) if md else 50
    streak = md.get("streak", 0) if md else 0
    streak_dir = md.get("streak_dir") if md else None

    # Three priority buckets
    now = []    # ACT NOW — things that need attention this morning
    watch = []  # WATCH TODAY — important context for decisions
    info = []   # REFERENCE — background data, less urgent

    # --- ACT NOW ---

    # Shipped but unpriced — highest urgency
    if risk and risk.get("sales_unpriced_shipped_lbs", 0) > 0:
        now.append(f"\u26A0 {risk['sales_unpriced_shipped_lbs']:,.0f} lbs shipped but UNPRICED \u2014 fix these first")

    # Ship schedule — urgent shipments
    schedule = load_ship_schedule()
    if schedule and pos:
        unpriced_orders = set()
        for s in pos.get("sales_unpriced_shipped", []) + pos.get("sales_unpriced_unshipped", []):
            unpriced_orders.add(str(s.get("order", "")))
        urgent = []
        upcoming = []
        for ship in schedule:
            so = str(ship.get("so", ""))
            if so in unpriced_orders:
                days_out = (datetime.strptime(ship["ship_date"], "%Y-%m-%d") - datetime.now()).days
                entry = f"SO {so} {ship.get('grade','')} \u2192 {ship.get('customer','')[:12]} ships {ship['ship_date'][5:]}"
                if days_out <= 0:
                    urgent.append(entry)
                elif days_out <= 7:
                    upcoming.append(entry)
        if urgent:
            now.append(f"\U0001F534 SHIPPING TODAY/OVERDUE UNPRICED: {'; '.join(urgent)}")
        if upcoming:
            now.append(f"\u26A0 Ships within 7d unpriced: {'; '.join(upcoming[:3])}")

    # Fix window — if strong, act now
    if fix_window:
        fw = fix_window
        if fw["score"] >= 65:
            factors_str = " + ".join(fw["factors"][:3]) if fw["factors"] else ""
            now.append(f"\U0001F7E2 FIX WINDOW: {fw['label']} ({fw['score']}/100) \u2014 {factors_str}")
        elif fw["score"] <= 30:
            factors_str = " + ".join(fw["factors"][:3]) if fw["factors"] else ""
            watch.append(f"\U0001F534 FIX WINDOW: {fw['label']} ({fw['score']}/100) \u2014 {factors_str}")

    # Above target — actionable
    if p >= ft and risk and risk["net_lbs"] > 0:
        now.append(f"ABOVE ${ft:.2f} TARGET \u2014 {risk.get('loads_unpriced',0)} loads unpriced. Fix some.")
    elif p >= ft - 0.10:
        watch.append(f"${p:.4f} \u2014 approaching ${ft:.2f}. GTCs in place.")

    # Position range — trim/add advice
    if risk:
        net = risk["net_lbs"]
        rmin = CFG["POSITION_RANGE_MIN"]
        rmax = CFG["POSITION_RANGE_MAX"]
        rng = rmax - rmin if rmax > rmin else 1
        pos_pct = max(0, min(100, ((net - rmin) / rng) * 100))
        tl = CFG["TRUCKLOAD_LBS"]
        loads_to_floor = max(0, round((net - rmin) / tl))
        loads_to_ceil = max(0, round((rmax - net) / tl))

        # Determine short-term market bias from signals
        bearish_count = 0; bullish_count = 0
        if intel and intel.get("signals"):
            for s in intel["signals"]:
                if s.get("short") == "bear": bearish_count += 1
                elif s.get("short") == "bull": bullish_count += 1
        if pct_30d >= 85: bearish_count += 1
        if streak >= 3 and streak_dir == "up": bearish_count += 1
        if pct_30d <= 25: bullish_count += 1
        if streak >= 3 and streak_dir == "down": bullish_count += 1
        bias = "bearish" if bearish_count > bullish_count else "bullish" if bullish_count > bearish_count else "neutral"

        # Over/under range is urgent
        if net > rmax:
            now.append(f"\U0001F534 OVER MAX ({net:,.0f} / {rmax:,.0f} lbs) \u2014 fix {round((net - rmax) / tl)}+ loads to get back in range")
        elif net < rmin:
            now.append(f"UNDER FLOOR ({net:,.0f} / {rmin:,.0f} lbs) \u2014 room to add {loads_to_ceil} loads")
        elif pos_pct >= 60 and bias == "bearish":
            target = rmin + int(rng * 0.2)
            trim_loads = max(1, round((net - target) / tl))
            now.append(f"POSITION {pos_pct:.0f}% of range ({net:,.0f} lbs) \u2014 bearish signals, trim ~{trim_loads} loads toward {target:,.0f}")
        elif pos_pct >= 60 and bias == "neutral":
            watch.append(f"POSITION {pos_pct:.0f}% of range ({net:,.0f} lbs) \u2014 mixed signals, hold but watch for trim")
        elif pos_pct <= 30 and bias == "bullish":
            watch.append(f"POSITION {pos_pct:.0f}% of range ({net:,.0f} lbs) \u2014 bullish signals, room to add {loads_to_ceil} loads")
        elif pos_pct <= 30 and bias == "neutral":
            info.append(f"POSITION {pos_pct:.0f}% of range ({net:,.0f} lbs) \u2014 near floor, good defensive positioning")
        else:
            watch.append(f"POSITION {pos_pct:.0f}% of range ({net:,.0f} lbs) \u2014 {bias} bias, {loads_to_floor} loads above floor")

    # Extreme move types — act now
    if sig["move_type"] == "LIQUIDATION":
        now.append("Liquidation event \u2014 competitors scared. Lean into buys.")
    elif sig["move_type"] == "FLASH_CRASH":
        now.append("Flash crash on thin volume \u2014 buying window.")
    elif sig["move_type"] == "BIG_DROP":
        now.append("Big drop \u2014 buying opportunity.")
    elif sig["move_type"] == "DIP":
        now.append("Dip day \u2014 outbid competitors, build relationships.")

    # Contract roll — urgent if critical
    if roll and roll.get("roll_urgency") in ("critical", "warning"):
        nm_ticker = roll.get("next_month", {}).get("ticker", "next contract")
        target = now if roll["roll_urgency"] == "critical" else watch
        target.append(f"\u26A0 {roll['front_month']['ticker']}: {roll['roll_status']} \u2014 liquidity migrating to {nm_ticker}")

    # --- WATCH TODAY ---

    # Month-end / Friday pressure
    today_dt = datetime.now()
    dom = today_dt.day
    dow = today_dt.weekday()
    last_day = calendar.monthrange(today_dt.year, today_dt.month)[1]
    trading_days_left = sum(1 for d in range(dom + 1, last_day + 1)
                           if datetime(today_dt.year, today_dt.month, d).weekday() < 5)
    if trading_days_left <= 3 and pct_30d >= 75:
        watch.append(f"Month-end in {trading_days_left} trading day{'s' if trading_days_left != 1 else ''} \u2014 rebalancing selling pressure likely at {pct_30d:.0f}th pctl")
    if dow == 4 and pct_30d >= 75:
        watch.append(f"Friday \u2014 week-end profit-taking likely after rally ({pct_30d:.0f}th pctl)")

    # S/R context
    sr = md.get("support_resistance", {}) if md else {}
    for ctx in sr.get("context", []):
        watch.append(ctx)

    # Broker intel headlines
    if intel and intel.get("signals"):
        for s in intel["signals"]:
            st = s.get("short", "watch"); lt = s.get("long", "watch")
            st_icon = "\U0001F7E2" if st == "bull" else "\U0001F534" if st == "bear" else "\u26A0"
            lt_icon = "\U0001F7E2" if lt == "bull" else "\U0001F534" if lt == "bear" else "\u26A0"
            if s.get("headline"):
                watch.append(f"INTEL: {s['headline']} [{st_icon} near-term / {lt_icon} long-term]")
                watch.append(f"INTEL_DETAIL: {st_icon} Next few days: {s.get('near', '')}")
                watch.append(f"INTEL_DETAIL: {lt_icon} Bigger picture: {s.get('far', '')}")

    # China session
    china = md.get("china", {}) if md else {}
    if china.get("thin_liquidity"):
        watch.append(f"\u26A0 {china.get('detail', 'SHFE closed')} \u2014 thin liquidity, watch for flash moves")
    elif china.get("status") == "CLOSED" and china.get("reason") == "Lunar New Year":
        watch.append(china.get("detail", "SHFE closed"))

    # Options OI walls — proximity alerts
    if options_oi and p > 0:
        pw = options_oi.get("put_wall")
        cw = options_oi.get("call_wall")
        mp = options_oi.get("max_pain")
        if pw:
            pw_dist = (p - pw["strike"])
            if 0 <= pw_dist <= 0.05:
                watch.append(f"\U0001F7E2 PUT WALL ${pw['strike']:.2f} ({pw['oi']:,} contracts) — {pw_dist * 100:.0f}c above institutional support")
            elif pw_dist < 0 and pw_dist >= -0.10:
                watch.append(f"\U0001F534 BELOW PUT WALL ${pw['strike']:.2f} ({pw['oi']:,} contracts) — {abs(pw_dist) * 100:.0f}c below support")
        if cw:
            cw_dist = (cw["strike"] - p)
            if 0 <= cw_dist <= 0.05:
                watch.append(f"\U0001F534 CALL WALL ${cw['strike']:.2f} ({cw['oi']:,} contracts) — {cw_dist * 100:.0f}c below institutional resistance")
            elif cw_dist < 0 and cw_dist >= -0.10:
                watch.append(f"\U0001F7E2 ABOVE CALL WALL ${cw['strike']:.2f} ({cw['oi']:,} contracts) — breakout {abs(cw_dist) * 100:.0f}c above resistance")
        if mp:
            mp_dist = abs(p - mp["strike"])
            if mp_dist <= 0.03:
                watch.append(f"MAX PAIN ${mp['strike']:.2f} — price at expiry magnet ({mp_dist * 100:.0f}c away)")

    # Extended streak
    if streak >= 4 and streak_dir == "up":
        watch.append(f"{streak}-day up streak \u2014 extended rally, pullback risk rising")
    elif streak >= 4 and streak_dir == "down":
        watch.append(f"{streak}-day down streak \u2014 oversold, bounce likely")

    # COT — extreme positioning is watch, otherwise info
    if cot:
        pct = cot.get("mm_pct_52w", 50)
        chg = cot.get("mm_weekly_change", 0)
        if pct >= 90:
            watch.append(f"COT: Managed Money EXTREMELY LONG ({pct:.0f}th pctl) \u2014 crowded trade, selloff risk")
        elif pct <= 10:
            watch.append(f"COT: Managed Money EXTREMELY SHORT ({pct:.0f}th pctl) \u2014 squeeze potential, buying opportunity")
        elif pct >= 70 and chg < -5000:
            info.append(f"COT: MM long but unwinding ({chg:+,} wk) \u2014 momentum selling watch")
        elif pct <= 30 and chg > 5000:
            info.append(f"COT: MM short but covering ({chg:+,} wk) \u2014 rally pressure")

    # --- REFERENCE ---

    ss = sig.get("spread_signal")
    if ss: info.append(ss["msg"])
    ds = sig.get("dxy_signal")
    if ds and ds["direction"] != "flat": info.append(ds["msg"])

    fed = md.get("fed", {}) if md else {}
    if fed.get("first_cut") and fed["first_cut"] != "None priced":
        info.append(f"Fed: {fed.get('total_cuts_2026', 0)} cut(s) priced by year-end, first in {fed['first_cut']}")
    elif fed.get("yield_10y"):
        info.append(f"10Y yield: {fed['yield_10y']}% ({fed.get('yield_10y_change', 0):+.2f})")

    wh = md.get("warehouse") if md else None
    if wh and wh.get("mt"):
        arrow = "\u2191" if wh["trend"] == "building" else "\u2193" if wh["trend"] == "drawing" else "\u2192"
        msg = f"COMEX: {wh['mt']:,} MT {arrow}"
        if wh.get("lme"):
            lw = wh["lme"]
            la = "\u2191" if lw["trend"] == "building" else "\u2193" if lw["trend"] == "drawing" else "\u2192"
            msg += f" / LME: {lw['mt']:,} MT {la}"
        if wh.get("global_mt"):
            msg += f" \u2014 Global: {wh['global_mt']:,} MT"
        info.append(msg)

    if sig.get("momentum_note"): info.append(sig["momentum_note"])

    if risk:
        uh = risk["unhedged_lbs"]; rd = risk["risk_per_dime"]
        if uh > 0: info.append(f"Long {uh:,.0f} lbs unpriced \u2014 ${abs(rd):,.0f} per 10c move")

    if risk:
        mf = CFG.get("MONTHLY_FLOW", {})
        sc = risk.get("sales_by_commodity", {})
        thin = []
        for grade, flow in mf.items():
            if flow > 0:
                sal = sc.get(grade, 0)
                months = sal / flow
                if months < 2:
                    thin.append(f"{grade} ({months:.1f}mo)")
        if thin:
            info.append(f"\u26A0 Sales pipeline thin: {', '.join(thin)} \u2014 need to sell")

    # Combine with section dividers
    dec = []
    if now:
        dec.append("SEC:ACT NOW")
        dec.extend(now)
    if watch:
        dec.append("SEC:WATCH TODAY")
        dec.extend(watch)
    if info:
        dec.append("SEC:REFERENCE")
        dec.extend(info)
    if not dec: dec.append("Markets stable \u2014 normal operations")
    return dec


_position_cache = {"path": None, "mtime": 0, "pos": None, "ts": 0}
_POSITION_CACHE_TTL = 60  # seconds — re-read xlsx at most once per minute

def load_position():
    # Spreadsheet is primary — Jorge's curated source of truth
    hf = find_latest_hedge_file()
    if hf:
        try:
            mtime = os.path.getmtime(hf)
        except OSError:
            mtime = 0
        now = time.time()
        cached = _position_cache
        if (cached["pos"] and cached["path"] == hf and cached["mtime"] == mtime
                and (now - cached["ts"]) < _POSITION_CACHE_TTL):
            return cached["pos"]

        print(f"[INFO] Reading: {os.path.basename(hf)}")
        pos = read_hedge_spreadsheet(hf)
        if pos:
            pos["data_source"] = pos.get("source_file", "spreadsheet")
            # Override avg_cost with ROM per-grade costs (grossed up for ICW)
            try:
                from rom import _fetch_inv_avg_costs, calc_blended_avg_cost
                grade_costs = _fetch_inv_avg_costs()
                if grade_costs:
                    inv_by_commodity = pos.get("inv_by_commodity", {})
                    icw_cu_lbs = pos.get("icw_cu_lbs", 0)
                    blended = calc_blended_avg_cost(grade_costs, inv_by_commodity, icw_cu_lbs)
                    if blended > 0:
                        pos["avg_cost"] = blended
                        pos["grade_costs"] = grade_costs
                        print(f"[INFO] Avg cost from ROM: ${blended:.4f}/lb (grades: {grade_costs})")
            except Exception as e:
                print(f"[WARN] ROM cost overlay failed, using spreadsheet avg_cost: {e}")
            _position_cache.update({"path": hf, "mtime": mtime, "pos": pos, "ts": now})
            return pos

    # Fallback to ROM if no spreadsheet available
    try:
        from rom import read_rom_position
        pos = read_rom_position()
        if pos:
            pos.setdefault("data_source", "ROM")
            return pos
    except Exception as e:
        print(f"[WARN] ROM unavailable: {e}")

    # Last resort: CSV
    if POSITION_CSV.exists():
        try:
            with open(POSITION_CSV, "r") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    r = rows[0]
                    return {"net_lbs": float(r.get("net_copper_lbs", 0)), "avg_cost": float(r.get("avg_cost_per_lb", 0)),
                            "hedge_lbs": float(r.get("hedge_lbs", 0)), "updated": r.get("date", "unknown"),
                            "source_file": "geomet_position.csv", "data_source": "CSV"}
        except Exception as e: print(f"[ERROR] CSV: {e}")
    return None


# ---------------------------------------------------------------------------
# AUTH — simple session-based login
# ---------------------------------------------------------------------------
_USERS = {
    "richard": hashlib.sha256(b"geomet").hexdigest(),
    "jorge": hashlib.sha256(b"geomet").hexdigest(),
    "mikel": hashlib.sha256(b"geomet").hexdigest(),
}
SESSIONS_FILE = DATA_DIR / "sessions.json"
SESSION_MAX_AGE = 86400 * 30  # 30 days

def _load_sessions():
    if not SESSIONS_FILE.exists():
        return {}
    try:
        with open(SESSIONS_FILE) as f:
            data = json.load(f)
        # Drop expired entries on load
        cutoff = time.time() - SESSION_MAX_AGE
        return {t: s for t, s in data.items() if s.get("created", 0) > cutoff}
    except Exception as e:
        print(f"[WARN] Failed to load sessions: {e}")
        return {}

def _save_sessions():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SESSIONS_FILE, "w") as f:
            json.dump(_sessions, f)
    except Exception as e:
        print(f"[WARN] Failed to save sessions: {e}")

_sessions = _load_sessions()  # token -> {"user": ..., "created": timestamp}

def _check_session(cookie_header):
    if not cookie_header: return None
    c = SimpleCookie()
    c.load(cookie_header)
    if "session" not in c: return None
    token = c["session"].value
    s = _sessions.get(token)
    if s and (time.time() - s["created"]) < SESSION_MAX_AGE:
        return s["user"]
    if token in _sessions:
        _sessions.pop(token, None)
        _save_sessions()
    return None

def _create_session(user):
    token = secrets.token_hex(32)
    _sessions[token] = {"user": user, "created": time.time()}
    _save_sessions()
    return token

LOGIN_PAGE = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Geomet — Login</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'IBM Plex Sans',sans-serif;background:#0a0e14;color:#e0e6ed;min-height:100vh;display:flex;align-items:center;justify-content:center}
.box{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:40px;width:320px}
.logo{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:3px;color:#d4845a;padding:4px 8px;border:1.5px solid #d4845a;border-radius:3px;display:inline-block;margin-bottom:20px}
h2{font-size:14px;font-weight:300;color:#6b7f99;margin-bottom:20px}
input{width:100%;padding:10px 12px;margin-bottom:12px;background:#1a2230;border:1px solid #1e2a3a;border-radius:4px;color:#e0e6ed;font-family:'IBM Plex Sans',sans-serif;font-size:13px}
input:focus{outline:none;border-color:#d4845a}
button{width:100%;padding:10px;background:rgba(212,132,90,.15);border:1px solid #d4845a;border-radius:4px;color:#d4845a;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;letter-spacing:1px;cursor:pointer}
button:hover{background:rgba(212,132,90,.25)}
.err{color:#ff4757;font-size:11px;margin-bottom:10px;display:none}
</style></head><body>
<div class="box"><div class="logo">GEOMET</div><h2>Copper Intelligence Dashboard</h2>
<div class="err" id="err">Invalid username or password</div>
<form id="login-form" method="POST" action="/login">
<input id="username" name="user" placeholder="Username" autocomplete="username" required>
<input id="password" name="pass" type="password" placeholder="Password" autocomplete="current-password" required>
<button type="submit">LOGIN</button></form></div>
<script>if(location.search.includes('err=1'))document.getElementById('err').style.display='block'</script>
</body></html>"""

THEME_FILE = DATA_DIR / "theme.json"

def get_theme():
    if THEME_FILE.exists():
        try:
            with open(THEME_FILE) as f: return json.load(f).get("theme", "dark")
        except: pass
    return "dark"

def set_theme(theme):
    try:
        with open(THEME_FILE, "w") as f: json.dump({"theme": theme}, f)
    except: pass

class Handler(SimpleHTTPRequestHandler):
    def _authed(self):
        return _check_session(self.headers.get("Cookie"))

    def do_GET(self):
        if self.path == "/login":
            self.send_response(200)
            self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(LOGIN_PAGE.encode()); return
        if self.path == "/logout":
            c = SimpleCookie()
            c.load(self.headers.get("Cookie") or "")
            if "session" in c and _sessions.pop(c["session"].value, None):
                _save_sessions()
            self.send_response(302)
            self.send_header("Set-Cookie", "session=; Path=/; Max-Age=0")
            self.send_header("Location", "/login"); self.end_headers(); return
        if not self._authed():
            self.send_response(302)
            self.send_header("Location", "/login"); self.end_headers(); return
        if self.path == "/api/theme":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"theme": get_theme()}).encode())
            return
        if self.path == "/api/intel":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            intel = load_broker_intel()
            self.wfile.write(json.dumps(intel or {}).encode())
            return
        if self.path == "/api/debug/rom-compare":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            compare = {"rom": None, "spreadsheet": None, "diff": {}}
            try:
                from rom import read_rom_position
                compare["rom"] = read_rom_position()
            except Exception as e:
                compare["rom_error"] = str(e)
            hf = find_latest_hedge_file()
            if hf:
                compare["spreadsheet"] = read_hedge_spreadsheet(hf)
            if compare["rom"] and compare["spreadsheet"]:
                r, s = compare["rom"], compare["spreadsheet"]
                for key in ["net_lbs", "avg_cost", "total_inv_po", "inventory_cu_lbs",
                            "po_lbs", "priced_sales_lbs", "unpriced_sales_lbs",
                            "comex_futures", "lme_futures"]:
                    rv = r.get(key, 0) or 0
                    sv = s.get(key, 0) or 0
                    compare["diff"][key] = {"rom": rv, "spreadsheet": sv,
                                            "delta": round(rv - sv, 2)}
                for grade in ["BB", "#1", "#2", "Chops"]:
                    ri = (r.get("inv_by_commodity") or {}).get(grade, 0)
                    si = (s.get("inv_by_commodity") or {}).get(grade, 0)
                    compare["diff"][f"inv_{grade}"] = {"rom": ri, "spreadsheet": si,
                                                       "delta": round(ri - si, 2)}
            self.wfile.write(json.dumps(compare, default=str).encode())
            return
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            md = fetch_copper_data()
            # Overlay real-time price on top of cached daily OHLC
            rt = _fetch_realtime_price()
            if rt.get("price") and md:
                md = dict(md)  # copy so we don't mutate the cache
                prev = rt.get("prev_close") or md.get("daily_prev_settle", md["prev_close"])
                md["price"] = rt["price"]
                md["prev_close"] = prev
                # Use TradingView's session-accurate change if available
                if rt.get("change") is not None and rt.get("source") == "tradingview":
                    md["change"] = rt["change"]
                    md["change_pct"] = rt["change_pct"] or 0
                else:
                    md["change"] = round(rt["price"] - prev, 4)
                    md["change_pct"] = round(((rt["price"] - prev) / prev) * 100, 2) if prev else 0
                md["copper_source"] = rt.get("source", md.get("copper_source", ""))
                # Tell frontend which contract the big price represents
                md["active_contract"] = rt.get("active_contract", "front")
                # Recalculate COMEX-LME spread with RT price
                # Only override if we don't have a locked peak-hours snapshot
                if md.get("lme_price_lb") and md.get("spread_timing") != "peak":
                    md["comex_lme_spread"] = round(rt["price"] - md["lme_price_lb"], 4)
            sig = compute_signals(md)
            pos = load_position()
            # Compare ROM vs spreadsheet — itemized discrepancies with dollar impact
            # Runs regardless of which source is primary, as long as ROM is reachable
            rom_note = None
            try:
                from rom import read_rom_position
                rom_pos = read_rom_position() if pos else None
                # Overlay ROM-derived live ship schedule onto pos (spreadsheet has none)
                if pos and rom_pos and rom_pos.get("ship_schedule"):
                    pos["ship_schedule"] = rom_pos["ship_schedule"]
                # Overlay ROM sales (has formula pricing data from OrderOverRide)
                if pos and rom_pos:
                    for sk in ("sales_unpriced_shipped", "sales_unpriced_unshipped", "sales_priced_unshipped"):
                        if rom_pos.get(sk):
                            pos[sk] = rom_pos[sk]
                ss_pos = pos if pos and pos.get("data_source") != "ROM" else None
                if not ss_pos:
                    _hf = find_latest_hedge_file()
                    if _hf:
                        ss_pos = read_hedge_spreadsheet(_hf)
                if rom_pos and ss_pos:
                    price = md.get("price", 5.0) if md else 5.0
                    items = []
                    # Priced sales — biggest potential impact
                    rp = rom_pos.get("priced_sales_lbs", 0) or 0
                    sp = ss_pos.get("priced_sales_lbs", 0) or 0
                    dp = rp - sp
                    if abs(dp) > 10000:
                        items.append({"label": "Priced Sales", "rom": round(rp), "ss": round(sp),
                            "delta_lbs": round(dp), "dollar": round(abs(dp) * price),
                            "why": "ROM likely includes fulfilled orders never closed in system",
                            "action": "Review old priced SOs in ROM — close completed ones"})
                    # Priced avg price
                    ra = rom_pos.get("priced_sales_avg", 0) or 0
                    sa = ss_pos.get("priced_sales_avg", 0) or 0
                    da = ra - sa
                    if abs(da) > 0.10 and rp > 0:
                        items.append({"label": "Avg Priced Price", "rom": round(ra, 4), "ss": round(sa, 4),
                            "delta_lbs": 0, "dollar": round(abs(da) * rp),
                            "why": "Old orders at lower historical prices drag ROM average down",
                            "action": "Affects margin projection — resolves when stale SOs are closed"})
                    # Unpriced sales
                    ru = rom_pos.get("unpriced_sales_lbs", 0) or 0
                    su = ss_pos.get("unpriced_sales_lbs", 0) or 0
                    du = ru - su
                    if abs(du) > 10000:
                        items.append({"label": "Unpriced Sales", "rom": round(ru), "ss": round(su),
                            "delta_lbs": round(du), "dollar": round(abs(du) * price),
                            "why": "Small variance in open unpriced order count",
                            "action": "Low priority — numbers are close"})
                    # Inventory
                    ri = rom_pos.get("inventory_cu_lbs", 0) or 0
                    si = ss_pos.get("inventory_cu_lbs", 0) or 0
                    di = ri - si
                    if abs(di) > 10000:
                        items.append({"label": "Inventory", "rom": round(ri), "ss": round(si),
                            "delta_lbs": round(di), "dollar": round(abs(di) * price),
                            "why": "Both from spreadsheet — rounding or unmapped items",
                            "action": "Check if minor grades are missing from grade mapping"})
                    # Sort by dollar impact descending
                    items.sort(key=lambda x: x["dollar"], reverse=True)
                    if items:
                        rom_note = {"items": items, "ss_file": ss_pos.get("source_file", "")}
            except Exception:
                pass
            risk = calc_risk(pos, md)
            if risk:
                risk["baseline_lbs"] = CFG["BASELINE_LBS"]
                risk["baseline_deviation"] = risk["net_lbs"] - CFG["BASELINE_LBS"]
            fix_window = calc_fix_window(md, sig)
            # When RT source is the next month (e.g. investing.com → May),
            # pass None as copper_price so contract_roll fetches front independently.
            # Then we override the spread with the correct direction.
            _ac = md.get("active_contract", "front") if md else "front"
            roll = get_contract_roll(md.get("price") if (_ac == "front" and md) else None)
            if roll and _ac == "next" and md:
                # HG1! has rolled to next month (Jul) — remap prices correctly
                # md.price (from HG1!) is actually next month (Jul)
                roll["next_price"] = md["price"]
                roll["next_source"] = md.get("source", "tradingview")
                # HG1! change includes the roll gap (Jul_now - May_prev_close) — wrong.
                # Compute real Jul change from Jul's own previous settle.
                _jul_ticker = roll.get("next_month", {}).get("ticker", "")
                _jul_yf = _jul_ticker.replace("HG", "HG", 1) + ".CMX" if _jul_ticker else None
                if _jul_yf:
                    _jul_prev = _fetch_prev_settle_yf(_jul_yf)
                    if _jul_prev and md["price"]:
                        roll["next_change"] = round(md["price"] - _jul_prev, 4)
                        roll["next_change_pct"] = round((roll["next_change"] / _jul_prev) * 100, 2)
                    else:
                        roll["next_change"] = md.get("change")
                        roll["next_change_pct"] = md.get("change_pct")
                else:
                    roll["next_change"] = md.get("change")
                    roll["next_change_pct"] = md.get("change_pct")
                # HG2! (in _tv_state_2) is now the third month (Sep) — remap it
                with _tv_lock:
                    _tv2p = _tv_state_2.get("price")
                    _tv2age = time.time() - _tv_state_2["timestamp"] if _tv_state_2.get("timestamp") else 999
                if _tv2p and _tv2age < 120:
                    roll["third_price"] = _tv2p
                if roll.get("front_price"):
                    spread = round(md["price"] - roll["front_price"], 4)
                    roll["calendar_spread"] = spread
                    roll["market_structure"] = "contango" if spread > 0.001 else "backwardation" if spread < -0.001 else "flat"
            cot = fetch_cot_data()
            options_oi = fetch_options_oi()
            # Price + OI change since last COT report date
            cot_context = None
            if cot and md and md.get("sparkline"):
                cot_date = cot["report_date"]
                price_at_cot = None
                for pt in md["sparkline"]:
                    if pt["date"] <= cot_date:
                        price_at_cot = pt["close"]
                if price_at_cot:
                    price_delta = round(md["price"] - price_at_cot, 4)
                    cot_context = {"price_delta": price_delta}
                    # OI comparison only when we have front-month history covering that date
                    if roll and roll.get("open_interest"):
                        oi_at_cot = _get_oi_on_date(cot_date)
                        current_oi = roll["open_interest"]["total"]
                        if oi_at_cot:
                            oi_delta = current_oi - oi_at_cot
                            cot_context["oi_at_cot"] = oi_at_cot
                            cot_context["oi_now"] = current_oi
                            cot_context["oi_delta"] = oi_delta
                            # Price-OI interpretation
                            if oi_delta > 500 and price_delta > 0.01:
                                cot_context["interp"] = "OI up + price up \u2192 new longs being added"
                            elif oi_delta > 500 and price_delta < -0.01:
                                cot_context["interp"] = "OI up + price down \u2192 new shorts being added"
                            elif oi_delta < -500 and price_delta > 0.01:
                                cot_context["interp"] = "OI down + price up \u2192 shorts covering"
                            elif oi_delta < -500 and price_delta < -0.01:
                                cot_context["interp"] = "OI down + price down \u2192 longs liquidating"
            outlook = calc_price_outlook(sig, md, cot, roll, options_oi)
            dec = gen_decisions(sig, risk, md, fix_window, roll, cot=cot, pos=pos, options_oi=options_oi)
            gtc = gen_gtc(pos, md)
            gtc_placed = enrich_gtc_orders(load_gtc_orders(), md)
            margin = calc_margin_projection(pos, md, risk)
            fixable = calc_fixable_orders(pos, md)
            # Prefer ROM-derived schedule (live from TransAppointments). Fall back
            # to file-based ship_schedule.json when ROM has none.
            schedule = (pos.get("ship_schedule") if pos else None) or load_ship_schedule()
            daily_insight = fetch_daily_insight(md, sig, cot, roll, outlook)
            # Merge market rates: JSON file overrides config.py defaults
            _mr_file = load_market_rates()
            _mr_rates = dict(CFG.get("MARKET_RATES", {}))
            _mr_date = CFG.get("MARKET_RATES_DATE", "")
            _mr_comex = CFG.get("MARKET_RATES_COMEX_STAMP", 0)
            _mr_source = ""
            _mr_history = []
            if _mr_file:
                _mr_rates.update(_mr_file.get("rates", {}))
                _mr_date = _mr_file.get("date", _mr_date)
                _mr_comex = _mr_file.get("comex_stamp", _mr_comex)
                _mr_source = _mr_file.get("source", "")
                _mr_history = _mr_file.get("history", [])
            payload = {
                "market": md, "signals": sig, "position": pos, "position_risk": risk,
                "decisions": dec, "gtc_suggestions": gtc, "gtc_placed": gtc_placed,
                "fix_window": fix_window, "fixable_orders": fixable, "outlook": outlook,
                "margin_projection": margin, "contract_roll": roll,
                "cot": cot, "cot_context": cot_context, "options_oi": options_oi,
                "ship_schedule": schedule, "daily_insight": daily_insight,
                "config": {"fix_target": CFG["FIX_TARGET"], "truckload_lbs": CFG["TRUCKLOAD_LBS"], "gtc_levels": CFG["GTC_LEVELS"], "baseline_lbs": CFG["BASELINE_LBS"], "monthly_flow": CFG["MONTHLY_FLOW"], "position_range_min": CFG["POSITION_RANGE_MIN"], "position_range_max": CFG["POSITION_RANGE_MAX"], "market_rates": _mr_rates, "market_rates_lme": CFG.get("MARKET_RATES_LME_AT_UPDATE", 0), "market_rates_date": _mr_date, "market_rates_comex_stamp": _mr_comex, "market_rates_source": _mr_source, "market_rates_stale": CFG.get("MARKET_RATES_STALE_THRESHOLD", 0.15), "icw_recovery": CFG.get("ICW_RECOVERY", {}), "custom_levels": CFG.get("CUSTOM_LEVELS", []), "quote_history": _mr_history},
                "data_source": pos.get("data_source", "unknown") if pos else "none",
                "rom_note": rom_note,
                "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.wfile.write(json.dumps(payload).encode())
            return
        # Strip query string before file lookup so ?param=1 still resolves
        path_only = self.path.split("?", 1)[0]
        if path_only in ("/", ""): path_only = "/index.html"
        fp = STATIC_DIR / path_only.lstrip("/")
        if fp.exists() and fp.is_file():
            self.send_response(200)
            ext = fp.suffix.lower()
            ct = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
                  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                  ".svg": "image/svg+xml", ".ico": "image/x-icon"}.get(ext, "application/octet-stream")
            self.send_header("Content-Type", ct); self.end_headers()
            self.wfile.write(fp.read_bytes())
        else: self.send_error(404)
    def do_POST(self):
        if self.path == "/login":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            from urllib.parse import parse_qs
            params = parse_qs(body)
            user = params.get("user", [""])[0].strip().lower()
            pwd = params.get("pass", [""])[0]
            pwd_hash = hashlib.sha256(pwd.encode()).hexdigest()
            if user in _USERS and _USERS[user] == pwd_hash:
                token = _create_session(user)
                self.send_response(302)
                self.send_header("Set-Cookie", f"session={token}; Path=/; Max-Age={SESSION_MAX_AGE}; HttpOnly; SameSite=Lax")
                self.send_header("Location", "/"); self.end_headers()
            else:
                self.send_response(302)
                self.send_header("Location", "/login?err=1"); self.end_headers()
            return
        if not self._authed():
            self.send_response(403)
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}'); return
        if self.path == "/api/theme":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            theme = body.get("theme", "dark")
            if theme not in ("dark", "light"): theme = "dark"
            set_theme(theme)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"theme": theme}).encode())
            return
        if self.path == "/api/market-rates":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            rates = body.get("rates", {})
            source = body.get("source", "")
            sources = body.get("sources", {})
            if rates:
                data = save_market_rates(rates, source, sources)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "data": data}).encode())
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":false,"error":"no rates provided"}')
            return
        if self.path == "/api/market-rates/delete-quote":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            idx = body.get("index")
            if idx is not None:
                existing = load_market_rates()
                history = existing.get("history", []) if existing else []
                if 0 <= idx < len(history):
                    removed = history.pop(idx)
                    existing["history"] = history
                    with open(MARKET_RATES_FILE, "w") as f:
                        json.dump(existing, f, indent=2)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": True, "removed": removed}).encode())
                else:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok":false,"error":"invalid index"}')
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":false,"error":"no index provided"}')
            return
        if self.path == "/api/intel":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            text = body.get("text", "").strip()
            if text:
                data = save_broker_intel(text)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            else:
                # Empty text = clear intel
                if BROKER_INTEL_FILE.exists():
                    BROKER_INTEL_FILE.unlink()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"cleared":true}')
            return
        if self.path == "/api/gtc":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            action = body.get("action", "")
            # Load ALL orders (including non-active) for persistence
            all_orders = []
            if GTC_ORDERS_FILE.exists():
                try:
                    with open(GTC_ORDERS_FILE) as f:
                        all_orders = json.load(f)
                except Exception:
                    all_orders = []
            resp = {"ok": False}
            if action == "add":
                new_id = max((o.get("id", 0) for o in all_orders), default=0) + 1
                order = {
                    "id": new_id,
                    "customer": body.get("customer", ""),
                    "grade": body.get("grade", ""),
                    "basis": body.get("basis", "LME_CASH"),
                    "price_mt": body.get("price_mt", 0),
                    "loads": body.get("loads", 1),
                    "status": "active",
                    "created": datetime.now().strftime("%Y-%m-%d"),
                }
                all_orders.append(order)
                save_gtc_orders(all_orders)
                resp = {"ok": True, "order": order}
            elif action == "remove":
                oid = body.get("id")
                all_orders = [o for o in all_orders if o.get("id") != oid]
                save_gtc_orders(all_orders)
                resp = {"ok": True}
            elif action == "fill":
                oid = body.get("id")
                for o in all_orders:
                    if o.get("id") == oid:
                        o["status"] = "filled"
                save_gtc_orders(all_orders)
                resp = {"ok": True}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
            return
        self.send_error(404)
    def log_message(self, *a): pass

def main():
    DATA_DIR.mkdir(exist_ok=True); STATIC_DIR.mkdir(exist_ok=True)
    hf = find_latest_hedge_file()
    lme_src = "metals.dev API" if CFG["METALS_DEV_API_KEY"] else ("manual" if CFG["LME_MANUAL_USD_MT"] else "NOT CONFIGURED")
    print()
    print("=" * 55)
    print("  GEOMET COPPER INTELLIGENCE DASHBOARD v5.0")
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
    class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True
    with ThreadedServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    main()
