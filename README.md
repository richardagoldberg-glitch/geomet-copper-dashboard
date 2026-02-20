# Geomet Copper Intelligence Dashboard

Real-time copper market dashboard for Geomet Recycling. Tracks COMEX and LME copper prices, hedging positions, and generates actionable trading signals.

## Features

- **Live COMEX/LME pricing** with 30-day spread tracking and COMEX-LME arbitrage signals
- **Fix Window scoring** — composite 0–100 score for timing price-fixings against unpriced longs
- **Hedge position reader** — auto-imports from Excel hedge worksheets (OneDrive or local)
- **GTC order suggestions** — distributes unpriced truckloads across target price levels
- **Risk metrics** — mark-to-market P&L, unhedged exposure, dollar risk per cent/dime
- **Macro context** — DXY, Fed Funds rate (via FRED API), rate cut expectations, SHFE/China session status, COMEX warehouse stocks
- **Technical signals** — moving averages, support/resistance, momentum, volume analysis

## Setup

### 1. Install dependencies

```
pip install yfinance openpyxl
```

### 2. Configure environment

Copy `.env.example` to `.env` and add your API keys:

```
METALS_DEV_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

- **METALS_DEV_API_KEY** — free at [metals.dev](https://metals.dev) (LME spot prices)
- **FRED_API_KEY** — free at [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) (Fed Funds rate auto-update)

Both are optional — the dashboard falls back to manual config values without them.

### 3. Edit config.py

Adjust trading parameters to match your position:

| Setting | Description |
|---|---|
| `FIX_TARGET` | COMEX price target for fixing unpriced longs |
| `GTC_LEVELS` | Price ladder for GTC order suggestions |
| `TRUCKLOAD_LBS` | Pounds per truckload |
| `LME_MANUAL_USD_MT` | Manual LME fallback ($/MT) if no API key |
| `COMEX_WAREHOUSE_MT` | COMEX warehouse stocks (update from CME daily) |
| `FED_FUNDS_RATE` | Static fallback if FRED API key not set |

### 4. Run

```
python3 app.py
```

Dashboard serves at **http://localhost:8777**.

## Hedge Worksheet

The dashboard reads `Hedge*.xlsx` files from `./data/`. Files should follow the naming convention `HedgeMMDDYYYY.xlsx` and contain `POSITION`, `SOSOLIDS`, and `O SOLID SALE` sheets.

A sync script automatically copies the latest hedge file from OneDrive every 5 minutes (see launchd setup below).

## launchd (macOS)

Two launchd services keep the dashboard running and hedge data in sync. Install both by copying the plists and loading them:

```bash
cp com.geomet.copper-dashboard.plist ~/Library/LaunchAgents/
cp com.geomet.sync-hedge.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.geomet.copper-dashboard.plist
launchctl load ~/Library/LaunchAgents/com.geomet.sync-hedge.plist
```

| Service | What it does |
|---|---|
| `com.geomet.copper-dashboard` | Runs `app.py` on port 8777. Auto-restarts on crash or reboot. |
| `com.geomet.sync-hedge` | Copies the latest `Hedge*.xlsx` from OneDrive to `./data/` every 5 minutes. Uses `osascript`/Finder to bypass macOS CloudStorage permissions. |

To check status:

```bash
launchctl list | grep geomet
```

To restart:

```bash
launchctl stop com.geomet.copper-dashboard
launchctl start com.geomet.copper-dashboard
```

Logs are written to `dashboard.log`, `dashboard_err.log`, and `sync_hedge.log` in the project directory.
