# Geomet Dashboard Configuration
# Edit these values as needed
# API keys are read from .env file (see .env.example)

import os

METALS_DEV_API_KEY = os.environ.get("METALS_DEV_API_KEY", "")

# Manual LME fallback ($/metric ton) — update daily if no API key
# Set to 0 to disable manual override
LME_MANUAL_USD_MT = 0

# Fix target — auto-adjusts if price moves >15% away
FIX_TARGET = 5.40

# GTC ladder — now auto-generated relative to current price
# These static levels are kept as fallback only
GTC_LEVELS = [5.40, 5.45, 5.50, 5.55]

# Truckload size in lbs
TRUCKLOAD_LBS = 42000

# Baseline position — your target long exposure in lbs (positive = long)
BASELINE_LBS = 300000

# Attention thresholds ($/lb daily move)
ATTENTION_MOVE = 0.10   # 10c gets your attention
BIG_MOVE = 0.20         # 20c is significant



# Monthly copper flow by grade (lbs/month) — from PBI 7/1/25-2/23/26
# Used to calculate "months of sales remaining" per grade
MONTHLY_FLOW = {
    "Chops": 171800,
    "BB": 162700,
    "#2": 106600,
    "#1": 82700,
}

# COMEX warehouse stocks (metric tons) — update from CME daily report
# https://www.cmegroup.com/delivery_reports/MetalsIssueAndStopsYTDReport.pdf
COMEX_WAREHOUSE_MT = 534000
COMEX_WAREHOUSE_DATE = "2026-02-14"
COMEX_WAREHOUSE_TREND = "building"   # "building" or "drawing"

# Customer fix-pricing availability windows (local time, 24h format)
# "start"/"end" in your local hours; overnight windows wrap (e.g. 19-10 = 7PM to 10AM)
# "basis" is the default pricing basis for that customer
CUSTOMER_HOURS = {
    "OM Commodities": {"start": 7, "end": 13, "basis": "LME"},
    "Citic":          {"start": 19, "end": 10, "basis": "LME"},   # Singapore — overnight
    "Sims":           {"start": 7, "end": 16, "basis": "COMEX"},
}

# Market rates by grade — % of LME that buyers are paying TODAY
# Update when the market moves. Dashboard nudges you when LME drifts.
# "basis": "3m" = % of LME 3-month, "cash" = % of LME cash settlement
MARKET_RATES = {
    "BB":    {"pct": 0.972, "basis": "3m"},      # 97.2% of 3M LME
    "#1":    {"pct": 0.947, "basis": "3m"},      # 94.7% of 3M LME
    "#2":    {"pct": 0.923, "basis": "3m"},      # 92.3% of 3M LME
    "Chops": {"pct": 0.935, "basis": "3m"},      # 93.5% of 3M LME
}

# Insulated wire grades — derived sale rate = Chops rate × recovery %
# These are not directly sold; shown as sub-chops references for what
# the wire would fetch at the chops-equivalent rate per lb of gross wire.
ICW_RECOVERY = {
    "MCM":    0.88,   # 88% Cu by weight
    "THHN":   0.78,   # 78% Cu
    "Romex":  0.65,   # 65% Cu (CUINS1 in ROM)
    "#2 Ins": 0.42,   # 42% Cu (CUINS2 in ROM)
}
MARKET_RATES_LME_AT_UPDATE = 0      # LME $/lb when you last set rates (0 = skip stale check)
MARKET_RATES_DATE = "2026-04-08"    # date you last updated
MARKET_RATES_STALE_THRESHOLD = 0.05 # nudge if LME moves > $0.05/lb (~$110/MT) from update price

# Custom price levels — pin levels from your broker, Bloomberg, or experience
# These get the same crossing/proximity alerts and visual treatment as auto S/R
CUSTOM_LEVELS = [
    {"price": 5.34, "label": "200 DMA (Bloomberg)"},
    {"price": 5.05, "label": "Major floor"},
]

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Current Fed Funds target rate (used as fallback when FRED_API_KEY is empty)
FED_FUNDS_RATE = "4.25-4.50"
FED_FUNDS_MIDPOINT = 4.375
