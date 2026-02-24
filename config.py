# Geomet Dashboard Configuration
# Edit these values as needed
# API keys are read from .env file (see .env.example)

import os

METALS_DEV_API_KEY = os.environ.get("METALS_DEV_API_KEY", "")

# Manual LME fallback ($/metric ton) — update daily if no API key
# Set to 0 to disable manual override
LME_MANUAL_USD_MT = 12850

# Fix target — COMEX price where you want to start pricing unpriced longs
FIX_TARGET = 6.00

# GTC ladder levels
GTC_LEVELS = [6.00, 6.05, 6.10, 6.15]

# Truckload size in lbs
TRUCKLOAD_LBS = 42000

# Baseline position — your target long exposure in lbs (positive = long)
BASELINE_LBS = 200000

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

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Current Fed Funds target rate (used as fallback when FRED_API_KEY is empty)
FED_FUNDS_RATE = "4.25-4.50"
FED_FUNDS_MIDPOINT = 4.375
