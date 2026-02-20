# Geomet Dashboard Configuration
# Edit these values as needed

# metals.dev API key (free at https://metals.dev — sign up, no credit card)
# Once you have it, paste it here and restart the app
METALS_DEV_API_KEY = "REDACTED"

# Manual LME fallback ($/metric ton) — update daily if no API key
# Set to 0 to disable manual override
LME_MANUAL_USD_MT = 12850

# Fix target — COMEX price where you want to start pricing unpriced longs
FIX_TARGET = 5.90

# GTC ladder levels
GTC_LEVELS = [5.90, 5.95, 6.00, 6.05]

# Truckload size in lbs
TRUCKLOAD_LBS = 42000

# Attention thresholds ($/lb daily move)
ATTENTION_MOVE = 0.10   # 10c gets your attention
BIG_MOVE = 0.20         # 20c is significant



# COMEX warehouse stocks (metric tons) — update from CME daily report
# https://www.cmegroup.com/delivery_reports/MetalsIssueAndStopsYTDReport.pdf
COMEX_WAREHOUSE_MT = 534000
COMEX_WAREHOUSE_DATE = "2026-02-14"
COMEX_WAREHOUSE_TREND = "building"   # "building" or "drawing"

# FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
# Used to auto-fetch the current Fed Funds target rate range
# Leave empty to use the static values below
FRED_API_KEY = ""

# Current Fed Funds target rate (used as fallback when FRED_API_KEY is empty)
FED_FUNDS_RATE = "4.25-4.50"
FED_FUNDS_MIDPOINT = 4.375
