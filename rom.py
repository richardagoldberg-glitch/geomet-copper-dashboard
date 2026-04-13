"""
ROM (RecyclingDB on PLATINUM SQL Server) integration for copper dashboard.

Hybrid approach:
  - Sales orders & POs: LIVE from ROM (real-time, no spreadsheet lag)
  - Inventory by grade: from spreadsheet (ROM has no reliable balance table)
  - Futures positions: from spreadsheet (broker-managed, not in ROM)
  - Average cost: from spreadsheet (hardcoded fallback)

ROM schema notes:
  - Inventory.InventoryID is INT; ItemName/ShortName are text
  - TruckScaleDTL: ShippedAsID (int FK), weight = Gross - Tare
  - TruckScaleHDR: ShipReceive=0 inbound, ShipReceive=1 outbound
  - OrderHeader: OrderType=0 purchase, OrderType=1 sale
  - Open orders: ClosedDate IS NULL AND Void = 0
  - OrderDetails: InventoryID (int), UnitsOrdered, UnitsShipped, Price
  - OrderShipWTTbl: OrderDetailID, ShippedWT
  - Dealers table: DealerID → CompanyName
"""

import os
import time
import threading
import pyodbc
from datetime import datetime

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
# We use pyodbc + Microsoft ODBC Driver 18 for SQL Server (installed via brew).
# This replaces pymssql/FreeTDS, which had a SIGABRT bug in tds_dataout_stream_write
# on macOS that crashed the entire Python process and triggered popups.
_conn = None
_conn_lock = threading.Lock()

CONNECT_TIMEOUT = 10  # seconds
QUERY_TIMEOUT = 30    # seconds

ODBC_DRIVER = "ODBC Driver 18 for SQL Server"


def _build_conn_str():
    server = os.environ.get("ROM_SQL_SERVER", "PLATINUM")
    database = os.environ.get("ROM_SQL_DATABASE", "RecyclingDB")
    user = os.environ.get("ROM_SQL_USER", "")
    password = os.environ.get("ROM_SQL_PASSWORD", "")
    if not user or not password:
        raise RuntimeError("ROM credentials not configured (ROM_SQL_USER / ROM_SQL_PASSWORD)")
    return (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={server};DATABASE={database};"
        f"UID={user};PWD={password};"
        "Encrypt=no;TrustServerCertificate=yes;"
    )


def _get_connection():
    """Get or create a pyodbc connection with auto-reconnect."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.cursor().execute("SELECT 1").fetchone()
                return _conn
            except Exception:
                try:
                    _conn.close()
                except Exception:
                    pass
                _conn = None

        _conn = pyodbc.connect(_build_conn_str(), timeout=CONNECT_TIMEOUT)
        _conn.timeout = QUERY_TIMEOUT
        return _conn


def query_rom(sql, params=None):
    """Execute a SQL query and return list of dicts. Thread-safe with auto-reconnect.
    pyodbc returns Row tuples; we convert to dicts using cursor.description."""
    global _conn
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        cols = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        cursor.close()
        return rows
    except Exception:
        # Drop the cached connection so the next call gets a fresh one.
        with _conn_lock:
            try:
                if _conn is not None:
                    _conn.close()
            except Exception:
                pass
            _conn = None
        raise


# ---------------------------------------------------------------------------
# Copper InventoryID mapping (ROM Inventory.InventoryID = int)
# Populated from scripts/explore_rom.py discovery
# ---------------------------------------------------------------------------
COPPER_INV_IDS = {
    "BB":    [1001],                    # Bare Bright Copper (CUBB)
    "#1":    [1003],                    # #1 Copper (CU1)
    "#2":    [1004, 1436],              # #2 Copper (CU2), CU2 Dirty (CU2DIRTY)
    "Chops": [1445, 1447, 1534],        # BB Chops (CUCHOP CUBB), #1A Medium (CUCHOP1A_M), #2 Chops (CUCHOPS2)
    "ICW":   [1141, 1010, 1216, 1384,   # CUINS1, CUINS2, CUINS1LITE, CUINS2HIGH
              1011, 1012, 1014, 1169,    # CUINSLOW, CUINSXMAS, HARNESS, JELLYWIRE
              1361,                      # Extra Low Grade Wire (<25%)
              1007, 1008, 1170,          # MCM, THHN, WAVEOPENCU
              1842, 1843],              # CUINS1FEED, CUINS2FEED (pre-chopped)
}

# ShortName lookup for item display (InventoryID → ShortName)
_INV_SHORTNAME = {
    1001: "CUBB", 1003: "CU1", 1004: "CU2", 1436: "CU2DIRTY",
    1445: "CUCHOP CUBB", 1447: "CUCHOP1A_M", 1534: "CUCHOPS2",
    1141: "CUINS1", 1010: "CUINS2", 1216: "CUINS1LITE", 1384: "CUINS2HIGH",
    1011: "CUINSLOW", 1012: "CUINSXMAS", 1014: "HARNESS",
    1169: "JELLYWIRE", 1361: "CUINS<25%",
    1007: "MCM", 1008: "THHN", 1170: "WAVEOPENCU",
    1842: "CUINS1FEED", 1843: "CUINS2FEED",
}

# Copper recovery rate per InventoryID (what % of as-is weight is copper)
# Pure copper grades = 1.0, insulated wire = fraction
# User-provided rates for main grades; estimates for low-volume items
CU_RECOVERY_PCT = {
    # Pure copper — 100%
    1001: 1.00,   # CUBB
    1003: 1.00,   # CU1
    1004: 1.00,   # CU2
    1436: 1.00,   # CU2DIRTY
    1445: 1.00,   # CUCHOP CUBB
    1447: 1.00,   # CUCHOP1A_M
    1534: 1.00,   # CUCHOPS2
    # Insulated — user-provided
    1007: 0.88,   # MCM
    1008: 0.78,   # THHN
    1141: 0.65,   # CUINS1
    1842: 0.65,   # CUINS1FEED (same as CUINS1, pre-chopped)
    1010: 0.42,   # CUINS2
    1843: 0.42,   # CUINS2FEED (same as CUINS2, pre-chopped)
    1384: 0.65,   # CUINS2HIGH
    1170: 0.55,   # WAVEOPENCU
    # Insulated — estimates (low volume)
    1216: 0.55,   # CUINS1LITE
    1011: 0.28,   # CUINSLOW
    1012: 0.12,   # CUINSXMAS
    1014: 0.45,   # HARNESS
    1169: 0.38,   # JELLYWIRE
    1361: 0.18,   # CUINS<25%
}


def _all_cu_ids():
    """All copper InventoryIDs flattened."""
    ids = []
    for grade_ids in COPPER_INV_IDS.values():
        ids.extend(grade_ids)
    return ids


def _cu_ids_csv():
    """Comma-separated list of copper InventoryIDs for inline SQL.
    Safe to inline because all values are hardcoded ints (no user input).
    Avoids pymssql/freetds parameterized-query crash on macOS.
    """
    return ",".join(str(i) for i in _all_cu_ids())


def _grade_for_inv_id(inv_id):
    """Return dashboard grade name for a ROM InventoryID (int)."""
    for grade, ids in COPPER_INV_IDS.items():
        if inv_id in ids:
            return grade
    return None


# ---------------------------------------------------------------------------
# Per-grade inventory cost from ROM
# ---------------------------------------------------------------------------
_inv_cost_cache = {"data": None, "timestamp": 0, "error_until": 0}
INV_COST_CACHE_TTL = 600       # 10 minutes — costs change slowly
INV_COST_ERROR_BACKOFF = 600   # 10 minutes after a failure

def _fetch_inv_avg_costs():
    """
    Query ROM InventoryEvalValues2 for AvgPurchPrice and InvPosition2 for
    actual inventory balances per copper item.
    Returns dict with:
      - per-grade cost-per-copper-lb (inventory-weighted for ICW)
      - "icw_detail": list of per-item ICW data for frontend display
    """
    now = time.time()
    if _inv_cost_cache["data"] is not None and (now - _inv_cost_cache["timestamp"]) < INV_COST_CACHE_TTL:
        return _inv_cost_cache["data"]
    if _inv_cost_cache["error_until"] > now and _inv_cost_cache["data"] is not None:
        return _inv_cost_cache["data"]

    all_ids = _all_cu_ids()
    if not all_ids:
        return _inv_cost_cache["data"] or {}

    cu_csv = _cu_ids_csv()
    try:
        cost_rows = query_rom(
            f"SELECT v.InventoryID, v.AvgPurchPrice "
            f"FROM InventoryEvalValues2 v "
            f"WHERE v.InventoryID IN ({cu_csv})"
        )
        pos_rows = query_rom(
            f"SELECT InventoryID, ShortName, TotalLbs "
            f"FROM InvPosition2 "
            f"WHERE InventoryID IN ({cu_csv})"
        )
    except Exception as e:
        print(f"[WARN] ROM inventory cost query failed: {e}")
        _inv_cost_cache["error_until"] = now + INV_COST_ERROR_BACKOFF
        return _inv_cost_cache["data"] or {}

    # Index position by InventoryID
    pos_by_id = {}
    for pr in pos_rows:
        pos_by_id[pr["InventoryID"]] = {
            "name": (pr.get("ShortName") or "").strip(),
            "lbs": float(pr.get("TotalLbs") or 0),
        }

    # Build per-grade cost: inventory-weighted for multi-item grades
    grade_items = {}  # grade -> list of {inv_id, cost_cu, lbs, recovery, avg_purch, name}
    for row in cost_rows:
        inv_id = row.get("InventoryID")
        avg_purch = float(row.get("AvgPurchPrice") or 0)
        if avg_purch <= 0:
            continue
        grade = _grade_for_inv_id(inv_id)
        if not grade:
            continue
        recovery = CU_RECOVERY_PCT.get(inv_id, 1.0)
        cost_per_cu_lb = avg_purch / recovery if recovery > 0 else 0
        if cost_per_cu_lb <= 0:
            continue
        p = pos_by_id.get(inv_id, {"name": f"ID:{inv_id}", "lbs": 0})
        grade_items.setdefault(grade, []).append({
            "inv_id": inv_id, "name": p["name"], "lbs": p["lbs"],
            "avg_purch": round(avg_purch, 4), "recovery": recovery,
            "cost_cu": round(cost_per_cu_lb, 4),
        })

    result = {}
    for grade, items in grade_items.items():
        # Weight by Cu lbs (gross lbs × recovery)
        total_cu = sum(it["lbs"] * it["recovery"] for it in items)
        total_val = sum(it["lbs"] * it["avg_purch"] for it in items)
        if total_cu > 0:
            result[grade] = round(total_val / total_cu, 4)
        else:
            # No inventory — fall back to simple average of costs
            result[grade] = round(sum(it["cost_cu"] for it in items) / len(items), 4)

    # Chops = ICW (chops are produced from insulated wire)
    if "ICW" in result:
        result["Chops"] = result["ICW"]

    # Attach ICW detail for frontend sub-rows
    icw_items = grade_items.get("ICW", [])
    result["icw_detail"] = sorted(
        [it for it in icw_items if it["lbs"] > 0],
        key=lambda x: x["lbs"], reverse=True,
    )

    _inv_cost_cache["data"] = result
    _inv_cost_cache["timestamp"] = now
    _inv_cost_cache["error_until"] = 0
    return result


def calc_blended_avg_cost(grade_costs, inv_by_commodity, icw_cu_lbs=0):
    """
    Compute blended avg cost per copper lb using ROM per-grade costs
    and spreadsheet inventory weights.

    grade_costs: {"BB": 5.41, "#1": 5.34, "#2": 5.12, "Chops": 5.41, "ICW": 5.20}
    inv_by_commodity: {"BB": 39000, "#1": 38000, "#2": 14000, "Chops": 7000}
    icw_cu_lbs: copper content lbs in insulated wire inventory
    """
    total_value = 0.0
    total_lbs = 0.0

    for grade in ["BB", "#1", "#2", "Chops"]:
        lbs = inv_by_commodity.get(grade, 0)
        cost = grade_costs.get(grade, 0)
        if lbs > 0 and cost > 0:
            total_value += lbs * cost
            total_lbs += lbs

    # ICW: use copper-content lbs (grossed up) with ICW cost per copper lb
    if icw_cu_lbs > 0 and grade_costs.get("ICW", 0) > 0:
        total_value += icw_cu_lbs * grade_costs["ICW"]
        total_lbs += icw_cu_lbs

    if total_lbs <= 0:
        return 0
    return round(total_value / total_lbs, 4)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_rom_cache = {"data": None, "timestamp": 0, "error_until": 0}
ROM_CACHE_TTL = 300        # 5 minutes
ROM_ERROR_BACKOFF = 300    # 5 minutes after failure


# ---------------------------------------------------------------------------
# Open Purchase Orders from ROM
# ---------------------------------------------------------------------------
def _fetch_open_pos():
    """
    Fetch open copper purchase orders (PO waiting to receive).
    ROM schema: OrderType=0 is purchase, ClosedDate IS NULL = open.
    """
    all_ids = _all_cu_ids()
    if not all_ids:
        return 0

    cu_csv = _cu_ids_csv()
    rows = query_rom(
        f"SELECT COALESCE(SUM(od.UnitsOrdered - ISNULL(od.UnitsShipped, 0)), 0) AS POWaiting "
        f"FROM OrderHeader oh "
        f"JOIN OrderDetails od ON oh.CompanyID = od.CompanyID AND oh.OrderID = od.OrderID "
        f"WHERE oh.OrderType = 0 AND oh.ClosedDate IS NULL AND oh.Void = 0 "
        f"AND od.InventoryID IN ({cu_csv})"
    )
    return float(rows[0]["POWaiting"]) if rows else 0


# ---------------------------------------------------------------------------
# Open Sales Orders from ROM
# ---------------------------------------------------------------------------
def _fetch_open_sales():
    """
    Fetch open copper sales orders with shipping status.
    ROM schema: OrderType=1 is sale, ClosedDate IS NULL = open.
    Returns sales lists and aggregate totals matching spreadsheet format.
    """
    all_ids = _all_cu_ids()
    if not all_ids:
        return _empty_sales()

    cu_csv = _cu_ids_csv()

    # Get open sales order details with customer name + pricing formula
    rows = query_rom(
        f"SELECT oh.OrderID, oh.OrderType, oh.ExternalOrderNum, oh.OrderNotes, "
        f"dl.CompanyName AS CustomerName, "
        f"od.InventoryID, od.UnitsOrdered, od.UnitsShipped, od.Price, "
        f"od.OrderDetailID, od.ItemText, "
        f"ovr.BasePriceType, ovr.FormulaFactor, ovr.FormulaAmount, "
        f"ovr.FormulaFactor2, ovr.FormulaAmount2, ovr.AgainstMarket "
        f"FROM OrderHeader oh "
        f"JOIN OrderDetails od ON oh.CompanyID = od.CompanyID AND oh.OrderID = od.OrderID "
        f"LEFT JOIN Dealers dl ON oh.CustomerID = dl.DealerID AND oh.CompanyID = dl.CompanyID "
        f"LEFT JOIN OrderOverRide ovr ON od.CompanyID = ovr.CompanyID "
        f"  AND od.OrderID = ovr.OrderID AND od.OrderDetailID = ovr.OrderDetailID "
        f"WHERE oh.OrderType = 1 AND oh.ClosedDate IS NULL AND oh.Void = 0 "
        f"AND od.InventoryID IN ({cu_csv}) "
        f"ORDER BY oh.OrderID"
    )

    # Get shipped weights from OrderShipWTTbl keyed by OrderDetailID
    shipped_by_dtl = {}
    try:
        ship_rows = query_rom(
            f"SELECT sw.OrderDetailID, SUM(sw.ShippedWT) AS Shipped "
            f"FROM OrderShipWTTbl sw "
            f"JOIN OrderDetails od ON sw.CompanyID = od.CompanyID "
            f"  AND sw.OrderID = od.OrderID AND sw.OrderDetailID = od.OrderDetailID "
            f"JOIN OrderHeader oh ON od.CompanyID = oh.CompanyID AND od.OrderID = oh.OrderID "
            f"WHERE oh.OrderType = 1 AND oh.ClosedDate IS NULL AND oh.Void = 0 "
            f"AND od.InventoryID IN ({cu_csv}) "
            f"GROUP BY sw.OrderDetailID"
        )
        for sr in ship_rows:
            shipped_by_dtl[sr["OrderDetailID"]] = float(sr.get("Shipped") or 0)
    except Exception:
        pass  # Shipping info is optional

    # Get actual ship dates from appointments keyed by OrderID
    actual_ship_by_order = {}
    try:
        appt_rows = query_rom(
            "SELECT ao.OrderID, ah.ActualShipDate "
            "FROM AppointmentOrder ao "
            "JOIN romAppointmentHDR ah ON ao.AppointmentID = ah.AppointmentID "
            "  AND ao.CompanyID = ah.CompanyID "
            "JOIN OrderHeader oh ON ao.CompanyID = oh.CompanyID AND ao.OrderID = oh.OrderID "
            "WHERE oh.OrderType = 1 AND oh.ClosedDate IS NULL AND oh.Void = 0 "
            "AND ah.ActualShipDate IS NOT NULL"
        )
        for ar in appt_rows:
            actual_ship_by_order[ar["OrderID"]] = ar["ActualShipDate"]
    except Exception:
        pass  # Appointment info is optional

    # Get scheduled future ship dates from TransAppointments (open OUT shipments).
    # Geomet doesn't fill romAppointmentHDR.ExpectedShipDate, but TransAppointments
    # exposes the appointment leg start date for open (not yet shipped) sales legs.
    expected_ship_by_so = {}      # so_num -> {"ship_date","delivery_date","customer_ref"}
    try:
        sched_rows = query_rom(
            f"SELECT SONum, FirstFromStartDate, FirstToStartDate, "
            f"WeightLB, CustRef1, ShortName, InventoryID "
            f"FROM TransAppointments "
            f"WHERE void = 0 AND InOut = 1 AND OpenCompleted = 'Open' "
            f"AND SONum IS NOT NULL "
            f"AND InventoryID IN ({cu_csv}) "
            f"ORDER BY FirstFromStartDate"
        )
        for sr in sched_rows:
            so_num = sr.get("SONum")
            if not so_num:
                continue
            ship_dt = sr.get("FirstFromStartDate")
            if not ship_dt:
                continue
            so_key = str(so_num)
            # Keep earliest scheduled ship date per SO if multiple legs
            existing = expected_ship_by_so.get(so_key)
            if existing and existing.get("_dt") and ship_dt >= existing["_dt"]:
                continue
            deliv_dt = sr.get("FirstToStartDate")
            expected_ship_by_so[so_key] = {
                "_dt": ship_dt,
                "ship_date": ship_dt.strftime("%Y-%m-%d"),
                "delivery_date": deliv_dt.strftime("%Y-%m-%d") if deliv_dt else "",
                "customer_ref": str(sr.get("CustRef1") or ""),
            }
    except Exception as e:
        print(f"[WARN] ROM scheduled ship date query failed: {e}")

    priced_total = 0
    priced_value = 0
    unpriced_total = 0
    sales_priced_unshipped = []
    sales_unpriced_shipped = []
    sales_unpriced_unshipped = []
    sales_by_commodity = {"BB": 0, "#1": 0, "#2": 0, "Chops": 0}

    for row in rows:
        order_id = str(row.get("OrderID", ""))
        customer = str(row.get("CustomerName") or "")
        inv_id = int(row.get("InventoryID") or 0)
        qty = float(row.get("UnitsOrdered") or 0)
        price = float(row.get("Price") or 0)
        units_shipped = float(row.get("UnitsShipped") or 0)
        dtl_id = row.get("OrderDetailID")
        po_ref = str(row.get("ExternalOrderNum") or "")
        shipped_wt = shipped_by_dtl.get(dtl_id, 0)
        item_text = str(row.get("ItemText") or "")

        if qty <= 0:
            continue

        # Map InventoryID to shortname for display
        shortname = _INV_SHORTNAME.get(inv_id, str(inv_id))

        # Determine basis from OrderOverRide formula, fallback to text parsing
        against_market = str(row.get("AgainstMarket") or "").upper()
        if "LME" in against_market:
            basis = "LME"
        elif "COMEX" in against_market or "CMX" in against_market:
            basis = "COMEX"
        else:
            basis = "COMEX"
            notes_upper = (po_ref + " " + str(row.get("OrderNotes") or "") + " " + item_text).upper()
            if "LME" in notes_upper:
                basis = "LME"

        grade = _grade_for_inv_id(inv_id)
        if grade and grade != "ICW":
            sales_by_commodity[grade] += qty
        elif grade == "ICW":
            sales_by_commodity["Chops"] += qty

        # Attach actual ship date if appointment exists
        actual_ship = actual_ship_by_order.get(row.get("OrderID"))
        actual_ship_str = ""
        if actual_ship:
            actual_ship_str = actual_ship.strftime("%Y-%m-%d") if hasattr(actual_ship, "strftime") else str(actual_ship)[:10]

        # Extract pricing formula from OrderOverRide
        formula_factor = int(row.get("FormulaFactor") or 0)
        formula_amount = float(row.get("FormulaAmount") or 0)
        formula_factor2 = int(row.get("FormulaFactor2") or 0)
        formula_amount2 = float(row.get("FormulaAmount2") or 0)
        against_mkt_raw = str(row.get("AgainstMarket") or "")

        # FormulaFactor: 4=multiply, 1=subtract, 0=none
        spread = 0
        if formula_factor == 4 and formula_amount > 0:
            spread = round(formula_amount, 4)  # e.g. 0.9425

        sale = {
            "order": order_id,
            "consumer": customer,
            "commodity": shortname,
            "lbs": qty,
            "basis": basis,
            "spread": spread,
            "poref": po_ref,
            "formula_factor": formula_factor,
            "formula_amount": round(formula_amount, 4) if formula_amount else 0,
            "formula_factor2": formula_factor2,
            "formula_amount2": round(formula_amount2, 4) if formula_amount2 else 0,
            "against_market": against_mkt_raw,
        }
        if actual_ship_str:
            sale["actual_ship_date"] = actual_ship_str

        # Attach scheduled future ship date if appointment exists
        sched = expected_ship_by_so.get(order_id)
        if sched:
            sale["expected_ship_date"] = sched["ship_date"]
            if sched.get("delivery_date"):
                sale["expected_delivery_date"] = sched["delivery_date"]

        if price > 0:
            priced_total += qty
            priced_value += qty * price
            sale["price"] = round(price, 4)
            sale["status"] = "PRICED"
            sale["priced_lbs"] = qty
            if shipped_wt <= 0 and units_shipped <= 0:
                sales_priced_unshipped.append(sale)
        else:
            unpriced_total += qty
            sale["status"] = "UNPRICED"
            sale["open_lbs"] = qty
            if shipped_wt > 0 or units_shipped > 0:
                sale["shipped_lbs"] = shipped_wt or units_shipped
                sales_unpriced_shipped.append(sale)
            else:
                sales_unpriced_unshipped.append(sale)

    priced_avg = round(priced_value / priced_total, 4) if priced_total > 0 else 0

    # Build ship_schedule list (frontend shape) from sales that have scheduled
    # ship dates. Include past-due (open but not yet shipped) so they surface
    # as overdue on the dashboard instead of silently disappearing.
    ship_schedule = []
    seen_sos = set()
    all_sales = sales_priced_unshipped + sales_unpriced_shipped + sales_unpriced_unshipped
    for s in all_sales:
        sd = s.get("expected_ship_date")
        if not sd:
            continue
        so_key = s.get("order", "")
        if so_key in seen_sos:
            continue
        seen_sos.add(so_key)
        try:
            so_int = int(so_key)
        except (TypeError, ValueError):
            so_int = so_key
        ship_schedule.append({
            "so": so_int,
            "ship_date": sd,
            "delivery_date": s.get("expected_delivery_date", ""),
            "customer": s.get("consumer", ""),
            "grade": s.get("commodity", ""),
            "lbs": s.get("lbs", 0),
            "po": s.get("poref", ""),
        })
    ship_schedule.sort(key=lambda x: x["ship_date"])

    return {
        "priced_sales_lbs": priced_total,
        "priced_sales_avg": priced_avg,
        "unpriced_sales_lbs": unpriced_total,
        "sales_priced_unshipped": sales_priced_unshipped,
        "sales_unpriced_shipped": sales_unpriced_shipped,
        "sales_unpriced_unshipped": sales_unpriced_unshipped,
        "sales_priced_unshipped_lbs": sum(s.get("priced_lbs", 0) for s in sales_priced_unshipped),
        "sales_unpriced_shipped_lbs": sum(s.get("open_lbs", 0) for s in sales_unpriced_shipped),
        "sales_unpriced_unshipped_lbs": sum(s.get("open_lbs", 0) for s in sales_unpriced_unshipped),
        "sales_by_commodity": sales_by_commodity,
        "ship_schedule": ship_schedule,
    }


def _empty_sales():
    return {
        "priced_sales_lbs": 0, "priced_sales_avg": 0, "unpriced_sales_lbs": 0,
        "sales_priced_unshipped": [], "sales_unpriced_shipped": [],
        "sales_unpriced_unshipped": [],
        "sales_priced_unshipped_lbs": 0, "sales_unpriced_shipped_lbs": 0,
        "sales_unpriced_unshipped_lbs": 0,
        "sales_by_commodity": {"BB": 0, "#1": 0, "#2": 0, "Chops": 0},
        "ship_schedule": [],
    }


# ---------------------------------------------------------------------------
# Inventory + Futures + Cost from spreadsheet
# ROM has no reliable current-inventory balance table — items get reclassified
# during processing (wire → chops) so TruckScale all-time net is wrong.
# ---------------------------------------------------------------------------
def _read_spreadsheet_inventory_and_futures():
    """
    Read inventory, avg cost, and futures from the Hedge spreadsheet.
    Returns dict or None.
    """
    try:
        import app
        hf = app.find_latest_hedge_file()
        if not hf:
            return None

        # Use the existing full spreadsheet reader
        pos = app.read_hedge_spreadsheet(hf)
        if not pos:
            return None

        return {
            "inv_by_commodity": pos.get("inv_by_commodity", {"BB": 0, "#1": 0, "#2": 0, "Chops": 0}),
            "icw_cu_lbs": pos.get("icw_cu_lbs", 0),
            "chops_solid_lbs": pos.get("chops_solid_lbs", 0),
            "avg_cost": pos.get("avg_cost", 0),
            "inventory_cu_lbs": pos.get("inventory_cu_lbs", 0),
            "comex_futures": pos.get("comex_futures", 0),
            "lme_futures": pos.get("lme_futures", 0),
            "total_inv_po": pos.get("total_inv_po", 0),
            "po_lbs": pos.get("po_lbs", 0),
            "source_file": pos.get("source_file", ""),
            "updated": pos.get("updated", ""),
        }
    except Exception as e:
        print(f"[WARN] ROM: spreadsheet read failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main entry point: read_rom_position()
# Hybrid: ROM sales/POs + spreadsheet inventory/futures/cost
# Returns same dict shape as read_hedge_spreadsheet() in app.py
# ---------------------------------------------------------------------------
def read_rom_position():
    """
    Pull sales & POs from ROM, inventory/futures/cost from spreadsheet.
    Returns dict compatible with read_hedge_spreadsheet() or None on failure.
    Uses 5-minute cache with 5-minute error backoff.
    """
    now = time.time()

    # Return cached data if fresh
    if _rom_cache["data"] and (now - _rom_cache["timestamp"]) < ROM_CACHE_TTL:
        return _rom_cache["data"]

    # Back off after errors
    if _rom_cache["error_until"] > now:
        print(f"[INFO] ROM in backoff until {datetime.fromtimestamp(_rom_cache['error_until']).strftime('%H:%M:%S')}")
        return None

    try:
        # Spreadsheet: inventory, cost, futures
        ss = _read_spreadsheet_inventory_and_futures()
        if not ss:
            print("[WARN] ROM: no spreadsheet data for inventory/futures")
            return None

        # ROM: live sales orders (POs come from spreadsheet — ROM has stale unclosed POs)
        sales_data = _fetch_open_sales()

        # Use spreadsheet inventory + spreadsheet POs
        # (ROM POs include stale/unclosed orders from years past)
        inv_by_commodity = ss["inv_by_commodity"]
        inv_total = sum(inv_by_commodity.values())
        total_inv_po = ss.get("total_inv_po", inv_total)
        po_lbs = ss.get("po_lbs", 0)
        rom_po_lbs = po_lbs  # store for debug comparison

        comex_futures = ss["comex_futures"]
        lme_futures = ss["lme_futures"]
        ss_avg_cost = ss["avg_cost"]  # spreadsheet fallback

        # Compute proper avg cost from ROM per-grade purchase prices
        # grossed up for insulated wire by recovery rate, weighted by
        # spreadsheet inventory balances
        grade_costs = _fetch_inv_avg_costs()
        icw_cu_lbs = ss.get("icw_cu_lbs", 0)
        avg_cost = calc_blended_avg_cost(grade_costs, inv_by_commodity, icw_cu_lbs)
        if avg_cost <= 0:
            avg_cost = ss_avg_cost  # fall back to spreadsheet
            print(f"[INFO] ROM avg_cost: using spreadsheet fallback ${ss_avg_cost:.4f}")
        else:
            print(f"[INFO] ROM avg_cost: ${avg_cost:.4f}/lb (grades: {grade_costs})")

        # Net position = inventory + PO - sales + futures
        total_sales = sales_data["priced_sales_lbs"] + sales_data["unpriced_sales_lbs"]
        net_lbs = total_inv_po - total_sales + comex_futures + lme_futures

        result = {
            "net_lbs": net_lbs,
            "avg_cost": avg_cost,
            "hedge_lbs": abs(comex_futures) + abs(lme_futures),
            "comex_futures": comex_futures,
            "lme_futures": lme_futures,
            "priced_sales_lbs": sales_data["priced_sales_lbs"],
            "priced_sales_avg": sales_data["priced_sales_avg"],
            "unpriced_sales_lbs": sales_data["unpriced_sales_lbs"],
            "total_inv_po": total_inv_po,
            "inventory_cu_lbs": inv_total,
            "po_lbs": po_lbs,
            "updated": datetime.now().strftime("%Y-%m-%d"),
            "source_file": f"ROM + {ss.get('source_file', 'spreadsheet')}",
            "data_source": "ROM",
            "futures_source": ss.get("source_file", ""),
            "sales_priced_unshipped": sales_data["sales_priced_unshipped"],
            "sales_unpriced_shipped": sales_data["sales_unpriced_shipped"],
            "sales_unpriced_unshipped": sales_data["sales_unpriced_unshipped"],
            "sales_priced_unshipped_lbs": sales_data["sales_priced_unshipped_lbs"],
            "sales_unpriced_shipped_lbs": sales_data["sales_unpriced_shipped_lbs"],
            "sales_unpriced_unshipped_lbs": sales_data["sales_unpriced_unshipped_lbs"],
            "ship_schedule": sales_data.get("ship_schedule", []),
            "inv_by_commodity": inv_by_commodity,
            "sales_by_commodity": sales_data["sales_by_commodity"],
            "icw_cu_lbs": ss.get("icw_cu_lbs", 0),
            "chops_solid_lbs": ss.get("chops_solid_lbs", 0),
            "grade_costs": grade_costs,
        }

        _rom_cache["data"] = result
        _rom_cache["timestamp"] = now
        _rom_cache["error_until"] = 0
        print(f"[INFO] Position from ROM — inv={inv_total:,.0f} lbs, PO={po_lbs:,.0f}, "
              f"sales(ROM)={total_sales:,.0f}, net={net_lbs:,.0f}")
        return result

    except Exception as e:
        print(f"[ERROR] ROM query failed: {e}")
        import traceback; traceback.print_exc()
        _rom_cache["error_until"] = now + ROM_ERROR_BACKOFF
        return None
