#!/usr/bin/env python3
"""
ROM Discovery Script for Copper Dashboard
Explores RecyclingDB on PLATINUM to find copper-related InventoryIDs,
order structures, and inventory balances.

ROM schema notes (from TAA/aluminum dashboard):
  - Inventory.InventoryID is INT, ItemName/ShortName are the text names
  - TruckScaleDTL uses ShippedAsID/ActualID (int FK to Inventory), weight = Gross - Tare
  - TruckScaleHDR: ShipReceive=0 is inbound (purchase), ShipReceive=1 is outbound (sale)
  - OrderHeader: Closed=0 is open, DealerID links to Dealers table
  - OrderDetails: InvID (not InventoryID), QtyOrdered, UnitsShipped, Price

Usage:
    python3 scripts/explore_rom.py
"""

import os
import sys
from pathlib import Path

# Load .env.rom
env_rom = Path(__file__).parent.parent / ".env.rom"
if env_rom.exists():
    with open(env_rom) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

import pymssql

SERVER = os.environ.get("ROM_SQL_SERVER", "PLATINUM")
DATABASE = os.environ.get("ROM_SQL_DATABASE", "RecyclingDB")
USER = os.environ.get("ROM_SQL_USER", "")
PASSWORD = os.environ.get("ROM_SQL_PASSWORD", "")


def connect():
    return pymssql.connect(
        server=SERVER, database=DATABASE, user=USER, password=PASSWORD,
        login_timeout=10, timeout=30, as_dict=True,
    )


def run(conn, sql, label=""):
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
    try:
        with conn.cursor(as_dict=True) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            if not rows:
                print("  (no rows)")
                return rows
            cols = list(rows[0].keys())
            print(f"  Columns: {cols}")
            print(f"  Row count: {len(rows)}")
            for i, row in enumerate(rows[:30]):
                vals = {k: row[k] for k in cols}
                print(f"  [{i}] {vals}")
            if len(rows) > 30:
                print(f"  ... ({len(rows) - 30} more rows)")
            return rows
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def main():
    if not USER or not PASSWORD:
        print("ERROR: ROM credentials not set. Check .env.rom file.")
        sys.exit(1)

    print(f"Connecting to {SERVER}/{DATABASE} as {USER}...")
    conn = connect()
    print("Connected!\n")

    # 1. All copper inventory items (InventoryID is int, names in ItemName/ShortName)
    run(conn,
        "SELECT InventoryID, ItemName, ShortName, InvGroupID, Active "
        "FROM Inventory "
        "WHERE ItemName LIKE '%copper%' OR ItemName LIKE '%bright%' "
        "OR ItemName LIKE '%chop%' OR ItemName LIKE '%insul%' "
        "OR ShortName LIKE '%CU%' OR ShortName LIKE '%copper%' "
        "OR ItemName LIKE '%CU %' OR ItemName LIKE 'CU%' "
        "OR ShortName LIKE 'CU%' "
        "ORDER BY ItemName",
        "COPPER INVENTORY ITEMS (by ItemName/ShortName)")

    # 2. Also search by InvGroupID — find what group copper items belong to
    run(conn,
        "SELECT DISTINCT ig.InvGroupID, ig.GroupName "
        "FROM InvGroup ig "
        "JOIN Inventory i ON ig.InvGroupID = i.InvGroupID AND ig.CompanyID = i.CompanyID "
        "WHERE i.ItemName LIKE '%copper%' OR i.ShortName LIKE 'CU%' "
        "ORDER BY ig.GroupName",
        "INVENTORY GROUPS CONTAINING COPPER")

    # 3. All items in copper groups (once we know the group IDs)
    copper_groups = run(conn,
        "SELECT DISTINCT i.InvGroupID "
        "FROM Inventory i "
        "WHERE i.ItemName LIKE '%copper%' OR i.ShortName LIKE 'CU%'",
        "COPPER GROUP IDS")
    if copper_groups:
        group_ids = ",".join(str(r["InvGroupID"]) for r in copper_groups if r.get("InvGroupID"))
        if group_ids:
            run(conn,
                f"SELECT InventoryID, ItemName, ShortName, InvGroupID, Active "
                f"FROM Inventory WHERE InvGroupID IN ({group_ids}) "
                f"ORDER BY InvGroupID, ItemName",
                f"ALL ITEMS IN COPPER GROUPS ({group_ids})")

    # 4. Active copper items — cross-ref with recent TruckScaleDTL (last 90 days)
    #    TruckScaleDTL uses ShippedAsID (int FK) and weight = Gross - Tare
    run(conn,
        "SELECT d.ShippedAsID, i.ItemName, i.ShortName, "
        "COUNT(*) AS TxnCount, "
        "SUM(CASE WHEN h.ShipReceive=0 THEN (d.Gross - d.Tare) ELSE 0 END) AS InWeight, "
        "SUM(CASE WHEN h.ShipReceive=1 THEN (d.Gross - d.Tare) ELSE 0 END) AS OutWeight "
        "FROM TruckScaleHDR h "
        "JOIN TruckScaleDTL d ON h.CompanyID = d.CompanyID AND h.TrackingID = d.TrackingID "
        "JOIN Inventory i ON d.ShippedAsID = i.InventoryID AND d.CompanyID = i.CompanyID "
        "WHERE h.Void = 0 AND h.Complete = 1 "
        "AND (i.ItemName LIKE '%copper%' OR i.ShortName LIKE 'CU%' "
        "     OR i.ItemName LIKE '%bright%' OR i.ItemName LIKE '%chop%' "
        "     OR i.ItemName LIKE '%insul%' OR i.ItemName LIKE '%wire%') "
        "AND h.StartedAt >= DATEADD(day, -90, GETDATE()) "
        "GROUP BY d.ShippedAsID, i.ItemName, i.ShortName "
        "ORDER BY TxnCount DESC",
        "ACTIVE COPPER ITEMS - Last 90 Days (TruckScale)")

    # 5. Also check PublicPurchActivity for copper items
    run(conn,
        "SELECT p.InvID, i.ItemName, i.ShortName, "
        "COUNT(*) AS TxnCount, SUM(p.Net) AS TotalWeight "
        "FROM PublicPurchActivity p "
        "JOIN Inventory i ON p.InvID = i.InventoryID AND p.CompanyID = i.CompanyID "
        "WHERE p.Complete = 1 "
        "AND (i.ItemName LIKE '%copper%' OR i.ShortName LIKE 'CU%' "
        "     OR i.ItemName LIKE '%bright%' OR i.ItemName LIKE '%chop%' "
        "     OR i.ItemName LIKE '%insul%' OR i.ItemName LIKE '%wire%') "
        "AND p.PurchDate >= DATEADD(day, -90, GETDATE()) "
        "GROUP BY p.InvID, i.ItemName, i.ShortName "
        "ORDER BY TxnCount DESC",
        "ACTIVE COPPER ITEMS - Last 90 Days (Public Purchases)")

    # 6. Net on-hand inventory (all-time inbound - outbound from TruckScale)
    #    ShipReceive=0 is inbound, ShipReceive=1 is outbound
    run(conn,
        "SELECT d.ShippedAsID, i.ItemName, i.ShortName, "
        "SUM(CASE WHEN h.ShipReceive=0 THEN (d.Gross - d.Tare) ELSE 0 END) AS TotalIn, "
        "SUM(CASE WHEN h.ShipReceive=1 THEN (d.Gross - d.Tare) ELSE 0 END) AS TotalOut, "
        "SUM(CASE WHEN h.ShipReceive=0 THEN (d.Gross - d.Tare) "
        "    ELSE -(d.Gross - d.Tare) END) AS NetBalance "
        "FROM TruckScaleHDR h "
        "JOIN TruckScaleDTL d ON h.CompanyID = d.CompanyID AND h.TrackingID = d.TrackingID "
        "JOIN Inventory i ON d.ShippedAsID = i.InventoryID AND d.CompanyID = i.CompanyID "
        "WHERE h.Void = 0 AND h.Complete = 1 "
        "AND (i.ItemName LIKE '%copper%' OR i.ShortName LIKE 'CU%' "
        "     OR i.ItemName LIKE '%bright%' OR i.ItemName LIKE '%chop%' "
        "     OR i.ItemName LIKE '%insul%' OR i.ItemName LIKE '%wire%') "
        "GROUP BY d.ShippedAsID, i.ItemName, i.ShortName "
        "HAVING ABS(SUM(CASE WHEN h.ShipReceive=0 THEN (d.Gross - d.Tare) "
        "    ELSE -(d.Gross - d.Tare) END)) > 100 "
        "ORDER BY i.ItemName",
        "NET COPPER ON-HAND (TruckScale In - Out, > 100 lbs)")

    # 7. Open copper sales orders
    run(conn,
        "SELECT TOP 15 oh.OrderID, oh.OrderType, oh.Closed, oh.OrderDate, "
        "oh.CustomerPONum, oh.ExternalOrderNum, oh.OrderNotes, "
        "dl.CompanyName AS CustomerName, "
        "od.InvID, i.ItemName, od.QtyOrdered, od.UnitsShipped, od.Price, od.ItemText "
        "FROM OrderHeader oh "
        "JOIN OrderDetails od ON oh.CompanyID = od.CompanyID AND oh.OrderID = od.OrderID "
        "JOIN Inventory i ON od.InvID = i.InventoryID AND od.CompanyID = i.CompanyID "
        "LEFT JOIN Dealers dl ON oh.DealerID = dl.DealerID AND oh.CompanyID = dl.CompanyID "
        "WHERE oh.Closed = 0 AND oh.Void = 0 "
        "AND oh.OrderType = 2 "  # 2 = sales order (smallint, not 'S')
        "AND (i.ItemName LIKE '%copper%' OR i.ShortName LIKE 'CU%' "
        "     OR i.ItemName LIKE '%bright%' OR i.ItemName LIKE '%chop%') "
        "ORDER BY oh.OrderID DESC",
        "OPEN COPPER SALES ORDERS (Top 15)")

    # 7b. Try OrderType=1 in case sales is 1
    run(conn,
        "SELECT TOP 15 oh.OrderID, oh.OrderType, oh.Closed, oh.OrderDate, "
        "dl.CompanyName AS CustomerName, "
        "od.InvID, i.ItemName, od.QtyOrdered, od.UnitsShipped, od.Price "
        "FROM OrderHeader oh "
        "JOIN OrderDetails od ON oh.CompanyID = od.CompanyID AND oh.OrderID = od.OrderID "
        "JOIN Inventory i ON od.InvID = i.InventoryID AND od.CompanyID = i.CompanyID "
        "LEFT JOIN Dealers dl ON oh.DealerID = dl.DealerID AND oh.CompanyID = dl.CompanyID "
        "WHERE oh.Closed = 0 AND oh.Void = 0 "
        "AND oh.OrderType = 1 "
        "AND (i.ItemName LIKE '%copper%' OR i.ShortName LIKE 'CU%' "
        "     OR i.ItemName LIKE '%bright%' OR i.ItemName LIKE '%chop%') "
        "ORDER BY oh.OrderID DESC",
        "OPEN COPPER ORDERS TYPE=1 (Top 15)")

    # 8. Check what OrderType values exist
    run(conn,
        "SELECT oh.OrderType, COUNT(*) AS Cnt "
        "FROM OrderHeader oh WHERE oh.Closed = 0 AND oh.Void = 0 "
        "GROUP BY oh.OrderType ORDER BY oh.OrderType",
        "ORDER TYPES (open orders)")

    # 9. Shipped weights for open orders
    run(conn,
        "SELECT TOP 10 sw.OrderID, sw.OrderType, sw.OrderDetailID, sw.ShippedWT, sw.UM "
        "FROM OrderShipWTTbl sw "
        "JOIN OrderDetails od ON sw.CompanyID = od.CompanyID AND sw.OrderID = od.OrderID "
        "    AND sw.OrderDetailID = od.OrderDTLID "
        "JOIN Inventory i ON od.InvID = i.InventoryID AND od.CompanyID = i.CompanyID "
        "WHERE (i.ItemName LIKE '%copper%' OR i.ShortName LIKE 'CU%') "
        "ORDER BY sw.OrderID DESC",
        "SHIPPED WEIGHTS (OrderShipWTTbl) for Copper")

    # 10. Check Adjustments table for inventory adjustments
    run(conn,
        "SELECT COLUMN_NAME, DATA_TYPE "
        "FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='Adjustments' "
        "ORDER BY ORDINAL_POSITION",
        "ADJUSTMENTS TABLE SCHEMA")

    # 11. Check for any inventory balance views
    run(conn,
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS "
        "WHERE TABLE_NAME LIKE '%Inv%' OR TABLE_NAME LIKE '%Balance%' "
        "OR TABLE_NAME LIKE '%Stock%' OR TABLE_NAME LIKE '%OnHand%' "
        "ORDER BY TABLE_NAME",
        "INVENTORY-RELATED VIEWS")

    # 12. Recent purchase prices for cost calculation
    run(conn,
        "SELECT TOP 20 d.ShippedAsID, i.ItemName, "
        "CONVERT(varchar, h.StartedAt, 23) AS PurchDate, "
        "(d.Gross - d.Tare) AS NetWeight, "
        "d.LBSTransToInv "
        "FROM TruckScaleHDR h "
        "JOIN TruckScaleDTL d ON h.CompanyID = d.CompanyID AND h.TrackingID = d.TrackingID "
        "JOIN Inventory i ON d.ShippedAsID = i.InventoryID AND d.CompanyID = i.CompanyID "
        "WHERE h.ShipReceive = 0 AND h.Void = 0 AND h.Complete = 1 "
        "AND (i.ShortName LIKE 'CU%') "
        "ORDER BY h.StartedAt DESC",
        "RECENT COPPER PURCHASES (for cost data)")

    # 13. Check if there's pricing/cost info in TruckScaleHDR or related tables
    run(conn,
        "SELECT COLUMN_NAME, DATA_TYPE "
        "FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_NAME = 'TruckScaleHDR' "
        "AND (COLUMN_NAME LIKE '%Price%' OR COLUMN_NAME LIKE '%Cost%' "
        "     OR COLUMN_NAME LIKE '%Amount%' OR COLUMN_NAME LIKE '%Total%') "
        "ORDER BY ORDINAL_POSITION",
        "PRICE/COST COLUMNS IN TruckScaleHDR")

    # 14. Check DealerPurchActivity for purchase prices
    run(conn,
        "SELECT COLUMN_NAME, DATA_TYPE "
        "FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='DealerPurchActivity' "
        "ORDER BY ORDINAL_POSITION",
        "DEALERPURCHACTIVITY SCHEMA")

    run(conn,
        "SELECT TOP 10 TrackingID, InvID, Net, PricePerLB, TotalAmount, "
        "CONVERT(varchar, PurchDate, 23) AS PurchDate "
        "FROM DealerPurchActivity "
        "WHERE InvID IN (SELECT InventoryID FROM Inventory WHERE ShortName LIKE 'CU%') "
        "ORDER BY PurchDate DESC",
        "RECENT COPPER DEALER PURCHASES WITH PRICES")

    # 15. Public purchase prices
    run(conn,
        "SELECT TOP 10 InvID, ItemName, Net, PricePerLB, TotalPrice, "
        "CONVERT(varchar, PurchDate, 23) AS PurchDate "
        "FROM PublicPurchActivity "
        "WHERE InvID IN (SELECT InventoryID FROM Inventory WHERE ShortName LIKE 'CU%') "
        "AND Complete = 1 "
        "ORDER BY PurchDate DESC",
        "RECENT COPPER PUBLIC PURCHASES WITH PRICES")

    conn.close()
    print("\n\nDone! Use the InventoryIDs above to populate COPPER_INV_IDS in rom.py")


if __name__ == "__main__":
    main()
