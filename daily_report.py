#!/usr/bin/env python3
"""Geomet Daily Yard Report — queries ROM and posts to Teams via webhook."""

import os
import sys
import json
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict

# Load ROM env — manual parser since dotenv may not be installed
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.rom")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue
            if "=" in _line:
                k, v = _line.split("=", 1)
                v = v.strip().strip("'").strip('"')
                os.environ.setdefault(k.strip(), v)

sys.path.insert(0, os.path.dirname(__file__))
from rom import query_rom

WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL", "")

# Copper inventory IDs
CU_BARE = {1001: "CUBB", 1003: "CU1", 1004: "CU2", 1436: "CU2DIRTY",
            1445: "CHOPS", 1447: "CHOPS2", 1534: "CHOPSFINE"}
CU_ICW = {1141: "CUINS1", 1010: "CUINS2", 1007: "MCM", 1008: "THHN",
           1170: "WAVEOPENCU", 1842: "CUINS1FEED", 1843: "CUINS2FEED",
           1014: "HARNESS", 1011: "CUINSLOW", 1012: "CUINSXMAS",
           1384: "CUINS2HIGH", 1216: "CUINS1LITE", 1169: "JELLYWIRE",
           1361: "CUINS<25%", 1362: "CUREFINERY", 1177: "ALUMBX"}
ALL_CU = {**CU_BARE, **CU_ICW}
RECOVERY = {1007: 0.88, 1008: 0.78, 1141: 0.65, 1170: 0.55, 1010: 0.42,
            1842: 0.65, 1843: 0.42, 1014: 0.45, 1011: 0.28, 1012: 0.12,
            1384: 0.65, 1216: 0.55, 1169: 0.38, 1361: 0.18, 1362: 1.0,
            1177: 0.63}

# Metal family grouping for non-copper
# TAA feedstock = 6063 extrusion that gets shredded into 6063SHRED for Tri-Arrows
TAA_FEEDSTOCK = {"6063BARE", "6063PTD", "6063IRONY", "6063POLY", "THERMAL",
                 "6063SHRED"}

METAL_FAMILIES = {
    "Aluminum": {"ALBRKG", "ALR", "ALRD", "ALCR", "ALSTART", "ALCUURD",
                 "CANS", "DIECAST", "ACCOMP", "ALTURN",
                 "6061EXT", "6061PTD", "6061IRONY", "6061SPC", "6061PUCKS"},
    "Ferrous (Tin & Iron)": {"REBAR", "CASTIRON", "CAST", "STEELSHAV", "L Iron", "S Iron",
                             "LOWBOARD", "STEEL", "SS304", "SSBRKG", "TIN"},
    "Auto/Wheels": {"WHEELSDIRTY", "CHRWHEELSDIR", "TRKWHEELS", "WHEELS", "Rotors",
                    "SEALED", "BUSH"},
    "Wire/Cable": {"ACSR/NEO", "NEOPRENE", "ELECMOT", "CAT5"},
    "Lead": {"LEAD"},
    "Brass": {"YELLOW", "90/10CUPRO"},
    "E-Scrap": {"PS", "TABLETS", "HARDDRBRD", "SHREDHARDDR"},
    "Transformers": {"CUTRANSF", "ALCUHEAT"},
}

# Ferrous-classified IDs — used for ferrous/nonferrous split in headline
FERROUS_FAMILIES = {"Ferrous (Tin & Iron)", "Auto/Wheels"}

def classify_metal(short_name):
    if short_name in TAA_FEEDSTOCK:
        return "TAA Feedstock"
    for family, items in METAL_FAMILIES.items():
        if short_name in items:
            return family
    return "Other"


def _date_filter(date_str=None):
    """Return SQL date expression: specific date or GETDATE()."""
    if date_str:
        return f"'{date_str}'"
    return "CAST(GETDATE() AS DATE)"


def fetch_daily_data(date_str=None):
    """Query ROM for scale tickets (dealer/contractor — NOT public)."""
    df = _date_filter(date_str)
    rows = query_rom(f'''
        SELECT h.TrackingID, h.StartedAt, d.CompanyName,
               i.ShortName, i.InventoryID,
               dtl.Gross, dtl.Tare, (dtl.Gross - dtl.Tare) as NetLbs,
               h.Complete, h.ShipReceive,
               d.DealerGroupID
        FROM TruckScaleHDR h
        JOIN TruckScaleDTL dtl ON h.CompanyID = dtl.CompanyID AND h.TrackingID = dtl.TrackingID
        JOIN Dealers d ON h.DealerID = d.DealerID
        JOIN Inventory i ON dtl.ShippedAsID = i.InventoryID
        WHERE CAST(h.StartedAt AS DATE) = {df}
        ORDER BY h.TrackingID
    ''')
    return rows


def fetch_public_tickets(date_str=None):
    """Count public walk-in tickets from PublicPurchActivity module.

    Public purchases are a separate ROM module — not in TruckScaleHDR.
    PubPurchID is the unique ticket identifier (one per walk-in transaction).
    """
    df = _date_filter(date_str)
    rows = query_rom(f'''
        SELECT COUNT(DISTINCT PubPurchID) AS ticket_count
        FROM PublicPurchActivity
        WHERE Complete = 1
          AND CAST(PurchDate AS DATE) = {df}
    ''')
    if rows:
        return rows[0].get("ticket_count", 0)
    return 0


def fetch_walkin_avg_same_weekday(date_str=None):
    """Average walk-in count for the same weekday over the prior 4 weeks."""
    if date_str:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
    else:
        target = datetime.now().date()
    prior_dates = [(target - timedelta(weeks=w)).strftime("%Y-%m-%d") for w in range(1, 5)]
    date_list = ",".join(f"'{d}'" for d in prior_dates)
    rows = query_rom(f'''
        SELECT CAST(PurchDate AS DATE) AS d, COUNT(DISTINCT PubPurchID) AS cnt
        FROM PublicPurchActivity
        WHERE Complete = 1
          AND CAST(PurchDate AS DATE) IN ({date_list})
        GROUP BY CAST(PurchDate AS DATE)
    ''')
    if not rows:
        return None
    return round(sum(r["cnt"] for r in rows) / len(rows))



def build_report(rows):
    """Build structured report data from scale ticket rows."""
    inbound = [r for r in rows if r["ShipReceive"] == 0]
    outbound = [r for r in rows if r["ShipReceive"] == 1]

    in_tickets = set(r["TrackingID"] for r in inbound)
    out_tickets = set(r["TrackingID"] for r in outbound)
    open_tickets = set(r["TrackingID"] for r in inbound if not r["Complete"])

    # Copper breakdown
    bare_lbs = 0
    icw_gross = 0
    icw_cu = 0
    cu_by_grade = defaultdict(float)
    for r in inbound:
        net = r["NetLbs"] or 0
        inv = r["InventoryID"]
        if inv in CU_BARE:
            bare_lbs += net
            cu_by_grade[r["ShortName"]] += net
        elif inv in CU_ICW:
            icw_gross += net
            rec = RECOVERY.get(inv, 0.50)
            icw_cu += net * rec

    # Non-copper by family
    family_totals = defaultdict(float)
    family_items = defaultdict(lambda: defaultdict(float))
    for r in inbound:
        net = r["NetLbs"] or 0
        inv = r["InventoryID"]
        if inv not in ALL_CU:
            fam = classify_metal(r["ShortName"])
            family_totals[fam] += net
            family_items[fam][r["ShortName"]] += net

    # By customer
    cust_data = defaultdict(lambda: {"lbs": 0, "tickets": set(), "items": defaultdict(float)})
    for r in inbound:
        net = r["NetLbs"] or 0
        c = r["CompanyName"]
        cust_data[c]["lbs"] += net
        cust_data[c]["tickets"].add(r["TrackingID"])
        cust_data[c]["items"][r["ShortName"]] += net

    # Outbound by customer
    out_data = defaultdict(lambda: {"lbs": 0, "items": defaultdict(float)})
    for r in outbound:
        net = r["NetLbs"] or 0
        out_data[r["CompanyName"]]["lbs"] += net
        out_data[r["CompanyName"]]["items"][r["ShortName"]] += net

    grand_in = sum(r["NetLbs"] or 0 for r in inbound)
    grand_out = sum(r["NetLbs"] or 0 for r in outbound)

    # Contractor tickets (DealerGroupID=10012) from TruckScaleHDR
    # Public walk-ins are in a separate module (PublicPurchActivity), counted separately
    contractor_tickets = set(r["TrackingID"] for r in inbound
                             if r.get("DealerGroupID") == 10012)

    # Ferrous vs nonferrous split
    # Ferrous = Steel/Iron/Tin + Auto/Wheels (rotors, cast iron wheels, etc.)
    # Nonferrous = everything else (copper, aluminum, brass, wire/cable, e-scrap, etc.)
    in_ferrous = 0
    in_nonferrous = 0
    for r in inbound:
        net = r["NetLbs"] or 0
        fam = classify_metal(r["ShortName"])
        if fam in FERROUS_FAMILIES:
            in_ferrous += net
        else:
            in_nonferrous += net

    out_ferrous = 0
    out_nonferrous = 0
    for r in outbound:
        net = r["NetLbs"] or 0
        fam = classify_metal(r["ShortName"])
        if fam in FERROUS_FAMILIES:
            out_ferrous += net
        else:
            out_nonferrous += net

    return {
        "in_tickets": len(in_tickets), "out_tickets": len(out_tickets),
        "dealer_tickets": len(in_tickets) - len(contractor_tickets),
        "open_tickets": len(open_tickets),
        "contractor_tickets": len(contractor_tickets),
        "grand_in": grand_in, "grand_out": grand_out,
        "in_ferrous": in_ferrous, "in_nonferrous": in_nonferrous,
        "out_ferrous": out_ferrous, "out_nonferrous": out_nonferrous,
        "bare_lbs": bare_lbs, "icw_gross": icw_gross, "icw_cu": icw_cu,
        "cu_by_grade": dict(cu_by_grade),
        "family_totals": dict(family_totals), "family_items": {k: dict(v) for k, v in family_items.items()},
        "cust_data": {k: {"lbs": v["lbs"], "tickets": len(v["tickets"]),
                          "items": dict(v["items"])} for k, v in cust_data.items()},
        "out_data": {k: {"lbs": v["lbs"], "items": dict(v["items"])} for k, v in out_data.items()},
    }


def fmt(n):
    """Format number with commas."""
    return f"{n:,.0f}"


def _bar(value, max_value, width=6):
    """Unicode bar chart: ████░░"""
    if max_value <= 0:
        return "░" * width
    filled = min(round(value / max_value * width), width)
    return "█" * filled + "░" * (width - filled)


def build_adaptive_card(rpt, public_tickets=0, walkin_avg=None, date_str=None):
    """Build Teams Adaptive Card JSON."""
    if date_str:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        dt = datetime.now()
    today = dt.strftime("%A, %B %d %Y")
    total_cu = rpt["bare_lbs"] + rpt["icw_cu"]

    body = []

    def section(text):
        body.append({"type": "TextBlock", "text": text,
                     "weight": "Bolder", "size": "Medium", "color": "Accent",
                     "spacing": "Large", "separator": True})

    def facts(items):
        body.append({"type": "FactSet", "facts": items, "spacing": "Small"})

    def note(text, color=None):
        block = {"type": "TextBlock", "text": text,
                 "size": "Small", "isSubtle": color is None, "wrap": True,
                 "spacing": "Small"}
        if color:
            block["color"] = color
        body.append(block)

    # ── HEADER ──
    body.append({"type": "TextBlock", "text": "GEOMET DAILY YARD REPORT",
                 "weight": "Bolder", "size": "Large", "color": "Good"})
    body.append({"type": "TextBlock", "text": today, "size": "Medium", "spacing": "None"})

    # ── HEADLINE — inbound/outbound side by side ──
    body.append({
        "type": "ColumnSet", "spacing": "Medium",
        "columns": [
            {"type": "Column", "width": "stretch", "items": [
                {"type": "TextBlock", "text": "📥 INBOUND", "weight": "Bolder", "size": "Small", "color": "Good"},
                {"type": "TextBlock", "text": f"**{fmt(rpt['grand_in'])}** lbs", "size": "Large", "spacing": "None"},
                {"type": "TextBlock", "text": f"🚛 {rpt['dealer_tickets']} dealer trucks", "size": "Small", "isSubtle": True, "spacing": "None"},
            ]},
            {"type": "Column", "width": "stretch", "items": [
                {"type": "TextBlock", "text": "📤 OUTBOUND", "weight": "Bolder", "size": "Small", "color": "Accent"},
                {"type": "TextBlock", "text": f"**{fmt(rpt['grand_out'])}** lbs", "size": "Large", "spacing": "None"},
                {"type": "TextBlock", "text": f"🚛 {rpt['out_tickets']} trucks", "size": "Small", "isSubtle": True, "spacing": "None"},
            ]},
        ]
    })
    note(f"In:  Nonferrous {fmt(rpt['in_nonferrous'])}  |  Ferrous {fmt(rpt['in_ferrous'])}")
    note(f"Out: Nonferrous {fmt(rpt['out_nonferrous'])}  |  Ferrous {fmt(rpt['out_ferrous'])}")

    # ── RETAIL WALK-INS ──
    contractor_ct = rpt.get("contractor_tickets", 0)
    retail_total = contractor_ct + public_tickets
    walkin_line = f"🛒 Walk-ins: {retail_total} (contractors {contractor_ct}, public {public_tickets})"
    if walkin_avg is not None and walkin_avg > 0:
        pct = round((public_tickets - walkin_avg) / walkin_avg * 100)
        if pct > 0:
            trend_color = "Good"
            arrow = "▲"
        elif pct < 0:
            trend_color = "Attention"
            arrow = "▼"
        else:
            trend_color = None
            arrow = "—"
        walkin_line += f"  |  4-wk avg {dt.strftime('%a')}: {walkin_avg} ({arrow}{abs(pct)}%)"
        note(walkin_line, color=trend_color)
    else:
        note(walkin_line)

    # ── COPPER ──
    section("🔶 COPPER")

    # Bar chart for copper grades
    all_cu_grades = sorted(rpt["cu_by_grade"].items(), key=lambda x: -x[1])
    max_cu = all_cu_grades[0][1] if all_cu_grades else 1
    cu_facts = []
    for grade, lbs in all_cu_grades:
        if lbs > 0:
            bar = _bar(lbs, max_cu)
            cu_facts.append({"title": f"  {grade}", "value": f"{bar}  {fmt(lbs)} lbs"})
    cu_facts.insert(0, {"title": "Bare Copper", "value": f"{fmt(rpt['bare_lbs'])} lbs"})
    cu_facts.append({"title": "ICW Feedstock (gross)", "value": f"{fmt(rpt['icw_gross'])} lbs"})
    cu_facts.append({"title": "ICW Cu Content", "value": f"{fmt(rpt['icw_cu'])} lbs"})
    facts(cu_facts)
    body.append({"type": "TextBlock", "text": f"⚡ Total Cu Content: {fmt(total_cu)} lbs",
                 "weight": "Bolder", "size": "Medium", "color": "Good", "spacing": "None"})

    # ── TAA FEEDSTOCK ──
    taa_total = rpt["family_totals"].get("TAA Feedstock", 0)
    if taa_total > 0:
        section("♻️ TAA FEEDSTOCK")
        taa_items = rpt["family_items"].get("TAA Feedstock", {})
        taa_facts = []
        for item, lbs in sorted(taa_items.items(), key=lambda x: -x[1]):
            if lbs > 0:
                taa_facts.append({"title": f"  {item}", "value": f"{fmt(lbs)} lbs"})
        facts(taa_facts)
        body.append({"type": "TextBlock", "text": f"Total TAA: {fmt(taa_total)} lbs",
                     "weight": "Bolder", "size": "Medium", "color": "Good", "spacing": "None"})

    # ── OTHER METALS BY CATEGORY ──
    section("🔩 OTHER METALS")

    fam_icons = {"Ferrous (Tin & Iron)": "🧲", "E-Scrap": "💻"}
    metal_facts = []
    for fam in ["Aluminum", "Ferrous (Tin & Iron)", "Auto/Wheels", "Wire/Cable",
                "E-Scrap", "Brass", "Lead", "Transformers", "Other"]:
        total = rpt["family_totals"].get(fam, 0)
        if total <= 0:
            continue
        icon = fam_icons.get(fam, "")
        prefix = f"{icon} " if icon else ""
        metal_facts.append({"title": f"{prefix}{fam}", "value": f"{fmt(total)} lbs"})
    facts(metal_facts)

    # ── CUSTOMERS ──
    section("👥 CUSTOMERS")

    sorted_custs = sorted(rpt["cust_data"].items(), key=lambda x: -x[1]["lbs"])
    cust_facts = []
    shown = 0
    for cust, d in sorted_custs:
        if d["lbs"] <= 0:
            continue
        if shown >= 15:
            break
        top_items = sorted(((n, l) for n, l in d["items"].items() if n.lower().strip("<> ") != "select"), key=lambda x: -x[1])[:3]
        detail = ", ".join(f"{n} {fmt(l)}" for n, l in top_items)
        cust_facts.append({
            "title": f"{cust[:25]} ({d['tickets']}x)",
            "value": f"{fmt(d['lbs'])} lbs  —  {detail}"
        })
        shown += 1
    facts(cust_facts)

    remaining = sorted_custs[15:]
    remaining_lbs = sum(d["lbs"] for _, d in remaining if d["lbs"] > 0)
    remaining_ct = sum(1 for _, d in remaining if d["lbs"] > 0)
    if remaining_ct > 0:
        note(f"+ {remaining_ct} more vendors, {fmt(remaining_lbs)} lbs")

    # ── OUTBOUND ──
    if rpt["out_data"]:
        section("📤 OUTBOUND SHIPMENTS")
        out_facts = []
        for cust, d in sorted(rpt["out_data"].items(), key=lambda x: -x[1]["lbs"]):
            if d["lbs"] <= 0:
                continue
            items_str = ", ".join(f"{n}" for n, l in
                                  sorted(d["items"].items(), key=lambda x: -x[1])[:2])
            out_facts.append({
                "title": f"{cust[:25]} ({items_str})",
                "value": f"{fmt(d['lbs'])} lbs"
            })
        facts(out_facts)

    # ── ALERTS ──
    alerts = []
    if rpt["open_tickets"] > 0:
        alerts.append(f"🚨 {rpt['open_tickets']} ticket{'s' if rpt['open_tickets']!=1 else ''} still open at report time")

    if alerts:
        section("⚠️ ALERTS")
        for a in alerts:
            body.append({"type": "TextBlock", "text": a, "color": "Attention",
                         "size": "Small", "wrap": True, "spacing": "Small"})

    # ── FOOTER ──
    body.append({"type": "TextBlock", "text": " ", "spacing": "Large"})
    body.append({"type": "TextBlock",
                 "text": f"Generated {datetime.now().strftime('%I:%M %p')} · Geomet Copper Intelligence ⚙️",
                 "size": "Small", "isSubtle": True, "horizontalAlignment": "Right"})

    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body
    }
    return card


def post_to_teams(card):
    """Send adaptive card to Teams webhook."""
    url = WEBHOOK_URL
    if not url:
        print("ERROR: TEAMS_WEBHOOK_URL not set")
        sys.exit(1)

    payload = json.dumps({
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "contentUrl": None,
            "content": card
        }]
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload,
                                headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"Teams webhook response: {resp.status}")
    return resp.status


def main():
    # Optional date override: python daily_report.py 2026-05-04
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    label = date_str or "today"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Generating daily yard report for {label}...")

    rows = fetch_daily_data(date_str)
    if not rows:
        print(f"No scale tickets for {label}. Skipping report.")
        return

    rpt = build_report(rows)
    public_ct = fetch_public_tickets(date_str)
    walkin_avg = fetch_walkin_avg_same_weekday(date_str)
    card = build_adaptive_card(rpt, public_tickets=public_ct, walkin_avg=walkin_avg, date_str=date_str)

    # Also save locally for debugging
    out_path = os.path.join(os.path.dirname(__file__), "data", "daily_report.json")
    with open(out_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Card saved to {out_path}")

    post_to_teams(card)
    print("Report posted to Teams.")


if __name__ == "__main__":
    main()
