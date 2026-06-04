#!/usr/bin/env python3
"""Geomet Weekly Yard Report — aggregates Mon-Fri (or Mon-Sat) and posts to Teams."""

import os
import sys
import json
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict

# Reuse daily_report infrastructure
sys.path.insert(0, os.path.dirname(__file__))

# Load ROM env
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

from rom import query_rom
from daily_report import (
    CU_BARE, CU_ICW, ALL_CU, RECOVERY, METAL_FAMILIES, TAA_FEEDSTOCK,
    FERROUS_FAMILIES, classify_metal, fmt, _bar, post_to_teams
)

WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL", "")


def _week_dates(end_date=None):
    """Return list of date strings for the business week ending on end_date.
    If end_date is Saturday, returns Mon-Sat. If Friday, returns Mon-Fri."""
    if end_date is None:
        end_date = datetime.now().date()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    # Walk back to Monday
    weekday = end_date.weekday()  # 0=Mon, 5=Sat
    monday = end_date - timedelta(days=weekday)
    dates = []
    d = monday
    while d <= end_date:
        if d.weekday() < 6:  # Skip Sunday
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def _prior_week_dates(current_week_dates):
    """Return date strings for the prior business week (same length)."""
    if not current_week_dates:
        return []
    first = datetime.strptime(current_week_dates[0], "%Y-%m-%d").date()
    last = datetime.strptime(current_week_dates[-1], "%Y-%m-%d").date()
    days = (last - first).days + 1
    prior_end = first - timedelta(days=1)
    # Walk back to find prior Monday
    while prior_end.weekday() == 6:  # skip Sunday
        prior_end -= timedelta(days=1)
    prior_start = prior_end - timedelta(days=days - 1)
    dates = []
    d = prior_start
    while d <= prior_end:
        if d.weekday() < 6:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def fetch_week_data(dates):
    """Fetch scale ticket data for a list of dates."""
    date_list = ",".join(f"'{d}'" for d in dates)
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
        WHERE CAST(h.StartedAt AS DATE) IN ({date_list})
        ORDER BY h.TrackingID
    ''')
    return rows


def fetch_week_walkins(dates):
    """Fetch public walk-in count for a list of dates."""
    date_list = ",".join(f"'{d}'" for d in dates)
    rows = query_rom(f'''
        SELECT COUNT(DISTINCT PubPurchID) AS ticket_count
        FROM PublicPurchActivity
        WHERE Complete = 1
          AND CAST(PurchDate AS DATE) IN ({date_list})
    ''')
    if rows:
        return rows[0].get("ticket_count", 0)
    return 0


def build_week_stats(rows):
    """Aggregate weekly stats from scale ticket rows."""
    inbound = [r for r in rows if r["ShipReceive"] == 0]
    outbound = [r for r in rows if r["ShipReceive"] == 1]

    grand_in = sum(r["NetLbs"] or 0 for r in inbound)
    grand_out = sum(r["NetLbs"] or 0 for r in outbound)
    in_tickets = set(r["TrackingID"] for r in inbound)
    out_tickets = set(r["TrackingID"] for r in outbound)

    # Ferrous / nonferrous split
    in_ferrous = 0
    in_nonferrous = 0
    for r in inbound:
        net = r["NetLbs"] or 0
        fam = classify_metal(r["ShortName"])
        inv = r["InventoryID"]
        if fam in FERROUS_FAMILIES:
            in_ferrous += net
        else:
            in_nonferrous += net

    # Copper
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

    # By customer
    cust_data = defaultdict(lambda: {"lbs": 0, "tickets": set()})
    for r in inbound:
        net = r["NetLbs"] or 0
        c = r["CompanyName"]
        cust_data[c]["lbs"] += net
        cust_data[c]["tickets"].add(r["TrackingID"])

    # Outbound by customer
    out_data = defaultdict(lambda: {"lbs": 0, "items": defaultdict(float)})
    for r in outbound:
        net = r["NetLbs"] or 0
        out_data[r["CompanyName"]]["lbs"] += net
        out_data[r["CompanyName"]]["items"][r["ShortName"]] += net

    # Day-by-day inbound
    daily_in = defaultdict(float)
    for r in inbound:
        day = r["StartedAt"]
        if hasattr(day, "strftime"):
            day = day.strftime("%Y-%m-%d")
        else:
            day = str(day)[:10]
        daily_in[day] += r["NetLbs"] or 0

    # Contractor tickets
    contractor_tickets = set(r["TrackingID"] for r in inbound
                             if r.get("DealerGroupID") == 10012)

    return {
        "grand_in": grand_in, "grand_out": grand_out,
        "in_tickets": len(in_tickets), "out_tickets": len(out_tickets),
        "in_ferrous": in_ferrous, "in_nonferrous": in_nonferrous,
        "bare_lbs": bare_lbs, "icw_gross": icw_gross, "icw_cu": icw_cu,
        "cu_by_grade": dict(cu_by_grade),
        "cust_data": dict(cust_data),
        "out_data": dict(out_data),
        "daily_in": dict(daily_in),
        "contractor_tickets": len(contractor_tickets),
    }


def _pct_change(current, prior):
    """Return formatted % change string with arrow and color."""
    if prior == 0:
        return "", None
    pct = round((current - prior) / abs(prior) * 100)
    if pct > 0:
        return f"▲{pct}%", "Good"
    elif pct < 0:
        return f"▼{abs(pct)}%", "Attention"
    return "—", None


def build_weekly_card(stats, prior_stats, walkins, prior_walkins, dates, prior_dates):
    """Build Teams Adaptive Card for the weekly summary."""
    first = datetime.strptime(dates[0], "%Y-%m-%d")
    last = datetime.strptime(dates[-1], "%Y-%m-%d")
    total_cu = stats["bare_lbs"] + stats["icw_cu"]
    prior_cu = prior_stats["bare_lbs"] + prior_stats["icw_cu"]

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
    body.append({"type": "TextBlock", "text": "GEOMET WEEKLY YARD REPORT",
                 "weight": "Bolder", "size": "Large", "color": "Good"})
    body.append({"type": "TextBlock",
                 "text": f"{first.strftime('%b %d')} – {last.strftime('%b %d, %Y')}  ({len(dates)} days)",
                 "size": "Medium", "spacing": "None"})

    # ── HEADLINE — side by side with % change ──
    in_chg, in_color = _pct_change(stats["grand_in"], prior_stats["grand_in"])
    out_chg, out_color = _pct_change(stats["grand_out"], prior_stats["grand_out"])

    body.append({
        "type": "ColumnSet", "spacing": "Medium",
        "columns": [
            {"type": "Column", "width": "stretch", "items": [
                {"type": "TextBlock", "text": "📥 INBOUND", "weight": "Bolder", "size": "Small", "color": "Good"},
                {"type": "TextBlock", "text": f"**{fmt(stats['grand_in'])}** lbs", "size": "Large", "spacing": "None"},
                {"type": "TextBlock", "text": f"🚛 {stats['in_tickets']} trucks  {in_chg} vs prior wk",
                 "size": "Small", "isSubtle": True, "spacing": "None"},
            ]},
            {"type": "Column", "width": "stretch", "items": [
                {"type": "TextBlock", "text": "📤 OUTBOUND", "weight": "Bolder", "size": "Small", "color": "Accent"},
                {"type": "TextBlock", "text": f"**{fmt(stats['grand_out'])}** lbs", "size": "Large", "spacing": "None"},
                {"type": "TextBlock", "text": f"🚛 {stats['out_tickets']} trucks  {out_chg} vs prior wk",
                 "size": "Small", "isSubtle": True, "spacing": "None"},
            ]},
        ]
    })
    note(f"Nonferrous {fmt(stats['in_nonferrous'])}  |  Ferrous {fmt(stats['in_ferrous'])}")

    # ── DAILY INBOUND SPARKLINE ──
    sorted_days = sorted(stats["daily_in"].items())
    if sorted_days:
        max_day = max(v for _, v in sorted_days)
        day_lines = []
        for d, lbs in sorted_days:
            dt = datetime.strptime(d, "%Y-%m-%d")
            bar = _bar(lbs, max_day)
            day_lines.append({"title": dt.strftime("%a"), "value": f"{bar}  {fmt(lbs)}"})
        section("📅 DAILY INBOUND")
        facts(day_lines)

    # ── WALK-INS ──
    walkin_chg, walkin_color = _pct_change(walkins, prior_walkins)
    walkin_line = f"🛒 Walk-ins: {walkins} this week  (prior week: {prior_walkins}  {walkin_chg})"
    note(walkin_line, color=walkin_color)

    # ── COPPER ──
    cu_chg, cu_color = _pct_change(total_cu, prior_cu)
    section("🔶 COPPER")
    cu_facts = []
    all_cu_grades = sorted(stats["cu_by_grade"].items(), key=lambda x: -x[1])
    max_cu = all_cu_grades[0][1] if all_cu_grades else 1
    for grade, lbs in all_cu_grades:
        if lbs > 0:
            bar = _bar(lbs, max_cu)
            cu_facts.append({"title": f"  {grade}", "value": f"{bar}  {fmt(lbs)} lbs"})
    cu_facts.insert(0, {"title": "Bare Copper", "value": f"{fmt(stats['bare_lbs'])} lbs"})
    cu_facts.append({"title": "ICW Feedstock (gross)", "value": f"{fmt(stats['icw_gross'])} lbs"})
    cu_facts.append({"title": "ICW Cu Content", "value": f"{fmt(stats['icw_cu'])} lbs"})
    facts(cu_facts)
    body.append({"type": "TextBlock",
                 "text": f"⚡ Total Cu Content: {fmt(total_cu)} lbs  ({cu_chg} vs prior wk)",
                 "weight": "Bolder", "size": "Medium", "color": "Good", "spacing": "None"})

    # ── TOP 10 CUSTOMERS ──
    section("👥 TOP CUSTOMERS")
    sorted_custs = sorted(stats["cust_data"].items(), key=lambda x: -x[1]["lbs"])
    cust_facts = []
    shown = 0
    for cust, d in sorted_custs:
        if d["lbs"] <= 0:
            continue
        if shown >= 10:
            break
        tickets = len(d["tickets"])
        cust_facts.append({
            "title": f"{cust[:25]} ({tickets}x)",
            "value": f"{fmt(d['lbs'])} lbs"
        })
        shown += 1
    facts(cust_facts)

    remaining = sorted_custs[10:]
    remaining_lbs = sum(d["lbs"] for _, d in remaining if d["lbs"] > 0)
    remaining_ct = sum(1 for _, d in remaining if d["lbs"] > 0)
    if remaining_ct > 0:
        note(f"+ {remaining_ct} more vendors, {fmt(remaining_lbs)} lbs")

    # ── OUTBOUND ──
    if stats["out_data"]:
        section("📤 OUTBOUND SHIPMENTS")
        out_facts = []
        for cust, d in sorted(stats["out_data"].items(), key=lambda x: -x[1]["lbs"]):
            if d["lbs"] <= 0:
                continue
            items_str = ", ".join(f"{n}" for n, l in
                                  sorted(d["items"].items(), key=lambda x: -x[1])[:2])
            out_facts.append({
                "title": f"{cust[:25]} ({items_str})",
                "value": f"{fmt(d['lbs'])} lbs"
            })
        facts(out_facts)

    # ── FOOTER ──
    body.append({"type": "TextBlock", "text": " ", "spacing": "Large"})
    body.append({"type": "TextBlock",
                 "text": f"Generated {datetime.now().strftime('%I:%M %p')} · Geomet Copper Intelligence ⚙️",
                 "size": "Small", "isSubtle": True, "horizontalAlignment": "Right"})

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body
    }


def main():
    # Optional: pass end date, e.g. python weekly_report.py 2026-05-17
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    if date_str:
        end_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    else:
        end_date = datetime.now().date()

    dates = _week_dates(end_date)
    prior_dates = _prior_week_dates(dates)
    label = f"{dates[0]} to {dates[-1]}"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Generating weekly report for {label}...")

    rows = fetch_week_data(dates)
    if not rows:
        print(f"No scale tickets for {label}. Skipping report.")
        return

    stats = build_week_stats(rows)
    walkins = fetch_week_walkins(dates)

    prior_rows = fetch_week_data(prior_dates)
    prior_stats = build_week_stats(prior_rows) if prior_rows else build_week_stats([])
    prior_walkins = fetch_week_walkins(prior_dates)

    card = build_weekly_card(stats, prior_stats, walkins, prior_walkins, dates, prior_dates)

    out_path = os.path.join(os.path.dirname(__file__), "data", "weekly_report.json")
    with open(out_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Card saved to {out_path}")

    post_to_teams(card)
    print("Weekly report posted to Teams.")


if __name__ == "__main__":
    main()
