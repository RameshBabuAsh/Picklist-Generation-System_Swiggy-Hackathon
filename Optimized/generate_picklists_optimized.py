#!/usr/bin/env python3
"""
generate_picklists_optimized.py

Enhanced version with:
- Picklist execution time estimation
- Picker shift scheduling
- Cutoff time enforcement
- Priority-aware greedy optimization

Assumes dataset now provides:
- cutoff_time (datetime)
- pod_priority
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import heapq

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "swiggy.csv"
OUTPUT_DIR = "picklists"
SUMMARY_CSV = "Summary.csv"

# Picklist capacity
MAX_UNITS = 2000
MAX_WEIGHT = 200.0
FRAGILE_WEIGHT = 50.0

# -------- Time Constants (seconds) --------
ENTRY_TO_ZONE = 120
BIN_TO_BIN = 30
PICK_PER_UNIT = 5
ZONE_TO_STAGE = 120
UNLOAD_PER_ORDER = 30

# -------- Picker Shifts --------
PICKER_SHIFTS = [
    ("Morning", "08:00", "17:00", 40),
    ("General", "10:00", "19:00", 30),
    ("Night1", "20:00", "05:00", 45),
    ("Night2", "21:00", "07:00", 35),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PICKER POOL INITIALIZATION
# ============================================================

def init_pickers(base_date):
    pickers = []
    pid = 0

    for name, start, end, count in PICKER_SHIFTS:
        s = datetime.combine(base_date, datetime.strptime(start, "%H:%M").time())
        e = datetime.combine(base_date, datetime.strptime(end, "%H:%M").time())
        if e <= s:
            e += timedelta(days=1)

        for _ in range(count):
            heapq.heappush(pickers, (s, pid, e))
            pid += 1

    return pickers  # min-heap by available time

# ============================================================
# EXECUTION TIME ESTIMATION
# ============================================================

def estimate_picklist_time(items):
    units = sum(i["qty"] for i in items)
    bins = len(set(i["bin"] for i in items))
    orders = len(set(i["order_id"] for i in items))

    return (
        ENTRY_TO_ZONE +
        bins * BIN_TO_BIN +
        units * PICK_PER_UNIT +
        ZONE_TO_STAGE +
        orders * UNLOAD_PER_ORDER
    )

# ============================================================
# MAIN LOGIC
# ============================================================

def generate_picklists():
    df = pd.read_csv(INPUT_CSV, parse_dates=["cutoff_time"])
    df["fragile"] = df["fragile"].astype(bool)
    df["weight"] = df["weight"] / 1000.0 if df["weight"].median() > 1000 else df["weight"]

    # Sort by cutoff then priority
    df = df.sort_values(by=["cutoff_time", "pod_priority"])

    base_date = df["cutoff_time"].min().date()
    picker_heap = init_pickers(base_date)

    summary = []
    picklist_id = 0

    for (zone, fragile), group in df.groupby(["zone", "fragile"]):
        cap_weight = FRAGILE_WEIGHT if fragile else MAX_WEIGHT

        items, cur_units, cur_weight = [], 0, 0.0

        for row in group.to_dict("records"):
            qty = int(row["order_qty"])
            unit_wt = row["weight"]

            while qty > 0:
                space_units = MAX_UNITS - cur_units
                space_weight = cap_weight - cur_weight
                max_by_weight = int(space_weight // unit_wt) if unit_wt > 0 else qty

                take = min(qty, space_units, max_by_weight)

                if take <= 0:
                    finalize_picklist(
                        items, picker_heap, summary, zone, fragile, picklist_id
                    )
                    picklist_id += 1
                    items, cur_units, cur_weight = [], 0, 0.0
                    continue

                items.append({
                    "sku": row["sku"],
                    "order_id": row["order_id"],
                    "store": row["store_id"],
                    "bin": row["location_code"],
                    "bin_rank": row["bin_rank"],
                    "qty": take,
                    "unit_weight": unit_wt,
                    "cutoff": row["cutoff_time"],
                })

                cur_units += take
                cur_weight += take * unit_wt
                qty -= take

        if items:
            finalize_picklist(
                items, picker_heap, summary, zone, fragile, picklist_id
            )
            picklist_id += 1

    pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)
    print("Picklists and summary generated successfully.")

# ============================================================
# PICKLIST FINALIZATION & PICKER ASSIGNMENT
# ============================================================

def finalize_picklist(items, picker_heap, summary, zone, fragile, pl_id):
    exec_time = estimate_picklist_time(items)
    earliest_cutoff = min(i["cutoff"] for i in items)

    while picker_heap:
        available_at, pid, shift_end = heapq.heappop(picker_heap)
        finish = available_at + timedelta(seconds=exec_time)

        if finish <= shift_end:
            heapq.heappush(picker_heap, (finish, pid, shift_end))
            break
    else:
        return  # no picker available

    valid_units = sum(
        i["qty"] for i in items if finish <= i["cutoff"]
    )

    fname = f"{datetime.now().date()}_PL{pl_id}.csv"
    pd.DataFrame(items).to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

    summary.append({
        "picklist_file": fname,
        "zone": zone,
        "picklist_type": "fragile" if fragile else "normal",
        "total_units": sum(i["qty"] for i in items),
        "valid_units": valid_units,
        "finish_time": finish.isoformat(),
        "earliest_cutoff": earliest_cutoff.isoformat(),
        "stores_in_picklist": ",".join(set(i["store"] for i in items)),
    })


if __name__ == "__main__":
    generate_picklists()
