#!/usr/bin/env python3
"""
generate_picklists.py

Reads `swiggy.csv` from the current directory and generates:
  - ./picklists/{date}_ZONE_{ZONE}_PL{n}.csv  (one CSV per picklist)
  - ./Summary.csv                              (one row per picklist)

Behavior:
  - Sorts rows by cutoff datetime (earliest first) then by priority (lower numeric value = higher priority).
  - Splits fragile items (truthy values in fragile column) and non-fragile items.
  - Creates zone-specific picklists greedily:
       max units per picklist = 2000
       max weight per picklist = 200.0 kg (fragile picklists: 50.0 kg)
  - Automatically maps commonly-named columns (order_id, store_id, sku, order_qty, zone, bin, bin_rank, pod_priority, weight_in_grams, fragile, dt/order_date for cutoff).
  - If you want full-file processing on very large files, consider setting MAX_ROWS=None or using chunking (see comment).
"""

import os
import pandas as pd
from datetime import datetime

# ---------------- CONFIG ----------------
INPUT_CSV = "swiggy.csv"
OUTPUT_DIR = "picklists"
SUMMARY_CSV = "Summary.csv"

# If MAX_ROWS is None -> process entire file. If integer, only process that many rows after sorting by cutoff+priority.
MAX_ROWS = None  # e.g., set to 200000 to cap for testing

# Picklist capacities
PICKLIST_UNIT_CAP = 2000
PICKLIST_WEIGHT_CAP = 200.0      # kg for normal
FRAGILE_WEIGHT_CAP = 50.0        # kg for fragile

# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_col(cols_lower, candidates):
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def read_and_normalize(input_path):
    print("Loading:", input_path)
    df = pd.read_csv(input_path, low_memory=True)
    print("Rows loaded:", len(df))
    cols_lower = {c.lower(): c for c in df.columns}

    # Best-effort column detection
    col_order_id = find_col(cols_lower, ["order_id", "order id", "order", "orderid"])
    col_store    = find_col(cols_lower, ["store_id", "store", "pod", "store id", "pod_id"])
    col_sku      = find_col(cols_lower, ["sku", "item", "product", "product_id", "sku_id"])
    col_qty      = find_col(cols_lower, ["order_qty", "qty", "quantity", "order_quantity"])
    col_zone     = find_col(cols_lower, ["zone", "zone_code", "zone id", "zone_id"])
    col_bin      = find_col(cols_lower, ["bin", "bin_code", "location_code", "bin_id", "location"])
    col_binrank  = find_col(cols_lower, ["bin_rank", "bin rank", "rank"])
    col_cutoff   = find_col(cols_lower, ["cutoff", "cutoff_time", "loading_cutoff", "dt", "order_date", "order_date_time"])
    col_priority = find_col(cols_lower, ["pod_priority", "priority", "pod priority", "pod_priority_id"])
    col_weight   = find_col(cols_lower, ["weight", "unit_weight", "wt", "item_weight", "weight_in_grams"])
    col_fragile  = find_col(cols_lower, ["fragile", "is_fragile", "isfragile", "fragility"])

    # Display mapping
    print("Auto-mapped columns:")
    mapping = {
        "order_id": col_order_id, "store": col_store, "sku": col_sku,
        "qty": col_qty, "zone": col_zone, "bin": col_bin, "bin_rank": col_binrank,
        "cutoff": col_cutoff, "priority": col_priority, "weight": col_weight, "fragile": col_fragile
    }
    for k,v in mapping.items():
        print(f"  {k:8s}: {v}")

    work = df.copy()

    # qty
    if col_qty is None:
        work["_qty"] = 1
        qty_col = "_qty"
    else:
        qty_col = col_qty
        work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce").fillna(1).astype(int)

    # weight (convert grams->kg heuristically)
    if col_weight is None:
        work["_weight"] = 0.0
        weight_col = "_weight"
    else:
        weight_col = col_weight
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0).astype(float)
        if work[weight_col].median() > 1000:
            # assume grams
            work[weight_col] = work[weight_col] / 1000.0

    # zone
    if col_zone is None:
        work["_zone"] = "UNKNOWN"
        zone_col = "_zone"
    else:
        zone_col = col_zone
        work[zone_col] = work[zone_col].fillna("UNKNOWN").astype(str)

    # sku
    if col_sku is None:
        work["_sku"] = "SKU_UNKNOWN"
        sku_col = "_sku"
    else:
        sku_col = col_sku
        work[sku_col] = work[sku_col].astype(str)

    # order id
    if col_order_id is None:
        work["_order"] = work.index.astype(str)
        order_col = "_order"
    else:
        order_col = col_order_id
        work[order_col] = work[order_col].astype(str)

    # store
    if col_store is None:
        work["_store"] = "STORE_UNKNOWN"
        store_col = "_store"
    else:
        store_col = col_store
        work[store_col] = work[store_col].astype(str)

    # bin
    if col_bin is None:
        work["_bin"] = "BIN_UNKNOWN"
        bin_col = "_bin"
    else:
        bin_col = col_bin
        work[bin_col] = work[bin_col].astype(str)

    # bin rank
    if col_binrank is None:
        work["_binrank"] = ""
        binrank_col = "_binrank"
    else:
        binrank_col = col_binrank
        work[binrank_col] = work[binrank_col].astype(str)

    # priority
    if col_priority is None:
        work["_priority"] = 9999
        priority_col = "_priority"
    else:
        priority_col = col_priority
        work[priority_col] = pd.to_numeric(work[priority_col], errors="coerce").fillna(9999).astype(int)

    # fragile
    if col_fragile is None:
        work["_fragile"] = False
        fragile_col = "_fragile"
    else:
        fragile_col = col_fragile
        s = work[fragile_col].astype(str).str.lower().str.strip()
        work["_fragile"] = s.isin(["1", "true", "yes", "y", "t"])
        fragile_col = "_fragile"

    # cutoff -> datetime; fallback to far future
    if col_cutoff is None:
        work["_cutoff"] = pd.NaT
        cutoff_col = "_cutoff"
    else:
        cutoff_col = col_cutoff
        work[cutoff_col] = pd.to_datetime(work[cutoff_col], errors="coerce", infer_datetime_format=True)
    work[cutoff_col] = work[cutoff_col].fillna(pd.Timestamp("2100-01-01"))

    # Final canonical column names returned
    canon = {
        "df": work,
        "qty_col": qty_col,
        "weight_col": weight_col,
        "zone_col": zone_col,
        "sku_col": sku_col,
        "order_col": order_col,
        "store_col": store_col,
        "bin_col": bin_col,
        "binrank_col": binrank_col,
        "priority_col": priority_col,
        "fragile_col": fragile_col,
        "cutoff_col": cutoff_col
    }
    return canon

def generate_picklists(canon, max_rows=None):
    df = canon["df"]
    qty_col = canon["qty_col"]
    weight_col = canon["weight_col"]
    zone_col = canon["zone_col"]
    sku_col = canon["sku_col"]
    order_col = canon["order_col"]
    store_col = canon["store_col"]
    bin_col = canon["bin_col"]
    binrank_col = canon["binrank_col"]
    priority_col = canon["priority_col"]
    fragile_col = canon["fragile_col"]
    cutoff_col = canon["cutoff_col"]

    # Sort by cutoff, then priority and maybe order_id for determinism
    df_sorted = df.sort_values(by=[cutoff_col, priority_col, order_col])
    if max_rows is not None:
        df_sorted = df_sorted.head(max_rows)
        print("Processing top", len(df_sorted), "rows (after sort).")
    else:
        print("Processing all rows:", len(df_sorted))

    fragile_df = df_sorted[df_sorted[fragile_col] == True]
    nonfrag_df = df_sorted[df_sorted[fragile_col] == False]

    summary_rows = []
    total_picklists = 0

    def process_zone_group(group_df, zone_name, fragile_flag=False):
        nonlocal summary_rows, total_picklists
        cap_units = PICKLIST_UNIT_CAP
        cap_weight = FRAGILE_WEIGHT_CAP if fragile_flag else PICKLIST_WEIGHT_CAP

        items = []   # current picklist items
        cur_units = 0
        cur_weight = 0.0
        cur_orders = set()
        cur_bins = set()
        pl_seq = 0

        def flush_current():
            nonlocal items, cur_units, cur_weight, cur_orders, cur_bins, pl_seq, total_picklists, summary_rows
            if not items:
                return
            pl_seq += 1
            total_picklists += 1
            fname = f"{datetime.now().date()}_ZONE_{zone_name}_PL{pl_seq}.csv"
            out_path = os.path.join(OUTPUT_DIR, fname)
            pd.DataFrame(items).to_csv(out_path, index=False)
            summary_rows.append({
                "picklist_file": fname,
                "zone": zone_name,
                "picklist_no": pl_seq,
                "picklist_type": "fragile" if fragile_flag else "normal",
                "total_units": cur_units,
                "total_weight": cur_weight,
                "distinct_orders": len(cur_orders),
                "distinct_bins": len(cur_bins),
                "earliest_cutoff": min([x["cutoff"] for x in items]) if items else pd.NaT,
                "created_at": datetime.now().isoformat()
            })
            # reset
            items = []
            cur_units = 0
            cur_weight = 0.0
            cur_orders = set()
            cur_bins = set()
            return

        # iterate rows in the group
        for row in group_df.to_dict(orient="records"):
            qty = int(row.get(qty_col, 1))
            unit_wt = float(row.get(weight_col, 0.0))
            sku = str(row.get(sku_col, "SKU_UNKNOWN"))
            order_id = str(row.get(order_col, ""))
            store = str(row.get(store_col, ""))
            binloc = str(row.get(bin_col, "BIN_UNKNOWN"))
            binrank = str(row.get(binrank_col, ""))
            cutoff = row.get(cutoff_col, pd.Timestamp("2100-01-01"))
            priority = int(row.get(priority_col, 9999))
            fragile_flag_row = bool(row.get(fragile_col, False))

            remaining = qty
            while remaining > 0:
                space_units = cap_units - cur_units
                if space_units <= 0:
                    flush_current()
                    continue
                # weight-limited units
                if unit_wt > 0:
                    max_by_weight = int((cap_weight - cur_weight) // unit_wt)
                    if max_by_weight < 0:
                        max_by_weight = 0
                else:
                    max_by_weight = remaining
                take = min(remaining, space_units, max_by_weight if max_by_weight > 0 else remaining)
                if take <= 0:
                    # if can't take due to weight, flush current list to try later
                    if not items:
                        # edge case: can't fit single unit due to unit weight > cap; force to take 1 to avoid infinite loop
                        take = 1
                    else:
                        flush_current()
                        continue
                items.append({
                    "sku": sku,
                    "order_id": order_id,
                    "store": store,
                    "bin": binloc,
                    "bin_rank": binrank,
                    "qty": take,
                    "unit_weight": unit_wt,
                    "cutoff": cutoff,
                    "priority": priority,
                    "fragile": fragile_flag_row
                })
                cur_units += take
                cur_weight += take * unit_wt
                cur_orders.add(order_id)
                cur_bins.add(binloc)
                remaining -= take
        # flush any leftover items
        if items:
            flush_current()

    # process non-fragile by zone
    for zone_name, group in nonfrag_df.groupby(zone_col):
        process_zone_group(group, str(zone_name), fragile_flag=False)
    # process fragile by zone
    for zone_name, group in fragile_df.groupby(zone_col):
        process_zone_group(group, str(zone_name), fragile_flag=True)

    # write summary
    if summary_rows:
        s_df = pd.DataFrame(summary_rows)
        s_df["earliest_cutoff"] = s_df["earliest_cutoff"].apply(lambda x: x.isoformat() if pd.notna(x) else "")
        s_df = s_df.sort_values(by=["earliest_cutoff", "zone", "picklist_no"])
        s_df.to_csv(SUMMARY_CSV, index=False)
        print("Wrote Summary:", SUMMARY_CSV)
    print("Total picklists generated:", total_picklists)

def main():
    if not os.path.exists(INPUT_CSV):
        print("Input file not found:", INPUT_CSV)
        return
    canon = read_and_normalize(INPUT_CSV)
    generate_picklists(canon, max_rows=MAX_ROWS)
    print("Picklists written to directory:", OUTPUT_DIR)
    print("Summary:", SUMMARY_CSV)

if __name__ == "__main__":
    main()
