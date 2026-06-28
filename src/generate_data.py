"""
NovaMart Synthetic Data Generator
Generates realistic operational data for 25 NovaMart retail stores across 12 weeks.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
# SEED FOR REPRODUCIBILITY
# ─────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STORE DEFINITIONS
# ─────────────────────────────────────────────
STORES = [
    # (store_id, city, state, region, sq_ft)
    ("S001", "Los Angeles",     "CA", "West",      14000),
    ("S002", "San Francisco",   "CA", "West",      11000),
    ("S003", "Seattle",         "WA", "West",      12500),
    ("S004", "Portland",        "OR", "West",      10500),
    ("S005", "San Diego",       "CA", "West",      13000),
    ("S006", "Phoenix",         "AZ", "Southwest", 13500),
    ("S007", "Las Vegas",       "NV", "Southwest", 12000),
    ("S008", "Denver",          "CO", "Southwest", 11500),
    ("S009", "Albuquerque",     "NM", "Southwest",  9500),
    ("S010", "Tucson",          "AZ", "Southwest",  9000),
    ("S011", "Chicago",         "IL", "Midwest",   13000),
    ("S012", "Minneapolis",     "MN", "Midwest",   11000),
    ("S013", "Detroit",         "MI", "Midwest",   10000),
    ("S014", "Columbus",        "OH", "Midwest",    9500),
    ("S015", "Kansas City",     "MO", "Midwest",    9000),
    ("S016", "New York",        "NY", "Northeast", 15000),
    ("S017", "Boston",          "MA", "Northeast", 11500),
    ("S018", "Philadelphia",    "PA", "Northeast", 10500),  # chronic underperformer
    ("S019", "Washington DC",   "DC", "Northeast", 12000),
    ("S020", "Baltimore",       "MD", "Northeast",  9500),
    ("S021", "Atlanta",         "GA", "Southeast", 12500),
    ("S022", "Miami",           "FL", "Southeast", 13000),  # chronic underperformer
    ("S023", "Charlotte",       "NC", "Southeast", 10000),
    ("S024", "Nashville",       "TN", "Southeast", 10500),
    ("S025", "Orlando",         "FL", "Southeast", 11500),
]

MANAGERS = [
    "James Ortega", "Priya Nair", "Marcus Webb", "Linda Chen", "Derek Simmons",
    "Aisha Patel", "Connor Walsh", "Fatima Hassan", "Ryan Kowalski", "Elena Rossi",
    "Trevor Banks", "Nadia Osei", "Justin Marsh", "Carmen Vega", "Patrick Liu",
    "Jasmine Ford", "Samuel Tran", "Nicole Russo", "Andre Mitchell", "Sonia Park",
    "Blake Turner", "Yara Khalil", "Ethan Brooks", "Monique Davis", "Chris Huang"
]

# Region performance multipliers (West outperforms Midwest by ~15%)
REGION_MULTIPLIER = {
    "West":      1.10,
    "Southwest": 1.00,
    "Midwest":   0.92,
    "Northeast": 1.08,
    "Southeast": 0.97,
}

# Stores with specific problems
UNDERPERFORMERS     = {"S018", "S022"}   # chronic low revenue + high shrinkage
LABOR_OVERSPEND     = {"S003", "S011", "S016"}  # labor cost consistently over target
BAD_NPS_GOOD_REV    = {"S001", "S007"}   # high revenue but terrible NPS

# ─────────────────────────────────────────────
# WEEK DEFINITIONS (12 weeks starting Jan 2024)
# ─────────────────────────────────────────────
START_DATE = datetime(2024, 1, 1)
WEEKS = [f"2024-W{str(i).zfill(2)}" for i in range(1, 13)]
WEEK_DATES = [(START_DATE + timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(12)]

# Week multipliers — weeks 9-10 are sale event spike
WEEK_MULTIPLIER = {
    "2024-W01": 1.00, "2024-W02": 0.97, "2024-W03": 0.99, "2024-W04": 1.02,
    "2024-W05": 1.01, "2024-W06": 0.98, "2024-W07": 1.03, "2024-W08": 1.05,
    "2024-W09": 1.22, "2024-W10": 1.18, "2024-W11": 1.04, "2024-W12": 1.06,
}


# ─────────────────────────────────────────────
# 1. STORE MASTER
# ─────────────────────────────────────────────
def generate_store_master():
    rows = []
    open_dates = pd.date_range("2017-01-01", "2022-12-31", periods=25)

    for i, (sid, city, state, region, sqft) in enumerate(STORES):
        rows.append({
            "store_id":       sid,
            "store_name":     f"NovaMart {city}",
            "city":           city,
            "state":          state,
            "region":         region,
            "square_footage": sqft,
            "opened_date":    open_dates[i].strftime("%Y-%m-%d"),
            "store_manager":  MANAGERS[i],
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/store_master.csv", index=False)
    print(f"✅ store_master.csv — {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# 2. WEEKLY SALES DATA
# ─────────────────────────────────────────────
def generate_sales_data():
    rows = []

    for sid, city, state, region, sqft in STORES:
        # Base weekly revenue target scaled to store size
        base_target = sqft * 7.5  # ~$7.50 revenue per sq ft per week

        # Apply region multiplier
        region_mult = REGION_MULTIPLIER[region]

        # Underperformers run 25% below target
        perf_mult = 0.75 if sid in UNDERPERFORMERS else 1.0

        prev_revenue = None

        for week in WEEKS:
            week_mult = WEEK_MULTIPLIER[week]
            target = round(base_target * region_mult * week_mult, 2)

            # Actual revenue: add noise, underperformers consistently miss
            noise = np.random.normal(0, 0.04)
            actual_mult = perf_mult + noise
            actual = round(target * actual_mult, 2)

            # Transactions and basket size
            avg_basket = round(np.random.uniform(38, 62), 2)
            transactions = int(actual / avg_basket)

            # Foot traffic: conversion rate 20-45%
            if sid in UNDERPERFORMERS:
                conversion_rate = round(np.random.uniform(18, 27), 1)
            else:
                conversion_rate = round(np.random.uniform(28, 44), 1)

            foot_traffic = int(transactions / (conversion_rate / 100))

            # Same store sales growth vs same week "last year" (simulated)
            if prev_revenue is None:
                sss_growth = round(np.random.uniform(-3, 5), 1)
            else:
                sss_growth = round(((actual - prev_revenue) / prev_revenue) * 100, 1)

            prev_revenue = actual

            rows.append({
                "store_id":               sid,
                "week":                   week,
                "week_start_date":        WEEK_DATES[WEEKS.index(week)],
                "revenue_actual":         actual,
                "revenue_target":         round(target, 2),
                "revenue_variance_pct":   round(((actual - target) / target) * 100, 1),
                "transactions":           transactions,
                "avg_basket_size":        avg_basket,
                "foot_traffic":           foot_traffic,
                "conversion_rate":        conversion_rate,
                "same_store_sales_growth": sss_growth,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/weekly_sales.csv", index=False)
    print(f"✅ weekly_sales.csv — {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# 3. INVENTORY DATA
# ─────────────────────────────────────────────
def generate_inventory_data():
    rows = []

    for sid, city, state, region, sqft in STORES:
        for week in WEEKS:
            week_mult = WEEK_MULTIPLIER[week]

            # Underperformers have higher shrinkage
            if sid in UNDERPERFORMERS:
                shrinkage = round(np.random.uniform(2.8, 4.5), 2)
                stockouts  = int(np.random.uniform(18, 35))
                turnover   = round(np.random.uniform(2.5, 3.5), 1)
            else:
                shrinkage = round(np.random.uniform(0.8, 2.0), 2)
                stockouts  = int(np.random.uniform(3, 15))
                turnover   = round(np.random.uniform(3.8, 6.2), 1)

            # Sale weeks drive more stockouts
            if week in ("2024-W09", "2024-W10"):
                stockouts = int(stockouts * 1.6)

            overstock_value   = round(np.random.uniform(3000, 12000), 2)
            days_supply       = int(np.random.uniform(14, 35))
            receiving_accuracy = round(np.random.uniform(94.0, 99.5), 1)

            rows.append({
                "store_id":           sid,
                "week":               week,
                "inventory_turnover": turnover,
                "shrinkage_rate":     shrinkage,
                "stockout_incidents": stockouts,
                "overstock_value":    overstock_value,
                "days_supply_on_hand": days_supply,
                "receiving_accuracy": receiving_accuracy,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/inventory.csv", index=False)
    print(f"✅ inventory.csv — {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# 4. LABOR DATA
# ─────────────────────────────────────────────
def generate_labor_data():
    rows = []

    for sid, city, state, region, sqft in STORES:
        base_headcount = max(10, int(sqft / 700))

        for week in WEEKS:
            week_mult = WEEK_MULTIPLIER[week]

            headcount        = base_headcount + np.random.randint(-1, 2)
            hours_scheduled  = headcount * np.random.randint(38, 43)

            # Labor overspenders run 15-25% over on actual hours
            if sid in LABOR_OVERSPEND:
                hours_worked = int(hours_scheduled * np.random.uniform(1.15, 1.25))
            else:
                hours_worked = int(hours_scheduled * np.random.uniform(0.97, 1.05))

            # Labor cost target: ~$22/hr blended rate
            labor_cost_target = round(hours_scheduled * 22.0, 2)
            labor_cost_actual = round(hours_worked    * np.random.uniform(21.5, 24.0), 2)

            # Revenue needed for sales_per_labor_hour — approximate from sales table
            approx_revenue    = sqft * 7.5 * REGION_MULTIPLIER[region] * week_mult
            sales_per_lh      = round(approx_revenue / hours_worked, 2)

            turnover_rate     = round(np.random.uniform(1.5, 5.5), 1)

            rows.append({
                "store_id":           sid,
                "week":               week,
                "headcount":          int(headcount),
                "hours_scheduled":    hours_scheduled,
                "hours_worked":       hours_worked,
                "labor_cost_target":  labor_cost_target,
                "labor_cost_actual":  labor_cost_actual,
                "labor_variance_pct": round(((labor_cost_actual - labor_cost_target) / labor_cost_target) * 100, 1),
                "sales_per_labor_hour": sales_per_lh,
                "turnover_rate":      turnover_rate,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/labor.csv", index=False)
    print(f"✅ labor.csv — {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# 5. CUSTOMER DATA
# ─────────────────────────────────────────────
def generate_customer_data():
    rows = []

    for sid, city, state, region, sqft in STORES:
        for week in WEEKS:

            # High revenue but terrible NPS stores
            if sid in BAD_NPS_GOOD_REV:
                nps = int(np.random.uniform(-10, 15))
                complaints = int(np.random.uniform(18, 30))
            elif sid in UNDERPERFORMERS:
                nps = int(np.random.uniform(5, 25))
                complaints = int(np.random.uniform(10, 20))
            else:
                nps = int(np.random.uniform(30, 65))
                complaints = int(np.random.uniform(2, 10))

            return_rate         = round(np.random.uniform(5.0, 14.0), 1)
            loyalty_active      = int(np.random.uniform(300, 900))
            new_loyalty_signups = int(np.random.uniform(15, 60))

            rows.append({
                "store_id":             sid,
                "week":                 week,
                "nps_score":            nps,
                "return_rate":          return_rate,
                "loyalty_members_active": loyalty_active,
                "new_loyalty_signups":  new_loyalty_signups,
                "complaints_logged":    complaints,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/customer.csv", index=False)
    print(f"✅ customer.csv — {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🏪 NovaMart Synthetic Data Generator\n" + "─" * 40)
    generate_store_master()
    generate_sales_data()
    generate_inventory_data()
    generate_labor_data()
    generate_customer_data()
    print("\n✅ All data written to /data folder")
    print("   Files: store_master.csv, weekly_sales.csv, inventory.csv, labor.csv, customer.csv")