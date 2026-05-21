"""
FMCG Retailer Intelligence — Dataset Profiling Script
Covers: column inspection, missing values, categorical detection,
        NLP column flagging, summary stats, duplicates, domain mapping,
        sample values, and CSV exports.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "data/raw/amul_retailer_data.xlsx"
EXPORT_PATH = "data/processed/"
os.makedirs(EXPORT_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

df = pd.read_excel(DATA_PATH, sheet_name=0)

print(f"✅  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"    File: {DATA_PATH}\n")

# ─────────────────────────────────────────────
# 2. COLUMN INVENTORY
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 2: COLUMN INVENTORY")
print("=" * 70)

col_info = pd.DataFrame({
    "Column"   : df.columns,
    "Dtype"    : df.dtypes.values,
    "Non-Null" : df.notna().sum().values,
    "Null"     : df.isna().sum().values,
    "Null_%"   : (df.isna().mean() * 100).round(1).values,
    "Unique"   : [df[c].nunique() for c in df.columns],
})

print(col_info.to_string(index=False))
col_info.to_csv(f"{EXPORT_PATH}column_inventory.csv", index=False)
print(f"\n💾  Saved → {EXPORT_PATH}column_inventory.csv\n")

# ─────────────────────────────────────────────
# 3. MISSING VALUE HEATMAP (TEXT)
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 3: MISSING VALUE SUMMARY (columns with any nulls)")
print("=" * 70)

missing = col_info[col_info["Null"] > 0][["Column", "Null", "Null_%"]]
if missing.empty:
    print("✅  No missing values found.")
else:
    print(missing.to_string(index=False))
missing.to_csv(f"{EXPORT_PATH}missing_values.csv", index=False)
print()

# ─────────────────────────────────────────────
# 4. DOMAIN MAPPING
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 4: BUSINESS DOMAIN MAPPING")
print("=" * 70)

DOMAIN_MAP = {
    "Domain A — Retailer Metadata": [
        "Sr. No.", "Data Source", "Shop Name", "Contact / Owner",
        "WhatsApp Number", "Area / Locality", "Zone", "Store Type",
        "Years Operating", "ADA / Distributor", "Timestamp"
    ],
    "Domain B — Infrastructure": [
        "Has Refrigerator", "Branded Display / Fridge Provided By",
        "Preferential Shelf Placement Brand"
    ],
    "Domain C — Product Penetration": [
        "Brands Stocked", "Stocks Amul C&T", "Aware of Amul C&T",
        "Why Not Stocking C&T", "C&T SKU Mix (80g / 180g / 850g)",
        "C&T Estimated Packs / Week"
    ],
    "Domain D — Demand & Sales": [
        "Amul Masti Dahi Packs / Week (est.)", "Pack Sizes Most in Demand",
        "Packaging Format Preferred", "#1 Selling Dahi Brand at Store",
        "Brand Ranking (brands stocked, by volume)",
        "Amul Dahi Sales Change vs 1 Year Ago",
        "Customers Ask for Amul Dahi by Name",
        "Customers Ask for Other Brand by Name"
    ],
    "Domain E — Competitor Intelligence": [
        "Primary Reason Customers Choose Competitor",
        "Best Trade Promotions Brand", "Best Rep Visits Brand",
        "Best Shelf Life / Freshness Brand", "Biggest 6-Month Growth Brand"
    ],
    "Domain F — Retailer Engagement": [
        "Margin on Amul Dahi (from price list)",
        "Trade Promotions Received on Amul Dahi",
        "Amul Rep Visit Frequency"
    ],
    "Domain G — NLP / Text Feedback": [
        "Customer Complaints About Amul Dahi",
        "Customer Feedback on Amul C&T Specifically",
        "Suggested Changes / Improvements for Amul",
        "Other Feedback for Amul"
    ],
}

for domain, cols in DOMAIN_MAP.items():
    present = [c for c in cols if c in df.columns]
    missing_cols = [c for c in cols if c not in df.columns]
    print(f"\n📂 {domain}")
    print(f"   ✅ Present ({len(present)}): {present}")
    if missing_cols:
        print(f"   ⚠️  Not found ({len(missing_cols)}): {missing_cols}")

print()

# ─────────────────────────────────────────────
# 5. CATEGORICAL COLUMNS ANALYSIS
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 5: CATEGORICAL COLUMNS (unique < 30 or dtype == object, short text)")
print("=" * 70)

CATEGORICAL_COLS = [
    "Zone", "Store Type", "Years Operating", "Has Refrigerator",
    "Stocks Amul C&T", "Aware of Amul C&T",
    "C&T Estimated Packs / Week", "Amul Masti Dahi Packs / Week (est.)",
    "Pack Sizes Most in Demand", "Packaging Format Preferred",
    "#1 Selling Dahi Brand at Store",
    "Amul Dahi Sales Change vs 1 Year Ago",
    "Customers Ask for Amul Dahi by Name",
    "Customers Ask for Other Brand by Name",
    "Margin on Amul Dahi (from price list)",
    "Best Trade Promotions Brand", "Best Rep Visits Brand",
    "Best Shelf Life / Freshness Brand", "Biggest 6-Month Growth Brand",
    "Trade Promotions Received on Amul Dahi",
    "Amul Rep Visit Frequency", "Data Source",
    "Branded Display / Fridge Provided By",
    "Preferential Shelf Placement Brand",
    "Primary Reason Customers Choose Competitor",
]

cat_summary = []
for col in CATEGORICAL_COLS:
    if col not in df.columns:
        continue
    vc = df[col].value_counts(dropna=False)
    top3 = " | ".join([f"{v}({c})" for v, c in vc.head(3).items()])
    cat_summary.append({
        "Column"  : col,
        "Uniques" : df[col].nunique(),
        "Null_%"  : round(df[col].isna().mean() * 100, 1),
        "Top 3 Values (count)" : top3
    })

cat_df = pd.DataFrame(cat_summary)
print(cat_df.to_string(index=False))
cat_df.to_csv(f"{EXPORT_PATH}categorical_summary.csv", index=False)
print(f"\n💾  Saved → {EXPORT_PATH}categorical_summary.csv\n")

# ─────────────────────────────────────────────
# 6. NLP / TEXT COLUMNS ANALYSIS
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 6: NLP / TEXT COLUMNS ANALYSIS")
print("=" * 70)

NLP_COLS = [
    "Customer Complaints About Amul Dahi",
    "Customer Feedback on Amul C&T Specifically",
    "Suggested Changes / Improvements for Amul",
    "Other Feedback for Amul",
    "Why Not Stocking C&T",
    "Brands Stocked",
    "Brand Ranking (brands stocked, by volume)",
    "C&T SKU Mix (80g / 180g / 850g)",
]

nlp_summary = []
for col in NLP_COLS:
    if col not in df.columns:
        continue
    non_null = df[col].dropna()
    avg_len  = int(non_null.astype(str).str.len().mean()) if len(non_null) else 0
    sample   = non_null.iloc[0] if len(non_null) else "N/A"
    nlp_summary.append({
        "Column"     : col,
        "Non-Null"   : len(non_null),
        "Null_%"     : round(df[col].isna().mean() * 100, 1),
        "Avg Chars"  : avg_len,
        "Sample Value": str(sample)[:80]
    })
    print(f"\n📝 {col}")
    print(f"   Non-null: {len(non_null)} | Avg length: {avg_len} chars")
    print(f"   Sample  : {str(sample)[:100]}")

pd.DataFrame(nlp_summary).to_csv(
    f"{EXPORT_PATH}nlp_column_summary.csv", index=False)
print(f"\n💾  Saved → {EXPORT_PATH}nlp_column_summary.csv\n")

# ─────────────────────────────────────────────
# 7. NUMERICAL / KPI SUMMARY
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 7: NUMERICAL SUMMARY STATISTICS")
print("=" * 70)

num_df = df.select_dtypes(include=[np.number])
if num_df.empty:
    print("ℹ️  No purely numeric columns found (most are encoded as text).")
else:
    print(num_df.describe().round(2).to_string())
    num_df.describe().round(2).to_csv(f"{EXPORT_PATH}numeric_summary.csv")
    print(f"\n💾  Saved → {EXPORT_PATH}numeric_summary.csv")
print()

# ─────────────────────────────────────────────
# 8. CORE BUSINESS KPIs
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 8: CORE BUSINESS KPIs (COMPUTED)")
print("=" * 70)

total = len(df)

# C&T Penetration
if "Stocks Amul C&T" in df.columns:
    ct_stockers = df["Stocks Amul C&T"].astype(str).str.strip().str.lower() == "yes"
    print(f"📊 C&T Penetration Rate   : {ct_stockers.sum():>4} / {total}  "
          f"= {ct_stockers.mean()*100:.1f}%")

# Awareness breakdown
if "Aware of Amul C&T" in df.columns:
    aware_counts = df["Aware of Amul C&T"].value_counts(dropna=False)
    for k, v in aware_counts.items():
        print(f"   Aware tier [{str(k)[:40]:<40}]: {v:>4}  ({v/total*100:.1f}%)")

# Refrigeration
if "Has Refrigerator" in df.columns:
    has_fridge = df["Has Refrigerator"].astype(str).str.lower().str.startswith("yes")
    print(f"\n🧊 Retailers with refrigerator: {has_fridge.sum()} / {total} "
          f"= {has_fridge.mean()*100:.1f}%")

# Rep visit frequency
if "Amul Rep Visit Frequency" in df.columns:
    print(f"\n🚗 Rep Visit Frequency Distribution:")
    rvf = df["Amul Rep Visit Frequency"].value_counts(dropna=False)
    for k, v in rvf.items():
        print(f"   {str(k):<45}: {v:>4}  ({v/total*100:.1f}%)")

# Trade promotions
if "Trade Promotions Received on Amul Dahi" in df.columns:
    promo_yes = df["Trade Promotions Received on Amul Dahi"].astype(str)\
                  .str.lower().str.contains("yes")
    print(f"\n🎁 Trade Promo Reach      : {promo_yes.sum():>4} / {total}  "
          f"= {promo_yes.mean()*100:.1f}%")

# Top competitor
if "#1 Selling Dahi Brand at Store" in df.columns:
    print(f"\n🏆 #1 Selling Brand Distribution:")
    top_brands = df["#1 Selling Dahi Brand at Store"].value_counts(dropna=False).head(6)
    for k, v in top_brands.items():
        print(f"   {str(k):<45}: {v:>4}  ({v/total*100:.1f}%)")

print()

# ─────────────────────────────────────────────
# 9. DUPLICATE DETECTION
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 9: DUPLICATE DETECTION")
print("=" * 70)

exact_dups = df.duplicated().sum()
print(f"Exact duplicate rows : {exact_dups}")

# Near-duplicate check on shop name + phone
id_cols = [c for c in ["Shop Name", "WhatsApp Number", "Area / Locality"] 
           if c in df.columns]
if id_cols:
    near_dups = df.duplicated(subset=id_cols, keep=False)
    nd_df = df[near_dups][id_cols + ["Zone"]].sort_values(id_cols)
    print(f"Near-duplicate rows (same shop+phone+area): {near_dups.sum()}")
    if near_dups.sum() > 0:
        print(nd_df.head(10).to_string(index=True))
        nd_df.to_csv(f"{EXPORT_PATH}near_duplicates.csv", index=True)
        print(f"💾  Saved → {EXPORT_PATH}near_duplicates.csv")
print()

# ─────────────────────────────────────────────
# 10. SAMPLE VALUES PER COLUMN
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 10: SAMPLE VALUES PER COLUMN (first 3 non-null)")
print("=" * 70)

samples = []
for col in df.columns:
    non_null_vals = df[col].dropna().unique()[:3]
    sample_str = " | ".join([str(v)[:60] for v in non_null_vals])
    samples.append({"Column": col, "Sample Values": sample_str})
    print(f"  {col:<50}: {sample_str}")

pd.DataFrame(samples).to_csv(f"{EXPORT_PATH}sample_values.csv", index=False)
print(f"\n💾  Saved → {EXPORT_PATH}sample_values.csv\n")

# ─────────────────────────────────────────────
# 11. ZONE-LEVEL SUMMARY
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 11: ZONE-LEVEL PENETRATION SUMMARY")
print("=" * 70)

if "Zone" in df.columns and "Stocks Amul C&T" in df.columns:
    df["_stocks_ct"] = df["Stocks Amul C&T"].astype(str).str.strip().str.lower() == "yes"
    zone_summary = df.groupby("Zone").agg(
        Total_Retailers = ("Zone", "count"),
        CT_Stockers     = ("_stocks_ct", "sum"),
    ).reset_index()
    zone_summary["Penetration_%"] = (
        zone_summary["CT_Stockers"] / zone_summary["Total_Retailers"] * 100
    ).round(1)
    zone_summary = zone_summary.sort_values("Penetration_%", ascending=False)
    print(zone_summary.to_string(index=False))
    zone_summary.to_csv(f"{EXPORT_PATH}zone_penetration.csv", index=False)
    print(f"\n💾  Saved → {EXPORT_PATH}zone_penetration.csv")
    df.drop(columns=["_stocks_ct"], inplace=True)

print()

# ─────────────────────────────────────────────
# EXPORT SUMMARY MANIFEST
# ─────────────────────────────────────────────
print("=" * 70)
print("✅  PROFILING COMPLETE — Exported Files:")
print("=" * 70)
exports = [
    "column_inventory.csv", "missing_values.csv", "categorical_summary.csv",
    "nlp_column_summary.csv", "numeric_summary.csv", "near_duplicates.csv",
    "sample_values.csv", "zone_penetration.csv"
]
for f in exports:
    path = f"{EXPORT_PATH}{f}"
    exists = "✅" if os.path.exists(path) else "⚠️  (not generated)"
    print(f"  {exists}  {path}")
print()
