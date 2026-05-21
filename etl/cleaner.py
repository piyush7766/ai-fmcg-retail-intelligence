"""
etl/cleaner.py
==============
Production-grade data cleaning pipeline for the Amul FMCG Retailer
Intelligence dataset.

Outputs (written to data/processed/):
  - retailers_clean.csv        : fully cleaned master table
  - segmentation_features.csv  : numeric/encoded features for ML
  - nlp_corpus.csv             : NLP-ready text columns

Run:
    python etl/cleaner.py
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"{LOG_DIR}/cleaner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

RAW_PATH    = "data/raw/amul_retailer_data.xlsx"
EXPORT_PATH = "data/processed/"
os.makedirs(EXPORT_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE MAPS
# ─────────────────────────────────────────────────────────────────────────────

ZONE_MAP = {
    "amer / periphery / ajmer road"        : "Amer_Periphery",
    "vaishali nagar / mansarovar"          : "Vaishali_Mansarovar",
    "malviya nagar / jagatpura / sanganer" : "Malviya_Sanganer",
    "jhotwara / murlipura"                 : "Jhotwara_Murlipura",
    "c-scheme / bani park / civil lines"   : "C_Scheme",
    "old city / walled city"               : "Old_City",
}

STORE_TYPE_MAP = {
    "kiryana/general store"      : "Kiryana",
    "kirana/general store"       : "Kiryana",
    "kiryana / general store"    : "Kiryana",
    "supermarket / modern trade" : "Supermarket",
    "supermarket/modern trade"   : "Supermarket",
    "dairy booth / parlour"      : "Dairy_Parlour",
    "dairy booth/parlour"        : "Dairy_Parlour",
    "bakery / convenience"       : "Bakery",
    "bakery/convenience"         : "Bakery",
    "tea stall"                  : "Tea_Stall",
}

YEARS_OP_MAP = {
    "less than 2 years"  : "lt_2yrs",
    "2\u20135 years"     : "2_5yrs",
    "2-5 years"          : "2_5yrs",
    "5\u201310 years"    : "5_10yrs",
    "5-10 years"         : "5_10yrs",
    "more than 10 years" : "gt_10yrs",
}

FRIDGE_MAP = {
    "yes, dedicated dairy refrigerator" : "dedicated",
    "yes, shared fridge"                : "shared",
    "no refrigeration"                  : "none",
    "no"                                : "none",
}

AWARENESS_MAP = {
    "yes, i stock it"             : "stocking",
    "yes, aware but not stocking" : "aware_only",
    "no, not aware at all"        : "unaware",
}

REP_VISIT_MAP = {
    "weekly"                          : "weekly",
    "fortnightly"                     : "fortnightly",
    "monthly"                         : "monthly",
    "rarely (once in 2\u20133 months)": "rarely",
    "rarely (once in 2-3 months)"     : "rarely",
    "rarely"                          : "rarely",
}

PROMO_RECEIVED_MAP = {
    "yes, occasionally" : "yes_occasionally",
    "yes, regularly"    : "yes_regularly",
    "no, never"         : "never",
    "never"             : "never",
}

VOLUME_MAP = {
    "less than 10 packs/week"  : 5,
    "less than 20 packs/week"  : 10,
    "10\u201320 packs/week"    : 15,
    "10-20 packs/week"         : 15,
    "20\u201350 packs/week"    : 35,
    "20-50 packs/week"         : 35,
    "50\u2013100 packs/week"   : 75,
    "50-100 packs/week"        : 75,
    "100\u2013200 packs/week"  : 150,
    "100-200 packs/week"       : 150,
    "200\u2013500 packs/week"  : 350,
    "200-500 packs/week"       : 350,
    "500+ packs/week"          : 600,
}

REP_VISIT_SCORE  = {"weekly": 4, "fortnightly": 3, "monthly": 2, "rarely": 1}
AWARENESS_SCORE  = {"stocking": 3, "aware_only": 1, "unaware": 0}
YEARS_ORD        = {"lt_2yrs": 1, "2_5yrs": 2, "5_10yrs": 3, "gt_10yrs": 4}

NULL_STRINGS = {"nan", "none", "n/a", "na", "-", "", "null", "not applicable"}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_lower(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def _map_with_fallback(series: pd.Series, mapping: dict,
                       fallback: str = "unknown") -> pd.Series:
    def _lookup(val):
        if pd.isna(val):
            return np.nan
        return mapping.get(str(val).strip().lower(), fallback)
    return series.apply(_lookup)


def _log_unmapped(original: pd.Series, cleaned: pd.Series, col: str):
    mask = (cleaned == "unknown") & original.notna()
    if mask.any():
        samples = list(original[mask].unique()[:5])
        log.warning(f"  [{col}] {mask.sum()} unmapped → 'unknown': {samples}")


def _clean_text_field(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
              .str.replace(r"[,]{2,}", ",", regex=True)
              .str.replace(r"[.]{2,}", ".", regex=True)
              .where(series.notna(), other=np.nan)
    )


def _nlp_clean(series: pd.Series) -> pd.Series:
    def _process(val):
        if pd.isna(val):
            return "no_data"
        s = str(val).strip().lower()
        if s in NULL_STRINGS or s == "no_data":
            return "no_data"
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s,./&'()\-]", "", s)
        return s.strip() or "no_data"
    return series.apply(_process)


def _parse_volume(val: str) -> int:
    if not val or val in NULL_STRINGS:
        return 0
    return VOLUME_MAP.get(val.strip().lower(), 0)


def _parse_pack_format(val) -> str:
    s = _safe_lower(val)
    if not s or s in NULL_STRINGS:
        return "unknown"
    has_cup   = "cup" in s or "tub" in s
    has_pouch = "pouch" in s
    if has_cup and has_pouch:
        return "both"
    if has_cup:
        return "cup"
    if has_pouch:
        return "pouch"
    if "no clear preference" in s:
        return "no_preference"
    return "unknown"


def _normalise_brand(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in NULL_STRINGS:
        return np.nan
    fixes = {
        "frubon"      : "FruBon",
        "fruboon"     : "FruBon",
        "fru bon"     : "FruBon",
        "rufill"      : "Rufil",
        "mother dairy": "Mother_Dairy",
        "amul masti"  : "Amul_Masti",
    }
    lower = s.lower()
    for k, v in fixes.items():
        lower = lower.replace(k, v)
    return lower.title()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    log.info(f"Loading raw data → {path}")
    df = pd.read_excel(path, sheet_name=0, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    before = len(df)
    df = df.dropna(how="all")
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} cols "
             f"(dropped {before - len(df)} fully-empty rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — STANDARDISE CATEGORICALS
# ─────────────────────────────────────────────────────────────────────────────

def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── STEP 2: Standardising categoricals ──")

    column_map = {
        "Zone"                                   : ("zone_clean",            ZONE_MAP),
        "Store Type"                             : ("store_type_clean",      STORE_TYPE_MAP),
        "Years Operating"                        : ("years_operating_clean", YEARS_OP_MAP),
        "Has Refrigerator"                       : ("fridge_type",           FRIDGE_MAP),
        "Aware of Amul C&T"                      : ("awareness_tier",        AWARENESS_MAP),
        "Amul Rep Visit Frequency"               : ("rep_visit_clean",       REP_VISIT_MAP),
        "Trade Promotions Received on Amul Dahi" : ("promo_received_clean",  PROMO_RECEIVED_MAP),
    }

    for raw_col, (new_col, mapping) in column_map.items():
        if raw_col in df.columns:
            df[new_col] = _map_with_fallback(df[raw_col], mapping)
            _log_unmapped(df[raw_col], df[new_col], raw_col)
            log.info(f"  ✓ '{raw_col}' → '{new_col}'")
        else:
            log.warning(f"  ⚠ Column not found: '{raw_col}'")

    if "Stocks Amul C&T" in df.columns:
        df["stocks_ct"] = df["Stocks Amul C&T"].apply(
            lambda v: True  if _safe_lower(v) == "yes"
                      else (False if _safe_lower(v) == "no" else np.nan)
        )
        log.info(f"  ✓ stocks_ct: {df['stocks_ct'].sum():.0f} stocking C&T")

    if "Packaging Format Preferred" in df.columns:
        df["pack_format_clean"] = df["Packaging Format Preferred"].apply(_parse_pack_format)
        log.info(f"  ✓ pack_format_clean: "
                 f"{df['pack_format_clean'].value_counts().to_dict()}")

    brand_cols = [
        "Best Trade Promotions Brand", "Best Rep Visits Brand",
        "Best Shelf Life / Freshness Brand", "Biggest 6-Month Growth Brand",
        "#1 Selling Dahi Brand at Store",
        "Branded Display / Fridge Provided By",
        "Preferential Shelf Placement Brand",
    ]
    for col in brand_cols:
        if col in df.columns:
            df[col] = df[col].apply(_normalise_brand)

    log.info("  Categorical standardisation complete.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── STEP 3: Handling missing values ──")

    nlp_raw_cols = [
        "Customer Complaints About Amul Dahi",
        "Customer Feedback on Amul C&T Specifically",
        "Suggested Changes / Improvements for Amul",
        "Other Feedback for Amul",
    ]
    for col in nlp_raw_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna("no_data")
            log.info(f"  NLP fill '{col[:45]}': {n} → 'no_data'")

    categorical_defaults = {
        "awareness_tier"       : "unaware",
        "rep_visit_clean"      : "rarely",
        "promo_received_clean" : "never",
        "pack_format_clean"    : "unknown",
        "store_type_clean"     : "unknown",
        "fridge_type"          : "unknown",
        "zone_clean"           : "unknown",
    }
    for col, default in categorical_defaults.items():
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].fillna(default)
            if before:
                log.info(f"  Default fill '{col}': {before} → '{default}'")

    if "Data Source" in df.columns:
        df["Data Source"] = df["Data Source"].fillna("Unknown")

    log.info("  Missing value handling complete.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CLEAN NAMES & IDENTIFIERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── STEP 4: Cleaning names and identifiers ──")

    text_id_cols = ["Shop Name", "Contact / Owner",
                    "Area / Locality", "ADA / Distributor"]

    for col in text_id_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(
            lambda v: np.nan if _safe_lower(v) in NULL_STRINGS else v
        )
        df[col] = _clean_text_field(df[col]).apply(
            lambda v: v.title() if pd.notna(v) else np.nan
        )
        log.info(f"  ✓ '{col}' cleaned and title-cased")

    if "WhatsApp Number" in df.columns:
        df["phone_clean"] = (
            df["WhatsApp Number"].astype(str)
              .str.replace(r"\D", "", regex=True)
              .str.lstrip("0")
        )
        df["phone_clean"] = df["phone_clean"].apply(
            lambda v: v if (v and v.lower() != "nan") else np.nan
        )
        df["phone_valid"] = df["phone_clean"].apply(
            lambda v: (len(str(v)) == 10) if pd.notna(v) else False
        )
        invalid = (~df["phone_valid"] & df["phone_clean"].notna()).sum()
        log.info(f"  Phone: {df['phone_valid'].sum()} valid | {invalid} invalid")

    log.info("  Name cleaning complete.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── STEP 5: Engineering analytical features ──")

    # 5a — Refrigerator flags
    if "fridge_type" in df.columns:
        df["feat_refrigerator_flag"] = df["fridge_type"].apply(
            lambda v: 1 if v in ("dedicated", "shared") else 0
        )
        df["feat_dedicated_fridge"] = df["fridge_type"].apply(
            lambda v: 1 if v == "dedicated" else 0
        )
        log.info(f"  feat_refrigerator_flag: "
                 f"{df['feat_refrigerator_flag'].sum()} / {len(df)} refrigerated")

    # 5b — Awareness score
    if "awareness_tier" in df.columns:
        df["feat_awareness_score"] = (
            df["awareness_tier"].map(AWARENESS_SCORE).fillna(0).astype(int)
        )

    # 5c — Rep visit score
    if "rep_visit_clean" in df.columns:
        df["feat_rep_visit_score"] = (
            df["rep_visit_clean"].map(REP_VISIT_SCORE).fillna(1).astype(int)
        )

    # 5d — Promo bonus
    if "promo_received_clean" in df.columns:
        df["feat_promo_bonus"] = df["promo_received_clean"].apply(
            lambda v: 2 if str(v).startswith("yes") else 0
        )
    else:
        df["feat_promo_bonus"] = 0

    # 5e — Engagement score
    if "feat_rep_visit_score" in df.columns:
        df["feat_engagement_score"] = (
            df["feat_rep_visit_score"] + df["feat_promo_bonus"]
        )

    # 5f — Infrastructure score
    if "fridge_type" in df.columns:
        df["feat_infrastructure_score"] = (
            df["fridge_type"]
            .map({"dedicated": 2, "shared": 1, "none": 0, "unknown": 0})
            .fillna(0).astype(int)
        )

    # 5g — Competitor presence flag
    rival_brands = ["frubon", "saras", "mother dairy", "ksheer", "rufil"]
    comp_signal_cols = [c for c in [
        "Best Trade Promotions Brand", "Best Rep Visits Brand",
        "#1 Selling Dahi Brand at Store",
    ] if c in df.columns]

    def _has_rival(row) -> int:
        for col in comp_signal_cols:
            val = _safe_lower(row.get(col, ""))
            if any(b in val for b in rival_brands):
                return 1
        return 0

    df["feat_competitor_presence"] = df.apply(_has_rival, axis=1)
    log.info(f"  feat_competitor_presence: "
             f"{df['feat_competitor_presence'].sum()} retailers with rival signals")

    # 5h — FruBon threat score
    def _frubon_threat(row) -> int:
        score = 0
        if "Best Trade Promotions Brand" in df.columns:
            if "frubon" in _safe_lower(row.get("Best Trade Promotions Brand", "")):
                score += 2
        if "Best Rep Visits Brand" in df.columns:
            if "frubon" in _safe_lower(row.get("Best Rep Visits Brand", "")):
                score += 1
        return score

    df["feat_frubon_threat_score"] = df.apply(_frubon_threat, axis=1)

    # 5i — Weekly volume (numeric)
    vol_pairs = [
        ("C&T Estimated Packs / Week",          "feat_ct_weekly_volume"),
        ("Amul Masti Dahi Packs / Week (est.)", "feat_masti_weekly_volume"),
    ]
    for raw_col, feat_col in vol_pairs:
        if raw_col in df.columns:
            df[feat_col] = df[raw_col].apply(
                lambda v: _parse_volume(_safe_lower(str(v)))
            )
            log.info(f"  {feat_col}: mean={df[feat_col].mean():.0f}  "
                     f"max={df[feat_col].max()}")

    # 5j — Years operating ordinal
    if "years_operating_clean" in df.columns:
        df["feat_years_operating_num"] = (
            df["years_operating_clean"].map(YEARS_ORD).fillna(0).astype(int)
        )

    # 5k — Adoption likelihood score (0–10 composite)
    components = {
        "feat_awareness_score"      : (0.35, 3),
        "feat_infrastructure_score" : (0.30, 2),
        "feat_engagement_score"     : (0.20, 6),
        "feat_years_operating_num"  : (0.15, 4),
    }
    df["feat_adoption_likelihood"] = 0.0
    for col, (weight, max_val) in components.items():
        if col in df.columns:
            normalised = df[col].clip(0, max_val) / max_val
            df["feat_adoption_likelihood"] += normalised * weight * 10
    df["feat_adoption_likelihood"] = df["feat_adoption_likelihood"].round(2)
    log.info(f"  feat_adoption_likelihood: "
             f"mean={df['feat_adoption_likelihood'].mean():.2f}  "
             f"max={df['feat_adoption_likelihood'].max():.2f}")

    log.info("  Feature engineering complete.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — NLP-READY TEXT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

def build_nlp_columns(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── STEP 6: Building NLP-ready text columns ──")

    nlp_source_cols = {
        "Customer Complaints About Amul Dahi"        : "nlp_complaints",
        "Customer Feedback on Amul C&T Specifically" : "nlp_ct_feedback",
        "Suggested Changes / Improvements for Amul"  : "nlp_suggestions",
        "Other Feedback for Amul"                    : "nlp_other_feedback",
        "Why Not Stocking C&T"                       : "nlp_why_not_stocking",
    }

    for src_col, nlp_col in nlp_source_cols.items():
        if src_col in df.columns:
            df[nlp_col] = _nlp_clean(df[src_col])
            non_empty = (df[nlp_col] != "no_data").sum()
            log.info(f"  ✓ {nlp_col}: {non_empty} non-empty records")

    nlp_cols = [c for c in df.columns if c.startswith("nlp_")]
    if nlp_cols:
        df["nlp_full_corpus"] = df[nlp_cols].apply(
            lambda row: " | ".join(
                v for v in row if pd.notna(v) and v != "no_data"
            ),
            axis=1,
        )
        df["nlp_full_corpus"] = df["nlp_full_corpus"].replace("", "no_data")
        log.info(f"  ✓ nlp_full_corpus assembled from {len(nlp_cols)} columns")

    log.info("  NLP column construction complete.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── STEP 7: Deduplication ──")

    before = len(df)
    df = df.drop_duplicates()
    exact_removed = before - len(df)
    log.info(f"  Exact duplicates removed: {exact_removed}")

    near_dup_cols = [c for c in ["Shop Name", "phone_clean", "Area / Locality"]
                     if c in df.columns]
    if near_dup_cols:
        before = len(df)
        df = df.drop_duplicates(subset=near_dup_cols, keep="first")
        near_removed = before - len(df)
        log.info(f"  Near-duplicates (shop+phone+area) removed: {near_removed}")

    log.info(f"  Final row count: {len(df)}\n")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_outputs(df: pd.DataFrame) -> None:
    log.info("── STEP 8: Exporting outputs ──")

    # 1) Master cleaned table
    master_path = os.path.join(EXPORT_PATH, "retailers_clean.csv")
    df.to_csv(master_path, index=False)
    log.info(f"  ✓ retailers_clean.csv → {len(df)} rows × {df.shape[1]} cols")

    # 2) Segmentation features (numeric / encoded only)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    id_cols = [c for c in ["Sr. No.", "Shop Name", "zone_clean",
                           "store_type_clean"] if c in df.columns]
    seg_df = df[id_cols + feat_cols].copy()
    seg_path = os.path.join(EXPORT_PATH, "segmentation_features.csv")
    seg_df.to_csv(seg_path, index=False)
    log.info(f"  ✓ segmentation_features.csv → "
             f"{len(seg_df)} rows × {len(feat_cols)} features")

    # 3) NLP corpus
    nlp_cols = [c for c in df.columns if c.startswith("nlp_")]
    nlp_id_cols = [c for c in ["Sr. No.", "Shop Name", "zone_clean"]
                   if c in df.columns]
    nlp_df = df[nlp_id_cols + nlp_cols].copy()
    nlp_path = os.path.join(EXPORT_PATH, "nlp_corpus.csv")
    nlp_df.to_csv(nlp_path, index=False)
    log.info(f"  ✓ nlp_corpus.csv → {len(nlp_df)} rows × {len(nlp_cols)} text cols")

    log.info("  Export complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    log.info("=" * 70)
    log.info("FMCG RETAILER INTELLIGENCE — CLEANING PIPELINE")
    log.info("=" * 70)

    df = load_raw()
    df = standardise_categoricals(df)
    df = handle_missing(df)
    df = clean_names(df)
    df = engineer_features(df)
    df = build_nlp_columns(df)
    df = remove_duplicates(df)
    export_outputs(df)

    log.info("=" * 70)
    log.info(f"✅  PIPELINE COMPLETE — {len(df)} retailers processed")
    log.info("=" * 70)
    return df


if __name__ == "__main__":
    run_pipeline()
