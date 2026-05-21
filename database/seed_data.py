# seed_data.py
# Production-grade data seeding script for the FMCG Retail Intelligence Platform

import os
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data/processed"))

DB_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', '')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'fmcg_retail_intelligence')}"
)

ZONE_MAP: dict = {}
STORE_TYPE_MAP: dict = {}
BRAND_MAP: dict = {}
DISTRIBUTOR_MAP: dict = {}
RETAILER_MAP: dict = {}
NLP_TOPIC_MAP: dict = {}
SEGMENT_MAP: dict = {}


# ─────────────────────────────────────────────
# ENGINE + SESSION
# ─────────────────────────────────────────────

def get_engine():
    engine = create_engine(DB_URL, pool_pre_ping=True, echo=False)
    log.info("Database engine created: %s", DB_URL.split("@")[-1])
    return engine


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_str(val, default=None):
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "no_data", "na"):
        return default
    return s


def safe_float(val, default=None):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default=None):
    try:
        if pd.isna(val):
            return default
        return int(float(val))
    except (TypeError, ValueError):
        return default


def safe_bool(val):
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return str(val).strip().lower() in ("true", "1", "yes", "t")


def _resolve_brand_id(raw: str):
    if not raw:
        return None
    r = raw.strip().upper()
    for code, bid in BRAND_MAP.items():
        if code.upper() in r or r in code.upper():
            return bid
    return None


def _sales_trend(raw: str):
    if not raw:
        return None
    r = raw.lower()
    if "significantly" in r and "increas" in r:
        return "increased_significantly"
    if "slightly" in r and "increas" in r:
        return "increased_slightly"
    if "stayed" in r or "same" in r:
        return "stayed_same"
    if "significantly" in r and "decreas" in r:
        return "decreased_significantly"
    if "slightly" in r and "decreas" in r:
        return "decreased_slightly"
    return None


def _pack_format(raw: str):
    if not raw:
        return None
    r = raw.lower()
    if "both" in r:
        return "both"
    if "cup" in r or "tub" in r:
        return "cup"
    if "pouch" in r:
        return "pouch"
    if "no clear" in r or "no_preference" in r:
        return "no_preference"
    return "other"


# ─────────────────────────────────────────────
# STEP 1 — Load reference lookups from DB
# ─────────────────────────────────────────────

def load_reference_maps(session):
    global ZONE_MAP, STORE_TYPE_MAP, BRAND_MAP, SEGMENT_MAP, NLP_TOPIC_MAP

    for row in session.execute(text("SELECT zone_id, zone_code FROM zones")).fetchall():
        ZONE_MAP[row[1]] = row[0]

    for row in session.execute(text("SELECT store_type_id, store_type_code FROM store_types")).fetchall():
        STORE_TYPE_MAP[row[1]] = row[0]

    for row in session.execute(text("SELECT brand_id, brand_code FROM brands")).fetchall():
        BRAND_MAP[row[1]] = row[0]

    for row in session.execute(text("SELECT segment_id, segment_code FROM retailer_segments")).fetchall():
        SEGMENT_MAP[row[1]] = row[0]

    for row in session.execute(
        text("SELECT topic_id, topic_type, lda_index, model_version FROM nlp_topics")
    ).fetchall():
        NLP_TOPIC_MAP[(row[1], row[2], row[3])] = row[0]

    log.info(
        "Reference maps loaded — zones:%d stores:%d brands:%d segments:%d topics:%d",
        len(ZONE_MAP), len(STORE_TYPE_MAP), len(BRAND_MAP), len(SEGMENT_MAP), len(NLP_TOPIC_MAP),
    )


# ─────────────────────────────────────────────
# STEP 2 — Distributors
# ─────────────────────────────────────────────

def seed_distributors(session, df: pd.DataFrame):
    inserted = 0
    for dist_name in df["ADA / Distributor"].dropna().unique():
        dist_name = str(dist_name).strip()
        if not dist_name or dist_name.lower() in ("nan", "none", ""):
            continue
        existing = session.execute(
            text("SELECT distributor_id FROM distributors WHERE distributor_name = :n"),
            {"n": dist_name},
        ).fetchone()
        if not existing:
            session.execute(
                text("INSERT INTO distributors (distributor_name) VALUES (:n) ON CONFLICT DO NOTHING"),
                {"n": dist_name},
            )
            inserted += 1

    session.flush()
    for row in session.execute(text("SELECT distributor_id, distributor_name FROM distributors")).fetchall():
        DISTRIBUTOR_MAP[row[1]] = row[0]
    log.info("Distributors seeded: %d new | %d total", inserted, len(DISTRIBUTOR_MAP))


# ─────────────────────────────────────────────
# STEP 3 — Retailers
# ─────────────────────────────────────────────

def seed_retailers(session, df: pd.DataFrame):
    global RETAILER_MAP
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        if sr_no is None:
            skipped += 1
            continue

        shop_name = safe_str(row.get("Shop Name"))
        zone_code = safe_str(row.get("zone_clean"))
        store_code = safe_str(row.get("store_type_clean"))
        dist_name = safe_str(row.get("ADA / Distributor"))

        zone_id = ZONE_MAP.get(zone_code)
        store_type_id = STORE_TYPE_MAP.get(store_code)
        distributor_id = DISTRIBUTOR_MAP.get(dist_name) if dist_name else None

        existing = session.execute(
            text(
                "SELECT retailer_id FROM retailers "
                "WHERE shop_name IS NOT DISTINCT FROM :s AND zone_id IS NOT DISTINCT FROM :z "
                "LIMIT 1"
            ),
            {"s": shop_name, "z": zone_id},
        ).fetchone()

        if existing:
            RETAILER_MAP[sr_no] = existing[0]
            skipped += 1
            continue

        ts_raw = safe_str(row.get("Timestamp") or row.get("survey_timestamp"))
        ts = None
        if ts_raw:
            try:
                ts = pd.to_datetime(ts_raw)
            except Exception:
                ts = None

        result = session.execute(
            text("""
                INSERT INTO retailers
                    (data_source, shop_name, owner_name, phone_raw, phone_clean,
                     phone_valid, area_locality, zone_id, store_type_id,
                     years_operating, years_operating_num, distributor_id, survey_timestamp)
                VALUES
                    (:data_source, :shop_name, :owner_name, :phone_raw, :phone_clean,
                     :phone_valid, :area_locality, :zone_id, :store_type_id,
                     :years_operating, :years_operating_num, :distributor_id, :survey_timestamp)
                RETURNING retailer_id
            """),
            {
                "data_source":         safe_str(row.get("Data Source"), "Unknown"),
                "shop_name":           shop_name,
                "owner_name":          safe_str(row.get("Contact / Owner")),
                "phone_raw":           safe_str(row.get("WhatsApp Number")),
                "phone_clean":         safe_str(row.get("phone_clean")),
                "phone_valid":         safe_bool(row.get("phone_valid")),
                "area_locality":       safe_str(row.get("Area / Locality")),
                "zone_id":             zone_id,
                "store_type_id":       store_type_id,
                "years_operating":     safe_str(row.get("Years Operating")),
                "years_operating_num": safe_float(row.get("feat_years_operating_num")),
                "distributor_id":      distributor_id,
                "survey_timestamp":    ts,
            },
        ).fetchone()
        session.flush()
        RETAILER_MAP[sr_no] = result[0]
        inserted += 1

    log.info("Retailers — inserted:%d  skipped/existing:%d", inserted, skipped)


# ─────────────────────────────────────────────
# STEP 4 — Infrastructure
# ─────────────────────────────────────────────

def seed_infrastructure(session, df: pd.DataFrame):
    inserted = 0
    for _, row in df.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        rid = RETAILER_MAP.get(sr_no)
        if rid is None:
            continue
        exists = session.execute(
            text("SELECT 1 FROM retailer_infrastructure WHERE retailer_id = :r"), {"r": rid}
        ).fetchone()
        if exists:
            continue

        fridge_raw = safe_str(row.get("Has Refrigerator"), "")
        if "dedicated" in fridge_raw.lower():
            fridge_type = "dedicated"
        elif "shared" in fridge_raw.lower():
            fridge_type = "shared"
        else:
            fridge_type = "none"

        session.execute(
            text("""
                INSERT INTO retailer_infrastructure
                    (retailer_id, has_refrigerator, fridge_type, fridge_flag,
                     dedicated_fridge_flag, infrastructure_score)
                VALUES
                    (:retailer_id, :has_refrigerator, :fridge_type, :fridge_flag,
                     :dedicated_fridge_flag, :infrastructure_score)
                ON CONFLICT DO NOTHING
            """),
            {
                "retailer_id":           rid,
                "has_refrigerator":      fridge_raw or None,
                "fridge_type":           fridge_type,
                "fridge_flag":           safe_bool(row.get("feat_refrigerator_flag")),
                "dedicated_fridge_flag": safe_bool(row.get("feat_dedicated_fridge")),
                "infrastructure_score":  safe_float(row.get("feat_infrastructure_score")),
            },
        )
        inserted += 1
    session.flush()
    log.info("Infrastructure rows inserted: %d", inserted)


# ─────────────────────────────────────────────
# STEP 5 — Product Intelligence
# ─────────────────────────────────────────────

def seed_product_intelligence(session, df: pd.DataFrame):
    inserted = 0
    for _, row in df.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        rid = RETAILER_MAP.get(sr_no)
        if rid is None:
            continue
        exists = session.execute(
            text("SELECT 1 FROM product_intelligence WHERE retailer_id = :r"), {"r": rid}
        ).fetchone()
        if exists:
            continue

        session.execute(
            text("""
                INSERT INTO product_intelligence
                    (retailer_id, stocks_ct, aware_ct, awareness_tier,
                     why_not_stocking_ct, ct_sku_mix, ct_packs_per_week_raw,
                     ct_weekly_volume, masti_packs_per_week, masti_weekly_volume,
                     pack_sizes_in_demand, packaging_format_pref,
                     sales_change_vs_last_yr, sales_trend)
                VALUES
                    (:retailer_id, :stocks_ct, :aware_ct, :awareness_tier,
                     :why_not_stocking_ct, :ct_sku_mix, :ct_packs_per_week_raw,
                     :ct_weekly_volume, :masti_packs_per_week, :masti_weekly_volume,
                     :pack_sizes_in_demand, :packaging_format_pref,
                     :sales_change_vs_last_yr, :sales_trend)
                ON CONFLICT DO NOTHING
            """),
            {
                "retailer_id":             rid,
                "stocks_ct":               safe_bool(row.get("stocks_ct")),
                "aware_ct":                safe_str(row.get("Aware of Amul C&T")),
                "awareness_tier":          safe_str(row.get("awareness_tier")),
                "why_not_stocking_ct":     safe_str(row.get("Why Not Stocking C&T")),
                "ct_sku_mix":              safe_str(row.get("C&T SKU Mix (80g / 180g / 850g)")),
                "ct_packs_per_week_raw":   safe_str(row.get("C&T Estimated Packs / Week")),
                "ct_weekly_volume":        safe_float(row.get("feat_ct_weekly_volume")),
                "masti_packs_per_week":    safe_str(row.get("Amul Masti Dahi Packs / Week (est.)")),
                "masti_weekly_volume":     safe_float(row.get("feat_masti_weekly_volume")),
                "pack_sizes_in_demand":    safe_str(row.get("Pack Sizes Most in Demand")),
                "packaging_format_pref":   _pack_format(safe_str(row.get("pack_format_clean"), "")),
                "sales_change_vs_last_yr": safe_str(row.get("Amul Dahi Sales Change vs 1 Year Ago")),
                "sales_trend":             _sales_trend(
                    safe_str(row.get("Amul Dahi Sales Change vs 1 Year Ago"), "")
                ),
            },
        )
        inserted += 1
    session.flush()
    log.info("Product intelligence rows inserted: %d", inserted)


# ─────────────────────────────────────────────
# STEP 6 — Competitor Intelligence
# ─────────────────────────────────────────────

def seed_competitor_intelligence(session, df: pd.DataFrame):
    inserted = 0
    for _, row in df.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        rid = RETAILER_MAP.get(sr_no)
        if rid is None:
            continue
        exists = session.execute(
            text("SELECT 1 FROM competitor_intelligence WHERE retailer_id = :r"), {"r": rid}
        ).fetchone()
        if exists:
            continue

        top_brand_raw  = safe_str(row.get("#1 Selling Dahi Brand at Store"))
        best_promo     = safe_str(row.get("Best Trade Promotions Brand"))
        best_rep       = safe_str(row.get("Best Rep Visits Brand"))
        best_shelf     = safe_str(row.get("Best Shelf Life / Freshness Brand"))
        best_display   = safe_str(row.get("Branded Display / Fridge Provided By"))
        biggest_growth = safe_str(row.get("Biggest 6-Month Growth Brand"))
        frubon_score   = safe_float(row.get("feat_frubon_threat_score"), 0.0)

        session.execute(
            text("""
                INSERT INTO competitor_intelligence
                    (retailer_id, top_brand_id, top_brand_raw, brand_ranking_raw,
                     competitor_choice_reason, choice_reason_primary,
                     customers_ask_amul, customers_ask_competitor,
                     best_trade_promotions_brand_id, best_rep_visits_brand_id,
                     best_shelf_life_brand_id, best_display_fridge_brand_id,
                     biggest_growth_brand_id, competitor_presence_flag,
                     frubon_threat_flag, frubon_threat_score)
                VALUES
                    (:retailer_id, :top_brand_id, :top_brand_raw, :brand_ranking_raw,
                     :competitor_choice_reason, :choice_reason_primary,
                     :customers_ask_amul, :customers_ask_competitor,
                     :best_trade_promotions_brand_id, :best_rep_visits_brand_id,
                     :best_shelf_life_brand_id, :best_display_fridge_brand_id,
                     :biggest_growth_brand_id, :competitor_presence_flag,
                     :frubon_threat_flag, :frubon_threat_score)
                ON CONFLICT DO NOTHING
            """),
            {
                "retailer_id":                    rid,
                "top_brand_id":                   _resolve_brand_id(top_brand_raw),
                "top_brand_raw":                  top_brand_raw,
                "brand_ranking_raw":              safe_str(row.get("Brand Ranking (brands stocked, by volume)")),
                "competitor_choice_reason":       safe_str(row.get("Primary Reason Customers Choose Competitor")),
                "choice_reason_primary":          safe_str(row.get("Primary Reason Customers Choose Competitor")),
                "customers_ask_amul":             safe_str(row.get("Customers Ask for Amul Dahi by Name")),
                "customers_ask_competitor":       safe_str(row.get("Customers Ask for Other Brand by Name")),
                "best_trade_promotions_brand_id": _resolve_brand_id(best_promo),
                "best_rep_visits_brand_id":       _resolve_brand_id(best_rep),
                "best_shelf_life_brand_id":       _resolve_brand_id(best_shelf),
                "best_display_fridge_brand_id":   _resolve_brand_id(best_display),
                "biggest_growth_brand_id":        _resolve_brand_id(biggest_growth),
                "competitor_presence_flag":       safe_bool(row.get("feat_competitor_presence")),
                "frubon_threat_flag":             frubon_score > 0,
                "frubon_threat_score":            frubon_score,
            },
        )
        inserted += 1
    session.flush()
    log.info("Competitor intelligence rows inserted: %d", inserted)


# ─────────────────────────────────────────────
# STEP 7 — Retailer Engagement
# ─────────────────────────────────────────────

def seed_engagement(session, df: pd.DataFrame):
    inserted = 0
    for _, row in df.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        rid = RETAILER_MAP.get(sr_no)
        if rid is None:
            continue
        exists = session.execute(
            text("SELECT 1 FROM retailer_engagement WHERE retailer_id = :r"), {"r": rid}
        ).fetchone()
        if exists:
            continue

        promo_raw   = safe_str(row.get("Trade Promotions Received on Amul Dahi"), "")
        promo_clean = safe_str(row.get("promo_received_clean"))
        promo_flag  = promo_clean not in (None, "never") if promo_clean else False

        session.execute(
            text("""
                INSERT INTO retailer_engagement
                    (retailer_id, margin_on_amul_dahi, trade_promotions_received,
                     promo_received_clean, promo_received_flag,
                     rep_visit_frequency, rep_visit_clean, rep_visit_score,
                     engagement_score)
                VALUES
                    (:retailer_id, :margin_on_amul_dahi, :trade_promotions_received,
                     :promo_received_clean, :promo_received_flag,
                     :rep_visit_frequency, :rep_visit_clean, :rep_visit_score,
                     :engagement_score)
                ON CONFLICT DO NOTHING
            """),
            {
                "retailer_id":              rid,
                "margin_on_amul_dahi":      safe_str(row.get("Margin on Amul Dahi (from price list)")),
                "trade_promotions_received": promo_raw,
                "promo_received_clean":     promo_clean,
                "promo_received_flag":      promo_flag,
                "rep_visit_frequency":      safe_str(row.get("Amul Rep Visit Frequency")),
                "rep_visit_clean":          safe_str(row.get("rep_visit_clean")),
                "rep_visit_score":          safe_float(row.get("feat_rep_visit_score")),
                "engagement_score":         safe_float(row.get("feat_engagement_score")),
            },
        )
        inserted += 1
    session.flush()
    log.info("Engagement rows inserted: %d", inserted)


# ─────────────────────────────────────────────
# STEP 8 — Retailer Feedback & NLP
# ─────────────────────────────────────────────

def _flag_complaint(corpus: str, keywords: list) -> bool:
    if not corpus:
        return False
    c = corpus.lower()
    return any(k in c for k in keywords)


def seed_feedback(session, df_main: pd.DataFrame, df_nlp: pd.DataFrame):
    # Merge NLP corpus into main df on Sr. No.
    nlp_cols = [
        "Sr. No.", "nlp_complaints", "nlp_ct_feedback",
        "nlp_suggestions", "nlp_other_feedback",
        "nlp_why_not_stocking", "nlp_full_corpus",
    ]
    available = [c for c in nlp_cols if c in df_nlp.columns]
    nlp = df_nlp[available].copy()
    merged = df_main.merge(nlp, on="Sr. No.", how="left", suffixes=("", "_nlp"))

    inserted = 0
    for _, row in merged.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        rid = RETAILER_MAP.get(sr_no)
        if rid is None:
            continue
        exists = session.execute(
            text("SELECT 1 FROM retailer_feedback WHERE retailer_id = :r"), {"r": rid}
        ).fetchone()
        if exists:
            continue

        # Prefer NLP-file columns; fall back to main CSV columns
        def pick(nlp_col, main_col):
            v = safe_str(row.get(nlp_col))
            return v if v else safe_str(row.get(main_col))

        nlp_complaints  = pick("nlp_complaints",        "Customer Complaints About Amul Dahi")
        nlp_ct_feedback = pick("nlp_ct_feedback",       "Customer Feedback on Amul C&T Specifically")
        nlp_suggestions = pick("nlp_suggestions",       "Suggested Changes / Improvements for Amul")
        nlp_other       = pick("nlp_other_feedback",    "Other Feedback for Amul")
        nlp_why         = pick("nlp_why_not_stocking",  "Why Not Stocking C&T")
        nlp_corpus      = pick("nlp_full_corpus",       "nlp_full_corpus")

        corpus_all = " ".join(filter(None, [nlp_corpus, nlp_complaints, nlp_suggestions]))

        session.execute(
            text("""
                INSERT INTO retailer_feedback
                    (retailer_id,
                     customer_complaints_raw, ct_specific_feedback_raw,
                     improvement_suggestions_raw, other_feedback_raw, why_not_stocking_raw,
                     nlp_complaints, nlp_ct_feedback, nlp_suggestions,
                     nlp_other_feedback, nlp_why_not_stocking, nlp_full_corpus,
                     flag_sour_taste, flag_texture_thin, flag_packaging_leak,
                     flag_short_shelf_life, flag_price_high,
                     flag_no_demand, flag_rep_absent)
                VALUES
                    (:retailer_id,
                     :customer_complaints_raw, :ct_specific_feedback_raw,
                     :improvement_suggestions_raw, :other_feedback_raw, :why_not_stocking_raw,
                     :nlp_complaints, :nlp_ct_feedback, :nlp_suggestions,
                     :nlp_other_feedback, :nlp_why_not_stocking, :nlp_full_corpus,
                     :flag_sour_taste, :flag_texture_thin, :flag_packaging_leak,
                     :flag_short_shelf_life, :flag_price_high,
                     :flag_no_demand, :flag_rep_absent)
                ON CONFLICT DO NOTHING
            """),
            {
                "retailer_id":                rid,
                "customer_complaints_raw":    safe_str(row.get("Customer Complaints About Amul Dahi")),
                "ct_specific_feedback_raw":   safe_str(row.get("Customer Feedback on Amul C&T Specifically")),
                "improvement_suggestions_raw": safe_str(row.get("Suggested Changes / Improvements for Amul")),
                "other_feedback_raw":         safe_str(row.get("Other Feedback for Amul")),
                "why_not_stocking_raw":       safe_str(row.get("Why Not Stocking C&T")),
                "nlp_complaints":             nlp_complaints,
                "nlp_ct_feedback":            nlp_ct_feedback,
                "nlp_suggestions":            nlp_suggestions,
                "nlp_other_feedback":         nlp_other,
                "nlp_why_not_stocking":       nlp_why,
                "nlp_full_corpus":            nlp_corpus,
                "flag_sour_taste":            _flag_complaint(corpus_all, ["sour", "acidic"]),
                "flag_texture_thin":          _flag_complaint(corpus_all, ["thin", "watery", "texture", "thick"]),
                "flag_packaging_leak":        _flag_complaint(corpus_all, ["leak", "packaging", "break", "torn"]),
                "flag_short_shelf_life":      _flag_complaint(corpus_all, ["shelf", "expiry", "fresh"]),
                "flag_price_high":            _flag_complaint(corpus_all, ["price", "expensive", "pricier", "costly"]),
                "flag_no_demand":             _flag_complaint(corpus_all, ["no demand", "no customer", "nobody asks"]),
                "flag_rep_absent":            _flag_complaint(corpus_all, ["rep hasn", "rep has not", "company rep"]),
            },
        )
        inserted += 1
    session.flush()
    log.info("Feedback/NLP rows inserted: %d", inserted)


# ─────────────────────────────────────────────
# STEP 9 — Retailer Scores (from segmentation_features.csv)
# ─────────────────────────────────────────────

def seed_scores(session, df_seg: pd.DataFrame):
    inserted = 0
    for _, row in df_seg.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        rid = RETAILER_MAP.get(sr_no)
        if rid is None:
            continue
        exists = session.execute(
            text("SELECT 1 FROM retailer_scores WHERE retailer_id = :r AND is_current = TRUE"),
            {"r": rid},
        ).fetchone()
        if exists:
            continue

        session.execute(
            text("""
                INSERT INTO retailer_scores
                    (retailer_id, awareness_score, rep_visit_score, promo_bonus,
                     engagement_score, infrastructure_score, competitor_presence,
                     frubon_threat_score, ct_weekly_volume, masti_weekly_volume,
                     years_operating_num, adoption_likelihood, model_version, is_current)
                VALUES
                    (:retailer_id, :awareness_score, :rep_visit_score, :promo_bonus,
                     :engagement_score, :infrastructure_score, :competitor_presence,
                     :frubon_threat_score, :ct_weekly_volume, :masti_weekly_volume,
                     :years_operating_num, :adoption_likelihood, :model_version, TRUE)
                ON CONFLICT DO NOTHING
            """),
            {
                "retailer_id":         rid,
                "awareness_score":     safe_float(row.get("feat_awareness_score")),
                "rep_visit_score":     safe_float(row.get("feat_rep_visit_score")),
                "promo_bonus":         safe_float(row.get("feat_promo_bonus")),
                "engagement_score":    safe_float(row.get("feat_engagement_score")),
                "infrastructure_score": safe_float(row.get("feat_infrastructure_score")),
                "competitor_presence": safe_float(row.get("feat_competitor_presence")),
                "frubon_threat_score": safe_float(row.get("feat_frubon_threat_score")),
                "ct_weekly_volume":    safe_float(row.get("feat_ct_weekly_volume")),
                "masti_weekly_volume": safe_float(row.get("feat_masti_weekly_volume")),
                "years_operating_num": safe_float(row.get("feat_years_operating_num")),
                "adoption_likelihood": safe_float(row.get("feat_adoption_likelihood")),
                "model_version":       "v1.0",
            },
        )
        inserted += 1
    session.flush()
    log.info("Retailer scores rows inserted: %d", inserted)


# ─────────────────────────────────────────────
# STEP 10 — Segment Assignments
# ─────────────────────────────────────────────

def _infer_segment_code(row) -> str:
    """Rule-based segment assignment matching the 5 personas."""
    awareness   = safe_float(row.get("feat_awareness_score"), 0)
    infra       = safe_float(row.get("feat_infrastructure_score"), 0)
    masti_vol   = safe_float(row.get("feat_masti_weekly_volume"), 0)
    frubon      = safe_float(row.get("feat_frubon_threat_score"), 0)
    engagement  = safe_float(row.get("feat_engagement_score"), 0)
    zone        = safe_str(row.get("zone_clean"), "")
    store       = safe_str(row.get("store_type_clean"), "")
    adoption    = safe_float(row.get("feat_adoption_likelihood"), 0)

    # Premium Urban Adopters — C-Scheme supermarkets, high adoption
    if zone == "C_Scheme" or (store == "Supermarket" and adoption >= 7.5):
        return "SEG_PREMIUM"

    # High-Potential Dormant — refrigerated, high masti, unaware of C&T
    if infra >= 2 and awareness == 0 and masti_vol >= 75:
        return "SEG_DORMANT"

    # Competitor-Dominated — high FruBon threat, low engagement
    if frubon >= 2 and engagement <= 2:
        return "SEG_COMPETITOR"

    # Loyal Amul Core — stocking or aware, decent engagement
    if awareness >= 1 and engagement >= 3:
        return "SEG_LOYAL"

    # Low-Infrastructure Risk — default fallback
    return "SEG_LOW_INFRA"


def seed_segment_assignments(session, df_seg: pd.DataFrame):
    inserted = 0
    for _, row in df_seg.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        rid = RETAILER_MAP.get(sr_no)
        if rid is None:
            continue
        exists = session.execute(
            text(
                "SELECT 1 FROM retailer_segment_assignments "
                "WHERE retailer_id = :r AND is_current = TRUE"
            ),
            {"r": rid},
        ).fetchone()
        if exists:
            continue

        seg_code = _infer_segment_code(row)
        seg_id   = SEGMENT_MAP.get(seg_code)
        if seg_id is None:
            log.warning("Segment code %s not found in DB — skipping retailer_id %d", seg_code, rid)
            continue

        session.execute(
            text("""
                INSERT INTO retailer_segment_assignments
                    (retailer_id, segment_id, cluster_id, model_version, confidence, is_current)
                VALUES
                    (:retailer_id, :segment_id, :cluster_id, :model_version, :confidence, TRUE)
                ON CONFLICT DO NOTHING
            """),
            {
                "retailer_id":   rid,
                "segment_id":    seg_id,
                "cluster_id":    None,
                "model_version": "v1.0",
                "confidence":    safe_float(row.get("feat_adoption_likelihood"), 5.0) / 10.0,
            },
        )
        inserted += 1
    session.flush()
    log.info("Segment assignments inserted: %d", inserted)


# ─────────────────────────────────────────────
# STEP 11 — Brand Rankings per Retailer
# ─────────────────────────────────────────────

def seed_brand_rankings(session, df: pd.DataFrame):
    inserted = 0
    for _, row in df.iterrows():
        sr_no = safe_int(row.get("Sr. No."))
        rid = RETAILER_MAP.get(sr_no)
        if rid is None:
            continue

        ranking_raw = safe_str(row.get("Brand Ranking (brands stocked, by volume)"))
        if not ranking_raw:
            continue

        # Parse "#1: Brand X, #2: Brand Y" format
        parts = [p.strip() for p in ranking_raw.split(",")]
        rank = 1
        for part in parts:
            # strip "#N:" prefix if present
            if ":" in part:
                part = part.split(":", 1)[1].strip()
            brand_id = _resolve_brand_id(part)
            if brand_id is None:
                rank += 1
                continue
            exists = session.execute(
                text(
                    "SELECT 1 FROM retailer_brand_rankings "
                    "WHERE retailer_id = :r AND brand_id = :b"
                ),
                {"r": rid, "b": brand_id},
            ).fetchone()
            if not exists:
                session.execute(
                    text("""
                        INSERT INTO retailer_brand_rankings
                            (retailer_id, brand_id, rank_position)
                        VALUES (:r, :b, :pos)
                        ON CONFLICT DO NOTHING
                    """),
                    {"r": rid, "b": brand_id, "pos": rank},
                )
                inserted += 1
            rank += 1

    session.flush()
    log.info("Brand ranking rows inserted: %d", inserted)


# ─────────────────────────────────────────────
# STEP 12 — KPI Snapshot (global)
# ─────────────────────────────────────────────

def seed_kpi_snapshot(session, df: pd.DataFrame):
    """Compute and store a single global KPI snapshot from the loaded data."""
    from datetime import date

    today = date.today()
    exists = session.execute(
        text(
            "SELECT 1 FROM kpi_snapshots "
            "WHERE snapshot_date = :d AND scope = 'global' AND scope_ref_id IS NULL"
        ),
        {"d": today},
    ).fetchone()
    if exists:
        log.info("Global KPI snapshot already exists for today — skipping.")
        return

    total       = len(df)
    ct_stockers = int(df["stocks_ct"].apply(safe_bool).sum()) if "stocks_ct" in df.columns else 0

    def rate(col, truthy_vals):
        if col not in df.columns:
            return None
        hits = df[col].apply(lambda v: safe_str(v) in truthy_vals).sum()
        return round(hits / total, 6) if total else None

    ct_pen   = round(ct_stockers / total, 6) if total else 0
    aware    = rate("awareness_tier", ["stocking", "aware_only"])
    promo_r  = df["promo_received_clean"].apply(
        lambda v: safe_str(v) not in (None, "never")
    ).sum() / total if "promo_received_clean" in df.columns else None

    weekly_r = df["rep_visit_clean"].apply(
        lambda v: safe_str(v) == "weekly"
    ).sum() / total if "rep_visit_clean" in df.columns else None

    frubon_p = (
        df["feat_frubon_threat_score"].apply(lambda v: safe_float(v, 0) > 0).sum() / total
        if "feat_frubon_threat_score" in df.columns else None
    )

    avg_adopt = (
        df["feat_adoption_likelihood"].apply(lambda v: safe_float(v, 0)).mean()
        if "feat_adoption_likelihood" in df.columns else None
    )

    session.execute(
        text("""
            INSERT INTO kpi_snapshots
                (snapshot_date, scope, scope_ref_id, scope_ref_label,
                 total_retailers, ct_stockers, ct_penetration_rate,
                 ct_awareness_rate, trade_promo_reach, weekly_rep_freq_pct,
                 frubon_threat_pct, avg_adoption_likelihood)
            VALUES
                (:snapshot_date, 'global', NULL, 'All Zones',
                 :total_retailers, :ct_stockers, :ct_penetration_rate,
                 :ct_awareness_rate, :trade_promo_reach, :weekly_rep_freq_pct,
                 :frubon_threat_pct, :avg_adoption_likelihood)
            ON CONFLICT DO NOTHING
        """),
        {
            "snapshot_date":        today,
            "total_retailers":      total,
            "ct_stockers":          ct_stockers,
            "ct_penetration_rate":  ct_pen,
            "ct_awareness_rate":    aware,
            "trade_promo_reach":    round(promo_r, 6) if promo_r is not None else None,
            "weekly_rep_freq_pct":  round(weekly_r, 6) if weekly_r is not None else None,
            "frubon_threat_pct":    round(frubon_p, 6) if frubon_p is not None else None,
            "avg_adoption_likelihood": round(avg_adopt, 6) if avg_adopt is not None else None,
        },
    )
    session.flush()
    log.info("Global KPI snapshot inserted for %s", today)


# ─────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────

def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        # Also try root project directory
        alt = Path(__file__).parent.parent / "data" / "processed" / filename
        if alt.exists():
            path = alt
        else:
            log.error("CSV not found: %s (also tried %s)", path, alt)
            sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %s — %d rows × %d cols", filename, len(df), len(df.columns))
    return df


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def main():
    log.info("=== FMCG Retail Intelligence — Seed Data ===")

    # Load CSVs
    df_main = load_csv("retailers_clean.csv")
    df_seg  = load_csv("segmentation_features.csv")
    df_nlp  = load_csv("nlp_corpus.csv")

    # Normalise Sr. No. across all dataframes
    for df in (df_main, df_seg, df_nlp):
        if "Sr. No." in df.columns:
            df["Sr. No."] = pd.to_numeric(df["Sr. No."], errors="coerce")

    engine  = get_engine()
    session = get_session(engine)

    try:
        log.info("--- Step 1: Loading reference maps ---")
        load_reference_maps(session)

        log.info("--- Step 2: Seeding distributors ---")
        seed_distributors(session, df_main)

        log.info("--- Step 3: Seeding retailers ---")
        seed_retailers(session, df_main)

        log.info("--- Step 4: Seeding infrastructure ---")
        seed_infrastructure(session, df_main)

        log.info("--- Step 5: Seeding product intelligence ---")
        seed_product_intelligence(session, df_main)

        log.info("--- Step 6: Seeding competitor intelligence ---")
        seed_competitor_intelligence(session, df_main)

        log.info("--- Step 7: Seeding engagement ---")
        seed_engagement(session, df_main)

        log.info("--- Step 8: Seeding feedback / NLP ---")
        seed_feedback(session, df_main, df_nlp)

        log.info("--- Step 9: Seeding retailer scores ---")
        seed_scores(session, df_seg)

        log.info("--- Step 10: Seeding segment assignments ---")
        seed_segment_assignments(session, df_seg)

        log.info("--- Step 11: Seeding brand rankings ---")
        seed_brand_rankings(session, df_main)

        log.info("--- Step 12: Seeding KPI snapshot ---")
        seed_kpi_snapshot(session, df_main)

        session.commit()
        log.info("=== All data committed successfully. ===")

    except SQLAlchemyError as exc:
        session.rollback()
        log.exception("Database error — transaction rolled back: %s", exc)
        sys.exit(1)
    except Exception as exc:
        session.rollback()
        log.exception("Unexpected error — transaction rolled back: %s", exc)
        sys.exit(1)
    finally:
        session.close()
        engine.dispose()
        log.info("Session closed.")


if __name__ == "__main__":
    main()
