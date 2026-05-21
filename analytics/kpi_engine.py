"""
================================================================================
kpi_engine.py — Production-Grade KPI Analytics Engine
AI-Powered FMCG Retail Intelligence Platform
Amul Retailer Intelligence System | Jaipur Market
================================================================================
Computes:
  · C&T penetration rate                · Awareness rate & funnel
  · Refrigeration coverage              · Competitor dominance index
  · Engagement effectiveness            · Retailer opportunity score
  · Zone-level performance              · Segment-level analytics
  · Top-performing retailers            · Underperforming regions
  · Promotion effectiveness             · Sales trend analysis
  · Pack size & format preferences      · Complaint & feedback signals

Outputs:
  · Typed KPISummary dataclasses        · Zone / segment DataFrames
  · Retailer-level opportunity table    · CSV exports
  · Visualization-ready dicts           · PostgreSQL snapshot writes

Author  : AI FMCG Intelligence Platform
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FMT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

logging.basicConfig(
    level   = LOG_LEVEL,
    format  = LOG_FMT,
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/kpi_engine.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("kpi_engine")

# ── Constants ─────────────────────────────────────────────────────────────────

EXPORTS_DIR = Path(os.getenv("EXPORTS_PATH", "data/exports/"))
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

ZONE_PRIORITY: dict[str, str] = {
    "C_Scheme":            "PREMIUM",
    "Vaishali_Mansarovar": "HIGH",
    "Malviya_Sanganer":    "HIGH",
    "Jhotwara_Murlipura":  "MEDIUM",
    "Amer_Periphery":      "MEDIUM",
    "Old_City":            "STANDARD",
}

FRUBON_THREAT_THRESHOLD = 2       # score ≥ this → at-risk retailer
OPPORTUNITY_SCORE_BINS   = [0, 3, 5, 7, float("inf")]
OPPORTUNITY_SCORE_LABELS = ["Low", "Medium", "High", "Critical"]

# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — Typed KPI containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PenetrationKPI:
    total_retailers:         int
    ct_stockers:             int
    ct_penetration_rate:     float   # %
    aware_only_count:        int
    unaware_count:           int
    awareness_rate:          float   # % who know the product (stocking + aware_only)
    unawareness_rate:        float   # %
    awareness_to_stocking_conversion: float  # % of aware_only who stock


@dataclass
class InfrastructureKPI:
    refrigerated_count:         int
    refrigeration_rate:         float   # %
    dedicated_fridge_count:     int
    dedicated_fridge_rate:      float   # %
    no_fridge_count:            int
    cold_chain_gap:             float   # % refrigerated but NOT stocking C&T


@dataclass
class CompetitorKPI:
    top_selling_brand:          str
    top_selling_brand_pct:      float
    amul_top_seller_pct:        float
    saras_top_seller_pct:       float
    frubon_top_seller_pct:      float
    best_promo_brand:           str
    best_promo_brand_pct:       float
    best_rep_brand:             str
    best_rep_brand_pct:         float
    frubon_threat_retailers:    int
    frubon_threat_pct:          float
    price_driven_loss_pct:      float   # % citing Price as competitor reason
    texture_driven_loss_pct:    float
    competitor_display_count:   int


@dataclass
class EngagementKPI:
    weekly_rep_visit_count:     int
    weekly_rep_visit_pct:       float
    monthly_rep_visit_pct:      float
    rarely_rep_visit_pct:       float
    promo_received_count:       int
    promo_reach_pct:            float
    avg_engagement_score:       float
    high_engagement_pct:        float   # engagement_score ≥ 5
    zero_engagement_count:      int


@dataclass
class SalesKPI:
    increased_significantly_pct: float
    increased_slightly_pct:      float
    stayed_same_pct:             float
    decreased_pct:               float
    net_growth_signal:           float   # (increased - decreased) / total * 100
    cup_format_preference_pct:   float
    pouch_format_preference_pct: float
    small_sku_demand_pct:        float   # 80g/180g


@dataclass
class ComplaintKPI:
    sourness_complaint_pct:      float
    texture_complaint_pct:       float
    packaging_complaint_pct:     float
    no_complaint_pct:            float
    top_complaint:               str
    top_complaint_pct:           float
    avg_complaint_sentiment:     float | None


@dataclass
class OpportunityKPI:
    avg_opportunity_score:       float
    critical_opportunity_count:  int
    high_opportunity_count:      int
    medium_opportunity_count:    int
    low_opportunity_count:       int
    refrigerated_not_stocking:   int   # Prime conversion targets
    top_10_opportunity_retailers: list[dict[str, Any]]


@dataclass
class GlobalKPISummary:
    computed_at:         str
    penetration:         PenetrationKPI
    infrastructure:      InfrastructureKPI
    competitor:          CompetitorKPI
    engagement:          EngagementKPI
    sales:               SalesKPI
    complaints:          ComplaintKPI
    opportunity:         OpportunityKPI
    headline_metrics:    dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# DATABASE — Connection & query helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_engine() -> Engine:
    """Build SQLAlchemy engine from environment variables."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        host     = os.getenv("DB_HOST",     "localhost")
        port     = os.getenv("DB_PORT",     "5432")
        db_name  = os.getenv("DB_NAME",     "fmcg_intelligence")
        user     = os.getenv("DB_USER",     "postgres")
        password = os.getenv("DB_PASSWORD", "")
        db_url   = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

    logger.info("Connecting to database…")
    engine = create_engine(
        db_url,
        pool_size      = 5,
        max_overflow   = 10,
        pool_pre_ping  = True,
        pool_recycle   = 3600,
        echo           = False,
    )
    return engine


@contextmanager
def db_session(engine: Engine) -> Generator[Any, None, None]:
    """Context manager that yields a connection and handles errors."""
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except SQLAlchemyError as exc:
        conn.rollback()
        logger.error("Database error: %s", exc, exc_info=True)
        raise
    finally:
        conn.close()


def query_to_df(engine: Engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    try:
        with db_session(engine) as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as exc:
        logger.error("Query failed: %s\nSQL: %s", exc, sql[:200])
        raise


# ──────────────────────────────────────────────────────────────────────────────
# SQL QUERIES — Pre-built analytical queries
# ──────────────────────────────────────────────────────────────────────────────

SQL_MASTER_VIEW = """
SELECT
    r.retailer_id,
    r.shop_name,
    r.area_locality,
    r.years_operating,
    r.years_operating_num,
    z.zone_code     AS zone_clean,
    z.zone_name,
    z.is_premium,
    st.store_type_code  AS store_type_clean,

    -- Infrastructure
    ri.fridge_type,
    ri.fridge_flag          AS feat_refrigerator_flag,
    ri.dedicated_fridge_flag AS feat_dedicated_fridge,
    ri.infrastructure_score AS feat_infrastructure_score,

    -- Product intelligence
    pi.stocks_ct,
    pi.awareness_tier,
    pi.ct_weekly_volume     AS feat_ct_weekly_volume,
    pi.masti_weekly_volume  AS feat_masti_weekly_volume,
    pi.packaging_format_pref,
    pi.pack_sizes_in_demand,
    pi.sales_trend,
    pi.sales_change_vs_last_yr,
    pi.why_not_stocking_ct,

    -- Competitor intelligence
    ci.top_brand_raw,
    ci.choice_reason_primary,
    ci.competitor_choice_reason,
    ci.competitor_presence_flag AS feat_competitor_presence,
    ci.frubon_threat_score  AS feat_frubon_threat_score,
    ci.frubon_threat_flag,
    ci.best_trade_promotions_brand_id,
    ci.best_rep_visits_brand_id,
    btb.brand_name          AS best_promo_brand_name,
    brb.brand_name          AS best_rep_brand_name,
    bdb.brand_name          AS best_display_brand_name,

    -- Engagement
    re.rep_visit_clean,
    re.promo_received_clean,
    re.promo_received_flag,
    re.rep_visit_score      AS feat_rep_visit_score,
    re.promo_received_flag::int AS feat_promo_bonus,
    re.engagement_score     AS feat_engagement_score,

    -- Feedback / NLP
    rf.nlp_complaints,
    rf.nlp_suggestions,
    rf.nlp_full_corpus,
    rf.complaint_sentiment,
    rf.suggestion_sentiment,
    rf.flag_sour_taste,
    rf.flag_texture_thin,
    rf.flag_packaging_leak,
    rf.flag_no_demand,
    rf.flag_rep_absent,
    rf.complaint_cluster_label,

    -- Scores
    rs.adoption_likelihood  AS feat_adoption_likelihood,
    rs.awareness_score      AS feat_awareness_score,

    -- Segmentation
    seg.segment_label,
    seg.segment_code,
    seg.priority            AS segment_priority

FROM retailers r
JOIN zones             z   ON r.zone_id         = z.zone_id
JOIN store_types       st  ON r.store_type_id   = st.store_type_id
LEFT JOIN retailer_infrastructure  ri  ON r.retailer_id = ri.retailer_id
LEFT JOIN product_intelligence     pi  ON r.retailer_id = pi.retailer_id
LEFT JOIN competitor_intelligence  ci  ON r.retailer_id = ci.retailer_id
LEFT JOIN retailer_engagement      re  ON r.retailer_id = re.retailer_id
LEFT JOIN retailer_feedback        rf  ON r.retailer_id = rf.retailer_id
LEFT JOIN retailer_scores          rs  ON r.retailer_id = rs.retailer_id AND rs.is_current = TRUE
LEFT JOIN brands                   btb ON ci.best_trade_promotions_brand_id = btb.brand_id
LEFT JOIN brands                   brb ON ci.best_rep_visits_brand_id       = brb.brand_id
LEFT JOIN brands                   bdb ON ci.best_display_fridge_brand_id   = bdb.brand_id
LEFT JOIN retailer_segment_assignments rsa ON r.retailer_id = rsa.retailer_id AND rsa.is_current = TRUE
LEFT JOIN retailer_segments        seg ON rsa.segment_id = seg.segment_id
WHERE r.is_active = TRUE
"""

SQL_ZONE_SUMMARY = """
SELECT
    z.zone_code,
    z.zone_name,
    z.is_premium,
    COUNT(r.retailer_id)                                            AS total_retailers,
    SUM(pi.stocks_ct::int)                                         AS ct_stockers,
    ROUND(AVG(pi.stocks_ct::int)::numeric * 100, 2)               AS ct_penetration_pct,
    ROUND(AVG(CASE WHEN pi.awareness_tier IN ('stocking','aware_only') THEN 1 ELSE 0 END)::numeric * 100, 2)
                                                                    AS awareness_rate_pct,
    ROUND(AVG(ri.fridge_flag::int)::numeric * 100, 2)             AS refrigeration_rate_pct,
    ROUND(AVG(re.promo_received_flag::int)::numeric * 100, 2)     AS promo_reach_pct,
    ROUND(AVG(ci.frubon_threat_score)::numeric, 3)                AS avg_frubon_threat,
    ROUND(AVG(rs.adoption_likelihood)::numeric, 3)                AS avg_adoption_likelihood,
    ROUND(AVG(pi.ct_weekly_volume)::numeric, 2)                   AS avg_ct_weekly_volume,
    ROUND(AVG(pi.masti_weekly_volume)::numeric, 2)                AS avg_masti_weekly_volume,
    ROUND(AVG(re.engagement_score)::numeric, 2)                   AS avg_engagement_score,
    SUM(CASE WHEN ci.frubon_threat_flag THEN 1 ELSE 0 END)        AS frubon_threat_count,
    SUM(CASE WHEN ri.fridge_flag AND NOT pi.stocks_ct THEN 1 ELSE 0 END)
                                                                    AS cold_chain_ready_not_stocking,
    MODE() WITHIN GROUP (ORDER BY ci.top_brand_raw)               AS modal_top_brand,
    MODE() WITHIN GROUP (ORDER BY re.rep_visit_clean)             AS modal_rep_frequency,
    ROUND(100.0 * SUM(CASE WHEN ci.choice_reason_primary ILIKE '%price%' THEN 1 ELSE 0 END)
          / NULLIF(COUNT(r.retailer_id), 0)::numeric, 2)          AS price_loss_pct
FROM retailers r
JOIN zones z                       ON r.zone_id         = z.zone_id
LEFT JOIN product_intelligence pi  ON r.retailer_id    = pi.retailer_id
LEFT JOIN retailer_infrastructure ri ON r.retailer_id  = ri.retailer_id
LEFT JOIN competitor_intelligence ci ON r.retailer_id  = ci.retailer_id
LEFT JOIN retailer_engagement re   ON r.retailer_id    = re.retailer_id
LEFT JOIN retailer_scores rs       ON r.retailer_id    = rs.retailer_id AND rs.is_current = TRUE
WHERE r.is_active = TRUE
GROUP BY z.zone_id, z.zone_code, z.zone_name, z.is_premium
ORDER BY ct_penetration_pct DESC
"""

SQL_SEGMENT_SUMMARY = """
SELECT
    seg.segment_code,
    seg.segment_label,
    seg.priority,
    COUNT(r.retailer_id)                                            AS retailer_count,
    ROUND(AVG(pi.stocks_ct::int)::numeric * 100, 2)               AS ct_penetration_pct,
    ROUND(AVG(CASE WHEN pi.awareness_tier IN ('stocking','aware_only') THEN 1 ELSE 0 END)::numeric * 100, 2)
                                                                    AS awareness_rate_pct,
    ROUND(AVG(re.engagement_score)::numeric, 2)                   AS avg_engagement_score,
    ROUND(AVG(rs.adoption_likelihood)::numeric, 3)                AS avg_adoption_likelihood,
    ROUND(AVG(pi.ct_weekly_volume)::numeric, 2)                   AS avg_ct_weekly_volume,
    ROUND(AVG(ci.frubon_threat_score)::numeric, 3)                AS avg_frubon_threat,
    ROUND(AVG(re.promo_received_flag::int)::numeric * 100, 2)     AS promo_reach_pct,
    SUM(CASE WHEN ri.fridge_flag AND NOT pi.stocks_ct THEN 1 ELSE 0 END)
                                                                    AS opportunity_targets
FROM retailer_segment_assignments rsa
JOIN retailer_segments seg         ON rsa.segment_id  = seg.segment_id
JOIN retailers r                   ON rsa.retailer_id = r.retailer_id
LEFT JOIN product_intelligence pi  ON r.retailer_id   = pi.retailer_id
LEFT JOIN retailer_infrastructure ri ON r.retailer_id = ri.retailer_id
LEFT JOIN competitor_intelligence ci ON r.retailer_id = ci.retailer_id
LEFT JOIN retailer_engagement re   ON r.retailer_id   = re.retailer_id
LEFT JOIN retailer_scores rs       ON r.retailer_id   = rs.retailer_id AND rs.is_current = TRUE
WHERE rsa.is_current = TRUE AND r.is_active = TRUE
GROUP BY seg.segment_id, seg.segment_code, seg.segment_label, seg.priority
ORDER BY avg_adoption_likelihood DESC
"""

SQL_TOP_RETAILERS = """
SELECT
    r.retailer_id,
    r.shop_name,
    r.area_locality,
    z.zone_code,
    st.store_type_code,
    pi.stocks_ct,
    pi.ct_weekly_volume,
    pi.masti_weekly_volume,
    rs.adoption_likelihood,
    rs.engagement_score,
    rs.awareness_score,
    ri.infrastructure_score,
    ci.frubon_threat_score,
    seg.segment_label,
    ROUND(
        (COALESCE(rs.adoption_likelihood,0) * 0.35 +
         COALESCE(rs.engagement_score,0)   * 0.25 +
         COALESCE(ri.infrastructure_score,0) * 0.20 +
         COALESCE(pi.masti_weekly_volume,0) / NULLIF(
             (SELECT MAX(masti_weekly_volume) FROM product_intelligence), 0) * 10 * 0.20
        )::numeric, 4
    ) AS opportunity_score
FROM retailers r
JOIN zones z                       ON r.zone_id         = z.zone_id
JOIN store_types st                ON r.store_type_id   = st.store_type_id
LEFT JOIN product_intelligence pi  ON r.retailer_id    = pi.retailer_id
LEFT JOIN retailer_infrastructure ri ON r.retailer_id  = ri.retailer_id
LEFT JOIN competitor_intelligence ci ON r.retailer_id  = ci.retailer_id
LEFT JOIN retailer_scores rs       ON r.retailer_id    = rs.retailer_id AND rs.is_current = TRUE
LEFT JOIN retailer_segment_assignments rsa ON r.retailer_id = rsa.retailer_id AND rsa.is_current = TRUE
LEFT JOIN retailer_segments seg    ON rsa.segment_id   = seg.segment_id
WHERE r.is_active = TRUE AND NOT pi.stocks_ct
ORDER BY opportunity_score DESC
LIMIT 50
"""

SQL_UNDERPERFORMING_ZONES = """
SELECT
    z.zone_code,
    z.zone_name,
    COUNT(r.retailer_id)                                            AS retailer_count,
    ROUND(AVG(pi.stocks_ct::int)::numeric * 100, 2)               AS ct_penetration_pct,
    ROUND(AVG(re.promo_received_flag::int)::numeric * 100, 2)     AS promo_reach_pct,
    ROUND(AVG(re.engagement_score)::numeric, 2)                   AS avg_engagement_score,
    ROUND(AVG(ci.frubon_threat_score)::numeric, 3)                AS avg_frubon_threat,
    -- Underperformance composite: low penetration + low promo + high threat
    ROUND((
        (1 - AVG(pi.stocks_ct::int)) * 0.4 +
        (1 - AVG(re.promo_received_flag::int)) * 0.3 +
        (AVG(ci.frubon_threat_score) / 3.0) * 0.3
    )::numeric, 4)                                                  AS underperformance_score
FROM retailers r
JOIN zones z                       ON r.zone_id         = z.zone_id
LEFT JOIN product_intelligence pi  ON r.retailer_id    = pi.retailer_id
LEFT JOIN retailer_engagement re   ON r.retailer_id    = re.retailer_id
LEFT JOIN competitor_intelligence ci ON r.retailer_id  = ci.retailer_id
WHERE r.is_active = TRUE
GROUP BY z.zone_id, z.zone_code, z.zone_name
ORDER BY underperformance_score DESC
"""

SQL_PROMOTION_EFFECTIVENESS = """
SELECT
    CASE
        WHEN re.promo_received_clean = 'yes_occasionally' THEN 'Promotion Received'
        ELSE 'No Promotion'
    END AS promo_group,
    COUNT(r.retailer_id)                                            AS retailer_count,
    ROUND(AVG(pi.stocks_ct::int)::numeric * 100, 2)               AS ct_penetration_pct,
    ROUND(AVG(rs.adoption_likelihood)::numeric, 3)                AS avg_adoption_likelihood,
    ROUND(AVG(re.engagement_score)::numeric, 2)                   AS avg_engagement_score,
    ROUND(AVG(pi.ct_weekly_volume)::numeric, 2)                   AS avg_ct_weekly_volume,
    ROUND(AVG(pi.masti_weekly_volume)::numeric, 2)                AS avg_masti_weekly_volume
FROM retailers r
LEFT JOIN product_intelligence pi  ON r.retailer_id    = pi.retailer_id
LEFT JOIN retailer_engagement re   ON r.retailer_id    = re.retailer_id
LEFT JOIN retailer_scores rs       ON r.retailer_id    = rs.retailer_id AND rs.is_current = TRUE
WHERE r.is_active = TRUE
GROUP BY promo_group
ORDER BY ct_penetration_pct DESC
"""

SQL_STORE_TYPE_BREAKDOWN = """
SELECT
    st.store_type_code,
    st.store_type_name,
    COUNT(r.retailer_id)                                            AS retailer_count,
    ROUND(AVG(pi.stocks_ct::int)::numeric * 100, 2)               AS ct_penetration_pct,
    ROUND(AVG(CASE WHEN pi.awareness_tier IN ('stocking','aware_only') THEN 1 ELSE 0 END)::numeric * 100, 2)
                                                                    AS awareness_rate_pct,
    ROUND(AVG(re.engagement_score)::numeric, 2)                   AS avg_engagement_score,
    ROUND(AVG(rs.adoption_likelihood)::numeric, 3)                AS avg_adoption_likelihood
FROM retailers r
JOIN store_types st                ON r.store_type_id  = st.store_type_id
LEFT JOIN product_intelligence pi  ON r.retailer_id    = pi.retailer_id
LEFT JOIN retailer_engagement re   ON r.retailer_id    = re.retailer_id
LEFT JOIN retailer_scores rs       ON r.retailer_id    = rs.retailer_id AND rs.is_current = TRUE
WHERE r.is_active = TRUE
GROUP BY st.store_type_id, st.store_type_code, st.store_type_name
ORDER BY ct_penetration_pct DESC
"""


# ──────────────────────────────────────────────────────────────────────────────
# PANDAS TRANSFORMATIONS — In-memory KPI computation from cleaned CSV / DB pull
# ──────────────────────────────────────────────────────────────────────────────

def _safe_pct(numerator: float | int, denominator: float | int, decimals: int = 2) -> float:
    """Return a safe percentage, rounded, handling zero denominators."""
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100, decimals)


def _top_value(series: pd.Series) -> str:
    vc = series.value_counts()
    return str(vc.index[0]) if len(vc) else "N/A"


def compute_penetration_kpi(df: pd.DataFrame) -> PenetrationKPI:
    """Compute C&T penetration and awareness funnel from master DataFrame."""
    logger.debug("Computing penetration KPI…")
    n = len(df)

    ct_stockers   = int(df["stocks_ct"].sum())
    aware_only    = int((df["awareness_tier"] == "aware_only").sum())
    unaware       = int((df["awareness_tier"] == "unaware").sum())
    total_aware   = ct_stockers + aware_only

    return PenetrationKPI(
        total_retailers               = n,
        ct_stockers                   = ct_stockers,
        ct_penetration_rate           = _safe_pct(ct_stockers, n),
        aware_only_count              = aware_only,
        unaware_count                 = unaware,
        awareness_rate                = _safe_pct(total_aware, n),
        unawareness_rate              = _safe_pct(unaware, n),
        awareness_to_stocking_conversion = _safe_pct(ct_stockers, total_aware) if total_aware else 0.0,
    )


def compute_infrastructure_kpi(df: pd.DataFrame) -> InfrastructureKPI:
    """Compute refrigeration and cold-chain coverage metrics."""
    logger.debug("Computing infrastructure KPI…")
    n = len(df)
    fridge_col = "Has Refrigerator"

    if fridge_col in df.columns:
        refrigerated     = df[fridge_col].str.contains("Yes", na=False)
        dedicated        = df[fridge_col].str.contains("dedicated", case=False, na=False)
        no_fridge        = (~refrigerated)
    else:
        refrigerated     = df.get("feat_refrigerator_flag", pd.Series(False, index=df.index)).astype(bool)
        dedicated        = df.get("feat_dedicated_fridge", pd.Series(False, index=df.index)).astype(bool)
        no_fridge        = ~refrigerated

    ref_count  = int(refrigerated.sum())
    ded_count  = int(dedicated.sum())
    nof_count  = int(no_fridge.sum())

    # Cold chain gap: retailers with fridge who don't stock C&T
    gap = int((refrigerated & (~df["stocks_ct"].astype(bool))).sum())

    return InfrastructureKPI(
        refrigerated_count      = ref_count,
        refrigeration_rate      = _safe_pct(ref_count, n),
        dedicated_fridge_count  = ded_count,
        dedicated_fridge_rate   = _safe_pct(ded_count, n),
        no_fridge_count         = nof_count,
        cold_chain_gap          = _safe_pct(gap, ref_count),
    )


def compute_competitor_kpi(df: pd.DataFrame) -> CompetitorKPI:
    """Compute competitor dominance and threat metrics."""
    logger.debug("Computing competitor KPI…")
    n = len(df)

    brand_col    = "#1 Selling Dahi Brand at Store"
    promo_col    = "Best Trade Promotions Brand"
    rep_col      = "Best Rep Visits Brand"
    reason_col   = "Primary Reason Customers Choose Competitor"
    display_col  = "Branded Display / Fridge Provided By"
    threat_col   = "feat_frubon_threat_score"

    top_brand_vc  = df[brand_col].value_counts() if brand_col in df.columns else pd.Series()
    promo_vc      = df[promo_col].value_counts() if promo_col in df.columns else pd.Series()
    rep_vc        = df[rep_col].value_counts()   if rep_col   in df.columns else pd.Series()

    top_brand      = str(top_brand_vc.index[0]) if len(top_brand_vc) else "N/A"
    top_brand_cnt  = int(top_brand_vc.iloc[0])  if len(top_brand_vc) else 0

    amul_brands    = ["Amul_Masti Dahi", "Amul", "Amul Dahi Creamy & Tasty"]
    amul_top_cnt   = int(df[brand_col].isin(amul_brands).sum()) if brand_col in df.columns else 0
    saras_top_cnt  = int(df[brand_col].str.contains("Saras", na=False).sum())  if brand_col in df.columns else 0
    frubon_top_cnt = int(df[brand_col].str.contains("FruBon|Frubon", na=False).sum()) if brand_col in df.columns else 0

    reason_s      = df[reason_col] if reason_col in df.columns else pd.Series()
    price_cnt     = int(reason_s.str.contains("Price",   na=False).sum())
    texture_cnt   = int(reason_s.str.contains("Thickness|texture|Taste", case=False, na=False).sum())

    frubon_threat = df[threat_col] if threat_col in df.columns else pd.Series(0, index=df.index)
    frubon_at_risk = int((frubon_threat >= FRUBON_THREAT_THRESHOLD).sum())

    display_cnt   = int(df[display_col].notna().sum()) if display_col in df.columns else 0

    return CompetitorKPI(
        top_selling_brand        = top_brand,
        top_selling_brand_pct   = _safe_pct(top_brand_cnt, n),
        amul_top_seller_pct     = _safe_pct(amul_top_cnt, n),
        saras_top_seller_pct    = _safe_pct(saras_top_cnt, n),
        frubon_top_seller_pct   = _safe_pct(frubon_top_cnt, n),
        best_promo_brand        = str(promo_vc.index[0]) if len(promo_vc) else "N/A",
        best_promo_brand_pct    = _safe_pct(promo_vc.iloc[0], n) if len(promo_vc) else 0.0,
        best_rep_brand          = str(rep_vc.index[0]) if len(rep_vc) else "N/A",
        best_rep_brand_pct      = _safe_pct(rep_vc.iloc[0], n) if len(rep_vc) else 0.0,
        frubon_threat_retailers = frubon_at_risk,
        frubon_threat_pct       = _safe_pct(frubon_at_risk, n),
        price_driven_loss_pct   = _safe_pct(price_cnt, n),
        texture_driven_loss_pct = _safe_pct(texture_cnt, n),
        competitor_display_count= display_cnt,
    )


def compute_engagement_kpi(df: pd.DataFrame) -> EngagementKPI:
    """Compute rep visit, promotion reach, and engagement effectiveness."""
    logger.debug("Computing engagement KPI…")
    n = len(df)

    rep_col  = "rep_visit_clean"  if "rep_visit_clean"  in df.columns else "Amul Rep Visit Frequency"
    promo_col= "promo_received_clean" if "promo_received_clean" in df.columns else "Trade Promotions Received on Amul Dahi"
    eng_col  = "feat_engagement_score"

    weekly_cnt    = int(df[rep_col].str.contains("weekly|Weekly",     na=False).sum())
    monthly_cnt   = int(df[rep_col].str.contains("monthly|Monthly",   na=False).sum())
    rarely_cnt    = int(df[rep_col].str.contains("rarely|Rarely",     na=False).sum())

    promo_cnt     = int(df[promo_col].str.contains("yes|Yes",         na=False).sum())

    eng_scores    = df[eng_col].dropna() if eng_col in df.columns else pd.Series()
    avg_eng       = round(float(eng_scores.mean()), 3) if len(eng_scores) else 0.0
    high_eng_cnt  = int((eng_scores >= 5).sum())
    zero_eng_cnt  = int((eng_scores == 0).sum())

    return EngagementKPI(
        weekly_rep_visit_count = weekly_cnt,
        weekly_rep_visit_pct   = _safe_pct(weekly_cnt, n),
        monthly_rep_visit_pct  = _safe_pct(monthly_cnt, n),
        rarely_rep_visit_pct   = _safe_pct(rarely_cnt, n),
        promo_received_count   = promo_cnt,
        promo_reach_pct        = _safe_pct(promo_cnt, n),
        avg_engagement_score   = avg_eng,
        high_engagement_pct    = _safe_pct(high_eng_cnt, n),
        zero_engagement_count  = zero_eng_cnt,
    )


def compute_sales_kpi(df: pd.DataFrame) -> SalesKPI:
    """Compute sales trend and format/SKU preference KPIs."""
    logger.debug("Computing sales KPI…")
    n = len(df)

    trend_col  = "Amul Dahi Sales Change vs 1 Year Ago"
    format_col = "Packaging Format Preferred"
    pack_col   = "Pack Sizes Most in Demand"

    if trend_col in df.columns:
        t = df[trend_col]
        inc_sig = int(t.str.contains("Increased significantly", na=False).sum())
        inc_slt = int(t.str.contains("Increased slightly",      na=False).sum())
        stayed  = int(t.str.contains("Stayed",                  na=False).sum())
        dec     = int(t.str.contains("Decreased",               na=False).sum())
    else:
        inc_sig = inc_slt = stayed = dec = 0

    net_growth = _safe_pct((inc_sig + inc_slt) - dec, n)

    cup_pct    = 0.0
    pouch_pct  = 0.0
    if format_col in df.columns:
        fmt = df[format_col]
        cup_pct   = _safe_pct(fmt.str.contains("Cup|cup", na=False).sum(), n)
        pouch_pct = _safe_pct(fmt.str.contains("Pouch|pouch", na=False).sum(), n)

    small_sku_pct = 0.0
    if pack_col in df.columns:
        small_sku_pct = _safe_pct(
            df[pack_col].str.contains("80g|180g", na=False).sum(), n
        )

    return SalesKPI(
        increased_significantly_pct = _safe_pct(inc_sig, n),
        increased_slightly_pct      = _safe_pct(inc_slt, n),
        stayed_same_pct             = _safe_pct(stayed, n),
        decreased_pct               = _safe_pct(dec, n),
        net_growth_signal           = net_growth,
        cup_format_preference_pct   = cup_pct,
        pouch_format_preference_pct = pouch_pct,
        small_sku_demand_pct        = small_sku_pct,
    )


def compute_complaint_kpi(df: pd.DataFrame) -> ComplaintKPI:
    """Compute complaint distribution and NLP sentiment KPIs."""
    logger.debug("Computing complaint KPI…")
    n = len(df)

    complaint_col = "Customer Complaints About Amul Dahi"
    sentiment_col = "complaint_sentiment" if "complaint_sentiment" in df.columns else None

    sour_cnt    = 0
    texture_cnt = 0
    pack_cnt    = 0
    no_comp_cnt = 0

    if complaint_col in df.columns:
        c = df[complaint_col]
        sour_cnt    = int(c.str.contains("sour",                       case=False, na=False).sum())
        texture_cnt = int(c.str.contains("thick|texture|watery|thin",  case=False, na=False).sum())
        pack_cnt    = int(c.str.contains("leak|packag",                case=False, na=False).sum())
        no_comp_cnt = int(c.str.contains("No complaint|no complaint",  case=False, na=False).sum())
        top_val     = _top_value(c)
    else:
        top_val = "N/A"

    top_cnt   = int(df[complaint_col].str.contains(
        re.escape(top_val), na=False).sum()) if complaint_col in df.columns and top_val != "N/A" else 0
    avg_sent  = None
    if sentiment_col and sentiment_col in df.columns:
        s = df[sentiment_col].dropna()
        avg_sent = round(float(s.mean()), 4) if len(s) else None

    return ComplaintKPI(
        sourness_complaint_pct   = _safe_pct(sour_cnt, n),
        texture_complaint_pct    = _safe_pct(texture_cnt, n),
        packaging_complaint_pct  = _safe_pct(pack_cnt, n),
        no_complaint_pct         = _safe_pct(no_comp_cnt, n),
        top_complaint            = top_val,
        top_complaint_pct        = _safe_pct(top_cnt, n),
        avg_complaint_sentiment  = avg_sent,
    )


def compute_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a retailer-level opportunity score for non-stockers.

    Score = adoption_likelihood × 0.35
          + engagement_score   × 0.25
          + infrastructure_score × 0.20
          + normalised_masti_volume × 0.20

    Higher = higher conversion priority.
    """
    logger.debug("Computing opportunity scores…")
    out = df.copy()

    adop  = out.get("feat_adoption_likelihood",  pd.Series(0.0, index=out.index))
    eng   = out.get("feat_engagement_score",     pd.Series(0.0, index=out.index))
    infra = out.get("feat_infrastructure_score", pd.Series(0.0, index=out.index))
    masti = out.get("feat_masti_weekly_volume",  pd.Series(0.0, index=out.index))

    max_masti = masti.max() or 1.0
    norm_masti = (masti / max_masti) * 10.0

    out["opportunity_score"] = (
        adop.fillna(0)       * 0.35 +
        eng.fillna(0)        * 0.25 +
        infra.fillna(0)      * 0.20 +
        norm_masti.fillna(0) * 0.20
    ).round(4)

    out["opportunity_tier"] = pd.cut(
        out["opportunity_score"],
        bins   = OPPORTUNITY_SCORE_BINS,
        labels = OPPORTUNITY_SCORE_LABELS,
        right  = True,
    )
    return out


def compute_opportunity_kpi(df: pd.DataFrame) -> OpportunityKPI:
    """Summarise opportunity scores into a typed KPI."""
    logger.debug("Computing opportunity KPI…")
    scored = compute_opportunity_score(df)
    non_stockers = scored[~scored["stocks_ct"].astype(bool)]

    ref_col = "Has Refrigerator" if "Has Refrigerator" in df.columns else None
    if ref_col:
        ref_not_stocking = int(
            (df[ref_col].str.contains("Yes", na=False) & ~df["stocks_ct"].astype(bool)).sum()
        )
    else:
        ref_not_stocking = int(
            (df.get("feat_refrigerator_flag", pd.Series(False)).astype(bool) &
             ~df["stocks_ct"].astype(bool)).sum()
        )

    tier_counts = non_stockers["opportunity_tier"].value_counts().to_dict()

    id_col   = "Sr. No."       if "Sr. No."     in non_stockers.columns else non_stockers.index.name or "index"
    name_col = "Shop Name"     if "Shop Name"   in non_stockers.columns else "shop_name"
    zone_col = "zone_clean"    if "zone_clean"  in non_stockers.columns else "Zone"

    top10 = (
        non_stockers.nlargest(10, "opportunity_score")
        [[id_col, name_col, zone_col, "opportunity_score", "opportunity_tier",
          "feat_adoption_likelihood", "feat_engagement_score"]]
        .rename(columns={id_col: "retailer_id", name_col: "shop_name", zone_col: "zone"})
        .to_dict("records")
    )

    return OpportunityKPI(
        avg_opportunity_score        = round(float(non_stockers["opportunity_score"].mean()), 4),
        critical_opportunity_count   = int(tier_counts.get("Critical", 0)),
        high_opportunity_count       = int(tier_counts.get("High", 0)),
        medium_opportunity_count     = int(tier_counts.get("Medium", 0)),
        low_opportunity_count        = int(tier_counts.get("Low", 0)),
        refrigerated_not_stocking    = ref_not_stocking,
        top_10_opportunity_retailers = top10,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ZONE-LEVEL & SEGMENT-LEVEL ANALYTICS (pandas path)
# ──────────────────────────────────────────────────────────────────────────────

def compute_zone_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-zone KPI table from master DataFrame."""
    logger.debug("Computing zone-level KPIs…")

    ref_series = (
        df["Has Refrigerator"].str.contains("Yes", na=False)
        if "Has Refrigerator" in df.columns
        else df.get("feat_refrigerator_flag", pd.Series(False, index=df.index)).astype(bool)
    )

    agg = df.assign(_ref=ref_series).groupby("zone_clean").agg(
        total_retailers         = ("stocks_ct",                "count"),
        ct_stockers             = ("stocks_ct",                "sum"),
        aware_or_stocking       = ("awareness_tier",           lambda s: s.isin(["stocking","aware_only"]).sum()),
        refrigerated            = ("_ref",                     "sum"),
        avg_adoption_likelihood = ("feat_adoption_likelihood", "mean"),
        avg_engagement_score    = ("feat_engagement_score",    "mean"),
        avg_ct_weekly_volume    = ("feat_ct_weekly_volume",    "mean"),
        avg_masti_volume        = ("feat_masti_weekly_volume", "mean"),
        avg_frubon_threat       = ("feat_frubon_threat_score", "mean"),
        avg_opportunity_score   = ("opportunity_score",        "mean") if "opportunity_score" in df.columns else ("stocks_ct", "count"),
    ).reset_index()

    agg["ct_penetration_pct"]  = (agg["ct_stockers"]       / agg["total_retailers"] * 100).round(2)
    agg["awareness_rate_pct"]  = (agg["aware_or_stocking"]  / agg["total_retailers"] * 100).round(2)
    agg["refrigeration_pct"]   = (agg["refrigerated"]       / agg["total_retailers"] * 100).round(2)
    agg["cold_chain_gap_pct"]  = ((agg["refrigerated"] - agg["ct_stockers"]) / agg["total_retailers"] * 100).round(2).clip(lower=0)
    agg["zone_priority"]       = agg["zone_clean"].map(ZONE_PRIORITY).fillna("UNKNOWN")

    return agg.sort_values("ct_penetration_pct", ascending=False)


def compute_zone_opportunity_matrix(zone_df: pd.DataFrame) -> pd.DataFrame:
    """Rank zones by opportunity gap (readiness − penetration)."""
    z = zone_df.copy()
    z["opportunity_gap"] = (z["refrigeration_pct"] - z["ct_penetration_pct"]).round(2)
    return z.sort_values("opportunity_gap", ascending=False)[
        ["zone_clean", "zone_priority", "total_retailers", "ct_penetration_pct",
         "refrigeration_pct", "opportunity_gap", "avg_frubon_threat", "avg_adoption_likelihood"]
    ]


def compute_segment_kpis(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute per-segment KPI table if segment data exists."""
    seg_col = "segment_label"
    if seg_col not in df.columns or df[seg_col].isna().all():
        logger.warning("No segment assignments found; skipping segment KPI.")
        return None

    logger.debug("Computing segment-level KPIs…")
    ref_series = (
        df["Has Refrigerator"].str.contains("Yes", na=False)
        if "Has Refrigerator" in df.columns
        else df.get("feat_refrigerator_flag", pd.Series(False, index=df.index)).astype(bool)
    )

    agg = df.assign(_ref=ref_series).groupby(seg_col).agg(
        retailer_count          = ("stocks_ct",                "count"),
        ct_penetration_pct      = ("stocks_ct",                lambda s: round(s.mean() * 100, 2)),
        awareness_rate_pct      = ("awareness_tier",           lambda s: round(s.isin(["stocking","aware_only"]).mean() * 100, 2)),
        avg_engagement_score    = ("feat_engagement_score",    "mean"),
        avg_adoption_likelihood = ("feat_adoption_likelihood", "mean"),
        avg_frubon_threat       = ("feat_frubon_threat_score", "mean"),
        refrigeration_pct       = ("_ref",                     lambda s: round(s.mean() * 100, 2)),
    ).reset_index()

    return agg.sort_values("avg_adoption_likelihood", ascending=False)


def compute_store_type_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-store-type KPI breakdown."""
    logger.debug("Computing store-type KPIs…")
    agg = df.groupby("store_type_clean").agg(
        retailer_count          = ("stocks_ct", "count"),
        ct_penetration_pct      = ("stocks_ct", lambda s: round(s.mean() * 100, 2)),
        awareness_rate_pct      = ("awareness_tier", lambda s: round(s.isin(["stocking","aware_only"]).mean() * 100, 2)),
        avg_engagement_score    = ("feat_engagement_score",    "mean"),
        avg_adoption_likelihood = ("feat_adoption_likelihood", "mean"),
    ).reset_index().sort_values("ct_penetration_pct", ascending=False)
    return agg


def compute_promotion_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Compare KPIs between promo-reached vs. non-promo retailers."""
    logger.debug("Computing promotion effectiveness…")
    promo_col = "promo_received_clean" if "promo_received_clean" in df.columns else "Trade Promotions Received on Amul Dahi"

    df = df.copy()
    df["_promo_group"] = df[promo_col].str.contains("yes|Yes", na=False).map(
        {True: "Promotion Received", False: "No Promotion"}
    )

    result = df.groupby("_promo_group").agg(
        retailer_count          = ("stocks_ct",                "count"),
        ct_penetration_pct      = ("stocks_ct",                lambda s: round(s.mean() * 100, 2)),
        avg_adoption_likelihood = ("feat_adoption_likelihood", "mean"),
        avg_engagement_score    = ("feat_engagement_score",    "mean"),
        avg_ct_weekly_volume    = ("feat_ct_weekly_volume",    "mean"),
        avg_masti_weekly_volume = ("feat_masti_weekly_volume", "mean"),
    ).reset_index()

    if len(result) == 2:
        promo_pen  = float(result.loc[result["_promo_group"] == "Promotion Received", "ct_penetration_pct"].values[0])
        no_pen     = float(result.loc[result["_promo_group"] == "No Promotion",       "ct_penetration_pct"].values[0])
        result["promotion_uplift_ppt"] = result["_promo_group"].map({
            "Promotion Received": round(promo_pen - no_pen, 2),
            "No Promotion":       0.0,
        })

    return result


# ──────────────────────────────────────────────────────────────────────────────
# HEADLINE METRICS BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_headline_metrics(
    pen:   PenetrationKPI,
    infra: InfrastructureKPI,
    comp:  CompetitorKPI,
    eng:   EngagementKPI,
    sales: SalesKPI,
    comp_kpi: ComplaintKPI,
) -> dict[str, Any]:
    """Flatten 5 critical headline numbers for dashboard cards."""
    return {
        "ct_penetration_rate":     pen.ct_penetration_rate,
        "awareness_rate":          pen.awareness_rate,
        "refrigeration_rate":      infra.refrigeration_rate,
        "cold_chain_gap":          infra.cold_chain_gap,
        "promo_reach_pct":         eng.promo_reach_pct,
        "avg_engagement_score":    eng.avg_engagement_score,
        "frubon_threat_pct":       comp.frubon_threat_pct,
        "best_promo_brand":        comp.best_promo_brand,
        "price_loss_pct":          comp.price_driven_loss_pct,
        "net_growth_signal":       sales.net_growth_signal,
        "top_complaint":           comp_kpi.top_complaint,
        "sourness_pct":            comp_kpi.sourness_complaint_pct,
        "opportunity_crisis": (
            f"{pen.unawareness_rate:.1f}% unaware | "
            f"{infra.cold_chain_gap:.1f}% cold-chain gap | "
            f"{100 - eng.promo_reach_pct:.1f}% without promotions"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL KPI ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def run_global_kpis(df: pd.DataFrame) -> GlobalKPISummary:
    """Compute all KPI blocks and return the unified GlobalKPISummary."""
    logger.info("▶ Computing global KPIs on %d retailers…", len(df))

    # Score all retailers first so opportunity metrics are available downstream
    df = compute_opportunity_score(df)

    pen   = compute_penetration_kpi(df)
    infra = compute_infrastructure_kpi(df)
    comp  = compute_competitor_kpi(df)
    eng   = compute_engagement_kpi(df)
    sales = compute_sales_kpi(df)
    compl = compute_complaint_kpi(df)
    opp   = compute_opportunity_kpi(df)
    headline = build_headline_metrics(pen, infra, comp, eng, sales, compl)

    summary = GlobalKPISummary(
        computed_at   = datetime.utcnow().isoformat(timespec="seconds") + "Z",
        penetration   = pen,
        infrastructure= infra,
        competitor    = comp,
        engagement    = eng,
        sales         = sales,
        complaints    = compl,
        opportunity   = opp,
        headline_metrics = headline,
    )
    logger.info("✔ Global KPI computation complete.")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# EXPORTS — CSV + PostgreSQL snapshot
# ──────────────────────────────────────────────────────────────────────────────

def export_kpi_summary(summary: GlobalKPISummary, tag: str = "") -> Path:
    """Flatten GlobalKPISummary to a single-row CSV and return the path."""
    flat: dict[str, Any] = {"computed_at": summary.computed_at}
    for block_name, block in [
        ("penetration",    summary.penetration),
        ("infrastructure", summary.infrastructure),
        ("competitor",     summary.competitor),
        ("engagement",     summary.engagement),
        ("sales",          summary.sales),
        ("complaints",     summary.complaints),
    ]:
        for k, v in asdict(block).items():
            flat[f"{block_name}__{k}"] = v

    # Opportunity (skip the top-10 list)
    opp = asdict(summary.opportunity)
    for k, v in opp.items():
        if k != "top_10_opportunity_retailers":
            flat[f"opportunity__{k}"] = v

    # Headline metrics
    for k, v in summary.headline_metrics.items():
        flat[f"headline__{k}"] = v

    suffix = f"_{tag}" if tag else ""
    ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path   = EXPORTS_DIR / f"kpi_global{suffix}_{ts}.csv"
    pd.DataFrame([flat]).to_csv(path, index=False)
    logger.info("📄 Global KPI summary exported → %s", path)
    return path


def export_zone_kpis(zone_df: pd.DataFrame, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path   = EXPORTS_DIR / f"kpi_zones{suffix}_{ts}.csv"
    zone_df.to_csv(path, index=False)
    logger.info("📄 Zone KPIs exported → %s", path)
    return path


def export_opportunity_targets(df: pd.DataFrame, tag: str = "") -> Path:
    """Export the top opportunity non-stockers ranked by score."""
    scored     = compute_opportunity_score(df)
    non_stock  = scored[~scored["stocks_ct"].astype(bool)].copy()
    non_stock  = non_stock.sort_values("opportunity_score", ascending=False)

    cols = [c for c in [
        "Sr. No.", "Shop Name", "Zone", "zone_clean", "store_type_clean",
        "Area / Locality", "Amul Rep Visit Frequency",
        "feat_adoption_likelihood", "feat_engagement_score",
        "feat_infrastructure_score", "feat_masti_weekly_volume",
        "opportunity_score", "opportunity_tier", "awareness_tier",
    ] if c in non_stock.columns]

    suffix = f"_{tag}" if tag else ""
    ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path   = EXPORTS_DIR / f"opportunity_targets{suffix}_{ts}.csv"
    non_stock[cols].to_csv(path, index=False)
    logger.info("📄 Opportunity targets exported → %s  (%d rows)", path, len(non_stock))
    return path


def export_segment_kpis(seg_df: pd.DataFrame, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path   = EXPORTS_DIR / f"kpi_segments{suffix}_{ts}.csv"
    seg_df.to_csv(path, index=False)
    logger.info("📄 Segment KPIs exported → %s", path)
    return path


def write_kpi_snapshot(engine: Engine, summary: GlobalKPISummary, scope: str = "global") -> None:
    """Persist current global KPI snapshot to the kpi_snapshots table."""
    p = summary.penetration
    e = summary.engagement
    c = summary.competitor
    cmpl = summary.complaints

    sql = text("""
        INSERT INTO kpi_snapshots (
            snapshot_date, scope, scope_ref_label,
            total_retailers, ct_stockers, ct_penetration_rate,
            ct_awareness_rate, trade_promo_reach, weekly_rep_freq_pct,
            frubon_threat_pct, complaint_rate, sourness_complaint_rate,
            cup_format_pref_pct, avg_adoption_likelihood
        ) VALUES (
            :snapshot_date, :scope, :scope_ref_label,
            :total_retailers, :ct_stockers, :ct_penetration_rate,
            :ct_awareness_rate, :trade_promo_reach, :weekly_rep_freq_pct,
            :frubon_threat_pct, :complaint_rate, :sourness_complaint_rate,
            :cup_format_pref_pct, :avg_adoption_likelihood
        )
        ON CONFLICT (snapshot_date, scope, scope_ref_id) DO UPDATE
            SET ct_penetration_rate = EXCLUDED.ct_penetration_rate,
                computed_at         = NOW()
    """)

    params = {
        "snapshot_date":          date.today().isoformat(),
        "scope":                  scope,
        "scope_ref_label":        "Jaipur Market",
        "total_retailers":        p.total_retailers,
        "ct_stockers":            p.ct_stockers,
        "ct_penetration_rate":    p.ct_penetration_rate / 100,
        "ct_awareness_rate":      p.awareness_rate / 100,
        "trade_promo_reach":      e.promo_reach_pct / 100,
        "weekly_rep_freq_pct":    e.weekly_rep_visit_pct / 100,
        "frubon_threat_pct":      c.frubon_threat_pct / 100,
        "complaint_rate":         (100 - cmpl.no_complaint_pct) / 100,
        "sourness_complaint_rate":cmpl.sourness_complaint_pct / 100,
        "cup_format_pref_pct":    summary.sales.cup_format_preference_pct / 100,
        "avg_adoption_likelihood":summary.opportunity.avg_opportunity_score,
    }

    try:
        with db_session(engine) as conn:
            conn.execute(sql, params)
        logger.info("✔ KPI snapshot written to database (scope=%s)", scope)
    except Exception as exc:
        logger.error("Failed to write KPI snapshot: %s", exc, exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION-READY OUTPUTS
# ──────────────────────────────────────────────────────────────────────────────

def awareness_funnel_chart_data(pen: PenetrationKPI) -> list[dict[str, Any]]:
    """Return list of dicts ready for a Plotly funnel chart."""
    return [
        {"stage": "Total Retailers",       "count": pen.total_retailers,   "pct": 100.0},
        {"stage": "Aware of C&T",          "count": pen.ct_stockers + pen.aware_only_count,
                                                                            "pct": pen.awareness_rate},
        {"stage": "Stocking C&T",          "count": pen.ct_stockers,       "pct": pen.ct_penetration_rate},
    ]


def zone_heatmap_data(zone_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return zone KPIs as a list of dicts for a Plotly choropleth / heatmap."""
    cols = ["zone_clean", "zone_priority", "total_retailers",
            "ct_penetration_pct", "awareness_rate_pct",
            "refrigeration_pct", "cold_chain_gap_pct",
            "avg_frubon_threat", "opportunity_gap"]
    cols = [c for c in cols if c in zone_df.columns]
    return zone_df[cols].to_dict("records")


def competitor_bar_chart_data(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Brand share data for a grouped bar chart."""
    brand_col = "#1 Selling Dahi Brand at Store"
    if brand_col not in df.columns:
        return []
    vc = df[brand_col].value_counts(normalize=True).mul(100).round(2).reset_index()
    vc.columns = ["brand", "share_pct"]
    return vc.to_dict("records")


def engagement_distribution_data(df: pd.DataFrame) -> dict[str, Any]:
    """Rep visit and promo distribution for stacked bar chart."""
    rep_col = "rep_visit_clean" if "rep_visit_clean" in df.columns else "Amul Rep Visit Frequency"
    return {
        "rep_visit": df[rep_col].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "promo_received": {
            "Yes": _safe_pct(df.get("promo_received_clean", pd.Series()).str.contains("yes", na=False).sum(), len(df)),
            "No":  _safe_pct(df.get("promo_received_clean", pd.Series()).str.contains("never", na=False).sum(), len(df)),
        },
    }


def complaint_donut_data(compl: ComplaintKPI) -> list[dict[str, Any]]:
    """Complaint distribution for a donut chart."""
    return [
        {"label": "Sourness",         "pct": compl.sourness_complaint_pct},
        {"label": "Texture",          "pct": compl.texture_complaint_pct},
        {"label": "Packaging Leak",   "pct": compl.packaging_complaint_pct},
        {"label": "No Complaints",    "pct": compl.no_complaint_pct},
        {"label": "Other",            "pct": max(0.0, 100 - compl.sourness_complaint_pct
                                                          - compl.texture_complaint_pct
                                                          - compl.packaging_complaint_pct
                                                          - compl.no_complaint_pct)},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def load_from_csv(path: str | Path) -> pd.DataFrame:
    """Load and lightly type-cast the cleaned retailers CSV."""
    logger.info("Loading data from CSV: %s", path)
    df = pd.read_csv(path, low_memory=False)

    bool_cols = ["stocks_ct", "feat_refrigerator_flag", "feat_dedicated_fridge"]
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
            df[col] = df[col].astype(bool)

    float_cols = [c for c in df.columns if c.startswith("feat_")]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


def load_from_db(engine: Engine) -> pd.DataFrame:
    """Pull the master analytical view from PostgreSQL."""
    logger.info("Loading master dataset from PostgreSQL…")
    df = query_to_df(engine, SQL_MASTER_VIEW)
    logger.info("Loaded %d rows from DB", len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def run_kpi_engine(
    source: str = "csv",
    csv_path: str | Path = "data/processed/retailers_clean.csv",
    export: bool = True,
    write_snapshot: bool = False,
    engine: Engine | None = None,
    tag: str = "",
) -> dict[str, Any]:
    """
    Top-level runner. Returns a dict of all computed outputs.

    Parameters
    ----------
    source         : 'csv' | 'db'
    csv_path       : Path to cleaned CSV (used when source='csv')
    export         : Whether to write CSV exports
    write_snapshot : Whether to persist snapshot to PostgreSQL
    engine         : SQLAlchemy engine (required for source='db' or write_snapshot=True)
    tag            : Optional label appended to export filenames

    Returns
    -------
    dict with keys:
        summary, zone_kpis, zone_opportunity, segment_kpis,
        store_type_kpis, promotion_effectiveness, opportunity_targets,
        chart_data, export_paths
    """
    logger.info("═══════════════════════════════════════════════")
    logger.info("  FMCG KPI Engine  |  source=%s  |  tag=%s", source, tag or "none")
    logger.info("═══════════════════════════════════════════════")

    # ── 1. Load data ──
    if source == "db":
        if engine is None:
            engine = build_engine()
        df = load_from_db(engine)
    else:
        df = load_from_csv(csv_path)

    # ── 2. Score all retailers ──
    df = compute_opportunity_score(df)

    # ── 3. Global KPIs ──
    summary = run_global_kpis(df)

    # ── 4. Dimensional tables ──
    zone_kpis          = compute_zone_kpis(df)
    zone_opportunity   = compute_zone_opportunity_matrix(zone_kpis)
    segment_kpis       = compute_segment_kpis(df)
    store_type_kpis    = compute_store_type_kpis(df)
    promo_effectiveness= compute_promotion_effectiveness(df)

    # ── 5. Top opportunity targets ──
    non_stockers = df[~df["stocks_ct"].astype(bool)].sort_values("opportunity_score", ascending=False)

    # ── 6. Chart-ready payloads ──
    chart_data = {
        "awareness_funnel":          awareness_funnel_chart_data(summary.penetration),
        "zone_heatmap":              zone_heatmap_data(zone_kpis),
        "competitor_brand_share":    competitor_bar_chart_data(df),
        "engagement_distribution":   engagement_distribution_data(df),
        "complaint_donut":           complaint_donut_data(summary.complaints),
        "headline_metrics":          summary.headline_metrics,
    }

    # ── 7. Exports ──
    export_paths: dict[str, Path] = {}
    if export:
        export_paths["global_kpi"]           = export_kpi_summary(summary, tag)
        export_paths["zone_kpis"]            = export_zone_kpis(zone_kpis, tag)
        export_paths["opportunity_targets"]  = export_opportunity_targets(df, tag)
        if segment_kpis is not None:
            export_paths["segment_kpis"]     = export_segment_kpis(segment_kpis, tag)
        p2 = EXPORTS_DIR / f"store_type_kpis_{tag}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        store_type_kpis.to_csv(p2, index=False)
        export_paths["store_type_kpis"] = p2
        p3 = EXPORTS_DIR / f"promo_effectiveness_{tag}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        promo_effectiveness.to_csv(p3, index=False)
        export_paths["promo_effectiveness"] = p3

    # ── 8. DB snapshot ──
    if write_snapshot:
        if engine is None:
            engine = build_engine()
        write_kpi_snapshot(engine, summary)

    logger.info("══ KPI Engine run complete. %d exports written. ══", len(export_paths))

    return {
        "summary":               summary,
        "zone_kpis":             zone_kpis,
        "zone_opportunity":      zone_opportunity,
        "segment_kpis":          segment_kpis,
        "store_type_kpis":       store_type_kpis,
        "promotion_effectiveness": promo_effectiveness,
        "opportunity_targets":   non_stockers,
        "chart_data":            chart_data,
        "export_paths":          export_paths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FMCG KPI Analytics Engine")
    parser.add_argument("--source",   choices=["csv", "db"], default="csv",
                        help="Data source: 'csv' (default) or 'db' (PostgreSQL)")
    parser.add_argument("--csv",      default="data/processed/retailers_clean.csv",
                        help="Path to cleaned retailers CSV (csv mode)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip CSV exports")
    parser.add_argument("--snapshot",  action="store_true",
                        help="Write KPI snapshot to PostgreSQL")
    parser.add_argument("--tag",       default="",
                        help="Label appended to export filenames")
    args = parser.parse_args()

    eng = build_engine() if args.source == "db" or args.snapshot else None

    results = run_kpi_engine(
        source          = args.source,
        csv_path        = args.csv,
        export          = not args.no_export,
        write_snapshot  = args.snapshot,
        engine          = eng,
        tag             = args.tag,
    )

    s: GlobalKPISummary = results["summary"]
    print("\n" + "═" * 60)
    print("  FMCG RETAIL INTELLIGENCE — KPI SUMMARY")
    print("═" * 60)
    p = s.penetration
    print(f"  Total Retailers      : {p.total_retailers}")
    print(f"  C&T Penetration      : {p.ct_penetration_rate:.1f}%")
    print(f"  Awareness Rate       : {p.awareness_rate:.1f}%")
    print(f"  Unawareness Rate     : {p.unawareness_rate:.1f}%")
    print(f"  Refrigeration Rate   : {s.infrastructure.refrigeration_rate:.1f}%")
    print(f"  Cold-Chain Gap       : {s.infrastructure.cold_chain_gap:.1f}%")
    print(f"  Promo Reach          : {s.engagement.promo_reach_pct:.1f}%")
    print(f"  Avg Engagement Score : {s.engagement.avg_engagement_score:.2f}")
    print(f"  FruBon Threat        : {s.competitor.frubon_threat_pct:.1f}%")
    print(f"  Best Promo Brand     : {s.competitor.best_promo_brand}")
    print(f"  Top Complaint        : {s.complaints.top_complaint}")
    print(f"  Net Growth Signal    : {s.sales.net_growth_signal:.1f}%")
    print("═" * 60)
    if results["export_paths"]:
        print("\n  EXPORTS:")
        for name, path in results["export_paths"].items():
            print(f"    {name:<25} → {path}")
    print()
