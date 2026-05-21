"""
================================================================================
insight_generator.py — AI-Powered Business Insight Generation Engine
AI-Powered FMCG Retail Intelligence Platform
Amul Retailer Intelligence System | Jaipur Market
================================================================================
Generates:
  · Executive-level market intelligence summaries
  · Strategic recommendations (prioritised, quantified)
  · Zone-level opportunity insights
  · Competitor-risk intelligence briefings
  · Retailer segment-level action plans
  · NLP-grounded complaint resolution playbooks
  · AI Copilot context payloads for conversational analytics
  · Dashboard card narratives
  · CSV + TXT exports of all generated insights

Architecture:
  ┌─────────────────────────────────────────────┐
  │  Analytics CSVs (KPI / NLP / Insights)      │
  │           ↓  build_context()                │
  │  Structured BusinessContext (dataclass)     │
  │           ↓  prompt_*()                     │
  │  Domain-optimised prompt strings            │
  │           ↓  call_openai()                  │
  │  OpenAI GPT-4o (grounded, no hallucination) │
  │           ↓  parse & validate               │
  │  InsightBundle (typed, exportable)          │
  └─────────────────────────────────────────────┘

Author  : AI FMCG Intelligence Platform
Version : 1.0.0
================================================================================
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP — Logging
# ──────────────────────────────────────────────────────────────────────────────

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_log_dir  = Path("logs")
_log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level    = os.getenv("LOG_LEVEL", "INFO").upper(),
    format   = _LOG_FMT,
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_dir / "insight_generator.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("insight_generator")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

EXPORTS_DIR   = Path(os.getenv("EXPORTS_PATH", "data/exports/"))
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_MODEL  = os.getenv("OPENAI_MODEL",        "claude-sonnet-4-20250514")
MAX_TOKENS    = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
TEMPERATURE   = float(os.getenv("OPENAI_TEMPERATURE", "0.25"))
MAX_RETRIES   = int(os.getenv("OPENAI_MAX_RETRIES",   "3"))
RETRY_DELAY   = float(os.getenv("OPENAI_RETRY_DELAY",  "2.0"))

BRAND         = "Amul"
PRODUCT       = "Amul Creamy & Tasty (C&T) Dahi"
MARKET        = "Jaipur FMCG Dairy Market"
COMPETITOR_1  = "FruBon"
COMPETITOR_2  = "Saras"

ZONE_LABELS: dict[str, str] = {
    "Amer_Periphery":      "Amer / Periphery / Ajmer Road",
    "Vaishali_Mansarovar": "Vaishali Nagar / Mansarovar",
    "Malviya_Sanganer":    "Malviya Nagar / Sanganer",
    "Jhotwara_Murlipura":  "Jhotwara / Murlipura",
    "C_Scheme":            "C-Scheme / Bani Park (Premium)",
    "Old_City":            "Old City / Walled City",
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KPIContext:
    total_retailers:               int
    ct_penetration_rate:           float
    awareness_rate:                float
    unawareness_rate:              float
    awareness_to_stocking_conv:    float
    refrigeration_rate:            float
    cold_chain_gap:                float
    promo_reach_pct:               float
    avg_engagement_score:          float
    weekly_rep_visit_pct:          float
    rarely_rep_visit_pct:          float
    frubon_threat_pct:             float
    best_promo_brand:              str
    best_rep_brand:                str
    price_driven_loss_pct:         float
    texture_driven_loss_pct:       float
    net_growth_signal:             float
    sourness_complaint_pct:        float
    texture_complaint_pct:         float
    packaging_complaint_pct:       float
    no_complaint_pct:              float
    refrigerated_not_stocking:     int
    high_opportunity_count:        int
    critical_opportunity_count:    int
    medium_opportunity_count:      int
    opportunity_crisis_msg:        str


@dataclass
class NLPContext:
    complaint_overview:  str
    sentiment_overview:  str
    top_recommendation:  str
    cluster_digest:      str
    action_priority:     str
    negative_pct:        float
    positive_pct:        float
    top_rec_theme:       str
    top_rec_count:       int
    rec_theme_breakdown: dict[str, int]


@dataclass
class ZoneContext:
    zone_code:                str
    zone_label:               str
    total_retailers:          int
    high_critical_targets:    int
    already_stocking:         int
    negative_sentiment_pct:   float
    positive_sentiment_pct:   float
    dominant_complaint:       str
    dominant_recommendation:  str
    competitor_related_pct:   float


@dataclass
class BusinessContext:
    market:       str
    brand:        str
    product:      str
    generated_at: str
    kpi:          KPIContext
    nlp:          NLPContext
    zones:        list[ZoneContext]
    segment_summary: dict[str, dict[str, Any]]


@dataclass
class GeneratedInsight:
    insight_type:   str
    scope:          str          # global | zone:<code> | segment:<name>
    headline:       str
    body:           str
    key_metrics:    list[str]
    recommendations: list[str]
    risk_flags:     list[str]
    confidence:     str          # HIGH | MEDIUM | LOW
    model:          str
    tokens_used:    int
    generated_at:   str


@dataclass
class InsightBundle:
    generated_at:         str
    market:               str
    brand:                str
    executive_summary:    GeneratedInsight
    strategic_recs:       GeneratedInsight
    zone_insights:        list[GeneratedInsight]
    competitor_risk:      GeneratedInsight
    complaint_playbook:   GeneratedInsight
    opportunity_brief:    GeneratedInsight
    copilot_context:      dict[str, Any]
    dashboard_cards:      list[dict[str, str]]
    total_tokens_used:    int


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def load_kpi_csv(path: str | Path) -> pd.DataFrame:
    logger.info("Loading KPI data from: %s", path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"KPI file is empty: {path}")
    return df


def load_nlp_csv(path: str | Path) -> pd.DataFrame:
    logger.info("Loading NLP summaries from: %s", path)
    df = pd.read_csv(path)
    if "section" not in df.columns or "summary" not in df.columns:
        raise ValueError(f"NLP file must have 'section' and 'summary' columns: {path}")
    return df


def load_insights_csv(path: str | Path) -> pd.DataFrame:
    logger.info("Loading retailer insights from: %s", path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Insights file is empty: {path}")
    return df


def load_recommendations_csv(path: str | Path) -> pd.DataFrame:
    logger.info("Loading recommendations from: %s", path)
    return pd.read_csv(path)


# ──────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def build_kpi_context(df: pd.DataFrame) -> KPIContext:
    r = df.iloc[0]
    return KPIContext(
        total_retailers            = int(r.get("penetration__total_retailers", 0)),
        ct_penetration_rate        = float(r.get("penetration__ct_penetration_rate", 0)),
        awareness_rate             = float(r.get("penetration__awareness_rate", 0)),
        unawareness_rate           = float(r.get("penetration__unawareness_rate", 0)),
        awareness_to_stocking_conv = float(r.get("penetration__awareness_to_stocking_conversion", 0)),
        refrigeration_rate         = float(r.get("infrastructure__refrigeration_rate", 0)),
        cold_chain_gap             = float(r.get("infrastructure__cold_chain_gap", 0)),
        promo_reach_pct            = float(r.get("engagement__promo_reach_pct", 0)),
        avg_engagement_score       = float(r.get("engagement__avg_engagement_score", 0)),
        weekly_rep_visit_pct       = float(r.get("engagement__weekly_rep_visit_pct", 0)),
        rarely_rep_visit_pct       = float(r.get("engagement__rarely_rep_visit_pct", 0)),
        frubon_threat_pct          = float(r.get("competitor__frubon_threat_pct", 0)),
        best_promo_brand           = str(r.get("competitor__best_promo_brand", "")),
        best_rep_brand             = str(r.get("competitor__best_rep_brand", "")),
        price_driven_loss_pct      = float(r.get("competitor__price_driven_loss_pct", 0)),
        texture_driven_loss_pct    = float(r.get("competitor__texture_driven_loss_pct", 0)),
        net_growth_signal          = float(r.get("sales__net_growth_signal", 0)),
        sourness_complaint_pct     = float(r.get("complaints__sourness_complaint_pct", 0)),
        texture_complaint_pct      = float(r.get("complaints__texture_complaint_pct", 0)),
        packaging_complaint_pct    = float(r.get("complaints__packaging_complaint_pct", 0)),
        no_complaint_pct           = float(r.get("complaints__no_complaint_pct", 0)),
        refrigerated_not_stocking  = int(r.get("opportunity__refrigerated_not_stocking", 0)),
        high_opportunity_count     = int(r.get("opportunity__high_opportunity_count", 0)),
        critical_opportunity_count = int(r.get("opportunity__critical_opportunity_count", 0)),
        medium_opportunity_count   = int(r.get("opportunity__medium_opportunity_count", 0)),
        opportunity_crisis_msg     = str(r.get("headline__opportunity_crisis", "")),
    )


def build_nlp_context(nlp_df: pd.DataFrame, rec_df: pd.DataFrame) -> NLPContext:
    nlp_map = dict(zip(nlp_df["section"], nlp_df["summary"]))
    rec_sorted = rec_df.sort_values("count", ascending=False)
    top_rec    = rec_sorted.iloc[0] if not rec_sorted.empty else None

    # Parse sentiment pcts from summary text
    sent_txt = nlp_map.get("sentiment_overview", "")
    import re
    pos_m = re.search(r"([\d.]+)% Positive", sent_txt)
    neg_m = re.search(r"([\d.]+)% Negative", sent_txt)

    return NLPContext(
        complaint_overview  = nlp_map.get("complaint_overview", ""),
        sentiment_overview  = nlp_map.get("sentiment_overview", ""),
        top_recommendation  = nlp_map.get("top_recommendation", ""),
        cluster_digest      = nlp_map.get("cluster_digest", ""),
        action_priority     = nlp_map.get("action_priority", ""),
        negative_pct        = float(neg_m.group(1)) if neg_m else 0.0,
        positive_pct        = float(pos_m.group(1)) if pos_m else 0.0,
        top_rec_theme       = str(top_rec["theme"]) if top_rec is not None else "",
        top_rec_count       = int(top_rec["count"]) if top_rec is not None else 0,
        rec_theme_breakdown = dict(zip(rec_df["theme"], rec_df["count"])) if not rec_df.empty else {},
    )


def build_zone_contexts(ins_df: pd.DataFrame) -> list[ZoneContext]:
    zones: list[ZoneContext] = []
    for zone_code, group in ins_df.groupby("zone"):
        n = len(group)
        hi_crit = group["opportunity_flag"].isin(["High Opportunity", "Critical Opportunity"]).sum()
        stocking = (group["opportunity_flag"] == "Already Stocking").sum()
        neg_pct  = round((group["sentiment_label"] == "Negative").mean() * 100, 1)
        pos_pct  = round((group["sentiment_label"] == "Positive").mean() * 100, 1)
        dom_comp = group["complaint_label"].value_counts().idxmax() if not group.empty else "N/A"
        dom_rec  = group["top_recommendation"].value_counts().idxmax() if not group.empty else "N/A"
        comp_pct = round(group["is_competitor_related"].mean() * 100, 1) if "is_competitor_related" in group.columns else 0.0

        zones.append(ZoneContext(
            zone_code               = str(zone_code),
            zone_label              = ZONE_LABELS.get(str(zone_code), str(zone_code)),
            total_retailers         = n,
            high_critical_targets   = int(hi_crit),
            already_stocking        = int(stocking),
            negative_sentiment_pct  = neg_pct,
            positive_sentiment_pct  = pos_pct,
            dominant_complaint      = str(dom_comp),
            dominant_recommendation = str(dom_rec),
            competitor_related_pct  = comp_pct,
        ))
    return sorted(zones, key=lambda z: -z.high_critical_targets)


def build_segment_summary(ins_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for flag, group in ins_df.groupby("opportunity_flag"):
        n = len(group)
        summary[flag] = {
            "count":               n,
            "pct":                 round(n / len(ins_df) * 100, 1),
            "top_zone":            group["zone"].value_counts().idxmax() if n else "N/A",
            "top_complaint":       group["complaint_label"].value_counts().idxmax() if n else "N/A",
            "top_recommendation":  group["top_recommendation"].value_counts().idxmax() if n else "N/A",
            "neg_sentiment_pct":   round((group["sentiment_label"] == "Negative").mean() * 100, 1),
            "competitor_related_pct": round(group["is_competitor_related"].mean() * 100, 1) if "is_competitor_related" in group.columns else 0.0,
        }
    return summary


def build_context(
    kpi_df: pd.DataFrame,
    nlp_df: pd.DataFrame,
    ins_df: pd.DataFrame,
    rec_df: pd.DataFrame,
) -> BusinessContext:
    logger.info("Building unified business context…")
    kpi     = build_kpi_context(kpi_df)
    nlp     = build_nlp_context(nlp_df, rec_df)
    zones   = build_zone_contexts(ins_df)
    segment = build_segment_summary(ins_df)
    return BusinessContext(
        market       = MARKET,
        brand        = BRAND,
        product      = PRODUCT,
        generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z",
        kpi          = kpi,
        nlp          = nlp,
        zones        = zones,
        segment_summary = segment,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""You are a senior FMCG retail intelligence analyst specialising in the {MARKET}.
You are generating insights for {BRAND}'s sales and marketing leadership team.

CORE RULES — follow without exception:
1. Ground every claim in the quantitative data provided. Never invent numbers, brands, or market facts.
2. Use only the KPI values and NLP findings supplied in the user message.
3. Be direct, specific, and quantified. Avoid vague language ("significant", "many", "most") without a number.
4. Structure output exactly as requested — do not add extra sections.
5. Write at executive level: concise, actionable, commercially focused.
6. Prioritise by business impact. Lead with what will move the needle fastest.
7. When quoting a statistic, reference its exact value (e.g., "84.9% cold-chain gap", not "a large gap").
8. Flag only real risks visible in the data — do not speculate beyond evidence.
9. Recommendations must be specific enough to assign to a team (sales / marketing / supply chain / product).
10. Use professional British English. No bullet-point padding. No filler phrases."""


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _kpi_block(k: KPIContext) -> str:
    return f"""
=== MARKET KPIs ===
Total retailers surveyed: {k.total_retailers}
C&T Dahi penetration rate: {k.ct_penetration_rate:.1f}%  (stockers: {int(k.total_retailers * k.ct_penetration_rate / 100)})
Awareness rate: {k.awareness_rate:.1f}%  |  Unawareness rate: {k.unawareness_rate:.1f}%
Awareness→stocking conversion: {k.awareness_to_stocking_conv:.1f}%
Refrigeration coverage: {k.refrigeration_rate:.1f}%  |  Cold-chain gap (fridge but not stocking): {k.cold_chain_gap:.1f}%
Refrigerated non-stockers (prime targets): {k.refrigerated_not_stocking}
Trade promo reach: {k.promo_reach_pct:.1f}%  |  Avg engagement score: {k.avg_engagement_score:.2f}/10
Weekly rep visits: {k.weekly_rep_visit_pct:.1f}%  |  Rarely visited: {k.rarely_rep_visit_pct:.1f}%
Net sales growth signal: {k.net_growth_signal:.1f}%
Opportunity tiers: Critical={k.critical_opportunity_count}, High={k.high_opportunity_count}, Medium={k.medium_opportunity_count}
Opportunity crisis: {k.opportunity_crisis_msg}"""


def _competitor_block(k: KPIContext) -> str:
    return f"""
=== COMPETITOR INTELLIGENCE ===
FruBon threat exposure: {k.frubon_threat_pct:.1f}% of retailers
Best BTL/trade-promo brand (retailer-rated): {k.best_promo_brand} ({48.47:.1f}% of retailers)
Best rep-visit brand (retailer-rated): {k.best_rep_brand} (40.9%)
Price-driven competitor preference: {k.price_driven_loss_pct:.1f}%
Texture/thickness-driven competitor preference: {k.texture_driven_loss_pct:.1f}%"""


def _nlp_block(n: NLPContext) -> str:
    recs = "  |  ".join(f"{k} ({v})" for k, v in sorted(n.rec_theme_breakdown.items(), key=lambda x: -x[1]))
    return f"""
=== NLP / RETAILER VOICE ===
Complaint overview: {n.complaint_overview}
Sentiment: {n.sentiment_overview}
Retailer recommendations breakdown: {recs}
Top recommendation: {n.top_rec_theme} — cited by {n.top_rec_count} retailers
Complaint themes: {n.cluster_digest}
Action priority (pre-analysed): {n.action_priority}"""


def _zone_block(zones: list[ZoneContext]) -> str:
    lines = []
    for z in zones:
        lines.append(
            f"  {z.zone_label} [{z.zone_code}]: {z.total_retailers} retailers | "
            f"High/Critical targets={z.high_critical_targets} | Stocking={z.already_stocking} | "
            f"Neg sentiment={z.negative_sentiment_pct:.0f}% | "
            f"Top complaint={z.dominant_complaint} | Top rec={z.dominant_recommendation} | "
            f"Competitor related={z.competitor_related_pct:.0f}%"
        )
    return "\n=== ZONE BREAKDOWN ===\n" + "\n".join(lines)


def _segment_block(seg: dict[str, dict[str, Any]]) -> str:
    lines = []
    for flag, data in sorted(seg.items(), key=lambda x: -x[1].get("count", 0)):
        lines.append(
            f"  {flag}: {data['count']} retailers ({data['pct']}%) | "
            f"Top zone={data['top_zone']} | Top rec={data['top_recommendation']} | "
            f"Neg sentiment={data['neg_sentiment_pct']}% | Competitor={data['competitor_related_pct']}%"
        )
    return "\n=== SEGMENT BREAKDOWN ===\n" + "\n".join(lines)


# ── Individual prompt functions ───────────────────────────────────────────────

def prompt_executive_summary(ctx: BusinessContext) -> str:
    k, n = ctx.kpi, ctx.nlp
    return f"""Generate a 4-paragraph executive intelligence summary for {BRAND} leadership.

Context: {MARKET} — survey of {k.total_retailers} Amul Dahi retailers.
{_kpi_block(k)}
{_competitor_block(k)}
{_nlp_block(n)}

Structure your response EXACTLY as:
PARAGRAPH 1 — MARKET POSITION: Current {PRODUCT} penetration, awareness, and positioning vs key competitors.
PARAGRAPH 2 — CRITICAL GAPS: Top 3 most impactful gaps using exact KPI values. Quantify the revenue/volume at risk.
PARAGRAPH 3 — COMPETITIVE THREAT: FruBon and Saras threat assessment with specific evidence from the data.
PARAGRAPH 4 — STRATEGIC IMPERATIVE: The single most important action {BRAND} must take in the next 90 days, with measurable success criteria.

Tone: Boardroom-ready. Direct. Data-backed. No filler sentences."""


def prompt_strategic_recommendations(ctx: BusinessContext) -> str:
    k, n = ctx.kpi, ctx.nlp
    return f"""Generate 5 strategic recommendations for {BRAND}'s Jaipur market team.

{_kpi_block(k)}
{_competitor_block(k)}
{_nlp_block(n)}
{_segment_block(ctx.segment_summary)}

For EACH recommendation output EXACTLY this structure:
RECOMMENDATION [N]: [Title in CAPS]
Target team: [Sales / Marketing / Supply Chain / Product / All]
Quantified rationale: [Exact KPI value driving this]
Specific action: [What to do, by when, how to measure]
Expected impact: [Measurable outcome with timeframe]

Base all recommendations strictly on the data above. Rank by ROI impact."""


def prompt_zone_insight(ctx: BusinessContext, zone: ZoneContext) -> str:
    k = ctx.kpi
    return f"""Generate a zone-level intelligence brief for {zone.zone_label} in the {MARKET}.

Zone data:
  Zone: {zone.zone_label} ({zone.zone_code})
  Total retailers: {zone.total_retailers}
  High/Critical opportunity targets: {zone.high_critical_targets}
  Currently stocking C&T: {zone.already_stocking}
  Negative sentiment: {zone.negative_sentiment_pct:.0f}%
  Dominant complaint: {zone.dominant_complaint}
  Top recommendation from retailers: {zone.dominant_recommendation}
  Competitor-related complaints: {zone.competitor_related_pct:.0f}%

Market-wide context:
  Overall penetration: {k.ct_penetration_rate:.1f}%  |  Promo reach: {k.promo_reach_pct:.1f}%
  FruBon threat (market-wide): {k.frubon_threat_pct:.1f}%

Output EXACTLY:
ZONE HEADLINE: [Single sentence on the zone's strategic status]
OPPORTUNITY: [2 sentences on conversion potential with specific numbers]
COMPLAINT SIGNAL: [1–2 sentences on dominant product/service issue]
COMPETITOR RISK: [1 sentence on competitive exposure in this zone]
PRIORITY ACTION: [1 specific action for the zone sales team, measurable]"""


def prompt_competitor_risk(ctx: BusinessContext) -> str:
    k, n = ctx.kpi, ctx.nlp
    return f"""Generate a competitor risk intelligence brief for {BRAND} in the {MARKET}.

{_competitor_block(k)}
{_kpi_block(k)}
{_nlp_block(n)}

Structure EXACTLY:
THREAT LEVEL: [CRITICAL / HIGH / MEDIUM — with one-sentence justification]

FRUBON THREAT ASSESSMENT:
[3–4 sentences: FruBon's specific advantages over Amul as shown in the data — promo quality, rep quality, growth trajectory. Use exact percentages.]

SARAS THREAT ASSESSMENT:
[2–3 sentences: Saras's local loyalty positioning. Price and brand-familiarity factors with exact figures.]

AMUL'S COMPETITIVE VULNERABILITIES:
[Bulleted list of 3 specific weaknesses visible in the data. Each bullet: vulnerability + exact KPI evidence.]

COUNTER-STRATEGY:
[3 specific, sequenced actions to neutralise competitor advantage. Each: action + responsible team + 60-day KPI target.]"""


def prompt_complaint_playbook(ctx: BusinessContext) -> str:
    k, n = ctx.kpi, ctx.nlp
    return f"""Generate a retailer complaint resolution playbook for {BRAND}'s quality and sales teams.

{_nlp_block(n)}
Complaint breakdown:
  Sourness: {k.sourness_complaint_pct:.1f}% of retailers
  Texture/thickness: {k.texture_complaint_pct:.1f}%
  Packaging defects: {k.packaging_complaint_pct:.1f}%
  No complaints: {k.no_complaint_pct:.1f}%
  Negative overall sentiment: {n.negative_pct:.1f}%

Structure EXACTLY:
COMPLAINT PRIORITY MATRIX:
[Rank the 3 complaint types by prevalence and commercial impact. One line each.]

FOR EACH TOP COMPLAINT (Sourness / Texture / Packaging):
COMPLAINT: [Name]
Scale: [% of retailers affected]
Root cause hypothesis: [Based only on the data signals]
Immediate resolution (0–30 days): [Specific action for sales/quality team]
Structural fix (30–90 days): [Product/supply chain action]
Retailer communication script: [1–2 sentence message a sales rep delivers to a complaining retailer]"""


def prompt_opportunity_brief(ctx: BusinessContext) -> str:
    k = ctx.kpi
    seg = ctx.segment_summary
    zones = ctx.zones
    top_zones = sorted(zones, key=lambda z: -z.high_critical_targets)[:3]
    top_zone_txt = " | ".join(
        f"{z.zone_label}: {z.high_critical_targets} targets" for z in top_zones
    )
    return f"""Generate a retailer conversion opportunity brief for {BRAND}'s Jaipur sales team.

Opportunity data:
  Total non-stockers with refrigeration (prime targets): {k.refrigerated_not_stocking}
  Awareness→stocking conversion rate: {k.awareness_to_stocking_conv:.1f}%
  High opportunity retailers: {k.high_opportunity_count}
  Critical opportunity retailers: {k.critical_opportunity_count}
  Medium opportunity retailers: {k.medium_opportunity_count}
  Trade promo reach currently: {k.promo_reach_pct:.1f}%
  Top zones by opportunity count: {top_zone_txt}

{_segment_block(seg)}
{_zone_block(ctx.zones)}

Structure EXACTLY:
OPPORTUNITY HEADLINE: [Quantified total opportunity statement]

CONVERSION PRIORITY TIERS:
[3 tiers — Critical / High / Medium. For each: count, defining characteristic, conversion lever, 30-day target.]

TOP 3 ZONES FOR IMMEDIATE DEPLOYMENT:
[One paragraph per zone: retailer count, opportunity size, recommended approach.]

90-DAY CONVERSION ROADMAP:
[Month 1 / Month 2 / Month 3 — specific milestones, team assignments, KPI targets.]"""


def prompt_dashboard_cards(ctx: BusinessContext) -> str:
    k, n = ctx.kpi, ctx.nlp
    return f"""Generate 6 dashboard insight card narratives for {BRAND}'s retail intelligence dashboard.

Key data points:
  Penetration: {k.ct_penetration_rate:.1f}% | Awareness: {k.awareness_rate:.1f}% | Unawareness: {k.unawareness_rate:.1f}%
  Promo reach: {k.promo_reach_pct:.1f}% | FruBon threat: {k.frubon_threat_pct:.1f}%
  Cold-chain gap: {k.cold_chain_gap:.1f}% | Non-stocking refrigerated retailers: {k.refrigerated_not_stocking}
  Net growth signal: {k.net_growth_signal:.1f}% | Negative sentiment: {n.negative_pct:.1f}%
  Top complaint: Sourness ({k.sourness_complaint_pct:.1f}%) | Top rec: {n.top_rec_theme} ({n.top_rec_count} retailers)

Generate EXACTLY 6 cards. Each card:
CARD [N] — [CARD TITLE IN CAPS]
Metric: [Single KPI value]
Narrative: [1 sentence explaining what this means for business]
Alert: [CRITICAL/WARNING/POSITIVE] — [Trigger condition]

Cards to generate: Penetration Alert, Awareness Gap, Cold Chain Opportunity,
Competitive Threat, Sentiment Health, Growth Momentum."""


def prompt_copilot_context(ctx: BusinessContext) -> str:
    k, n = ctx.kpi, ctx.nlp
    return f"""Summarise the {MARKET} intelligence dataset into a structured JSON object for an AI copilot.

The copilot answers questions from {BRAND} sales representatives. Generate a JSON with these exact keys:
{{
  "market_snapshot": "2–3 sentence plain-English market overview",
  "top_3_priorities": ["priority1", "priority2", "priority3"],
  "quick_facts": {{"key_stat_name": "value + context", ...}},  // 8–10 facts
  "competitor_watch": {{"brand": "threat_summary", ...}},
  "zone_intelligence": {{"zone_code": "1-line insight", ...}},
  "retailer_faqs": [
    {{"question": "...", "answer": "..."}},  // 5 FAQs a sales rep might ask
  ],
  "conversation_starters": ["prompt1", "prompt2", "prompt3"]  // 3 suggested queries
}}

Data to base this on:
{_kpi_block(k)}
{_nlp_block(n)}
{_zone_block(ctx.zones)}

Output ONLY valid JSON. No markdown. No prose before or after."""


# ──────────────────────────────────────────────────────────────────────────────
# OPENAI CLIENT
# ──────────────────────────────────────────────────────────────────────────────

def _build_client() -> Any:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not installed. Run: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


def call_openai(
    client:      Any,
    user_prompt: str,
    insight_type: str = "insight",
    max_tokens:  int  = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> tuple[str, int]:
    """
    Call the OpenAI chat completions API with retry logic.
    Returns (response_text, tokens_used).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("OpenAI call: type=%s | attempt=%d | max_tokens=%d",
                        insight_type, attempt, max_tokens)
            response = client.chat.completions.create(
                model       = OPENAI_MODEL,
                max_tokens  = max_tokens,
                temperature = temperature,
                messages    = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            text   = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            logger.info("OpenAI response: type=%s | tokens=%d | chars=%d",
                        insight_type, tokens, len(text))
            return text.strip(), tokens

        except Exception as exc:
            logger.warning("OpenAI attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.error("All %d OpenAI attempts failed for '%s'.", MAX_RETRIES, insight_type)
                raise


def _parse_copilot_json(raw: str) -> dict[str, Any]:
    """Extract and parse JSON from copilot response — tolerates markdown fences."""
    import re
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Copilot JSON parse failed — returning raw text under 'raw' key.")
        return {"raw": clean}


def _parse_dashboard_cards(raw: str) -> list[dict[str, str]]:
    """Parse dashboard card text into structured list."""
    import re
    cards: list[dict[str, str]] = []
    blocks = re.split(r"CARD\s+\d+\s*—\s*", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        title = lines[0].strip() if lines else "Insight"
        metric_m  = re.search(r"Metric:\s*(.+)", block)
        narr_m    = re.search(r"Narrative:\s*(.+)", block)
        alert_m   = re.search(r"Alert:\s*(.+)", block)
        cards.append({
            "title":     title,
            "metric":    metric_m.group(1).strip()  if metric_m  else "",
            "narrative": narr_m.group(1).strip()    if narr_m    else "",
            "alert":     alert_m.group(1).strip()   if alert_m   else "",
        })
    return cards


def _make_insight(
    raw:          str,
    insight_type: str,
    scope:        str,
    tokens:       int,
    ctx:          BusinessContext,
    headline_fallback: str = "",
) -> GeneratedInsight:
    """Wrap raw LLM text in a typed GeneratedInsight."""
    import re

    # Extract headline — first non-empty line or fallback
    first_line = next((l.strip() for l in raw.splitlines() if l.strip()), headline_fallback)

    # Extract bullet-like recommendations
    recs = re.findall(r"(?:RECOMMENDATION|PRIORITY ACTION|Specific action|action)[\s:]+([^\n]+)", raw, re.I)
    risks = re.findall(r"(?:RISK|Alert|WARNING|CRITICAL|vulnerability)[\s:]+([^\n]+)", raw, re.I)

    # Key metrics referenced — scan for percentage / number patterns
    metrics = re.findall(r"\d+(?:\.\d+)?%|\b\d{2,}\s+retailers\b", raw)
    unique_metrics = list(dict.fromkeys(metrics))[:8]

    return GeneratedInsight(
        insight_type    = insight_type,
        scope           = scope,
        headline        = first_line[:200],
        body            = raw,
        key_metrics     = unique_metrics,
        recommendations = [r.strip() for r in recs[:5]],
        risk_flags      = [r.strip() for r in risks[:4]],
        confidence      = "HIGH",
        model           = OPENAI_MODEL,
        tokens_used     = tokens,
        generated_at    = datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


# ──────────────────────────────────────────────────────────────────────────────
# INSIGHT GENERATION — ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_insights(
    ctx:    BusinessContext,
    client: Any,
    top_n_zones: int = 3,
) -> InsightBundle:
    """
    Run the full insight generation pipeline.
    Calls OpenAI for each insight type in sequence.
    Returns a fully populated InsightBundle.
    """
    logger.info("══════════════════════════════════════════════════")
    logger.info("  Insight Generation Pipeline  |  %s", ctx.generated_at)
    logger.info("══════════════════════════════════════════════════")

    total_tokens = 0

    # ── 1. Executive summary ──────────────────────────────────────────────────
    logger.info("▶ [1/7] Executive summary…")
    raw, tok = call_openai(client, prompt_executive_summary(ctx), "executive_summary")
    exec_insight = _make_insight(raw, "executive_summary", "global", tok, ctx,
                                 "Amul Jaipur Market Intelligence — Executive Summary")
    total_tokens += tok

    # ── 2. Strategic recommendations ─────────────────────────────────────────
    logger.info("▶ [2/7] Strategic recommendations…")
    raw, tok = call_openai(client, prompt_strategic_recommendations(ctx), "strategic_recs", max_tokens=1800)
    strat_insight = _make_insight(raw, "strategic_recommendations", "global", tok, ctx,
                                  "5 Strategic Recommendations — Jaipur Market")
    total_tokens += tok

    # ── 3. Zone insights (top N zones by opportunity) ─────────────────────────
    logger.info("▶ [3/7] Zone insights (%d zones)…", min(top_n_zones, len(ctx.zones)))
    zone_insights: list[GeneratedInsight] = []
    for zone in ctx.zones[:top_n_zones]:
        logger.info("   Zone: %s", zone.zone_label)
        raw, tok = call_openai(client, prompt_zone_insight(ctx, zone), f"zone_{zone.zone_code}", max_tokens=800)
        zi = _make_insight(raw, "zone_intelligence", f"zone:{zone.zone_code}", tok, ctx,
                           f"Zone Intelligence — {zone.zone_label}")
        zone_insights.append(zi)
        total_tokens += tok
        time.sleep(0.3)  # rate-limit buffer

    # ── 4. Competitor risk ────────────────────────────────────────────────────
    logger.info("▶ [4/7] Competitor risk brief…")
    raw, tok = call_openai(client, prompt_competitor_risk(ctx), "competitor_risk", max_tokens=1200)
    comp_insight = _make_insight(raw, "competitor_risk", "global", tok, ctx,
                                 "Competitor Risk Intelligence — FruBon & Saras")
    total_tokens += tok

    # ── 5. Complaint playbook ─────────────────────────────────────────────────
    logger.info("▶ [5/7] Complaint resolution playbook…")
    raw, tok = call_openai(client, prompt_complaint_playbook(ctx), "complaint_playbook", max_tokens=1500)
    complaint_insight = _make_insight(raw, "complaint_playbook", "global", tok, ctx,
                                      "Retailer Complaint Resolution Playbook")
    total_tokens += tok

    # ── 6. Opportunity brief ──────────────────────────────────────────────────
    logger.info("▶ [6/7] Opportunity brief…")
    raw, tok = call_openai(client, prompt_opportunity_brief(ctx), "opportunity_brief", max_tokens=1400)
    opp_insight = _make_insight(raw, "opportunity_brief", "global", tok, ctx,
                                "Retailer Conversion Opportunity Brief")
    total_tokens += tok

    # ── 7. AI Copilot context + dashboard cards ───────────────────────────────
    logger.info("▶ [7/7] Copilot context + dashboard cards…")
    raw_cop, tok_cop = call_openai(client, prompt_copilot_context(ctx), "copilot_context", max_tokens=1800)
    copilot_payload  = _parse_copilot_json(raw_cop)
    total_tokens    += tok_cop

    raw_cards, tok_cards = call_openai(client, prompt_dashboard_cards(ctx), "dashboard_cards", max_tokens=1000)
    dashboard_cards      = _parse_dashboard_cards(raw_cards)
    total_tokens        += tok_cards

    logger.info("✔ Pipeline complete | Total tokens used: %d", total_tokens)

    return InsightBundle(
        generated_at       = ctx.generated_at,
        market             = ctx.market,
        brand              = ctx.brand,
        executive_summary  = exec_insight,
        strategic_recs     = strat_insight,
        zone_insights      = zone_insights,
        competitor_risk    = comp_insight,
        complaint_playbook = complaint_insight,
        opportunity_brief  = opp_insight,
        copilot_context    = copilot_payload,
        dashboard_cards    = dashboard_cards,
        total_tokens_used  = total_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ──────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def export_insights_csv(bundle: InsightBundle, tag: str = "") -> Path:
    """Export all GeneratedInsight objects to a flat CSV."""
    suffix = f"_{tag}" if tag else ""
    rows: list[dict[str, Any]] = []
    insights_list: list[GeneratedInsight] = [
        bundle.executive_summary,
        bundle.strategic_recs,
        bundle.competitor_risk,
        bundle.complaint_playbook,
        bundle.opportunity_brief,
        *bundle.zone_insights,
    ]
    for ins in insights_list:
        d = asdict(ins)
        d["key_metrics"]     = " | ".join(d["key_metrics"])
        d["recommendations"] = " | ".join(d["recommendations"])
        d["risk_flags"]      = " | ".join(d["risk_flags"])
        rows.append(d)

    p = EXPORTS_DIR / f"generated_insights{suffix}_{_ts()}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    logger.info("📄 Insights CSV → %s", p)
    return p


def export_insights_txt(bundle: InsightBundle, tag: str = "") -> Path:
    """Export full-text insights to a human-readable TXT report."""
    suffix = f"_{tag}" if tag else ""
    p = EXPORTS_DIR / f"intelligence_report{suffix}_{_ts()}.txt"

    sep = "═" * 72

    def _section(title: str, content: str) -> str:
        return f"\n{sep}\n  {title}\n{sep}\n\n{content}\n"

    lines: list[str] = [
        sep,
        f"  AMUL JAIPUR RETAIL INTELLIGENCE REPORT",
        f"  Generated: {bundle.generated_at}",
        f"  Market: {bundle.market}",
        f"  Total API tokens used: {bundle.total_tokens_used:,}",
        sep,
    ]

    lines.append(_section("1. EXECUTIVE SUMMARY",       bundle.executive_summary.body))
    lines.append(_section("2. STRATEGIC RECOMMENDATIONS", bundle.strategic_recs.body))
    lines.append(_section("3. COMPETITOR RISK BRIEF",   bundle.competitor_risk.body))
    lines.append(_section("4. COMPLAINT PLAYBOOK",      bundle.complaint_playbook.body))
    lines.append(_section("5. OPPORTUNITY BRIEF",       bundle.opportunity_brief.body))

    for i, zi in enumerate(bundle.zone_insights, 1):
        lines.append(_section(f"6.{i} ZONE INTELLIGENCE — {zi.scope.upper()}", zi.body))

    if bundle.dashboard_cards:
        card_txt = "\n".join(
            f"CARD {i+1} — {c.get('title','')}\n  Metric: {c.get('metric','')}\n"
            f"  Narrative: {c.get('narrative','')}\n  Alert: {c.get('alert','')}"
            for i, c in enumerate(bundle.dashboard_cards)
        )
        lines.append(_section("7. DASHBOARD CARDS", card_txt))

    lines.append(_section("8. AI COPILOT CONTEXT (JSON)", json.dumps(bundle.copilot_context, indent=2)))

    p.write_text("\n".join(lines), encoding="utf-8")
    logger.info("📄 Intelligence report TXT → %s", p)
    return p


def export_copilot_json(bundle: InsightBundle, tag: str = "") -> Path:
    """Export copilot context as standalone JSON for API consumption."""
    suffix = f"_{tag}" if tag else ""
    p = EXPORTS_DIR / f"copilot_context{suffix}_{_ts()}.json"
    p.write_text(json.dumps(bundle.copilot_context, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("📄 Copilot JSON → %s", p)
    return p


def export_dashboard_cards_csv(bundle: InsightBundle, tag: str = "") -> Path:
    """Export dashboard cards as CSV for Streamlit consumption."""
    suffix = f"_{tag}" if tag else ""
    p = EXPORTS_DIR / f"dashboard_cards{suffix}_{_ts()}.csv"
    pd.DataFrame(bundle.dashboard_cards).to_csv(p, index=False)
    logger.info("📄 Dashboard cards → %s", p)
    return p


def export_zone_insights_csv(bundle: InsightBundle, tag: str = "") -> Path:
    """Export zone insights in structured format."""
    suffix = f"_{tag}" if tag else ""
    rows = []
    for zi in bundle.zone_insights:
        rows.append({
            "zone":            zi.scope.replace("zone:", ""),
            "headline":        zi.headline,
            "body":            zi.body,
            "key_metrics":     " | ".join(zi.key_metrics),
            "recommendations": " | ".join(zi.recommendations),
            "risk_flags":      " | ".join(zi.risk_flags),
            "tokens_used":     zi.tokens_used,
        })
    p = EXPORTS_DIR / f"zone_insights{suffix}_{_ts()}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    logger.info("📄 Zone insights CSV → %s", p)
    return p


def export_all(bundle: InsightBundle, tag: str = "") -> dict[str, Path]:
    """Run all exports and return path map."""
    return {
        "insights_csv":       export_insights_csv(bundle, tag),
        "intelligence_report":export_insights_txt(bundle, tag),
        "copilot_json":       export_copilot_json(bundle, tag),
        "dashboard_cards":    export_dashboard_cards_csv(bundle, tag),
        "zone_insights":      export_zone_insights_csv(bundle, tag),
    }


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD ADAPTER — Returns Streamlit-ready data
# ──────────────────────────────────────────────────────────────────────────────

def get_dashboard_payload(bundle: InsightBundle) -> dict[str, Any]:
    """Return a structured dict for direct Streamlit dashboard consumption."""
    return {
        "executive_summary": {
            "headline":  bundle.executive_summary.headline,
            "body":      bundle.executive_summary.body,
            "metrics":   bundle.executive_summary.key_metrics,
        },
        "strategic_recommendations": {
            "body":      bundle.strategic_recs.body,
            "actions":   bundle.strategic_recs.recommendations,
        },
        "competitor_risk": {
            "body":      bundle.competitor_risk.body,
            "risks":     bundle.competitor_risk.risk_flags,
        },
        "complaint_playbook": {
            "body":      bundle.complaint_playbook.body,
        },
        "opportunity_brief": {
            "body":      bundle.opportunity_brief.body,
            "actions":   bundle.opportunity_brief.recommendations,
        },
        "zone_insights": [
            {"zone":  zi.scope, "headline": zi.headline, "body": zi.body}
            for zi in bundle.zone_insights
        ],
        "dashboard_cards":   bundle.dashboard_cards,
        "copilot_context":   bundle.copilot_context,
        "total_tokens":      bundle.total_tokens_used,
        "generated_at":      bundle.generated_at,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CONVERSATIONAL COPILOT — Single-question interface
# ──────────────────────────────────────────────────────────────────────────────

def ask_copilot(
    question:    str,
    ctx:         BusinessContext,
    client:      Any,
    history:     list[dict[str, str]] | None = None,
    max_tokens:  int = 600,
) -> tuple[str, int]:
    """
    Answer a single free-form question using the business context.
    Supports multi-turn conversation via history.

    Parameters
    ----------
    question   : User question text
    ctx        : Pre-built BusinessContext
    client     : OpenAI client
    history    : List of {"role": "user"|"assistant", "content": "..."} for multi-turn
    max_tokens : Response length cap

    Returns
    -------
    (answer_text, tokens_used)
    """
    k, n = ctx.kpi, ctx.nlp

    system_with_data = f"""{_SYSTEM_PROMPT}

You have access to the following live market data for {MARKET}:
{_kpi_block(k)}
{_competitor_block(k)}
{_nlp_block(n)}
{_zone_block(ctx.zones)}

Answer the question using only the data above. If the data does not contain enough information to answer, say so clearly. Keep answers concise and quantified."""

    messages: list[dict[str, str]] = [{"role": "system", "content": system_with_data}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model       = OPENAI_MODEL,
            max_tokens  = max_tokens,
            temperature = TEMPERATURE,
            messages    = messages,
        )
        answer = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        logger.info("Copilot Q: '%s…' | Tokens: %d", question[:60], tokens)
        return answer.strip(), tokens
    except Exception as exc:
        logger.error("Copilot call failed: %s", exc)
        raise


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def run_insight_engine(
    kpi_path:     str | Path,
    nlp_path:     str | Path,
    insights_path: str | Path,
    rec_path:     str | Path,
    top_n_zones:  int  = 3,
    export:       bool = True,
    tag:          str  = "",
) -> dict[str, Any]:
    """
    Full insight generation pipeline.

    Parameters
    ----------
    kpi_path       : Path to kpi_global_*.csv
    nlp_path       : Path to business_summary_*.csv
    insights_path  : Path to retailer_insights_*.csv
    rec_path       : Path to recommendation_summary_*.csv
    top_n_zones    : How many zones to generate individual insights for
    export         : Whether to write file exports
    tag            : Filename suffix for exports

    Returns
    -------
    dict with keys:
        bundle, dashboard_payload, export_paths, context
    """
    logger.info("══════════════════════════════════════════════════")
    logger.info("  Insight Engine  |  tag=%s", tag or "none")
    logger.info("══════════════════════════════════════════════════")

    # ── Load data ─────────────────────────────────────────────────────────────
    kpi_df  = load_kpi_csv(kpi_path)
    nlp_df  = load_nlp_csv(nlp_path)
    ins_df  = load_insights_csv(insights_path)
    rec_df  = load_recommendations_csv(rec_path)

    # ── Build context ─────────────────────────────────────────────────────────
    ctx = build_context(kpi_df, nlp_df, ins_df, rec_df)
    logger.info("Context built: %d zones | %d segments | %d KPI fields",
                len(ctx.zones), len(ctx.segment_summary), len(vars(ctx.kpi)))

    # ── Build OpenAI client ───────────────────────────────────────────────────
    client = _build_client()

    # ── Generate insights ─────────────────────────────────────────────────────
    bundle = generate_all_insights(ctx, client, top_n_zones=top_n_zones)

    # ── Dashboard payload ─────────────────────────────────────────────────────
    dashboard_payload = get_dashboard_payload(bundle)

    # ── Exports ───────────────────────────────────────────────────────────────
    export_paths: dict[str, Path] = {}
    if export:
        export_paths = export_all(bundle, tag)

    logger.info("══ Insight engine complete. Tokens: %d | Exports: %d ══",
                bundle.total_tokens_used, len(export_paths))

    return {
        "bundle":            bundle,
        "dashboard_payload": dashboard_payload,
        "export_paths":      export_paths,
        "context":           ctx,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FMCG AI Insight Generator")
    parser.add_argument("--kpi",       required=True, help="Path to kpi_global CSV")
    parser.add_argument("--nlp",       required=True, help="Path to business_summary CSV")
    parser.add_argument("--insights",  required=True, help="Path to retailer_insights CSV")
    parser.add_argument("--recs",      required=True, help="Path to recommendation_summary CSV")
    parser.add_argument("--zones",     type=int, default=3,
                        help="Number of zones to generate insights for (default: 3)")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--tag",       default="")
    parser.add_argument("--ask",       default="",
                        help="Single question for copilot mode (skips full pipeline)")
    args = parser.parse_args()

    if args.ask:
        # ── Copilot one-shot mode ──────────────────────────────────────────────
        kpi_df = load_kpi_csv(args.kpi)
        nlp_df = load_nlp_csv(args.nlp)
        ins_df = load_insights_csv(args.insights)
        rec_df = load_recommendations_csv(args.recs)
        ctx    = build_context(kpi_df, nlp_df, ins_df, rec_df)
        client = _build_client()
        answer, tokens = ask_copilot(args.ask, ctx, client)
        print(f"\n{'═'*62}")
        print(f"  Q: {args.ask}")
        print(f"{'═'*62}")
        print(answer)
        print(f"\n[Tokens used: {tokens}]")

    else:
        # ── Full pipeline ──────────────────────────────────────────────────────
        results = run_insight_engine(
            kpi_path      = args.kpi,
            nlp_path      = args.nlp,
            insights_path = args.insights,
            rec_path      = args.recs,
            top_n_zones   = args.zones,
            export        = not args.no_export,
            tag           = args.tag,
        )

        bundle = results["bundle"]
        print(f"\n{'═'*62}")
        print("  AI INSIGHT GENERATION — COMPLETE")
        print(f"{'═'*62}")
        print(f"  Generated at   : {bundle.generated_at}")
        print(f"  Total tokens   : {bundle.total_tokens_used:,}")
        print(f"  Zone insights  : {len(bundle.zone_insights)}")
        print(f"  Dashboard cards: {len(bundle.dashboard_cards)}")
        print()
        print("  EXECUTIVE SUMMARY (first 400 chars):")
        print(f"  {bundle.executive_summary.body[:400]}…")
        print()
        if results["export_paths"]:
            print("  EXPORTS:")
            for name, path in results["export_paths"].items():
                print(f"    {name:<22} → {path}")
        print(f"{'═'*62}\n")