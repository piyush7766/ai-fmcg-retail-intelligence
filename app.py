# ============================================================
# AI-Powered FMCG Retail Intelligence Dashboard
# Amul Creamy & Tasty Dahi — Jaipur Market
# Production-Grade Streamlit Application
# ============================================================

import os
import html
import re
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv

# ── Load .env (project root) ─────────────────────────────────
load_dotenv()

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI FMCG Retail Intelligence | Amul C&T Jaipur",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Enterprise Dark CSS ──────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: #0D1117;
      color: #C9D1D9;
  }
  .main { background-color: #0D1117; }
  section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
  .block-container { padding: 1.5rem 2rem; }

  /* ── KPI Cards ── */
  .kpi-card {
      background: #1C2333; border: 1px solid #30363D; border-radius: 10px;
      padding: 18px 20px; margin-bottom: 12px; transition: all 0.25s ease;
      position: relative; overflow: hidden;
  }
  .kpi-card:hover { border-color: #1F6FEB; box-shadow: 0 0 18px rgba(31,111,235,0.25); transform: translateY(-2px); }
  .kpi-card .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #8B949E; margin-bottom: 6px; }
  .kpi-card .kpi-value { font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 6px; }
  .kpi-card .kpi-alert { font-size: 11px; font-weight: 600; border-radius: 4px; padding: 3px 8px; display: inline-block; }
  .kpi-card .kpi-narrative { font-size: 12px; color: #8B949E; margin-top: 8px; line-height: 1.5; }
  .kpi-card.critical { border-left: 4px solid #F85149; }
  .kpi-card.warning  { border-left: 4px solid #D29922; }
  .kpi-card.positive { border-left: 4px solid #3FB950; }
  .alert-critical { background: rgba(248,81,73,0.2); color: #F85149; }
  .alert-warning  { background: rgba(210,153,34,0.2); color: #D29922; }
  .alert-positive { background: rgba(63,185,80,0.2); color: #3FB950; }

  /* ── Section Headers ── */
  .section-header {
      border-left: 4px solid #1F6FEB; padding-left: 14px; margin-bottom: 20px; margin-top: 8px;
  }
  .section-header h2 { font-size: 20px; font-weight: 700; color: #E6EDF3; margin: 0 0 4px 0; }
  .section-header p  { font-size: 13px; color: #8B949E; margin: 0; }

  /* ── Insight Panels ── */
  .panel {
      background: #1C2333; border: 1px solid #30363D; border-radius: 10px;
      padding: 18px 20px; margin-bottom: 14px;
  }
  .panel h4 { font-size: 13px; font-weight: 700; letter-spacing: 0.8px; margin: 0 0 10px 0; }
  .panel p  { font-size: 13px; color: #C9D1D9; margin: 0; line-height: 1.65; }
  .panel.d  { border-left: 4px solid #F85149; } .panel.d h4 { color: #F85149; }
  .panel.w  { border-left: 4px solid #D29922; } .panel.w h4 { color: #D29922; }
  .panel.s  { border-left: 4px solid #3FB950; } .panel.s h4 { color: #3FB950; }
  .panel.i  { border-left: 4px solid #1F6FEB; } .panel.i h4 { color: #58A6FF; }

  /* ── Zone Cards ── */
  .zone-card {
      background: #1C2333; border: 1px solid #30363D; border-radius: 10px;
      padding: 18px 20px; margin-bottom: 14px;
  }
  .zone-card h3 { font-size: 15px; font-weight: 700; color: #58A6FF; margin: 0 0 10px 0; }
  .zone-card .zone-metric { display: inline-block; background: #21262D; border-radius: 4px;
      padding: 3px 10px; font-size: 12px; font-weight: 600; margin: 3px 4px 3px 0; color: #C9D1D9; }

  /* ── Strategy Cards ── */
  .rec-card {
      background: #1C2333; border: 1px solid #30363D; border-radius: 10px;
      padding: 16px 20px; margin-bottom: 12px; display: flex; gap: 14px; align-items: flex-start;
  }
  .rec-num { font-size: 24px; font-weight: 800; color: #1F6FEB; min-width: 32px; }
  .rec-body h4 { font-size: 13px; font-weight: 700; color: #E6EDF3; margin: 0 0 4px 0; }
  .rec-body .rec-meta { font-size: 11px; color: #8B949E; margin-bottom: 6px; }
  .rec-body p  { font-size: 12px; color: #C9D1D9; margin: 0; line-height: 1.55; }

  /* ── Chat ── */
  .chat-wrap { max-height: 420px; overflow-y: auto; padding: 4px 0; margin-bottom: 12px; }
  .msg-u { background: #1F6FEB; color: #fff; border-radius: 12px 12px 4px 12px;
      padding: 10px 14px; margin: 6px 0 6px 20%; font-size: 13px; line-height: 1.5; }
  .msg-a { background: #1C2333; color: #C9D1D9; border: 1px solid #30363D;
      border-radius: 12px 12px 12px 4px; padding: 10px 14px; margin: 6px 20% 6px 0; font-size: 13px; line-height: 1.55; }
  .msg-label { font-size: 10px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 4px; }
  .msg-u .msg-label { color: rgba(255,255,255,0.7); text-align: right; }
  .msg-a .msg-label { color: #58A6FF; }

  /* ── Confidence Badge ── */
  .badge { display: inline-block; border-radius: 4px; padding: 2px 10px;
      font-size: 10px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
  .badge.high { background: rgba(63,185,80,0.2); color: #3FB950; border: 1px solid #3FB950; }
  .badge.med  { background: rgba(210,153,34,0.2); color: #D29922; border: 1px solid #D29922; }

  /* ── Divider ── */
  hr.divider { border: none; border-top: 1px solid #21262D; margin: 20px 0; }

  /* ── Sidebar ── */
  .sidebar-logo { text-align: center; padding: 12px 0 20px 0; }
  .sidebar-logo h1 { font-size: 16px; font-weight: 800; color: #1F6FEB; margin: 0; }
  .sidebar-logo p  { font-size: 11px; color: #8B949E; margin: 2px 0 0 0; }
  .sidebar-snapshot { background: #21262D; border-radius: 8px; padding: 12px 14px; margin-top: 12px; }
  .sidebar-snapshot h5 { font-size: 11px; font-weight: 600; color: #8B949E; letter-spacing: 0.8px;
      text-transform: uppercase; margin: 0 0 8px 0; }
  .sidebar-snapshot .snap-row { display: flex; justify-content: space-between;
      font-size: 12px; padding: 3px 0; border-bottom: 1px solid #30363D; }
  .sidebar-snapshot .snap-row:last-child { border-bottom: none; }
  .snap-key { color: #8B949E; }
  .snap-val { color: #58A6FF; font-weight: 600; }

  /* ── Metric overrides ── */
  [data-testid="stMetric"] { background: #1C2333; border: 1px solid #30363D; border-radius: 8px; padding: 14px; }
  [data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 12px !important; }
  [data-testid="stMetricValue"] { color: #E6EDF3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly Template ──────────────────────────────────────────
PLTARGS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(28,35,51,0.4)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#C9D1D9"),
    margin=dict(l=20, r=20, t=40, b=20),
)
GRID = dict(gridcolor="#21262D", zerolinecolor="#21262D")
NO_BAR = dict(displayModeBar=False)

# ── Helpers ─────────────────────────────────────────────────
def clean_text(t: str) -> str:
    """Strip newlines and collapse extra spaces from CSV multi-line fields."""
    if pd.isna(t):
        return ""
    return re.sub(r"\s+", " ", str(t).replace("\n", " ").replace("\r", " ")).strip()

def esc(t: str) -> str:
    return html.escape(clean_text(t))

def find_file(folder: str, prefix: str) -> str | None:
    """Return first file in folder whose name starts with prefix."""
    try:
        for fn in os.listdir(folder):
            if fn.startswith(prefix):
                return os.path.join(folder, fn)
    except FileNotFoundError:
        pass
    return None

# ── Data Loading ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exports")

@st.cache_data(show_spinner=False)
def load_data():
    def _read(prefix):
        fp = find_file(DATA_DIR, prefix)
        if fp:
            return pd.read_csv(fp)
        return pd.DataFrame()

    kpi_df       = _read("kpi_global_")
    cards_df     = _read("dashboard_cards_")
    insights_df  = _read("generated_insights_")
    retailer_df  = _read("retailer_insights_")
    zone_df      = _read("zone_insights_")
    return kpi_df, cards_df, insights_df, retailer_df, zone_df

kpi_df, cards_df, insights_df, retailer_df, zone_df = load_data()

# ── Extract scalar KPIs (fall back to hardcoded values) ─────
def _kpi(col, fallback):
    try:
        return kpi_df.iloc[0][col]
    except Exception:
        return fallback

total_retailers        = int(_kpi("penetration__total_retailers", 491))
ct_stockers            = int(_kpi("penetration__ct_stockers", 74))
penetration_rate       = float(_kpi("penetration__ct_penetration_rate", 15.07))
awareness_rate         = float(_kpi("penetration__awareness_rate", 43.18))
unawareness_rate       = float(_kpi("penetration__unawareness_rate", 56.82))
awareness_conv         = float(_kpi("penetration__awareness_to_stocking_conversion", 34.91))
refrigerated_count     = int(_kpi("infrastructure__refrigerated_count", 489))
cold_chain_gap         = float(_kpi("infrastructure__cold_chain_gap", 84.87))
frubon_threat_pct      = float(_kpi("competitor__frubon_threat_pct", 50.71))
best_promo_brand       = str(_kpi("competitor__best_promo_brand", "Frubon"))
best_promo_pct         = float(_kpi("competitor__best_promo_brand_pct", 48.47))
best_rep_brand         = str(_kpi("competitor__best_rep_brand", "Frubon"))
best_rep_pct           = float(_kpi("competitor__best_rep_brand_pct", 40.94))
amul_ts_pct            = float(_kpi("competitor__amul_top_seller_pct", 79.23))
saras_ts_pct           = float(_kpi("competitor__saras_top_seller_pct", 17.52))
frubon_ts_pct          = float(_kpi("competitor__frubon_top_seller_pct", 2.44))
price_driven_loss      = float(_kpi("competitor__price_driven_loss_pct", 48.27))
weekly_rep_pct         = float(_kpi("engagement__weekly_rep_visit_pct", 5.91))
monthly_rep_pct        = float(_kpi("engagement__monthly_rep_visit_pct", 54.58))
rarely_rep_pct         = float(_kpi("engagement__rarely_rep_visit_pct", 17.31))
promo_reach_pct        = float(_kpi("engagement__promo_reach_pct", 18.33))
avg_engagement         = float(_kpi("engagement__avg_engagement_score", 2.534))
increased_sig_pct      = float(_kpi("sales__increased_significantly_pct", 17.92))
increased_sl_pct       = float(_kpi("sales__increased_slightly_pct", 41.96))
stayed_same_pct        = float(_kpi("sales__stayed_same_pct", 30.75))
decreased_pct          = float(_kpi("sales__decreased_pct", 9.37))
net_growth_signal      = float(_kpi("sales__net_growth_signal", 50.51))
cup_format_pct         = float(_kpi("sales__cup_format_preference_pct", 49.49))
pouch_format_pct       = float(_kpi("sales__pouch_format_preference_pct", 38.49))
sourness_pct           = float(_kpi("complaints__sourness_complaint_pct", 35.03))
texture_pct            = float(_kpi("complaints__texture_complaint_pct", 18.74))
packaging_pct          = float(_kpi("complaints__packaging_complaint_pct", 18.53))
no_complaint_pct       = float(_kpi("complaints__no_complaint_pct", 31.57))
top_complaint          = str(_kpi("complaints__top_complaint", "Curd is too sour"))
critical_opp           = int(_kpi("opportunity__critical_opportunity_count", 0))
high_opp               = int(_kpi("opportunity__high_opportunity_count", 15))
medium_opp             = int(_kpi("opportunity__medium_opportunity_count", 317))
low_opp                = int(_kpi("opportunity__low_opportunity_count", 85))
refrigerated_not_stock = int(_kpi("opportunity__refrigerated_not_stocking", 415))

# ── Helper: get insight by type ──────────────────────────────
def get_insight(insight_type: str) -> str:
    try:
        row = insights_df[insights_df["insight_type"] == insight_type].iloc[0]
        return clean_text(row.get("body", row.get("headline", "")))
    except Exception:
        return ""

executive_summary  = get_insight("executive_summary")
comp_risk_body     = get_insight("competitor_risk")
complaint_playbook = get_insight("complaint_playbook")
opp_brief          = get_insight("opportunity_brief")
strat_recs_body    = get_insight("strategic_recommendations")

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-logo">
  <h1>🧠 FMCG Intelligence</h1>
  <p>Amul C&amp;T Dahi — Jaipur Market</p>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Overview",
        "🗺️ Zone Intelligence",
        "⚔️ Competitor Intelligence",
        "💬 NLP Insights",
        "🎯 Retailer Opportunity Engine",
        "🤖 AI Copilot",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div class="sidebar-snapshot">
  <h5>Data Snapshot</h5>
  <div class="snap-row"><span class="snap-key">Total Retailers</span><span class="snap-val">{total_retailers}</span></div>
  <div class="snap-row"><span class="snap-key">C&T Stockers</span><span class="snap-val">{ct_stockers}</span></div>
  <div class="snap-row"><span class="snap-key">Penetration</span><span class="snap-val">{penetration_rate:.1f}%</span></div>
  <div class="snap-row"><span class="snap-key">Awareness</span><span class="snap-val">{awareness_rate:.1f}%</span></div>
  <div class="snap-row"><span class="snap-key">Cold Chain Gap</span><span class="snap-val">{cold_chain_gap:.1f}%</span></div>
  <div class="snap-row"><span class="snap-key">FruBon Threat</span><span class="snap-val">{frubon_threat_pct:.1f}%</span></div>
  <div class="snap-row"><span class="snap-key">Net Growth</span><span class="snap-val">{net_growth_signal:.1f}%</span></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<hr class='divider'>
<div style='font-size:10px;color:#8B949E;text-align:center;padding-top:4px;'>
  Survey Date: May 2026 &nbsp;|&nbsp; Confidence: HIGH<br>
  Powered by GPT-4o · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════
if nav == "📊 Executive Overview":
    st.markdown("""
    <div class="section-header">
      <h2>Executive Overview</h2>
      <p>Amul Creamy &amp; Tasty Dahi · Jaipur Market Intelligence · May 2026</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 6 KPI Cards ───────────────────────────────────────────
    alert_map = {"WARNING": ("warning", "alert-warning"), "CRITICAL": ("critical", "alert-critical"), "POSITIVE": ("positive", "alert-positive")}
    rows = [cards_df.iloc[i:i+3] for i in range(0, min(6, len(cards_df)), 3)] if not cards_df.empty else []

    for row_cards in rows:
        cols = st.columns(3)
        for col, (_, card) in zip(cols, row_cards.iterrows()):
            alert_raw = str(card.get("alert", "")).split("—")[0].strip().upper()
            css_class, badge_class = alert_map.get(alert_raw, ("warning", "alert-warning"))
            col.markdown(f"""
            <div class="kpi-card {css_class}">
              <div class="kpi-label">{esc(card.get('title',''))}</div>
              <div class="kpi-value">{esc(card.get('metric',''))}</div>
              <span class="kpi-alert {badge_class}">{esc(alert_raw)}</span>
              <div class="kpi-narrative">{esc(card.get('narrative',''))[:120]}…</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Charts Row ────────────────────────────────────────────
    c1, c2 = st.columns([1, 1])

    # Market Funnel
    with c1:
        aware_stocking = round(total_retailers * awareness_rate / 100)
        fig_funnel = go.Figure(go.Funnel(
            y=["Total Retailers", "Refrigerated", "Aware / Aware+Stocking", "C&T Stockers"],
            x=[total_retailers, refrigerated_count, aware_stocking, ct_stockers],
            textinfo="value+percent initial",
            marker=dict(color=["#1F6FEB", "#388BFD", "#79C0FF", "#3FB950"]),
            connector=dict(line=dict(color="#30363D", width=2)),
        ))
        fig_funnel.update_layout(**PLTARGS, height=360, title="Market Conversion Funnel")
        st.plotly_chart(fig_funnel, use_container_width=True, config=NO_BAR)

    # Sales Performance Donut
    with c2:
        fig_donut = go.Figure(go.Pie(
            labels=["Increased Significantly", "Increased Slightly", "Stayed Same", "Decreased"],
            values=[increased_sig_pct, increased_sl_pct, stayed_same_pct, decreased_pct],
            hole=0.55,
            marker=dict(colors=["#3FB950", "#388BFD", "#D29922", "#F85149"]),
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_donut.add_annotation(
            text=f"<b>{net_growth_signal}%</b><br>Net Growth",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#3FB950"),
        )
        fig_donut.update_layout(**PLTARGS, height=360, title="Sales Performance Distribution",
                                showlegend=True, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_donut, use_container_width=True, config=NO_BAR)

    # ── Key Metrics + AI Summary ──────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Penetration Rate", f"{penetration_rate:.1f}%", delta=f"Target: 30%")
    m2.metric("Retailer Awareness", f"{awareness_rate:.1f}%", delta=f"-{unawareness_rate:.1f}% unaware")
    m3.metric("Promo Reach", f"{promo_reach_pct:.1f}%", delta="Target: 40%")
    m4.metric("Cold Chain Gap", f"{cold_chain_gap:.1f}%", delta=f"{refrigerated_not_stock} at risk", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    body_text = executive_summary or (
        "Amul's current penetration rate for Creamy & Tasty (C&T) Dahi stands at 15.1%, with awareness among retailers at 43.2%. "
        "Competitor FruBon leads BTL promotions (48.5%) and rep visits (40.9%). "
        "Critical gaps: 84.9% cold-chain gap, 56.8% unaware retailers, only 18.3% promo reach."
    )
    st.markdown(f"""
    <div class="panel i">
      <h4>🧠 AI EXECUTIVE SUMMARY &nbsp;<span class="badge high">HIGH CONFIDENCE</span></h4>
      <p>{esc(body_text)}</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ZONE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
elif nav == "🗺️ Zone Intelligence":
    st.markdown("""
    <div class="section-header">
      <h2>Zone Intelligence</h2>
      <p>Geographic opportunity analysis across Jaipur retail clusters</p>
    </div>
    """, unsafe_allow_html=True)

    # Zone data (hardcoded from zone_insights + prompt)
    zones_meta = {
        "Amer_Periphery":     {"label": "Amer / Periphery", "total": 291, "high_crit": 116, "stocking": 33,
                               "neg_sent": 53, "comp_pct": 98, "conv_rate": 71.6,
                               "crit": 0, "med": 116, "low": 109, "already": 33},
        "Malviya_Sanganer":   {"label": "Malviya / Sanganer", "total": 68, "high_crit": 39, "stocking": 7,
                               "neg_sent": 52, "comp_pct": 100, "conv_rate": 57.4,
                               "crit": 2, "med": 37, "low": 22, "already": 7},
        "Vaishali_Mansarovar":{"label": "Vaishali / Mansarovar", "total": 77, "high_crit": 31, "stocking": 12,
                               "neg_sent": 60, "comp_pct": 99, "conv_rate": 39.9,
                               "crit": 0, "med": 31, "low": 34, "already": 12},
    }

    # ── Zone Comparison Bar ───────────────────────────────────
    zone_labels = [v["label"] for v in zones_meta.values()]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Total Retailers", x=zone_labels,
                             y=[v["total"] for v in zones_meta.values()],
                             marker_color="#1F6FEB", text=[v["total"] for v in zones_meta.values()], textposition="outside"))
    fig_bar.add_trace(go.Bar(name="High-Crit Opportunities", x=zone_labels,
                             y=[v["high_crit"] for v in zones_meta.values()],
                             marker_color="#D29922", text=[v["high_crit"] for v in zones_meta.values()], textposition="outside"))
    fig_bar.add_trace(go.Bar(name="Currently Stocking", x=zone_labels,
                             y=[v["stocking"] for v in zones_meta.values()],
                             marker_color="#3FB950", text=[v["stocking"] for v in zones_meta.values()], textposition="outside"))
    fig_bar.update_layout(**PLTARGS, height=360, title="Zone Retailer Overview",
                          barmode="group", xaxis=dict(**GRID), yaxis=dict(**GRID, title="Retailers"))
    st.plotly_chart(fig_bar, use_container_width=True, config=NO_BAR)

    # ── Stacked Horizontal Bar ────────────────────────────────
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        fig_stacked = go.Figure()
        tiers = ["Critical", "High / Med Opp", "Low", "Already Stocking"]
        colors = ["#F85149", "#D29922", "#388BFD", "#3FB950"]
        vals_map = [
            [v["crit"] for v in zones_meta.values()],
            [v["med"]  for v in zones_meta.values()],
            [v["low"]  for v in zones_meta.values()],
            [v["already"] for v in zones_meta.values()],
        ]
        for tier, color, vals in zip(tiers, colors, vals_map):
            fig_stacked.add_trace(go.Bar(
                name=tier, y=zone_labels, x=vals,
                orientation="h", marker_color=color,
                text=vals, textposition="inside",
            ))
        fig_stacked.update_layout(**PLTARGS, height=360, barmode="stack",
                                  title="Opportunity Tiers by Zone",
                                  xaxis=dict(**GRID, title="Retailers"),
                                  yaxis=dict(**GRID))
        st.plotly_chart(fig_stacked, use_container_width=True, config=NO_BAR)

    # ── Radar Chart ───────────────────────────────────────────
    with c2:
        categories = ["Neg Sentiment", "Stocking %", "Comp Risk", "Conv Rate", "Opp Score"]
        fig_radar = go.Figure()
        radar_data = {
            "Amer / Periphery":     [53, 33/291*100, 98, 71.6, 75],
            "Malviya / Sanganer":   [52, 7/68*100,   100, 57.4, 80],
            "Vaishali / Mansarovar":[60, 12/77*100,  99, 39.9, 65],
        }
        rcolors = ["#1F6FEB", "#D29922", "#F85149"]
        for (zone_lbl, vals), col in zip(radar_data.items(), rcolors):
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill="toself", name=zone_lbl, line=dict(color=col),
                fillcolor=col.replace("#", "rgba(") + ",0.12)".replace("rgba(", "rgba(").replace(",0.12)", ",0.12)"),
            ))
        # Fix fillcolor properly
        for trace, col in zip(fig_radar.data, rcolors):
            r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
            trace.fillcolor = f"rgba({r},{g},{b},0.15)"
        fig_radar.update_layout(**PLTARGS, height=360, title="Zone Comparison Radar",
                                polar=dict(radialaxis=dict(visible=True, range=[0, 110], gridcolor="#21262D"),
                                           bgcolor="rgba(28,35,51,0.4)"))
        st.plotly_chart(fig_radar, use_container_width=True, config=NO_BAR)

    # ── Zone Insight Panels ───────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    panel_cols = st.columns(3)
    panel_styles = ["d", "w", "d"]
    zone_keys = list(zones_meta.keys())

    for idx, (pstyle, col) in enumerate(zip(panel_styles, panel_cols)):
        z = zones_meta[zone_keys[idx]]
        # Try to get AI insight for this zone
        try:
            zi_row = zone_df[zone_df["zone"] == zone_keys[idx]].iloc[0]
            zone_body = clean_text(zi_row.get("body", ""))[:400] + "…"
        except Exception:
            zone_body = f"This zone has {z['total']} retailers with {z['high_crit']} high-opportunity targets and {z['stocking']} currently stocking C&T Dahi."
        col.markdown(f"""
        <div class="zone-card">
          <h3>📍 {esc(z['label'])}</h3>
          <span class="zone-metric">🏪 {z['total']} Retailers</span>
          <span class="zone-metric">🎯 {z['high_crit']} Opp Targets</span>
          <span class="zone-metric">✅ {z['stocking']} Stocking</span>
          <span class="zone-metric">😞 {z['neg_sent']}% Neg Sentiment</span>
          <span class="zone-metric">⚔️ {z['comp_pct']}% Comp Risk</span>
          <span class="zone-metric">🔄 {z['conv_rate']}% Conv Rate</span>
          <hr class='divider' style='margin:10px 0;'>
          <p style='font-size:12px;color:#C9D1D9;'>{esc(zone_body)}</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — COMPETITOR INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
elif nav == "⚔️ Competitor Intelligence":
    st.markdown("""
    <div class="section-header">
      <h2>Competitor Intelligence</h2>
      <p>Competitive landscape analysis · FruBon threat assessment · Counter-strategy</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    # Market Share Donut
    with c1:
        fig_ms = go.Figure(go.Pie(
            labels=["Amul Masti Dahi", "Saras", "FruBon", "Others"],
            values=[amul_ts_pct, saras_ts_pct, frubon_ts_pct, max(0, 100 - amul_ts_pct - saras_ts_pct - frubon_ts_pct)],
            hole=0.55,
            marker=dict(colors=["#1F6FEB", "#3FB950", "#F85149", "#8B949E"]),
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_ms.add_annotation(text="<b>Top Seller</b><br>Market Share", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=12, color="#C9D1D9"))
        fig_ms.update_layout(**PLTARGS, height=360, title="Top Selling Brand — Market Share",
                             showlegend=True, legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig_ms, use_container_width=True, config=NO_BAR)

    # Competitive Positioning Bar
    with c2:
        comp_cats  = ["BTL Promo %", "Rep Quality %", "Threat Presence %", "Price Preference %"]
        frubon_v   = [48.47, 40.94, 50.71, 15.0]
        amul_v     = [18.33, 5.91,  0.0,   0.0]
        saras_v    = [0.0,   0.0,   0.0,   48.27]
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="FruBon",  x=comp_cats, y=frubon_v, marker_color="#F85149",
                                  text=[f"{v}%" for v in frubon_v], textposition="outside"))
        fig_comp.add_trace(go.Bar(name="Amul",    x=comp_cats, y=amul_v,   marker_color="#1F6FEB",
                                  text=[f"{v}%" for v in amul_v], textposition="outside"))
        fig_comp.add_trace(go.Bar(name="Saras",   x=comp_cats, y=saras_v,  marker_color="#3FB950",
                                  text=[f"{v}%" for v in saras_v], textposition="outside"))
        fig_comp.update_layout(**PLTARGS, height=360, title="Competitive Positioning Analysis",
                               barmode="group", xaxis=dict(**GRID), yaxis=dict(**GRID, title="%", range=[0, 70]))
        st.plotly_chart(fig_comp, use_container_width=True, config=NO_BAR)

    # FruBon Threat Gauge + Vulnerability Heatmap
    c3, c4 = st.columns([0.9, 1.1])
    with c3:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=frubon_threat_pct,
            delta={"reference": 40, "increasing": {"color": "#F85149"}},
            title={"text": "FruBon Threat Exposure %", "font": {"size": 14, "color": "#C9D1D9"}},
            number={"suffix": "%", "font": {"size": 32, "color": "#F85149"}},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#8B949E", tickfont=dict(color="#8B949E")),
                bar=dict(color="#F85149"),
                steps=[
                    dict(range=[0, 30], color="#21262D"),
                    dict(range=[30, 60], color="rgba(210,153,34,0.2)"),
                    dict(range=[60, 100], color="rgba(248,81,73,0.15)"),
                ],
                threshold=dict(line=dict(color="#F85149", width=3), thickness=0.75, value=frubon_threat_pct),
            ),
        ))
        fig_gauge.update_layout(**PLTARGS, height=300)
        st.plotly_chart(fig_gauge, use_container_width=True, config=NO_BAR)

    with c4:
        vuln_cats  = ["Texture Complaints", "Low Promo Reach", "Rare Rep Visits", "Price Disadvantage"]
        vuln_scores= [31.6, 100 - promo_reach_pct, rarely_rep_pct, price_driven_loss]
        fig_vuln = go.Figure(go.Bar(
            x=vuln_scores, y=vuln_cats, orientation="h",
            marker=dict(color=["#F85149", "#D29922", "#D29922", "#D29922"],
                        line=dict(color="#30363D", width=1)),
            text=[f"{v:.1f}%" for v in vuln_scores], textposition="outside",
        ))
        fig_vuln.update_layout(**PLTARGS, height=300, title="Amul Competitive Vulnerabilities",
                               xaxis=dict(**GRID, range=[0, 100], title="%"), yaxis=dict(**GRID))
        st.plotly_chart(fig_vuln, use_container_width=True, config=NO_BAR)

    # AI Competitor Risk Panel
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    risk_text = comp_risk_body or (
        "THREAT LEVEL: HIGH — FruBon's 50.7% threat exposure. Counter-strategy: "
        "(1) Product reformulation targeting 20% reduction in texture complaints within 60 days; "
        "(2) Trade promotion expansion from 18.3% to 35% within 60 days; "
        "(3) Rep visit boost from 5.9% to 15%."
    )
    st.markdown(f"""
    <div class="panel d">
      <h4>⚠️ AI COMPETITOR RISK ASSESSMENT &nbsp;<span class="badge high">HIGH CONFIDENCE</span></h4>
      <p>{esc(risk_text[:600])}…</p>
    </div>
    """, unsafe_allow_html=True)

    # Counter-strategy Cards
    counter_strategies = [
        ("Product Reformulation", "Product Team", "60 days",
         "Immediate quality checks on taste and texture. Target 20% reduction in texture-related complaints."),
        ("Trade Promotion Expansion", "Marketing", "60 days",
         f"Scale promo reach from {promo_reach_pct:.1f}% → 35%. Launch targeted BTL campaigns to match FruBon."),
        ("Rep Visit Frequency", "Sales", "60 days",
         f"Increase weekly visits from {weekly_rep_pct:.1f}% → 15%. Boost retailer engagement score above 4.0."),
    ]
    c_cols = st.columns(3)
    for col, (title, team, timeline, desc) in zip(c_cols, counter_strategies):
        col.markdown(f"""
        <div class="panel w">
          <h4>🛡️ {esc(title)}</h4>
          <p><b>Team:</b> {esc(team)} &nbsp;|&nbsp; <b>Timeline:</b> {esc(timeline)}<br><br>{esc(desc)}</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — NLP INSIGHTS
# ═══════════════════════════════════════════════════════════════
elif nav == "💬 NLP Insights":
    st.markdown("""
    <div class="section-header">
      <h2>NLP Insights</h2>
      <p>Retailer complaint analysis · Sentiment intelligence · Resolution playbook</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    # Complaint Cluster Donut
    with c1:
        if not retailer_df.empty and "complaint_label" in retailer_df.columns:
            comp_counts = retailer_df["complaint_label"].value_counts().reset_index()
            comp_counts.columns = ["Complaint", "Count"]
        else:
            comp_counts = pd.DataFrame({
                "Complaint": ["Texture & Thickness", "Sourness & Taste", "Packaging", "No Complaint", "Other"],
                "Count":     [155, 172, 91, 155, 18],
            })
        fig_comp_pie = go.Figure(go.Pie(
            labels=comp_counts["Complaint"], values=comp_counts["Count"],
            hole=0.5,
            marker=dict(colors=["#F85149", "#D29922", "#388BFD", "#3FB950", "#8B949E"]),
            textinfo="label+percent", textfont=dict(size=11),
        ))
        fig_comp_pie.update_layout(**PLTARGS, height=360, title="Complaint Cluster Distribution")
        st.plotly_chart(fig_comp_pie, use_container_width=True, config=NO_BAR)

    # Sentiment Distribution by Zone
    with c2:
        if not retailer_df.empty and "zone" in retailer_df.columns and "sentiment_label" in retailer_df.columns:
            sent_zone = retailer_df.groupby(["zone", "sentiment_label"]).size().reset_index(name="count")
            fig_sent = px.bar(sent_zone, x="zone", y="count", color="sentiment_label",
                              barmode="stack",
                              color_discrete_map={"Negative": "#F85149", "Neutral": "#D29922", "Positive": "#3FB950"},
                              title="Sentiment Distribution by Zone")
            fig_sent.update_layout(**PLTARGS, height=360,
                                   xaxis=dict(**GRID, title="Zone"), yaxis=dict(**GRID, title="Retailers"))
        else:
            fig_sent = go.Figure()
            zones_s = ["Amer_Periphery", "Malviya_Sanganer", "Vaishali_Mansarovar"]
            neg_v, neu_v, pos_v = [154, 35, 46], [85, 22, 18], [52, 11, 13]
            for name, vals, col in [("Negative", neg_v, "#F85149"),
                                    ("Neutral",  neu_v, "#D29922"),
                                    ("Positive", pos_v, "#3FB950")]:
                fig_sent.add_trace(go.Bar(name=name, x=zones_s, y=vals,
                                          marker_color=col, text=vals, textposition="inside"))
            fig_sent.update_layout(**PLTARGS, height=360, barmode="stack",
                                   title="Sentiment Distribution by Zone",
                                   xaxis=dict(**GRID), yaxis=dict(**GRID, title="Retailers"))
        st.plotly_chart(fig_sent, use_container_width=True, config=NO_BAR)

    # Top Complaint Themes + Top Recommendations
    c3, c4 = st.columns([1, 1])
    with c3:
        if not retailer_df.empty and "top_complaint_theme" in retailer_df.columns:
            theme_counts = retailer_df["top_complaint_theme"].value_counts().head(8).reset_index()
            theme_counts.columns = ["Theme", "Count"]
        else:
            theme_counts = pd.DataFrame({
                "Theme": ["sourness", "texture", "packaging", "positive", "freshness"],
                "Count": [172, 155, 91, 45, 28],
            })
        fig_themes = go.Figure(go.Bar(
            x=theme_counts["Count"], y=theme_counts["Theme"], orientation="h",
            marker_color="#388BFD",
            text=theme_counts["Count"], textposition="outside",
        ))
        fig_themes.update_layout(**PLTARGS, height=340, title="Top Complaint Themes",
                                 xaxis=dict(**GRID, title="Count"), yaxis=dict(**GRID))
        st.plotly_chart(fig_themes, use_container_width=True, config=NO_BAR)

    with c4:
        if not retailer_df.empty and "top_recommendation" in retailer_df.columns:
            rec_counts = retailer_df["top_recommendation"].value_counts().head(8).reset_index()
            rec_counts.columns = ["Recommendation", "Count"]
        else:
            rec_counts = pd.DataFrame({
                "Recommendation": ["Pricing", "Promotions", "Freshness", "Quality", "Other"],
                "Count":          [188, 120, 80, 60, 44],
            })
        fig_recs = go.Figure(go.Bar(
            x=rec_counts["Count"], y=rec_counts["Recommendation"], orientation="h",
            marker_color="#D29922",
            text=rec_counts["Count"], textposition="outside",
        ))
        fig_recs.update_layout(**PLTARGS, height=340, title="Top Retailer Recommendations",
                               xaxis=dict(**GRID, title="Count"), yaxis=dict(**GRID))
        st.plotly_chart(fig_recs, use_container_width=True, config=NO_BAR)

    # Sentiment Scatter
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if not retailer_df.empty and "sentiment_score" in retailer_df.columns:
        plot_df = retailer_df.dropna(subset=["sentiment_score", "zone"]).copy()
        fig_scatter = px.strip(
            plot_df, x="zone", y="sentiment_score",
            color="sentiment_label" if "sentiment_label" in plot_df.columns else None,
            color_discrete_map={"Negative": "#F85149", "Neutral": "#D29922", "Positive": "#3FB950"},
            title="Retailer Sentiment Scores by Zone",
            hover_data=["shop_name"] if "shop_name" in plot_df.columns else None,
        )
        fig_scatter.update_layout(**PLTARGS, height=360,
                                  xaxis=dict(**GRID, title="Zone"),
                                  yaxis=dict(**GRID, title="Sentiment Score", range=[-1.1, 1.1]))
        st.plotly_chart(fig_scatter, use_container_width=True, config=NO_BAR)

    # Complaint Playbook Panels
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header" style="margin-top:0;">
      <h2>Complaint Resolution Playbook</h2>
      <p>Prioritised actions for top complaint themes</p>
    </div>
    """, unsafe_allow_html=True)

    playbook_data = [
        ("d", "🔴 SOURNESS (35.0% of retailers)",
         "<b>Root cause:</b> Production inconsistency / over-fermentation.<br>"
         "<b>0–30 days:</b> Immediate quality assessment &amp; reformulation plan.<br>"
         "<b>30–90 days:</b> Rigorous consistency check protocol in production.<br>"
         "<b>Script:</b> <i>\"We are actively reformulating our products to address sourness. Your feedback is invaluable.\"</i>"),
        ("w", "🟠 TEXTURE &amp; THICKNESS (31.6% of retailers)",
         "<b>Root cause:</b> Production variance in batch consistency.<br>"
         "<b>0–30 days:</b> Quality control review; identify and rectify variance.<br>"
         "<b>30–90 days:</b> Revised QA protocol; standardise texture/thickness across batches.<br>"
         "<b>Script:</b> <i>\"We recognise texture issues and are reviewing our production processes.\"</i>"),
        ("i", "🔵 PACKAGING DEFECTS (18.5% of retailers)",
         "<b>Root cause:</b> Pouch design flaw — leak points.<br>"
         "<b>0–30 days:</b> Audit current packaging materials and identify failure points.<br>"
         "<b>30–90 days:</b> Redesign pouch for leak-proof integrity; rollout new design.<br>"
         "<b>Script:</b> <i>\"We are redesigning our pouches for improved quality. Thank you for your support.\"</i>"),
    ]
    pb_cols = st.columns(3)
    for col, (style, title, body) in zip(pb_cols, playbook_data):
        col.markdown(f"""
        <div class="panel {style}">
          <h4>{title}</h4>
          <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — RETAILER OPPORTUNITY ENGINE
# ═══════════════════════════════════════════════════════════════
elif nav == "🎯 Retailer Opportunity Engine":
    st.markdown("""
    <div class="section-header">
      <h2>Retailer Opportunity Engine</h2>
      <p>Conversion intelligence · 90-day roadmap · Filterable retailer targets</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    # Opportunity Funnel
    with c1:
        conversion_target = round(refrigerated_not_stock * awareness_conv / 100)
        fig_opp_funnel = go.Figure(go.Funnel(
            y=["Refrigerated Non-Stockers", f"Qualified Opportunities (High+Med)", "Conversion Target"],
            x=[refrigerated_not_stock, high_opp + medium_opp, conversion_target],
            textinfo="value+percent initial",
            marker=dict(color=["#D29922", "#1F6FEB", "#3FB950"]),
            connector=dict(line=dict(color="#30363D", width=2)),
        ))
        fig_opp_funnel.update_layout(**PLTARGS, height=360, title="Opportunity Conversion Funnel")
        st.plotly_chart(fig_opp_funnel, use_container_width=True, config=NO_BAR)

    # Opportunity by Zone
    with c2:
        opp_zones = ["Amer / Periphery", "Malviya / Sanganer", "Vaishali / Mansarovar"]
        opp_high  = [116, 39, 31]
        opp_med   = [142, 15, 21]
        opp_low   = [0, 7, 13]
        fig_opp_zone = go.Figure()
        for name, vals, col in [("High Opp", opp_high, "#F85149"), ("Med Opp", opp_med, "#D29922"), ("Low", opp_low, "#388BFD")]:
            fig_opp_zone.add_trace(go.Bar(name=name, x=opp_zones, y=vals,
                                          marker_color=col, text=vals, textposition="inside"))
        fig_opp_zone.update_layout(**PLTARGS, height=360, barmode="stack",
                                   title="Opportunity Distribution by Zone",
                                   xaxis=dict(**GRID), yaxis=dict(**GRID, title="Retailers"))
        st.plotly_chart(fig_opp_zone, use_container_width=True, config=NO_BAR)

    # Conversion Priority Cards
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    pri_cols = st.columns(3)
    pri_data = [
        ("d", "🔴 CRITICAL TIER", f"{critical_opp} retailers",
         "Immediate attention required. Deploy sales team within 7 days. Personalised outreach with pricing incentives."),
        ("w", "🟠 HIGH TIER", f"{high_opp} retailers",
         "59% negative sentiment. 98.3% competitor presence. 30-day target: Convert 5 retailers. Leverage pricing adjustments."),
        ("i", "🔵 MEDIUM TIER", f"{medium_opp} retailers",
         "46.7% negative sentiment. 97.6% competitor presence. 30-day target: Convert 30 retailers. Focus on promotions + quality messaging."),
    ]
    for col, (style, title, metric, body) in zip(pri_cols, pri_data):
        col.markdown(f"""
        <div class="panel {style}">
          <h4>{title}</h4>
          <div style='font-size:22px;font-weight:800;color:#E6EDF3;margin:8px 0;'>{metric}</div>
          <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

    # 90-Day Roadmap
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header" style="margin-top:0;">
      <h2>90-Day Conversion Roadmap</h2>
      <p>Phased deployment plan across priority zones</p>
    </div>
    """, unsafe_allow_html=True)
    roadmap = [
        ("Month 1 — Deploy", "Sales Team",
         "Retailer visits in Amer / Periphery. Implement pricing strategies. Gather product feedback. "
         "Target: Convert 5 high-opportunity retailers. KPI: 5 new C&T listings."),
        ("Month 2 — Expand", "Sales + Marketing",
         "Expand to Malviya/Sanganer and Vaishali/Mansarovar. Pricing adjustments + quality communication. "
         "Develop promotional materials. Target: 15 additional medium-opportunity conversions."),
        ("Month 3 — Evaluate", "Sales + Marketing + Product",
         "Evaluate conversion success. Refine strategies based on feedback. "
         "Target: 30 total conversions across all zones. Comprehensive review and strategy adjustment."),
    ]
    r_cols = st.columns(3)
    for col, (title, team, desc) in zip(r_cols, roadmap):
        col.markdown(f"""
        <div class="panel i">
          <h4>📅 {esc(title)}</h4>
          <p><b>Owners:</b> {esc(team)}<br><br>{esc(desc)}</p>
        </div>
        """, unsafe_allow_html=True)

    # Filterable Retailer Table
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header" style="margin-top:0;">
      <h2>Retailer Target List</h2>
      <p>Filterable database of conversion targets</p>
    </div>
    """, unsafe_allow_html=True)

    if not retailer_df.empty:
        table_cols = [c for c in ["shop_name", "zone", "complaint_label", "sentiment_label",
                                   "opportunity_flag", "top_recommendation"] if c in retailer_df.columns]
        tbl = retailer_df[table_cols].copy() if table_cols else retailer_df.copy()

        fc1, fc2, fc3 = st.columns(3)
        zone_opts = ["All"] + sorted(tbl["zone"].dropna().unique().tolist()) if "zone" in tbl.columns else ["All"]
        opp_opts  = ["All"] + sorted(tbl["opportunity_flag"].dropna().unique().tolist()) if "opportunity_flag" in tbl.columns else ["All"]
        sent_opts = ["All"] + sorted(tbl["sentiment_label"].dropna().unique().tolist()) if "sentiment_label" in tbl.columns else ["All"]

        sel_zone = fc1.selectbox("Filter by Zone", zone_opts)
        sel_opp  = fc2.selectbox("Filter by Opportunity Flag", opp_opts)
        sel_sent = fc3.selectbox("Filter by Sentiment", sent_opts)

        if sel_zone != "All" and "zone" in tbl.columns:
            tbl = tbl[tbl["zone"] == sel_zone]
        if sel_opp != "All" and "opportunity_flag" in tbl.columns:
            tbl = tbl[tbl["opportunity_flag"] == sel_opp]
        if sel_sent != "All" and "sentiment_label" in tbl.columns:
            tbl = tbl[tbl["sentiment_label"] == sel_sent]

        st.dataframe(tbl.fillna("—").head(200), use_container_width=True, height=360)
        st.caption(f"Showing {min(200, len(tbl))} of {len(tbl)} retailers")
    else:
        st.info("Retailer data not available. Place retailer_insights CSV in data/exports/.")

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — AI COPILOT  (OpenAI GPT-4o-mini · Live)
# ═══════════════════════════════════════════════════════════════
elif nav == "🤖 AI Copilot":
    st.markdown("""
    <div class="section-header">
      <h2>AI Copilot</h2>
      <p>GPT-4o-mini · Real-time conversational analytics · Grounded in 491-retailer survey</p>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key Resolution (.env → env var → sidebar fallback) ──
    def _resolve_api_key() -> str:
        """
        Priority:
          1. OPENAI_API_KEY in .env (loaded by load_dotenv() at startup)
          2. OPENAI_API_KEY already present in os.environ (e.g. CI/CD / Docker)
          3. Manual entry via sidebar input (development fallback only)
        """
        return os.getenv("OPENAI_API_KEY", "")

    copilot_api_key = _resolve_api_key()
    if not copilot_api_key:
        with st.sidebar:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:11px;color:#F85149;margin-bottom:4px;'>"
                "⚠️ OPENAI_API_KEY not found in .env</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='font-size:10px;color:#8B949E;margin-bottom:6px;'>"
                "Add <code>OPENAI_API_KEY=sk-…</code> to your <code>.env</code> file "
                "or enter it below for this session only.</p>",
                unsafe_allow_html=True,
            )
            copilot_api_key = st.text_input(
                "OpenAI API Key (session only)",
                type="password",
                key="oai_key_input",
                placeholder="sk-…",
                label_visibility="collapsed",
            )

    # ── Build System Prompt (cached in session_state) ────────
    def _build_system_prompt() -> str:
        ctx: dict = {}
        ctx_fp = find_file(DATA_DIR, "copilot_context_")
        if ctx_fp:
            try:
                with open(ctx_fp, encoding="utf-8") as fh:
                    ctx = json.load(fh)
            except Exception:
                pass

        zone_lines = ""
        if not zone_df.empty:
            for _, zrow in zone_df.iterrows():
                zone_lines += (
                    f"\n  [{zrow.get('zone', '?')}] "
                    f"{clean_text(str(zrow.get('headline', ''))[:160])} | "
                    f"Metrics: {clean_text(str(zrow.get('key_metrics', ''))[:120])}"
                )

        comp_bdown = rec_bdown = ""
        if not retailer_df.empty:
            if "complaint_label" in retailer_df.columns:
                top_c = retailer_df["complaint_label"].value_counts().head(5)
                comp_bdown = " | ".join(f"{k}: {v}" for k, v in top_c.items())
            if "top_recommendation" in retailer_df.columns:
                top_r = retailer_df["top_recommendation"].value_counts().head(5)
                rec_bdown = " | ".join(f"{k}: {v}" for k, v in top_r.items())

        faqs_block = ""
        for faq in ctx.get("retailer_faqs", []):
            faqs_block += f"\n  Q: {faq.get('question', '')}  →  A: {faq.get('answer', '')}"

        conv_target = round(refrigerated_not_stock * awareness_conv / 100)

        return f"""You are an expert FMCG retail analytics AI assistant specialised in the Jaipur dairy market. \
You have deep knowledge of a field survey of {total_retailers} retailers for Amul Creamy & Tasty (C&T) Dahi. \
Respond in clear, concise business prose. Keep answers under 180 words unless a detailed breakdown is explicitly requested. \
Ground every answer in the specific figures below. Vary phrasing across conversation turns — never repeat a prior response verbatim. \
Use Markdown bold (**text**) for key metrics and bullet points where helpful.

=== PENETRATION & AWARENESS ===
C&T penetration: {penetration_rate:.1f}% ({ct_stockers} stockers / {total_retailers} retailers)
Awareness: {awareness_rate:.1f}% aware | {unawareness_rate:.1f}% ({round(total_retailers * unawareness_rate / 100)}) unaware
Awareness-to-stocking conversion: {awareness_conv:.1f}%
Net growth signal: {net_growth_signal:.1f}%

=== INFRASTRUCTURE ===
Refrigerated: {refrigerated_count}/{total_retailers} ({100 - cold_chain_gap:.1f}% have refrigeration)
Cold-chain gap: {cold_chain_gap:.1f}% | Refrigerated non-stockers: {refrigerated_not_stock} (prime targets)
Conversion potential from non-stockers: ~{conv_target} retailers

=== COMPETITOR INTELLIGENCE ===
FruBon threat exposure: {frubon_threat_pct:.1f}%
FruBon BTL promo lead: {best_promo_pct:.1f}% vs Amul {promo_reach_pct:.1f}%
FruBon rep quality lead: {best_rep_pct:.1f}% vs Amul weekly {weekly_rep_pct:.1f}%
Top selling brand share: Amul Masti {amul_ts_pct:.1f}% | Saras {saras_ts_pct:.1f}% | FruBon {frubon_ts_pct:.1f}%
Price-driven loss to competitors: {price_driven_loss:.1f}%

=== ENGAGEMENT & PROMOTIONS ===
Rep visits — Weekly: {weekly_rep_pct:.1f}% | Monthly: {monthly_rep_pct:.1f}% | Rarely: {rarely_rep_pct:.1f}%
Promo reach: {promo_reach_pct:.1f}% ({round(total_retailers * promo_reach_pct / 100)} retailers)
Avg engagement score: {avg_engagement:.2f}/10

=== SALES PERFORMANCE ===
Increased significantly: {increased_sig_pct:.1f}% | Slightly: {increased_sl_pct:.1f}%
Stayed same: {stayed_same_pct:.1f}% | Decreased: {decreased_pct:.1f}%
Format preference — Cup: {cup_format_pct:.1f}% | Pouch: {pouch_format_pct:.1f}%

=== NLP / COMPLAINTS ===
Top complaint: "{top_complaint}" ({sourness_pct:.1f}%)
Texture & Thickness: 31.6% | Packaging defects: {packaging_pct:.1f}% | No complaint: {no_complaint_pct:.1f}%
Cluster breakdown: {comp_bdown}
Top retailer recommendations: {rec_bdown}

=== OPPORTUNITY TIERS ===
Critical: {critical_opp} | High: {high_opp} | Medium: {medium_opp} | Low: {low_opp}

=== ZONE INTELLIGENCE ==={zone_lines}
  Amer/Periphery: 291 total | 116 high-opp | 33 stocking | 53% neg sentiment | 98% comp risk | conv 71.6%
  Malviya/Sanganer: 68 total | 39 high/crit | 7 stocking (10.3%) | 52% neg | 100% comp risk
  Vaishali/Mansarovar: 77 total | 31 high-opp | 12 stocking | 60% neg | 99% comp risk

=== AI INSIGHTS (HIGH CONFIDENCE) ===
Executive summary: {(executive_summary or '')[:420]}
Competitor risk: {(comp_risk_body or '')[:320]}
Opportunity brief: {(opp_brief or '')[:320]}
Complaint playbook: {(complaint_playbook or '')[:320]}
Strategic recs: {(strat_recs_body or '')[:420]}

=== MARKET CONTEXT ===
{ctx.get('market_snapshot', '')}
Top 3 priorities: {'; '.join(ctx.get('top_3_priorities', []))}

=== RETAILER FAQs ==={faqs_block}

BEHAVIOUR RULES:
1. Always cite exact survey numbers (% + absolute count where available).
2. For strategy/action questions: give 2-4 numbered bullet actions with owner and timeline.
3. Never copy a previous answer verbatim; add new depth or a different analytical angle each turn.
4. Confidence: HIGH (field survey of {total_retailers} retailers, GPT-4o-mini grounded analysis, temperature 0.3)."""

    if "copilot_system_prompt" not in st.session_state:
        st.session_state["copilot_system_prompt"] = _build_system_prompt()
    SYSTEM_PROMPT = st.session_state["copilot_system_prompt"]

    # ── OpenAI API Call ───────────────────────────────────────
    def _call_gpt(history: list) -> str:
        """Call GPT-4o-mini with full conversation history; return response text."""
        try:
            client = OpenAI(api_key=copilot_api_key)
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=full_messages,
                temperature=0.3,
                max_tokens=450,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"⚠️ GPT-4o-mini API error: {exc}"

    # ── Submit helper ─────────────────────────────────────────
    def _submit(query: str) -> None:
        if not copilot_api_key:
            st.warning("Enter your OpenAI API key in the sidebar to enable live AI.", icon="🔑")
            return
        q = query.strip()
        if not q:
            return
        st.session_state["chat_display"].append({"role": "user", "text": q})
        st.session_state["oai_history"].append({"role": "user", "content": q})
        with st.spinner("GPT-4o-mini generating response…"):
            reply = _call_gpt(st.session_state["oai_history"])
        st.session_state["chat_display"].append({"role": "ai", "text": reply})
        st.session_state["oai_history"].append({"role": "assistant", "content": reply})

    # ── Session State Init ────────────────────────────────────
    if "oai_history" not in st.session_state:
        st.session_state["oai_history"] = []
    if "chat_display" not in st.session_state:
        st.session_state["chat_display"] = []

    # ── Strategic Recommendation Cards ───────────────────────
    st.markdown(
        "<div style='margin-bottom:8px;'><b style='color:#58A6FF;font-size:13px;'>"
        "STRATEGIC RECOMMENDATIONS &nbsp;<span class=\"badge high\">HIGH CONFIDENCE</span>"
        "</b></div>",
        unsafe_allow_html=True,
    )
    recs = [
        ("IMPROVE PRODUCT TEXTURE & THICKNESS", "Product Team", "Q1 2024",
         f"31.6% of retailers cite texture complaints (avg sentiment −1.0). "
         f"Conduct reformulation check by Q1 2024. Target: 50% complaint reduction, 20% stocking rate increase."),
        ("ENHANCE PACKAGING INTEGRITY", "Product Team", "Q2 2024",
         "Packaging defects affect 18.5% of retailers. Redesign pouch for leak-proof integrity by Q2 2024. "
         "Target: 75% reduction in packaging complaints, 15% sales uplift."),
        ("EXPAND TRADE PROMOTION REACH", "Marketing", "Q1 2024",
         f"Promo reach at {promo_reach_pct:.1f}% vs FruBon's {best_promo_pct:.1f}%. "
         f"Scale to 40% of retailers by Q1 2024. Projected: 10% sales volume increase."),
        ("INCREASE REP VISIT FREQUENCY", "Sales", "Q2 2024",
         f"Weekly visits at {weekly_rep_pct:.1f}%; {rarely_rep_pct:.1f}% rarely visited. "
         f"Target 10% weekly by Q2 2024. Expected: 15% sales increase."),
        ("INTRODUCE AFFORDABLE SKU OPTIONS", "Product", "Q3 2024",
         f"Pricing top recommendation from 188+ retailers ({price_driven_loss:.1f}% price-driven loss). "
         f"Launch 80g & 180g SKUs by Q3 2024. Projected: 12% volume increase."),
    ]
    for i, (title, team, timeline, rationale) in enumerate(recs, 1):
        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-num">{i}</div>
          <div class="rec-body">
            <h4>{esc(title)}</h4>
            <div class="rec-meta">🏢 {esc(team)} &nbsp;|&nbsp; 📅 {esc(timeline)} &nbsp;|&nbsp;
              <span class="badge high">HIGH</span></div>
            <p>{esc(rationale)}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Live Chat UI ──────────────────────────────────────────
    st.markdown(
        "<b style='color:#58A6FF;font-size:13px;'>💬 LIVE AI ANALYTICS CHAT &nbsp;"
        "<span class=\"badge high\">GPT-4o-mini · REAL-TIME</span></b>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size:12px;color:#8B949E;margin:10px 0 6px 0;'>💡 Suggested Questions</p>",
        unsafe_allow_html=True,
    )
    sq_cols = st.columns(4)
    suggested_qs = [
        "What is our penetration status and key gaps?",
        "Analyse the FruBon competitive threat",
        "Which zone should we prioritise first?",
        "What is causing retailer complaints?",
    ]
    for _col, _sq in zip(sq_cols, suggested_qs):
        if _col.button(_sq, use_container_width=True):
            _submit(_sq)
            st.rerun()

    # Chat history display
    if st.session_state["chat_display"]:
        chat_html = '<div class="chat-wrap">'
        for msg in st.session_state["chat_display"]:
            if msg["role"] == "user":
                chat_html += (
                    f'<div class="msg-u">'
                    f'<div class="msg-label">You</div>'
                    f'{esc(msg["text"])}'
                    f'</div>'
                )
            else:
                ai_body = html.escape(msg["text"]).replace("\n", "<br>")
                ai_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", ai_body)
                chat_html += (
                    f'<div class="msg-a">'
                    f'<div class="msg-label">🧠 AI Copilot &nbsp;'
                    f'<span class="badge high">GPT-4o-mini · LIVE</span></div>'
                    f'{ai_body}'
                    f'</div>'
                )
        chat_html += '<div class="cf"></div></div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-wrap" style="display:flex;align-items:center;justify-content:center;min-height:140px;">
          <div style="text-align:center;color:#8B949E;">
            <div style="font-size:28px;margin-bottom:8px;">🧠</div>
            <div style="font-size:13px;font-weight:600;color:#C9D1D9;">Ask me anything about the Jaipur FMCG retail data.</div>
            <div style="font-size:11px;margin-top:4px;">
              Powered by GPT-4o-mini &nbsp;·&nbsp; Grounded in 491-retailer survey &nbsp;·&nbsp; Temperature 0.3
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    user_input = st.chat_input("Ask about penetration, competitors, zones, complaints, strategy…")
    if user_input:
        _submit(user_input)
        st.rerun()

    ctl1, ctl2, _ = st.columns([1, 1, 5])
    if ctl1.button("🗑️ Clear Chat"):
        st.session_state["oai_history"] = []
        st.session_state["chat_display"] = []
        if "copilot_system_prompt" in st.session_state:
            del st.session_state["copilot_system_prompt"]
        st.rerun()
    if ctl2.button("📋 Export Chat") and st.session_state["chat_display"]:
        lines = [f"[{m['role'].upper()}]: {m['text']}" for m in st.session_state["chat_display"]]
        st.download_button(
            label="⬇️ Download",
            data="\n\n".join(lines),
            file_name="copilot_chat_export.txt",
            mime="text/plain",
            key="dl_chat_btn",
        )
