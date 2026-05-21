"""
================================================================================
feedback_analyzer.py — Production-Grade NLP Analytics Engine
AI-Powered FMCG Retail Intelligence Platform
Amul Retailer Intelligence System | Jaipur Market
================================================================================
Analyses:
  · Retailer complaint themes (TF-IDF + K-Means clustering)
  · Sentiment detection (lexicon-based, no external API required)
  · Keyword extraction (TF-IDF ranked, unigram + bigram)
  · Topic modeling (LDA + NMF dual-method)
  · Competitor-related complaint identification
  · Retailer recommendation extraction from suggestion corpus
  · Business-friendly summaries per cluster / zone / segment
  · Visualization-ready exports (CSV + dict payloads)

Inputs  : nlp_corpus.csv  +  retailers_clean.csv
Outputs : complaint_clusters.csv, sentiment_summary.csv,
          keyword_frequency.csv, topic_clusters.csv,
          retailer_insights.csv, competitor_complaints.csv,
          recommendation_summary.csv, business_summary.csv

Author  : AI FMCG Intelligence Platform
Version : 1.0.0
Dependencies: pandas, numpy, scipy, scikit-learn (no NLTK / VADER required)
================================================================================
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import logging
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP — Logging
# ──────────────────────────────────────────────────────────────────────────────

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_log_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

_log_dir = Path("logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_log_handlers.append(
    logging.FileHandler(_log_dir / "feedback_analyzer.log", mode="a", encoding="utf-8")
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=LOG_FMT,
    handlers=_log_handlers,
)
logger = logging.getLogger("feedback_analyzer")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

EXPORTS_DIR = Path(os.getenv("EXPORTS_PATH", "data/exports/"))
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Text cleaning ─────────────────────────────────────────────────────────────
STOP_WORDS: set[str] = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","it","its","this","that","these","those","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","need","dare","ought","used","able",
    "i","we","you","he","she","they","me","us","him","her","them","my","our",
    "your","his","their","what","which","who","when","where","why","how","all",
    "each","every","both","few","more","most","other","some","such","no","not",
    "only","same","so","than","too","very","just","also","then","there","from",
    "as","by","about","into","through","during","before","after","above",
    "below","between","out","off","over","under","again","further","up","down",
    "data","no_data","specify","generally","sometimes","often","always","never",
    "yes","dont","doesnt","cant","won","isn","wasn","didn","hasn","haven",
    "couldn","wouldn","shouldn","get","got","getting","give","given","go","going",
    "come","coming","take","taking","make","making","know","think","see","look",
    "want","need","like","use","using","used","one","two","three","four","five",
}

# ── Complaint lexicon (domain-specific) ──────────────────────────────────────
COMPLAINT_LEXICON: dict[str, list[str]] = {
    "sourness":    ["sour", "acidic", "fermented", "tangy", "tart"],
    "texture":     ["thin", "watery", "thick", "texture", "whey", "separation",
                    "watered", "runny", "lumpy", "creamy"],
    "packaging":   ["leak", "leaks", "leaking", "break", "breaks", "broken",
                    "package", "packaging", "pouch", "torn", "damaged", "spill"],
    "freshness":   ["shelf", "expiry", "expired", "fresh", "stale", "old",
                    "stock", "date", "freshness", "older", "delivered"],
    "pricing":     ["price", "expensive", "costly", "cheap", "margin", "margins",
                    "higher", "costly", "affordable", "saras", "cheaper"],
    "availability":["demand", "supply", "availability", "available", "stock",
                    "absent", "missing", "rep", "visit", "offered"],
    "competitor":  ["frubon", "saras", "mother", "dairy", "ksheer", "rufil",
                    "competitor", "brand", "other"],
    "positive":    ["good", "fine", "okay", "nice", "great", "excellent",
                    "appreciate", "like", "happy", "satisfied", "no complaints",
                    "well", "best", "better", "love", "prefer"],
}

# ── Sentiment lexicon (valence scores) ───────────────────────────────────────
# Positive polarity words → +score, negative → −score
SENTIMENT_LEXICON: dict[str, float] = {
    # Positive
    "good": 0.6, "great": 0.8, "excellent": 0.9, "fine": 0.4, "okay": 0.3,
    "nice": 0.6, "happy": 0.7, "satisfied": 0.7, "appreciate": 0.6, "love": 0.8,
    "best": 0.8, "better": 0.5, "well": 0.5, "fresh": 0.5, "quality": 0.4,
    "prefer": 0.4, "liked": 0.5, "enjoying": 0.6, "creamy": 0.3, "thick": 0.3,
    "demand": 0.4, "popular": 0.5, "growth": 0.5, "improved": 0.5,
    # Negative
    "sour": -0.7, "acidic": -0.6, "thin": -0.5, "watery": -0.6, "whey": -0.4,
    "leak": -0.7, "leaks": -0.7, "break": -0.6, "broken": -0.6, "damaged": -0.7,
    "expired": -0.8, "stale": -0.7, "old": -0.4, "expensive": -0.5,
    "costly": -0.6, "overpriced": -0.7, "problem": -0.6, "issue": -0.5,
    "poor": -0.7, "bad": -0.8, "terrible": -0.9, "horrible": -0.9, "awful": -0.9,
    "complaint": -0.3, "complain": -0.4, "unhappy": -0.7, "dissatisfied": -0.8,
    "absent": -0.5, "missing": -0.5, "never": -0.3, "no": -0.2,
    "separation": -0.4, "runny": -0.5, "lumpy": -0.5, "torn": -0.6,
    "short": -0.3, "low": -0.3, "unclear": -0.4, "insufficient": -0.5,
    # Intensifiers
    "very": 1.5, "extremely": 1.8, "too": 1.3, "really": 1.4, "quite": 1.2,
    "not": -1.0, "never": -1.0, "no": -0.8,
}

# ── Competitor keywords ───────────────────────────────────────────────────────
COMPETITOR_KEYWORDS: list[str] = [
    "frubon", "saras", "mother dairy", "mother_dairy", "ksheer", "rufil",
    "lotus", "competitor", "other brand", "local brand", "rajasthan",
    "local pride", "local", "regional",
]

# ── Recommendation pattern seeds ─────────────────────────────────────────────
RECOMMENDATION_SEEDS: dict[str, list[str]] = {
    "pricing":       ["better margin", "higher margin", "margin increase", "affordable",
                      "cheaper", "price reduction", "competitive price"],
    "promotions":    ["offer", "offers", "free trial", "promotion", "consumer facing",
                      "discount", "scheme", "gift", "reward"],
    "advertising":   ["advertising", "advertisement", "awareness", "jaipur", "campaign",
                      "visibility", "marketing", "promotion"],
    "rep_visits":    ["rep visit", "sales rep", "frequent visit", "more visit",
                      "field visit", "representative", "agent"],
    "packaging":     ["better packaging", "leak proof", "prevents leak", "packaging",
                      "seal", "improved pack"],
    "sku":           ["smaller sku", "affordable sku", "single use", "small pack",
                      "80g", "180g", "trial pack", "new size"],
    "freshness":     ["fresh stock", "longer shelf", "shelf life", "fresher",
                      "expiry", "cold chain"],
    "quality":       ["improve quality", "thicker", "texture", "taste improvement",
                      "less sour", "consistency"],
}

# ── Cluster label mappings ────────────────────────────────────────────────────
COMPLAINT_CLUSTER_LABELS: dict[int, str] = {
    0: "Sourness & Taste",
    1: "Texture & Thickness",
    2: "Packaging Defects",
    3: "Price & Margin Sensitivity",
    4: "No Complaints / Positive",
    5: "Shelf Life & Freshness",
    6: "Demand & Availability Gap",
    7: "Competitor Dominance",
}

SUGGESTION_TOPIC_LABELS: dict[int, str] = {
    0: "Advertising & Brand Visibility",
    1: "SKU Affordability & Pack Sizes",
    2: "Retailer Margin Improvement",
    3: "Consumer Promotions & Free Trials",
    4: "Rep Visit Frequency",
    5: "Packaging Quality",
    6: "Freshness & Cold Chain",
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterProfile:
    cluster_id:       int
    label:            str
    size:             int
    pct_of_total:     float
    top_keywords:     list[str]
    avg_sentiment:    float
    zone_distribution: dict[str, int]
    sample_texts:     list[str]
    business_summary: str


@dataclass
class SentimentResult:
    text:           str
    score:          float      # −1.0 … +1.0
    label:          str        # Positive / Neutral / Negative
    confidence:     float      # 0.0 … 1.0
    dominant_theme: str


@dataclass
class KeywordFrequency:
    term:            str
    frequency:       int
    tfidf_weight:    float
    corpus_section:  str       # complaints / suggestions / why_not_stocking / full


@dataclass
class TopicCluster:
    topic_id:        int
    label:           str
    top_terms:       list[str]
    term_weights:    list[float]
    document_count:  int
    method:          str       # LDA | NMF


@dataclass
class RetailerInsight:
    retailer_id:         int
    shop_name:           str
    zone:                str
    complaint_cluster:   int
    complaint_label:     str
    sentiment_score:     float
    sentiment_label:     str
    top_complaint_theme: str
    top_recommendation:  str
    is_competitor_related: bool
    opportunity_flag:    str


@dataclass
class AnalysisReport:
    computed_at:             str
    total_documents:         int
    complaint_clusters:      list[ClusterProfile]
    suggestion_topics:       list[TopicCluster]
    global_sentiment:        dict[str, Any]
    top_keywords:            list[KeywordFrequency]
    competitor_complaint_pct: float
    recommendation_themes:   dict[str, int]
    zone_sentiment_map:      dict[str, float]
    business_summary:        dict[str, str]


# ──────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Lowercase, strip special chars, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _remove_stopwords(text: str) -> str:
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def _is_empty_text(text: str) -> bool:
    cleaned = _normalize_text(text)
    return (not cleaned or
            cleaned in {"no_data", "no", "na", "none", "nan", "-"} or
            len(cleaned.split()) < 1)


def preprocess_corpus(series: pd.Series, remove_stops: bool = True) -> pd.Series:
    """Full preprocessing pipeline on a text Series."""
    result = series.fillna("").apply(_normalize_text)
    if remove_stops:
        result = result.apply(_remove_stopwords)
    # Replace empty after cleaning with placeholder
    result = result.apply(lambda t: t if t.strip() else "no data")
    return result


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to all NLP columns and add _proc variants."""
    logger.info("Preprocessing NLP columns…")
    nlp_cols = [c for c in df.columns if c.startswith("nlp_")]
    for col in nlp_cols:
        df[col + "_proc"] = preprocess_corpus(df[col])
        logger.debug("  Preprocessed %s", col)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SENTIMENT ANALYSIS — Lexicon-based (no external dependencies)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_sentiment_score(text: str) -> float:
    """
    Compute a sentiment score in [−1, +1] using the domain lexicon.
    Handles simple negation (not / never / no preceding a sentiment word).
    """
    if _is_empty_text(text):
        return 0.0

    tokens = _normalize_text(text).split()
    score = 0.0
    weight = 0.0
    n = len(tokens)

    i = 0
    while i < n:
        tok = tokens[i]
        val = SENTIMENT_LEXICON.get(tok, 0.0)

        # Intensifier: amplify next word
        if tok in {"very", "extremely", "really", "quite", "too"} and i + 1 < n:
            next_val = SENTIMENT_LEXICON.get(tokens[i + 1], 0.0)
            score  += val * next_val
            weight += abs(next_val) if next_val else 0.1
            i += 2
            continue

        # Negation: flip next word
        if tok in {"not", "never", "no"} and i + 1 < n:
            next_val = SENTIMENT_LEXICON.get(tokens[i + 1], 0.0)
            score  += val * next_val * -1
            weight += abs(next_val) if next_val else 0.1
            i += 2
            continue

        if val != 0.0:
            score  += val
            weight += abs(val)

        i += 1

    if weight == 0:
        return 0.0

    raw = score / weight
    return float(np.clip(raw, -1.0, 1.0))


def _sentiment_label(score: float) -> str:
    if score >= 0.20:
        return "Positive"
    if score <= -0.15:
        return "Negative"
    return "Neutral"


def _sentiment_confidence(score: float) -> float:
    return round(min(abs(score) * 1.5, 1.0), 3)


def _dominant_theme(text: str) -> str:
    """Return the dominant complaint theme from the lexicon."""
    text_lower = text.lower()
    hits: dict[str, int] = {}
    for theme, keywords in COMPLAINT_LEXICON.items():
        hits[theme] = sum(1 for kw in keywords if kw in text_lower)
    # Exclude 'positive' theme from complaint dominance
    complaint_themes = {k: v for k, v in hits.items() if k != "positive" and v > 0}
    if not complaint_themes:
        return "positive" if hits.get("positive", 0) > 0 else "general"
    return max(complaint_themes, key=complaint_themes.get)


def analyze_sentiment(series: pd.Series) -> pd.DataFrame:
    """
    Run sentiment analysis on a text Series.
    Returns DataFrame with score, label, confidence, dominant_theme.
    """
    logger.info("Running sentiment analysis on %d documents…", len(series))
    results = []
    for text in series:
        score = _compute_sentiment_score(str(text))
        results.append({
            "text":           text,
            "sentiment_score":  round(score, 4),
            "sentiment_label":  _sentiment_label(score),
            "confidence":       _sentiment_confidence(score),
            "dominant_theme":   _dominant_theme(str(text)),
        })
    df_out = pd.DataFrame(results)
    dist = df_out["sentiment_label"].value_counts().to_dict()
    logger.info("Sentiment distribution: %s", dist)
    return df_out


# ──────────────────────────────────────────────────────────────────────────────
# KEYWORD EXTRACTION — TF-IDF ranked
# ──────────────────────────────────────────────────────────────────────────────

def extract_keywords(
    series: pd.Series,
    corpus_name: str = "corpus",
    max_features: int = 200,
    ngram_range: tuple[int, int] = (1, 2),
    top_n: int = 40,
) -> list[KeywordFrequency]:
    """
    Extract top-N keywords by TF-IDF weight from a text corpus.
    Also computes raw term frequency for context.
    """
    logger.info("Extracting keywords from '%s' (%d docs)…", corpus_name, len(series))

    docs = preprocess_corpus(series, remove_stops=True).tolist()
    # Filter genuinely empty docs
    docs = [d if d != "no data" else "" for d in docs]

    tfidf = TfidfVectorizer(
        max_features    = max_features,
        ngram_range     = ngram_range,
        min_df          = 2,
        max_df          = 0.95,
        sublinear_tf    = True,
    )
    try:
        X = tfidf.fit_transform(docs)
    except ValueError as exc:
        logger.warning("TF-IDF failed for '%s': %s", corpus_name, exc)
        return []

    feature_names = tfidf.get_feature_names_out()
    # Mean TF-IDF across docs
    mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
    # Raw document frequency (how many docs contain the term)
    doc_freq    = np.asarray((X > 0).sum(axis=0)).flatten()

    # Build counter for raw frequency in full text
    all_text  = " ".join(series.fillna("").tolist()).lower()
    raw_counts = Counter(re.findall(r"\b[a-z]{3,}\b", all_text))

    top_idx = np.argsort(mean_tfidf)[::-1][:top_n]
    results: list[KeywordFrequency] = []
    for idx in top_idx:
        term = feature_names[idx]
        # Raw freq: for bigrams, use first word's count as proxy
        first_word = term.split()[0]
        freq = raw_counts.get(first_word, int(doc_freq[idx]))
        results.append(KeywordFrequency(
            term           = term,
            frequency      = freq,
            tfidf_weight   = round(float(mean_tfidf[idx]), 5),
            corpus_section = corpus_name,
        ))

    logger.info("  Top keyword: '%s' (weight=%.4f)", results[0].term, results[0].tfidf_weight)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# COMPLAINT CLUSTERING — TF-IDF + K-Means
# ──────────────────────────────────────────────────────────────────────────────

def _optimal_k(X, k_range: range, random_state: int = 42) -> int:
    """Select optimal K using silhouette score."""
    best_k, best_score = k_range[0], -1.0
    for k in k_range:
        if k >= X.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels, sample_size=min(500, X.shape[0]))
            logger.debug("  k=%d → silhouette=%.4f", k, score)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    logger.info("Optimal K selected: %d (silhouette=%.4f)", best_k, best_score)
    return best_k


def _cluster_top_terms(km: KMeans, vectorizer: TfidfVectorizer, top_n: int = 8) -> dict[int, list[str]]:
    """Extract top TF-IDF terms per cluster centroid."""
    feature_names = vectorizer.get_feature_names_out()
    result: dict[int, list[str]] = {}
    for idx, centroid in enumerate(km.cluster_centers_):
        top_indices = centroid.argsort()[::-1][:top_n]
        result[idx] = [feature_names[i] for i in top_indices]
    return result


def _auto_label_cluster(top_terms: list[str]) -> str:
    """Map cluster top terms to a business-friendly label."""
    term_set = set(" ".join(top_terms).lower().split())
    scores: dict[str, int] = {}
    mapping = {
        "Sourness & Taste":          {"sour", "acidic", "taste", "ferment", "tart"},
        "Texture & Thickness":       {"thin", "thick", "watery", "texture", "whey", "separation", "runny"},
        "Packaging Defects":         {"leak", "break", "package", "pouch", "packaging", "seal", "torn"},
        "Price & Margin Sensitivity":{"price", "expensive", "margin", "cheaper", "saras", "higher", "cost"},
        "No Complaints / Positive":  {"complaint", "generally", "fine", "okay", "good", "happy", "satisfied"},
        "Shelf Life & Freshness":    {"shelf", "expiry", "fresh", "stale", "old", "date"},
        "Demand & Availability Gap": {"demand", "enquiry", "supply", "rep", "offered", "visit"},
        "Competitor Dominance":      {"frubon", "saras", "mother", "competitor", "brand", "local"},
    }
    for label, keywords in mapping.items():
        scores[label] = len(term_set & keywords)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Feedback"


def cluster_complaints(
    df: pd.DataFrame,
    text_col: str = "nlp_complaints_proc",
    zone_col: str = "zone_clean",
    k_range: range = range(4, 9),
    max_features: int = 300,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[ClusterProfile], TfidfVectorizer, KMeans]:
    """
    Cluster retailer complaints using TF-IDF + K-Means.
    Returns (df_with_clusters, profiles, vectorizer, km_model).
    """
    logger.info("Clustering complaints (col='%s', k_range=%s)…", text_col, list(k_range))

    texts = df[text_col].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = (1, 2),
        min_df       = 2,
        max_df       = 0.90,
        sublinear_tf = True,
    )
    X = vectorizer.fit_transform(texts)
    X_norm = normalize(X)

    # Select optimal K
    k = _optimal_k(X_norm, k_range, random_state)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=15, max_iter=500)
    labels = km.fit_predict(X_norm)
    df = df.copy()
    df["complaint_cluster_id"] = labels

    top_terms_map = _cluster_top_terms(km, vectorizer, top_n=10)
    label_map     = {cid: _auto_label_cluster(terms) for cid, terms in top_terms_map.items()}
    df["complaint_cluster_label"] = df["complaint_cluster_id"].map(label_map)

    # Sentiment per document
    sentiment_df = analyze_sentiment(df[text_col.replace("_proc", "")] if text_col.replace("_proc","") in df.columns else df[text_col])
    df["sentiment_score"] = sentiment_df["sentiment_score"].values
    df["sentiment_label"] = sentiment_df["sentiment_label"].values
    df["dominant_theme"]  = sentiment_df["dominant_theme"].values

    # Build cluster profiles
    n_total  = len(df)
    profiles: list[ClusterProfile] = []
    for cid in sorted(df["complaint_cluster_id"].unique()):
        mask    = df["complaint_cluster_id"] == cid
        sub     = df[mask]
        terms   = top_terms_map.get(cid, [])
        label   = label_map.get(cid, f"Cluster {cid}")
        zone_dist = sub[zone_col].value_counts().to_dict() if zone_col in sub.columns else {}
        samples = sub[text_col].dropna().head(3).tolist()
        avg_sent = float(sub["sentiment_score"].mean()) if "sentiment_score" in sub.columns else 0.0

        # Auto business summary
        top2_terms = ", ".join(terms[:4])
        biz_summary = (
            f"{label}: {len(sub)} retailers ({len(sub)/n_total*100:.1f}%). "
            f"Key signals: [{top2_terms}]. "
            f"Avg sentiment: {avg_sent:+.2f}. "
            f"Dominant zone: {max(zone_dist, key=zone_dist.get) if zone_dist else 'N/A'}."
        )

        profiles.append(ClusterProfile(
            cluster_id       = cid,
            label            = label,
            size             = len(sub),
            pct_of_total     = round(len(sub) / n_total * 100, 2),
            top_keywords     = terms,
            avg_sentiment    = round(avg_sent, 4),
            zone_distribution= zone_dist,
            sample_texts     = samples,
            business_summary = biz_summary,
        ))

    logger.info("Formed %d complaint clusters. Distribution: %s",
                k, {p.label: p.size for p in profiles})
    return df, profiles, vectorizer, km


# ──────────────────────────────────────────────────────────────────────────────
# TOPIC MODELING — LDA + NMF dual method
# ──────────────────────────────────────────────────────────────────────────────

def _top_topic_terms(model, vectorizer, n_top: int = 10) -> list[list[str]]:
    feature_names = vectorizer.get_feature_names_out()
    result = []
    for topic in model.components_:
        top_idx = topic.argsort()[::-1][:n_top]
        result.append([feature_names[i] for i in top_idx])
    return result


def _top_topic_weights(model, n_top: int = 10) -> list[list[float]]:
    result = []
    for topic in model.components_:
        sorted_weights = sorted(topic, reverse=True)[:n_top]
        total = sum(sorted_weights) or 1.0
        result.append([round(w / total, 4) for w in sorted_weights])
    return result


def _assign_topic_label(terms: list[str]) -> str:
    term_str = " ".join(terms).lower()
    for seed_label, seeds in RECOMMENDATION_SEEDS.items():
        if any(s.split()[0] in term_str for s in seeds):
            return seed_label.replace("_", " ").title()
    return _auto_label_cluster(terms)


def model_topics(
    series: pd.Series,
    corpus_name: str = "suggestions",
    n_topics: int = 6,
    max_features: int = 200,
    method: str = "lda",
) -> tuple[list[TopicCluster], np.ndarray]:
    """
    Run LDA or NMF topic modeling on a text corpus.
    Returns (list of TopicCluster, document-topic matrix).
    """
    logger.info("Topic modeling ('%s', n=%d, method=%s)…", corpus_name, n_topics, method.upper())

    docs = preprocess_corpus(series, remove_stops=True).tolist()
    docs = [d if d != "no data" else "" for d in docs]

    if method == "lda":
        vec = CountVectorizer(
            max_features = max_features,
            ngram_range  = (1, 2),
            min_df       = 2,
            max_df       = 0.95,
        )
    else:
        vec = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = (1, 2),
            min_df       = 2,
            max_df       = 0.95,
        )

    try:
        X = vec.fit_transform(docs)
    except ValueError as exc:
        logger.warning("Vectorisation failed for '%s': %s", corpus_name, exc)
        return [], np.zeros((len(docs), n_topics))

    if method == "lda":
        model = LatentDirichletAllocation(
            n_components     = n_topics,
            max_iter         = 30,
            learning_method  = "online",
            random_state     = 42,
            doc_topic_prior  = 0.1,
            topic_word_prior = 0.01,
        )
    else:
        model = NMF(
            n_components = n_topics,
            random_state = 42,
            max_iter     = 400,
        )

    doc_topic = model.fit_transform(X)
    top_terms   = _top_topic_terms(model, vec, n_top=10)
    top_weights = _top_topic_weights(model, n_top=10)

    # Count documents with dominant topic
    dominant_topics = doc_topic.argmax(axis=1)
    topic_counts    = Counter(dominant_topics.tolist())

    topics: list[TopicCluster] = []
    for tid in range(n_topics):
        terms  = top_terms[tid]
        label  = _assign_topic_label(terms)
        topics.append(TopicCluster(
            topic_id       = tid,
            label          = label,
            top_terms      = terms,
            term_weights   = top_weights[tid],
            document_count = int(topic_counts.get(tid, 0)),
            method         = method.upper(),
        ))

    logger.info("Topics formed: %s", {t.label: t.document_count for t in topics})
    return topics, doc_topic


# ──────────────────────────────────────────────────────────────────────────────
# COMPETITOR COMPLAINT DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_competitor_complaints(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    """
    Flag and extract retailer records that mention competitors in their feedback.
    """
    logger.info("Detecting competitor-related complaints…")

    def _mentions_competitor(text: str) -> bool:
        t = str(text).lower()
        return any(kw in t for kw in COMPETITOR_KEYWORDS)

    def _competitor_brands(text: str) -> list[str]:
        t = str(text).lower()
        found = []
        brand_map = {
            "frubon":        "FruBon",
            "saras":         "Saras",
            "mother dairy":  "Mother Dairy",
            "mother_dairy":  "Mother Dairy",
            "ksheer":        "Ksheer",
            "rufil":         "Rufil",
        }
        for kw, brand in brand_map.items():
            if kw in t and brand not in found:
                found.append(brand)
        return found

    df = df.copy()

    # Combine all text cols for detection
    combined = df[text_cols].fillna("").agg(" | ".join, axis=1)
    df["is_competitor_related"]    = combined.apply(_mentions_competitor)
    df["competitor_brands_mentioned"] = combined.apply(
        lambda t: ", ".join(_competitor_brands(t)) if _competitor_brands(t) else ""
    )

    n_flagged = df["is_competitor_related"].sum()
    logger.info("Competitor-related complaints: %d / %d (%.1f%%)",
                n_flagged, len(df), n_flagged / len(df) * 100)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_recommendations(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """
    Classify each suggestion document into a recommendation theme.
    Returns (per-doc theme Series, theme frequency dict).
    """
    logger.info("Extracting recommendation themes from %d documents…", len(series))

    def _classify(text: str) -> str:
        text_lower = str(text).lower()
        for theme, seeds in RECOMMENDATION_SEEDS.items():
            if any(seed.lower() in text_lower for seed in seeds):
                return theme.replace("_", " ").title()
        return "General Improvement"

    themes = series.fillna("").apply(_classify)
    freq   = dict(Counter(themes.tolist()).most_common())
    logger.info("Recommendation themes: %s", freq)
    return themes, freq


# ──────────────────────────────────────────────────────────────────────────────
# ZONE SENTIMENT MAP
# ──────────────────────────────────────────────────────────────────────────────

def compute_zone_sentiment(df: pd.DataFrame, zone_col: str = "zone_clean") -> dict[str, float]:
    """Average sentiment score per zone, sorted descending."""
    if zone_col not in df.columns or "sentiment_score" not in df.columns:
        return {}
    zone_sent = (
        df.groupby(zone_col)["sentiment_score"]
        .mean()
        .round(4)
        .sort_values()
        .to_dict()
    )
    logger.info("Zone sentiment: %s", zone_sent)
    return zone_sent


# ──────────────────────────────────────────────────────────────────────────────
# RETAILER-LEVEL INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────

def build_retailer_insights(
    df: pd.DataFrame,
    cluster_col: str      = "complaint_cluster_id",
    label_col: str        = "complaint_cluster_label",
    sentiment_col: str    = "sentiment_score",
    sentiment_label_col: str = "sentiment_label",
    theme_col: str        = "dominant_theme",
    rec_col: str          = "recommendation_theme",
    competitor_col: str   = "is_competitor_related",
) -> list[RetailerInsight]:
    """Build per-retailer insight records."""
    logger.info("Building retailer-level insights…")

    def _opportunity_flag(row: pd.Series) -> str:
        score = row.get("feat_adoption_likelihood", 0) or 0
        if row.get("stocks_ct", False):
            return "Already Stocking"
        if score >= 7.0:
            return "Critical Opportunity"
        if score >= 5.5:
            return "High Opportunity"
        if score >= 4.0:
            return "Medium Opportunity"
        return "Low Opportunity"

    insights: list[RetailerInsight] = []
    for _, row in df.iterrows():
        insights.append(RetailerInsight(
            retailer_id          = int(row.get("Sr. No.", 0) or 0),
            shop_name            = str(row.get("Shop Name", "Unknown")),
            zone                 = str(row.get("zone_clean", "")),
            complaint_cluster    = int(row.get(cluster_col, -1) or -1),
            complaint_label      = str(row.get(label_col, "")),
            sentiment_score      = float(row.get(sentiment_col, 0.0) or 0.0),
            sentiment_label      = str(row.get(sentiment_label_col, "")),
            top_complaint_theme  = str(row.get(theme_col, "")),
            top_recommendation   = str(row.get(rec_col, "")),
            is_competitor_related= bool(row.get(competitor_col, False)),
            opportunity_flag     = _opportunity_flag(row),
        ))

    logger.info("Built %d retailer insights", len(insights))
    return insights


# ──────────────────────────────────────────────────────────────────────────────
# BUSINESS SUMMARY GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_business_summary(
    profiles:           list[ClusterProfile],
    sentiment_global:   dict[str, Any],
    rec_themes:         dict[str, int],
    kpis:               pd.DataFrame | None = None,
) -> dict[str, str]:
    """
    Generate concise business-language summaries for executive reporting.
    Returns a keyed dict of summary paragraphs.
    """
    logger.info("Generating business summaries…")

    # Top complaint cluster
    top_cluster = max(profiles, key=lambda p: p.size, default=None)

    # Sentiment breakdown
    pos_pct = sentiment_global.get("Positive_pct", 0)
    neg_pct = sentiment_global.get("Negative_pct", 0)
    neu_pct = sentiment_global.get("Neutral_pct", 0)

    # Top recommendation
    top_rec = max(rec_themes, key=rec_themes.get) if rec_themes else "N/A"
    top_rec_cnt = rec_themes.get(top_rec, 0)

    summary: dict[str, str] = {}

    summary["complaint_overview"] = (
        f"The dominant retailer complaint theme is '{top_cluster.label}' "
        f"({top_cluster.size} retailers, {top_cluster.pct_of_total:.1f}%). "
        f"Key product signals include: {', '.join(top_cluster.top_keywords[:5])}. "
        f"Immediate product/quality intervention is recommended for this segment."
    ) if top_cluster else "No complaint clusters computed."

    summary["sentiment_overview"] = (
        f"Overall feedback sentiment: {pos_pct:.1f}% Positive, "
        f"{neg_pct:.1f}% Negative, {neu_pct:.1f}% Neutral. "
        f"{'Sentiment skews negative — immediate corrective action required.' if neg_pct > 40 else ''}"
        f"{'Majority feedback is positive — maintain product quality.' if pos_pct > 50 else ''}"
    )

    summary["top_recommendation"] = (
        f"The most requested retailer improvement is '{top_rec}' "
        f"(cited by {top_rec_cnt} retailers). "
        f"Top 3 priorities: "
        + ", ".join(f"'{k}' ({v})" for k, v in sorted(rec_themes.items(), key=lambda x: -x[1])[:3])
        + ". These represent the highest-ROI intervention areas."
    )

    cluster_summary_lines = []
    for p in sorted(profiles, key=lambda x: -x.size)[:4]:
        cluster_summary_lines.append(
            f"• {p.label}: {p.size} retailers, avg sentiment {p.avg_sentiment:+.2f}"
        )
    summary["cluster_digest"] = "\n".join(cluster_summary_lines)

    summary["action_priority"] = (
        "Priority 1 — Taste/Sourness: Product reformulation or consistency check required.\n"
        "Priority 2 — Packaging: Leak-proof redesign for pouch format is critical.\n"
        "Priority 3 — Promotions: Extend trade promotion reach (currently ~18% of retailers).\n"
        "Priority 4 — Rep Coverage: Increase visit frequency; FruBon outperforms Amul on rep quality.\n"
        "Priority 5 — SKU Affordability: Introduce 80g/180g SKUs with competitive margin structure."
    )

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ──────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def export_complaint_clusters(
    df: pd.DataFrame, profiles: list[ClusterProfile], tag: str = ""
) -> tuple[Path, Path]:
    """Export cluster-annotated retailer table + cluster profile summary."""
    suffix = f"_{tag}" if tag else ""

    # Retailer-level cluster assignments
    cols = [c for c in [
        "Sr. No.", "Shop Name", "zone_clean", "store_type_clean",
        "nlp_complaints", "complaint_cluster_id", "complaint_cluster_label",
        "sentiment_score", "sentiment_label", "dominant_theme",
        "is_competitor_related", "competitor_brands_mentioned",
    ] if c in df.columns]
    p1 = EXPORTS_DIR / f"complaint_clusters{suffix}_{_ts()}.csv"
    df[cols].to_csv(p1, index=False)
    logger.info("📄 Complaint clusters → %s", p1)

    # Cluster profile summary
    rows = []
    for p in profiles:
        rows.append({
            "cluster_id":       p.cluster_id,
            "label":            p.label,
            "size":             p.size,
            "pct_of_total":     p.pct_of_total,
            "top_keywords":     " | ".join(p.top_keywords[:8]),
            "avg_sentiment":    p.avg_sentiment,
            "top_zone":         max(p.zone_distribution, key=p.zone_distribution.get) if p.zone_distribution else "",
            "business_summary": p.business_summary,
        })
    p2 = EXPORTS_DIR / f"complaint_cluster_profiles{suffix}_{_ts()}.csv"
    pd.DataFrame(rows).to_csv(p2, index=False)
    logger.info("📄 Cluster profiles → %s", p2)
    return p1, p2


def export_sentiment_summary(df: pd.DataFrame, tag: str = "") -> Path:
    """Export sentiment summary: global + zone breakdown + distribution."""
    suffix = f"_{tag}" if tag else ""

    # Per-retailer sentiment
    cols = [c for c in ["Sr. No.", "Shop Name", "zone_clean",
                        "sentiment_score", "sentiment_label", "dominant_theme"] if c in df.columns]
    p = EXPORTS_DIR / f"sentiment_summary{suffix}_{_ts()}.csv"
    df[cols].to_csv(p, index=False)
    logger.info("📄 Sentiment summary → %s", p)
    return p


def export_keyword_frequency(keywords: list[KeywordFrequency], tag: str = "") -> Path:
    """Export keyword frequency table across all corpus sections."""
    suffix = f"_{tag}" if tag else ""
    rows = [asdict(k) for k in keywords]
    p = EXPORTS_DIR / f"keyword_frequency{suffix}_{_ts()}.csv"
    pd.DataFrame(rows).sort_values("tfidf_weight", ascending=False).to_csv(p, index=False)
    logger.info("📄 Keyword frequency → %s  (%d terms)", p, len(rows))
    return p


def export_topic_clusters(topics: list[TopicCluster], tag: str = "") -> Path:
    """Export topic modeling results."""
    suffix = f"_{tag}" if tag else ""
    rows = []
    for t in topics:
        rows.append({
            "topic_id":      t.topic_id,
            "label":         t.label,
            "method":        t.method,
            "document_count":t.document_count,
            "top_terms":     " | ".join(t.top_terms[:8]),
            "top_weights":   " | ".join(str(w) for w in t.term_weights[:8]),
        })
    p = EXPORTS_DIR / f"topic_clusters{suffix}_{_ts()}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    logger.info("📄 Topic clusters → %s", p)
    return p


def export_retailer_insights(insights: list[RetailerInsight], tag: str = "") -> Path:
    """Export per-retailer NLP insight records."""
    suffix = f"_{tag}" if tag else ""
    rows = [asdict(i) for i in insights]
    p = EXPORTS_DIR / f"retailer_insights{suffix}_{_ts()}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    logger.info("📄 Retailer insights → %s  (%d rows)", p, len(rows))
    return p


def export_competitor_complaints(df: pd.DataFrame, tag: str = "") -> Path:
    """Export flagged competitor-related feedback records."""
    suffix = f"_{tag}" if tag else ""
    comp_df = df[df.get("is_competitor_related", pd.Series(False, index=df.index))].copy() \
        if "is_competitor_related" in df.columns else pd.DataFrame()
    cols = [c for c in [
        "Sr. No.", "Shop Name", "zone_clean", "nlp_complaints", "nlp_full_corpus",
        "competitor_brands_mentioned", "sentiment_score", "sentiment_label",
        "#1 Selling Dahi Brand at Store", "Primary Reason Customers Choose Competitor",
    ] if c in comp_df.columns]
    p = EXPORTS_DIR / f"competitor_complaints{suffix}_{_ts()}.csv"
    if not comp_df.empty:
        comp_df[cols].to_csv(p, index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(p, index=False)
    logger.info("📄 Competitor complaints → %s  (%d rows)", p, len(comp_df))
    return p


def export_recommendation_summary(
    df: pd.DataFrame, themes: dict[str, int], tag: str = ""
) -> Path:
    """Export recommendation themes: per-retailer + frequency summary."""
    suffix = f"_{tag}" if tag else ""
    rows = [{"theme": k, "count": v, "pct": round(v / len(df) * 100, 2)}
            for k, v in sorted(themes.items(), key=lambda x: -x[1])]
    p = EXPORTS_DIR / f"recommendation_summary{suffix}_{_ts()}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    logger.info("📄 Recommendation summary → %s", p)
    return p


def export_business_summary(summary: dict[str, str], tag: str = "") -> Path:
    """Export human-readable business summary as CSV."""
    suffix = f"_{tag}" if tag else ""
    rows = [{"section": k, "summary": v} for k, v in summary.items()]
    p = EXPORTS_DIR / f"business_summary{suffix}_{_ts()}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    logger.info("📄 Business summary → %s", p)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION-READY PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────

def sentiment_donut_data(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Sentiment distribution for a donut/pie chart."""
    if "sentiment_label" not in df.columns:
        return []
    vc = df["sentiment_label"].value_counts()
    total = len(df)
    return [{"label": k, "count": int(v), "pct": round(v / total * 100, 2)}
            for k, v in vc.items()]


def complaint_cluster_bar_data(profiles: list[ClusterProfile]) -> list[dict[str, Any]]:
    """Cluster sizes for a horizontal bar chart."""
    return [{"label": p.label, "count": p.size, "pct": p.pct_of_total,
             "avg_sentiment": p.avg_sentiment}
            for p in sorted(profiles, key=lambda x: -x.size)]


def keyword_cloud_data(keywords: list[KeywordFrequency], top_n: int = 30) -> list[dict[str, Any]]:
    """Keyword cloud data (term + weight for sizing)."""
    return [{"text": k.term, "value": k.frequency, "tfidf": k.tfidf_weight}
            for k in keywords[:top_n]]


def topic_heatmap_data(topics: list[TopicCluster]) -> list[dict[str, Any]]:
    """Topic × term weight matrix for a heatmap chart."""
    rows = []
    for t in topics:
        for term, weight in zip(t.top_terms[:8], t.term_weights[:8]):
            rows.append({"topic": t.label, "term": term, "weight": weight})
    return rows


def zone_sentiment_bar_data(zone_map: dict[str, float]) -> list[dict[str, Any]]:
    """Zone sentiment for a ranked bar chart."""
    return [{"zone": z, "avg_sentiment": s} for z, s in sorted(zone_map.items(), key=lambda x: x[1])]


def recommendation_treemap_data(themes: dict[str, int]) -> list[dict[str, Any]]:
    """Recommendation theme frequency for a treemap."""
    total = sum(themes.values()) or 1
    return [{"label": k, "value": v, "pct": round(v / total * 100, 2)}
            for k, v in sorted(themes.items(), key=lambda x: -x[1])]


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def load_corpus(corpus_path: str | Path) -> pd.DataFrame:
    """Load and validate the NLP corpus CSV."""
    logger.info("Loading NLP corpus from: %s", corpus_path)
    df = pd.read_csv(corpus_path, low_memory=False)
    required = {"nlp_complaints", "nlp_suggestions", "nlp_full_corpus"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NLP corpus missing required columns: {missing}")
    logger.info("Corpus loaded: %d rows × %d columns", *df.shape)
    return df


def load_retailers(retailers_path: str | Path) -> pd.DataFrame:
    """Load and validate the retailers CSV."""
    logger.info("Loading retailers from: %s", retailers_path)
    df = pd.read_csv(retailers_path, low_memory=False)
    # Cast boolean columns
    for col in ["stocks_ct", "feat_refrigerator_flag", "feat_dedicated_fridge"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False}).astype(bool)
    for col in [c for c in df.columns if c.startswith("feat_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Retailers loaded: %d rows × %d columns", *df.shape)
    return df


def merge_datasets(corpus: pd.DataFrame, retailers: pd.DataFrame) -> pd.DataFrame:
    """
    Merge corpus with retailer metadata on Sr. No.
    Falls back to index-based merge if Sr. No. is absent in either.
    """
    key = "Sr. No."
    if key in corpus.columns and key in retailers.columns:
        # Keep all retailer columns, bring in NLP cols from corpus
        nlp_cols = [c for c in corpus.columns if c.startswith("nlp_")]
        merge_cols = [key] + nlp_cols
        overlap = set(retailers.columns) & set(nlp_cols)
        if overlap:
            retailers = retailers.drop(columns=list(overlap))
        df = retailers.merge(corpus[[key] + [c for c in nlp_cols if c not in retailers.columns]],
                             on=key, how="left")
    else:
        logger.warning("Sr. No. not found in both datasets — using index-based merge")
        nlp_cols = [c for c in corpus.columns if c.startswith("nlp_")]
        for col in nlp_cols:
            if col not in retailers.columns:
                retailers[col] = corpus[col].values if len(corpus) == len(retailers) else None
        df = retailers

    logger.info("Merged dataset: %d rows × %d columns", *df.shape)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def run_nlp_engine(
    corpus_path:    str | Path = "data/processed/nlp_corpus.csv",
    retailers_path: str | Path = "data/processed/retailers_clean.csv",
    n_complaint_clusters: int | None = None,   # None → auto-select
    n_suggestion_topics:  int = 6,
    topic_method:   str  = "lda",
    export:         bool = True,
    tag:            str  = "",
) -> dict[str, Any]:
    """
    Full NLP analytics pipeline.

    Parameters
    ----------
    corpus_path            : Path to nlp_corpus.csv
    retailers_path         : Path to retailers_clean.csv
    n_complaint_clusters   : Fixed K for complaint clustering (None = auto)
    n_suggestion_topics    : Number of suggestion topics for LDA/NMF
    topic_method           : 'lda' | 'nmf'
    export                 : Write CSV exports
    tag                    : Label suffix for exported files

    Returns
    -------
    dict with keys:
        df, complaint_profiles, suggestion_topics, complaint_topics,
        sentiment_global, all_keywords, rec_themes, zone_sentiment,
        retailer_insights, chart_data, export_paths, business_summary
    """
    logger.info("══════════════════════════════════════════════════")
    logger.info("  FMCG NLP Engine  |  tag=%s", tag or "none")
    logger.info("══════════════════════════════════════════════════")

    # ── 1. Load & merge ──────────────────────────────────────────────────────
    corpus    = load_corpus(corpus_path)
    retailers = load_retailers(retailers_path)
    df        = merge_datasets(corpus, retailers)

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    df = preprocess_dataframe(df)

    # ── 3. Competitor detection (needs raw text) ──────────────────────────────
    raw_text_cols = [c for c in [
        "nlp_complaints", "nlp_ct_feedback", "nlp_suggestions",
        "nlp_other_feedback", "nlp_full_corpus",
        "Primary Reason Customers Choose Competitor",
        "Best Trade Promotions Brand", "Best Rep Visits Brand",
        "#1 Selling Dahi Brand at Store",
    ] if c in df.columns]
    df = detect_competitor_complaints(df, raw_text_cols)

    # ── 4. Complaint clustering ───────────────────────────────────────────────
    k_range = range(4, 9) if n_complaint_clusters is None else range(n_complaint_clusters, n_complaint_clusters + 1)
    df, complaint_profiles, comp_vec, comp_km = cluster_complaints(
        df,
        text_col  = "nlp_complaints_proc",
        zone_col  = "zone_clean",
        k_range   = k_range,
    )

    # ── 5. Recommendation extraction ─────────────────────────────────────────
    df["recommendation_theme"], rec_themes = extract_recommendations(
        df.get("nlp_suggestions", pd.Series("", index=df.index))
    )

    # ── 6. Sentiment analysis (on raw complaints) ─────────────────────────────
    # Already computed inside cluster_complaints; enrich with suggestion sentiment
    sug_sent = analyze_sentiment(
        df.get("nlp_suggestions", pd.Series("", index=df.index))
    )
    df["suggestion_sentiment_score"] = sug_sent["sentiment_score"].values

    # Global sentiment distribution
    vc = df["sentiment_label"].value_counts()
    n  = len(df)
    sentiment_global: dict[str, Any] = {
        "total":          n,
        "Positive_count": int(vc.get("Positive", 0)),
        "Neutral_count":  int(vc.get("Neutral", 0)),
        "Negative_count": int(vc.get("Negative", 0)),
        "Positive_pct":   round(vc.get("Positive", 0) / n * 100, 2),
        "Neutral_pct":    round(vc.get("Neutral",  0) / n * 100, 2),
        "Negative_pct":   round(vc.get("Negative", 0) / n * 100, 2),
        "avg_sentiment":  round(float(df["sentiment_score"].mean()), 4),
        "min_sentiment":  round(float(df["sentiment_score"].min()), 4),
        "max_sentiment":  round(float(df["sentiment_score"].max()), 4),
    }
    logger.info("Global sentiment: %s", sentiment_global)

    # ── 7. Zone sentiment map ─────────────────────────────────────────────────
    zone_sentiment = compute_zone_sentiment(df)

    # ── 8. Keyword extraction ─────────────────────────────────────────────────
    all_keywords: list[KeywordFrequency] = []
    for col_raw, section in [
        ("nlp_complaints",      "complaints"),
        ("nlp_suggestions",     "suggestions"),
        ("nlp_why_not_stocking","why_not_stocking"),
        ("nlp_full_corpus",     "full_corpus"),
    ]:
        if col_raw in df.columns:
            kws = extract_keywords(df[col_raw], corpus_name=section, top_n=30)
            all_keywords.extend(kws)

    # ── 9. Topic modeling ─────────────────────────────────────────────────────
    suggestion_topics, sug_doc_topic = model_topics(
        df.get("nlp_suggestions", pd.Series("", index=df.index)),
        corpus_name = "suggestions",
        n_topics    = n_suggestion_topics,
        method      = topic_method,
    )
    df["suggestion_topic_id"]    = sug_doc_topic.argmax(axis=1)
    df["suggestion_topic_label"] = pd.Series(
        df["suggestion_topic_id"].map({t.topic_id: t.label for t in suggestion_topics})
    )

    # Also topic-model the complaint corpus for cross-validation
    complaint_topics, _ = model_topics(
        df.get("nlp_complaints", pd.Series("", index=df.index)),
        corpus_name = "complaints",
        n_topics    = min(6, len(complaint_profiles)),
        method      = "nmf",
    )

    # ── 10. Retailer-level insights ───────────────────────────────────────────
    retailer_insights = build_retailer_insights(df)

    # ── 11. Business summaries ────────────────────────────────────────────────
    biz_summary = generate_business_summary(
        complaint_profiles, sentiment_global, rec_themes
    )

    # ── 12. Visualization payloads ────────────────────────────────────────────
    chart_data: dict[str, Any] = {
        "sentiment_donut":       sentiment_donut_data(df),
        "complaint_cluster_bar": complaint_cluster_bar_data(complaint_profiles),
        "keyword_cloud":         keyword_cloud_data(all_keywords, top_n=30),
        "topic_heatmap":         topic_heatmap_data(suggestion_topics),
        "zone_sentiment_bar":    zone_sentiment_bar_data(zone_sentiment),
        "recommendation_treemap":recommendation_treemap_data(rec_themes),
        "sentiment_global":      sentiment_global,
    }

    # ── 13. CSV exports ───────────────────────────────────────────────────────
    export_paths: dict[str, Path] = {}
    if export:
        p1, p2 = export_complaint_clusters(df, complaint_profiles, tag)
        export_paths["complaint_clusters"]        = p1
        export_paths["complaint_cluster_profiles"]= p2
        export_paths["sentiment_summary"]         = export_sentiment_summary(df, tag)
        export_paths["keyword_frequency"]         = export_keyword_frequency(all_keywords, tag)
        export_paths["suggestion_topics"]         = export_topic_clusters(suggestion_topics, tag)
        export_paths["complaint_topics"]          = export_topic_clusters(
            complaint_topics, tag=f"{tag}_complaint")
        export_paths["retailer_insights"]         = export_retailer_insights(retailer_insights, tag)
        export_paths["competitor_complaints"]     = export_competitor_complaints(df, tag)
        export_paths["recommendation_summary"]    = export_recommendation_summary(df, rec_themes, tag)
        export_paths["business_summary"]          = export_business_summary(biz_summary, tag)

    logger.info("══ NLP Engine complete. %d exports written. ══", len(export_paths))

    return {
        "df":                    df,
        "complaint_profiles":    complaint_profiles,
        "suggestion_topics":     suggestion_topics,
        "complaint_topics":      complaint_topics,
        "sentiment_global":      sentiment_global,
        "all_keywords":          all_keywords,
        "rec_themes":            rec_themes,
        "zone_sentiment":        zone_sentiment,
        "retailer_insights":     retailer_insights,
        "chart_data":            chart_data,
        "export_paths":          export_paths,
        "business_summary":      biz_summary,
        "vectorizer":            comp_vec,
        "cluster_model":         comp_km,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FMCG NLP Feedback Analyzer")
    parser.add_argument("--corpus",    default="data/processed/nlp_corpus.csv")
    parser.add_argument("--retailers", default="data/processed/retailers_clean.csv")
    parser.add_argument("--k",         type=int, default=None,
                        help="Number of complaint clusters (default: auto)")
    parser.add_argument("--topics",    type=int, default=6,
                        help="Number of suggestion topics (default: 6)")
    parser.add_argument("--method",    choices=["lda", "nmf"], default="lda",
                        help="Topic modeling method (default: lda)")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--tag",       default="")
    args = parser.parse_args()

    results = run_nlp_engine(
        corpus_path           = args.corpus,
        retailers_path        = args.retailers,
        n_complaint_clusters  = args.k,
        n_suggestion_topics   = args.topics,
        topic_method          = args.method,
        export                = not args.no_export,
        tag                   = args.tag,
    )

    sg = results["sentiment_global"]
    print("\n" + "═" * 62)
    print("  FMCG NLP ANALYTICS — RESULTS SUMMARY")
    print("═" * 62)
    print(f"  Total Retailers Analysed : {sg['total']}")
    print(f"  Positive Sentiment       : {sg['Positive_pct']:.1f}%")
    print(f"  Neutral  Sentiment       : {sg['Neutral_pct']:.1f}%")
    print(f"  Negative Sentiment       : {sg['Negative_pct']:.1f}%")
    print(f"  Avg Sentiment Score      : {sg['avg_sentiment']:+.4f}")
    print()
    print("  COMPLAINT CLUSTERS:")
    for p in sorted(results["complaint_profiles"], key=lambda x: -x.size):
        print(f"    [{p.cluster_id}] {p.label:<35} {p.size:>3} retailers "
              f"({p.pct_of_total:.1f}%) | sentiment {p.avg_sentiment:+.2f}")
    print()
    print("  SUGGESTION TOPICS:")
    for t in sorted(results["suggestion_topics"], key=lambda x: -x.document_count):
        print(f"    [{t.topic_id}] {t.label:<35} {t.document_count:>3} docs")
    print()
    print("  TOP RECOMMENDATIONS:")
    for theme, cnt in sorted(results["rec_themes"].items(), key=lambda x: -x[1])[:5]:
        print(f"    {theme:<35} {cnt:>3}")
    print()
    print("  ZONE SENTIMENT:")
    for zone, score in results["zone_sentiment"].items():
        print(f"    {zone:<25} {score:+.4f}")
    print()
    if results["export_paths"]:
        print("  EXPORTS:")
        for name, path in results["export_paths"].items():
            print(f"    {name:<35} → {path}")
    print()
    print("  BUSINESS SUMMARY:")
    for section, text in results["business_summary"].items():
        print(f"\n  [{section.upper()}]")
        print(f"  {text}")
    print("\n" + "═" * 62)