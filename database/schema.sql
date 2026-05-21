-- ============================================================
-- FMCG RETAIL INTELLIGENCE PLATFORM
-- Production-Grade PostgreSQL Schema
-- Amul Retailer Intelligence System | Jaipur Market
-- ============================================================

-- ============================================================
-- EXTENSIONS
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- LOOKUP / DIMENSION TABLES
-- ============================================================

CREATE TABLE zones (
    zone_id         SERIAL PRIMARY KEY,
    zone_name       VARCHAR(200) NOT NULL UNIQUE,
    zone_code       VARCHAR(50)  NOT NULL UNIQUE,
    zone_type       VARCHAR(50)  CHECK (zone_type IN ('urban', 'semi-urban', 'peripheral', 'old_city')),
    is_premium      BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE store_types (
    store_type_id   SERIAL PRIMARY KEY,
    store_type_name VARCHAR(100) NOT NULL UNIQUE,
    store_type_code VARCHAR(50)  NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE distributors (
    distributor_id   SERIAL PRIMARY KEY,
    distributor_name VARCHAR(200) NOT NULL,
    zone_id          INT          REFERENCES zones(zone_id) ON DELETE SET NULL,
    contact_name     VARCHAR(200),
    contact_phone    VARCHAR(20),
    is_active        BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE brands (
    brand_id        SERIAL PRIMARY KEY,
    brand_name      VARCHAR(200) NOT NULL UNIQUE,
    brand_code      VARCHAR(50)  NOT NULL UNIQUE,
    parent_company  VARCHAR(200),
    is_amul         BOOLEAN      NOT NULL DEFAULT FALSE,
    is_competitor   BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE sku_types (
    sku_id          SERIAL PRIMARY KEY,
    brand_id        INT          NOT NULL REFERENCES brands(brand_id) ON DELETE CASCADE,
    sku_name        VARCHAR(200) NOT NULL,
    sku_code        VARCHAR(100),
    weight_grams    INT,
    format          VARCHAR(50)  CHECK (format IN ('cup', 'pouch', 'both', 'other')),
    mrp             NUMERIC(10, 2),
    margin_pct      NUMERIC(5, 2),
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (brand_id, sku_code)
);

-- ============================================================
-- CORE RETAILER TABLE
-- ============================================================

CREATE TABLE retailers (
    retailer_id         SERIAL PRIMARY KEY,
    data_source         VARCHAR(100) NOT NULL DEFAULT 'Primary Field Survey',
    shop_name           VARCHAR(200),
    owner_name          VARCHAR(200),
    phone_raw           VARCHAR(50),
    phone_clean         VARCHAR(20),
    phone_valid         BOOLEAN      NOT NULL DEFAULT FALSE,
    area_locality       VARCHAR(200),
    zone_id             INT          REFERENCES zones(zone_id) ON DELETE SET NULL,
    store_type_id       INT          REFERENCES store_types(store_type_id) ON DELETE SET NULL,
    years_operating     VARCHAR(50),
    years_operating_num NUMERIC(4, 1),
    distributor_id      INT          REFERENCES distributors(distributor_id) ON DELETE SET NULL,
    survey_timestamp    TIMESTAMPTZ,
    is_active           BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- INFRASTRUCTURE & COLD CHAIN
-- ============================================================

CREATE TABLE retailer_infrastructure (
    infra_id                    SERIAL PRIMARY KEY,
    retailer_id                 INT          NOT NULL UNIQUE REFERENCES retailers(retailer_id) ON DELETE CASCADE,
    has_refrigerator            VARCHAR(100),
    fridge_type                 VARCHAR(50)  CHECK (fridge_type IN ('dedicated', 'shared', 'none')),
    fridge_flag                 BOOLEAN      NOT NULL DEFAULT FALSE,
    dedicated_fridge_flag       BOOLEAN      NOT NULL DEFAULT FALSE,
    branded_display_brand_id    INT          REFERENCES brands(brand_id) ON DELETE SET NULL,
    cold_chain_support_needed   BOOLEAN,
    infrastructure_score        NUMERIC(4, 2),
    created_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- PRODUCT INTELLIGENCE
-- ============================================================

CREATE TABLE product_intelligence (
    record_id               SERIAL PRIMARY KEY,
    retailer_id             INT          NOT NULL UNIQUE REFERENCES retailers(retailer_id) ON DELETE CASCADE,
    stocks_ct               BOOLEAN      NOT NULL DEFAULT FALSE,
    aware_ct                VARCHAR(100),
    awareness_tier          VARCHAR(50)  CHECK (awareness_tier IN ('stocking', 'aware_only', 'unaware')),
    why_not_stocking_ct     TEXT,
    ct_sku_mix              VARCHAR(200),
    ct_packs_per_week_raw   VARCHAR(100),
    ct_weekly_volume        NUMERIC(8, 2),
    masti_packs_per_week    VARCHAR(100),
    masti_weekly_volume     NUMERIC(8, 2),
    pack_sizes_in_demand    TEXT,
    packaging_format_pref   VARCHAR(50)  CHECK (packaging_format_pref IN ('cup', 'pouch', 'both', 'no_preference', 'other')),
    sales_change_vs_last_yr VARCHAR(100),
    sales_trend             VARCHAR(50)  CHECK (sales_trend IN ('increased_significantly', 'increased_slightly', 'stayed_same', 'decreased_slightly', 'decreased_significantly')),
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- COMPETITOR INTELLIGENCE
-- ============================================================

CREATE TABLE competitor_intelligence (
    record_id                   SERIAL PRIMARY KEY,
    retailer_id                 INT          NOT NULL UNIQUE REFERENCES retailers(retailer_id) ON DELETE CASCADE,
    top_brand_id                INT          REFERENCES brands(brand_id) ON DELETE SET NULL,
    top_brand_raw               VARCHAR(200),
    brand_ranking_raw           TEXT,
    competitor_choice_reason    TEXT,
    choice_reason_primary       VARCHAR(100),
    customers_ask_amul          VARCHAR(100),
    customers_ask_competitor    VARCHAR(100),
    best_trade_promotions_brand_id  INT      REFERENCES brands(brand_id) ON DELETE SET NULL,
    best_rep_visits_brand_id        INT      REFERENCES brands(brand_id) ON DELETE SET NULL,
    best_shelf_life_brand_id        INT      REFERENCES brands(brand_id) ON DELETE SET NULL,
    best_display_fridge_brand_id    INT      REFERENCES brands(brand_id) ON DELETE SET NULL,
    biggest_growth_brand_id         INT      REFERENCES brands(brand_id) ON DELETE SET NULL,
    competitor_presence_flag    BOOLEAN      NOT NULL DEFAULT FALSE,
    frubon_threat_flag          BOOLEAN      NOT NULL DEFAULT FALSE,
    frubon_threat_score         NUMERIC(4, 2) NOT NULL DEFAULT 0,
    created_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- RETAILER ENGAGEMENT
-- ============================================================

CREATE TABLE retailer_engagement (
    record_id                   SERIAL PRIMARY KEY,
    retailer_id                 INT          NOT NULL UNIQUE REFERENCES retailers(retailer_id) ON DELETE CASCADE,
    margin_on_amul_dahi         VARCHAR(100),
    trade_promotions_received   VARCHAR(100),
    promo_received_clean        VARCHAR(50)  CHECK (promo_received_clean IN ('yes_occasionally', 'yes_regularly', 'never')),
    promo_received_flag         BOOLEAN      NOT NULL DEFAULT FALSE,
    rep_visit_frequency         VARCHAR(100),
    rep_visit_clean             VARCHAR(50)  CHECK (rep_visit_clean IN ('weekly', 'fortnightly', 'monthly', 'rarely', 'never')),
    rep_visit_score             NUMERIC(4, 2) NOT NULL DEFAULT 0,
    engagement_score            NUMERIC(6, 2) NOT NULL DEFAULT 0,
    created_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- RETAILER FEEDBACK & NLP
-- ============================================================

CREATE TABLE retailer_feedback (
    feedback_id             SERIAL PRIMARY KEY,
    retailer_id             INT          NOT NULL UNIQUE REFERENCES retailers(retailer_id) ON DELETE CASCADE,

    -- Raw text fields
    customer_complaints_raw TEXT,
    ct_specific_feedback_raw TEXT,
    improvement_suggestions_raw TEXT,
    other_feedback_raw      TEXT,
    why_not_stocking_raw    TEXT,

    -- NLP-cleaned / normalised
    nlp_complaints          TEXT,
    nlp_ct_feedback         TEXT,
    nlp_suggestions         TEXT,
    nlp_other_feedback      TEXT,
    nlp_why_not_stocking    TEXT,
    nlp_full_corpus         TEXT,

    -- Sentiment scores (-1.0 to 1.0)
    complaint_sentiment     NUMERIC(5, 4),
    suggestion_sentiment    NUMERIC(5, 4),
    overall_sentiment       NUMERIC(5, 4),

    -- NLP-derived clusters & topics
    complaint_cluster_id    INT,
    complaint_cluster_label VARCHAR(200),
    suggestion_topic_id     INT,
    suggestion_topic_label  VARCHAR(200),
    keyword_tags            TEXT[],

    -- Top complaint flags
    flag_sour_taste         BOOLEAN NOT NULL DEFAULT FALSE,
    flag_texture_thin       BOOLEAN NOT NULL DEFAULT FALSE,
    flag_packaging_leak     BOOLEAN NOT NULL DEFAULT FALSE,
    flag_short_shelf_life   BOOLEAN NOT NULL DEFAULT FALSE,
    flag_price_high         BOOLEAN NOT NULL DEFAULT FALSE,
    flag_no_demand          BOOLEAN NOT NULL DEFAULT FALSE,
    flag_rep_absent         BOOLEAN NOT NULL DEFAULT FALSE,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- RETAILER SCORING (ML-derived, refreshable)
-- ============================================================

CREATE TABLE retailer_scores (
    score_id                SERIAL PRIMARY KEY,
    retailer_id             INT          NOT NULL REFERENCES retailers(retailer_id) ON DELETE CASCADE,

    awareness_score         NUMERIC(5, 2) NOT NULL DEFAULT 0,
    rep_visit_score         NUMERIC(5, 2) NOT NULL DEFAULT 0,
    promo_bonus             NUMERIC(5, 2) NOT NULL DEFAULT 0,
    engagement_score        NUMERIC(5, 2) NOT NULL DEFAULT 0,
    infrastructure_score    NUMERIC(5, 2) NOT NULL DEFAULT 0,
    competitor_presence     NUMERIC(5, 2) NOT NULL DEFAULT 0,
    frubon_threat_score     NUMERIC(5, 2) NOT NULL DEFAULT 0,
    ct_weekly_volume        NUMERIC(10, 2) NOT NULL DEFAULT 0,
    masti_weekly_volume     NUMERIC(10, 2) NOT NULL DEFAULT 0,
    years_operating_num     NUMERIC(5, 2),
    adoption_likelihood     NUMERIC(5, 4),

    model_version           VARCHAR(50),
    scored_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_current              BOOLEAN     NOT NULL DEFAULT TRUE,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- RETAILER SEGMENTATION
-- ============================================================

CREATE TABLE retailer_segments (
    segment_id      SERIAL PRIMARY KEY,
    segment_label   VARCHAR(100) NOT NULL UNIQUE,
    segment_code    VARCHAR(50)  NOT NULL UNIQUE,
    description     TEXT,
    priority        VARCHAR(50)  CHECK (priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MONITOR')),
    color_hex       VARCHAR(10),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE retailer_segment_assignments (
    assignment_id   SERIAL PRIMARY KEY,
    retailer_id     INT          NOT NULL REFERENCES retailers(retailer_id) ON DELETE CASCADE,
    segment_id      INT          NOT NULL REFERENCES retailer_segments(segment_id) ON DELETE CASCADE,
    cluster_id      INT,
    model_version   VARCHAR(50),
    confidence      NUMERIC(5, 4),
    assigned_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    is_current      BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (retailer_id, model_version)
);

-- ============================================================
-- BRAND RANKINGS AT RETAILER LEVEL
-- ============================================================

CREATE TABLE retailer_brand_rankings (
    ranking_id      SERIAL PRIMARY KEY,
    retailer_id     INT NOT NULL REFERENCES retailers(retailer_id) ON DELETE CASCADE,
    brand_id        INT NOT NULL REFERENCES brands(brand_id) ON DELETE CASCADE,
    rank_position   INT NOT NULL CHECK (rank_position >= 1),
    volume_share    NUMERIC(5, 2),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (retailer_id, brand_id)
);

-- ============================================================
-- AI INSIGHTS
-- ============================================================

CREATE TABLE ai_insights (
    insight_id          SERIAL PRIMARY KEY,
    insight_uid         UUID         NOT NULL DEFAULT uuid_generate_v4() UNIQUE,
    scope               VARCHAR(50)  NOT NULL CHECK (scope IN ('global', 'zone', 'store_type', 'segment', 'retailer')),
    scope_ref_id        INT,
    scope_ref_label     VARCHAR(200),
    intent_type         VARCHAR(100) CHECK (intent_type IN (
                            'penetration', 'awareness', 'competitor', 'complaints',
                            'zone', 'recommendation', 'executive_summary', 'general'
                        )),
    headline            TEXT,
    evidence            TEXT,
    strategic_implication TEXT,
    full_response       TEXT,
    model_used          VARCHAR(100),
    prompt_version      VARCHAR(50),
    input_context       JSONB,
    token_usage         INT,
    generated_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    is_pinned           BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- CONVERSATIONAL ANALYTICS (Chat sessions & turns)
-- ============================================================

CREATE TABLE chat_sessions (
    session_id      UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_name    VARCHAR(200),
    user_identifier VARCHAR(200),
    active_filters  JSONB,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE chat_turns (
    turn_id         SERIAL PRIMARY KEY,
    session_id      UUID         NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    turn_number     INT          NOT NULL,
    user_query      TEXT         NOT NULL,
    detected_intent VARCHAR(100),
    data_context    TEXT,
    ai_response     TEXT,
    generated_sql   TEXT,
    model_used      VARCHAR(100),
    latency_ms      INT,
    token_usage     INT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, turn_number)
);

-- ============================================================
-- KPI SNAPSHOTS (time-series store for dashboard)
-- ============================================================

CREATE TABLE kpi_snapshots (
    snapshot_id             SERIAL PRIMARY KEY,
    snapshot_date           DATE         NOT NULL,
    scope                   VARCHAR(50)  NOT NULL CHECK (scope IN ('global', 'zone', 'store_type', 'segment')),
    scope_ref_id            INT,
    scope_ref_label         VARCHAR(200),

    total_retailers         INT,
    ct_stockers             INT,
    ct_penetration_rate     NUMERIC(7, 4),
    ct_awareness_rate       NUMERIC(7, 4),
    trade_promo_reach       NUMERIC(7, 4),
    weekly_rep_freq_pct     NUMERIC(7, 4),
    frubon_threat_pct       NUMERIC(7, 4),
    complaint_rate          NUMERIC(7, 4),
    sourness_complaint_rate NUMERIC(7, 4),
    cup_format_pref_pct     NUMERIC(7, 4),
    avg_adoption_likelihood NUMERIC(7, 4),

    computed_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    UNIQUE (snapshot_date, scope, scope_ref_id)
);

-- ============================================================
-- NLP TOPIC REGISTRY
-- ============================================================

CREATE TABLE nlp_topics (
    topic_id        SERIAL PRIMARY KEY,
    topic_type      VARCHAR(50)  NOT NULL CHECK (topic_type IN ('complaint_cluster', 'suggestion_topic')),
    topic_label     VARCHAR(200) NOT NULL,
    topic_keywords  TEXT[],
    lda_index       INT,
    model_version   VARCHAR(50),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (topic_type, lda_index, model_version)
);

-- ============================================================
-- AUDIT / CHANGE LOG
-- ============================================================

CREATE TABLE audit_log (
    log_id          BIGSERIAL    PRIMARY KEY,
    table_name      VARCHAR(100) NOT NULL,
    record_id       INT          NOT NULL,
    operation       VARCHAR(10)  NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    changed_by      VARCHAR(200),
    old_data        JSONB,
    new_data        JSONB,
    changed_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- INDEXES — Core lookups
-- ============================================================

CREATE INDEX idx_retailers_zone        ON retailers(zone_id);
CREATE INDEX idx_retailers_store_type  ON retailers(store_type_id);
CREATE INDEX idx_retailers_distributor ON retailers(distributor_id);
CREATE INDEX idx_retailers_phone       ON retailers(phone_clean);
CREATE INDEX idx_retailers_active      ON retailers(is_active);
CREATE INDEX idx_retailers_shop_name   ON retailers USING gin (shop_name gin_trgm_ops);

-- ============================================================
-- INDEXES — Product intelligence
-- ============================================================

CREATE INDEX idx_pi_stocks_ct          ON product_intelligence(stocks_ct);
CREATE INDEX idx_pi_awareness_tier     ON product_intelligence(awareness_tier);
CREATE INDEX idx_pi_masti_volume       ON product_intelligence(masti_weekly_volume);
CREATE INDEX idx_pi_ct_volume          ON product_intelligence(ct_weekly_volume);

-- ============================================================
-- INDEXES — Competitor intelligence
-- ============================================================

CREATE INDEX idx_ci_top_brand          ON competitor_intelligence(top_brand_id);
CREATE INDEX idx_ci_frubon_threat      ON competitor_intelligence(frubon_threat_flag);
CREATE INDEX idx_ci_competitor_flag    ON competitor_intelligence(competitor_presence_flag);
CREATE INDEX idx_ci_best_promo_brand   ON competitor_intelligence(best_trade_promotions_brand_id);
CREATE INDEX idx_ci_best_rep_brand     ON competitor_intelligence(best_rep_visits_brand_id);

-- ============================================================
-- INDEXES — Engagement
-- ============================================================

CREATE INDEX idx_re_rep_visit_clean    ON retailer_engagement(rep_visit_clean);
CREATE INDEX idx_re_promo_flag         ON retailer_engagement(promo_received_flag);

-- ============================================================
-- INDEXES — Feedback / NLP
-- ============================================================

CREATE INDEX idx_rf_complaint_cluster  ON retailer_feedback(complaint_cluster_id);
CREATE INDEX idx_rf_suggestion_topic   ON retailer_feedback(suggestion_topic_id);
CREATE INDEX idx_rf_flag_sour          ON retailer_feedback(flag_sour_taste);
CREATE INDEX idx_rf_flag_packaging     ON retailer_feedback(flag_packaging_leak);
CREATE INDEX idx_rf_corpus_fts         ON retailer_feedback USING gin (to_tsvector('english', coalesce(nlp_full_corpus, '')));

-- ============================================================
-- INDEXES — Segmentation
-- ============================================================

CREATE INDEX idx_rsa_segment_current   ON retailer_segment_assignments(segment_id) WHERE is_current = TRUE;
CREATE INDEX idx_rsa_retailer_current  ON retailer_segment_assignments(retailer_id) WHERE is_current = TRUE;

-- ============================================================
-- INDEXES — Scoring
-- ============================================================

CREATE INDEX idx_rs_retailer_current   ON retailer_scores(retailer_id) WHERE is_current = TRUE;
CREATE INDEX idx_rs_adoption           ON retailer_scores(adoption_likelihood DESC);

-- ============================================================
-- INDEXES — AI Insights
-- ============================================================

CREATE INDEX idx_ai_scope              ON ai_insights(scope, scope_ref_id);
CREATE INDEX idx_ai_intent             ON ai_insights(intent_type);
CREATE INDEX idx_ai_generated_at       ON ai_insights(generated_at DESC);

-- ============================================================
-- INDEXES — Chat
-- ============================================================

CREATE INDEX idx_ct_session            ON chat_turns(session_id);
CREATE INDEX idx_ct_intent             ON chat_turns(detected_intent);

-- ============================================================
-- INDEXES — KPI Snapshots
-- ============================================================

CREATE INDEX idx_kpi_date_scope        ON kpi_snapshots(snapshot_date DESC, scope, scope_ref_id);

-- ============================================================
-- INDEXES — Audit
-- ============================================================

CREATE INDEX idx_audit_table_record    ON audit_log(table_name, record_id);
CREATE INDEX idx_audit_changed_at      ON audit_log(changed_at DESC);

-- ============================================================
-- SEED: Zone lookup
-- ============================================================

INSERT INTO zones (zone_name, zone_code, zone_type, is_premium) VALUES
    ('Amer / Periphery / Ajmer Road',            'Amer_Periphery',      'peripheral', FALSE),
    ('Vaishali Nagar / Mansarovar',               'Vaishali_Mansarovar', 'urban',      FALSE),
    ('Malviya Nagar / Jagatpura / Sanganer',      'Malviya_Sanganer',    'urban',      FALSE),
    ('Jhotwara / Murlipura',                      'Jhotwara_Murlipura',  'semi-urban', FALSE),
    ('C-Scheme / Bani Park / Civil Lines',        'C_Scheme',            'urban',      TRUE),
    ('Old City / Walled City',                    'Old_City',            'old_city',   FALSE);

-- ============================================================
-- SEED: Store type lookup
-- ============================================================

INSERT INTO store_types (store_type_name, store_type_code) VALUES
    ('Kiryana/General store',       'Kiryana'),
    ('Supermarket / Modern trade',  'Supermarket'),
    ('Dairy booth / Parlour',       'Dairy_Parlour'),
    ('Bakery / Convenience',        'Bakery'),
    ('Tea stall',                   'Tea_Stall'),
    ('Restaurant / Canteen',        'Restaurant');

-- ============================================================
-- SEED: Brands
-- ============================================================

INSERT INTO brands (brand_name, brand_code, parent_company, is_amul, is_competitor) VALUES
    ('Amul Masti Dahi',       'AMUL_MASTI',   'GCMMF',           TRUE,  FALSE),
    ('Amul Dahi Creamy & Tasty', 'AMUL_CT',   'GCMMF',           TRUE,  FALSE),
    ('Saras Dahi',            'SARAS',         'RCDF Rajasthan',  FALSE, TRUE),
    ('FruBon Dahi',           'FRUBON',        'FruBon',          FALSE, TRUE),
    ('Mother Dairy Curd',     'MOTHER_DAIRY',  'Mother Dairy',    FALSE, TRUE),
    ('Ksheer Dahi',           'KSHEER',        'Ksheer',          FALSE, TRUE),
    ('Rufil Dahi',            'RUFIL',         'Rufil',           FALSE, TRUE),
    ('Lotus Dahi',            'LOTUS',         'Lotus',           FALSE, TRUE);

-- ============================================================
-- SEED: Retailer segments
-- ============================================================

INSERT INTO retailer_segments (segment_label, segment_code, description, priority, color_hex) VALUES
    ('High-Potential Dormant',    'SEG_DORMANT',    'Refrigerated, high-volume Masti sellers, unaware of C&T',              'CRITICAL', '#E74C3C'),
    ('Competitor-Dominated',      'SEG_COMPETITOR', 'FruBon/Saras primary, weak Amul rep engagement',                       'HIGH',     '#F39C12'),
    ('Loyal Amul Core',           'SEG_LOYAL',      'Amul Masti primary, aware of C&T, moderate engagement',                'MEDIUM',   '#27AE60'),
    ('Low-Infrastructure Risk',   'SEG_LOW_INFRA',  'Limited cold chain, low volume, price-sensitive',                      'LOW',      '#95A5A6'),
    ('Premium Urban Adopters',    'SEG_PREMIUM',    'Supermarkets, C-Scheme zone, high-margin potential',                   'HIGH',     '#3498DB');

-- ============================================================
-- SEED: NLP topics
-- ============================================================

INSERT INTO nlp_topics (topic_type, topic_label, topic_keywords, lda_index, model_version) VALUES
    ('complaint_cluster', 'Sourness & Taste',       ARRAY['sour', 'acidic', 'taste', 'watery'],          0, 'v1'),
    ('complaint_cluster', 'Texture & Thickness',    ARRAY['thick', 'thin', 'texture', 'watery', 'whey'], 1, 'v1'),
    ('complaint_cluster', 'Packaging Issues',       ARRAY['leak', 'package', 'break', 'pouch'],          2, 'v1'),
    ('complaint_cluster', 'Price Sensitivity',      ARRAY['price', 'expensive', 'margin'],               3, 'v1'),
    ('complaint_cluster', 'No Complaints',          ARRAY['good', 'fine', 'okay'],                       4, 'v1'),
    ('complaint_cluster', 'Shelf Life & Freshness', ARRAY['shelf', 'fresh', 'expiry', 'old'],            5, 'v1'),
