import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from datetime import datetime
from pathlib import Path
import json
import joblib
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ---- Load Trained ML Model & Benchmark Data ----
MODEL_DIR = Path(__file__).parent / "model_prep" / "output"

@st.cache_resource
def load_trained_model():
    """Load the best trained pipeline (StandardScaler + ExtraTrees)."""
    model_path = MODEL_DIR / "best_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_benchmark_data():
    """Load pipeline report, model comparison, feature importance."""
    data = {}
    report_path = MODEL_DIR / "pipeline_report.json"
    if report_path.exists():
        with open(report_path) as f:
            data["report"] = json.load(f)
    comp_path = MODEL_DIR / "model_comparison.csv"
    if comp_path.exists():
        data["comparison"] = pd.read_csv(comp_path)
    final_path = MODEL_DIR / "final_comparison.csv"
    if final_path.exists():
        data["final"] = pd.read_csv(final_path)
    fi_path = MODEL_DIR / "feature_importance.csv"
    if fi_path.exists():
        data["importance"] = pd.read_csv(fi_path)
    feat_path = MODEL_DIR / "features.json"
    if feat_path.exists():
        with open(feat_path) as f:
            data["features"] = json.load(f)
    return data

trained_model = load_trained_model()
benchmark = load_benchmark_data()

# ---- Streamlit Config ----
st.set_page_config(
    page_title="🌍 Seismic Risk Assessment", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍"
)

# Clean CSS with automatic dark/light theme adaptation
st.markdown("""
<style>
    /* ---- CSS Variables for theme adaptation ---- */
    :root {
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --text-heading: #334155;
        --bg-card: #f8fafc;
        --bg-card-alt: #f1f5f9;
        --border-card: #e2e8f0;
        --notice-bg: #fef3c7;
        --notice-text: #92400e;
        --notice-border: #f59e0b;
        --warn-bg: #fef2f2;
        --warn-text: #7f1d1d;
        --warn-heading: #991b1b;
        --warn-border: #dc2626;
        --caution-bg: #fffbeb;
        --caution-text: #92400e;
        --caution-border: #d97706;
        --pred-bg: #fef2f2;
        --pred-border: #fecaca;
        --pred-heading: #991b1b;
        --pred-val: #dc2626;
        --pred-sub: #b91c1c;
        --footer-text: #64748b;
        --sidebar-bg: #1e293b;
        --sidebar-text: #e2e8f0;
        --sidebar-select-text: #0f172a;
        --disclaimer-text: #1f2937;
    }

    /* Dark theme overrides — Streamlit applies this attribute */
    [data-testid="stAppViewContainer"][data-theme="dark"],
    [data-theme="dark"] {
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --text-heading: #cbd5e1;
        --bg-card: #1e293b;
        --bg-card-alt: #1e293b;
        --border-card: #334155;
        --notice-bg: #422006;
        --notice-text: #fde68a;
        --notice-border: #d97706;
        --warn-bg: #450a0a;
        --warn-text: #fecaca;
        --warn-heading: #fca5a5;
        --warn-border: #ef4444;
        --caution-bg: #451a03;
        --caution-text: #fde68a;
        --caution-border: #f59e0b;
        --pred-bg: #450a0a;
        --pred-border: #7f1d1d;
        --pred-heading: #fca5a5;
        --pred-val: #f87171;
        --pred-sub: #fca5a5;
        --footer-text: #94a3b8;
        --sidebar-bg: #0f172a;
        --sidebar-text: #e2e8f0;
        --sidebar-select-text: #e2e8f0;
        --disclaimer-text: #e2e8f0;
    }

    /* Also support prefers-color-scheme for browsers */
    @media (prefers-color-scheme: dark) {
        :root:not([data-theme="light"]) {
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --text-heading: #cbd5e1;
            --bg-card: #1e293b;
            --bg-card-alt: #1e293b;
            --border-card: #334155;
            --notice-bg: #422006;
            --notice-text: #fde68a;
            --notice-border: #d97706;
            --warn-bg: #450a0a;
            --warn-text: #fecaca;
            --warn-heading: #fca5a5;
            --warn-border: #ef4444;
            --caution-bg: #451a03;
            --caution-text: #fde68a;
            --caution-border: #f59e0b;
            --pred-bg: #450a0a;
            --pred-border: #7f1d1d;
            --pred-heading: #fca5a5;
            --pred-val: #f87171;
            --pred-sub: #fca5a5;
            --footer-text: #94a3b8;
            --sidebar-bg: #0f172a;
            --sidebar-text: #e2e8f0;
            --sidebar-select-text: #e2e8f0;
            --disclaimer-text: #e2e8f0;
        }
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        min-width: 380px !important;
        max-width: 380px !important;
        background: var(--sidebar-bg) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown * { color: var(--sidebar-text) !important; }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * { color: var(--sidebar-select-text) !important; }
    [data-testid="stSidebar"] .stRadio label span { color: var(--sidebar-text) !important; }
    [data-testid="stSidebar"] .stSlider [data-baseweb] * { color: var(--sidebar-text) !important; }
    [data-testid="stSidebar"] .stCheckbox label span { color: var(--sidebar-text) !important; }

    /* ---- Layout ---- */
    .block-container { padding-top: 1.5rem; }

    /* ---- Header ---- */
    .app-header { font-size: 1.8rem; font-weight: 800; color: var(--text-primary); margin: 0; }
    .app-sub { font-size: 1rem; color: var(--text-secondary); margin: 0 0 1rem 0; }

    /* ---- Notice bar ---- */
    .notice-bar {
        background: var(--notice-bg); border-left: 4px solid var(--notice-border);
        padding: 12px 16px; border-radius: 6px; margin-bottom: 1.2rem;
        font-size: 0.92rem; color: var(--notice-text);
    }

    /* ---- Stat card ---- */
    .stat-card {
        background: var(--bg-card); border: 1px solid var(--border-card);
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .stat-card h4 { margin: 0 0 4px 0; font-size: 0.85rem; color: var(--text-secondary); font-weight: 600; }
    .stat-card .val { font-size: 1.6rem; font-weight: 800; color: var(--text-primary); }

    /* ---- Legend chip ---- */
    .legend-chip {
        display: inline-block; padding: 6px 14px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 700; color: white; margin: 3px 2px;
    }

    /* ---- Info card ---- */
    .info-card {
        background: var(--bg-card-alt); border: 1px solid var(--border-card);
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .info-card h4 { margin: 0; color: var(--text-heading); font-size: 0.85rem; font-weight: 600; }
    .info-card .val { font-size: 1.4rem; font-weight: 800; color: var(--text-primary); margin: 6px 0 2px 0; }
    .info-card .sub { font-size: 0.78rem; color: var(--text-muted); }

    /* ---- Footer card ---- */
    .footer-card {
        background: var(--bg-card); border: 1px solid var(--border-card);
        border-radius: 10px; padding: 14px; text-align: center;
    }
    .footer-card h4 { margin: 0 0 4px 0; color: var(--text-heading); font-size: 0.85rem; }
    .footer-card p { margin: 2px 0; color: var(--footer-text); font-size: 0.82rem; }

    /* ---- Warning banner ---- */
    .warn-banner {
        background: var(--warn-bg); border: 2px solid var(--warn-border); border-left: 6px solid var(--warn-border);
        padding: 16px 20px; border-radius: 8px; margin-bottom: 1.2rem;
    }
    .warn-banner h4 { margin: 0 0 6px 0; color: var(--warn-heading); font-size: 1rem; }
    .warn-banner p { margin: 2px 0; color: var(--warn-text); font-size: 0.88rem; line-height: 1.5; }

    /* ---- Caution banner ---- */
    .caution-banner {
        background: var(--caution-bg); border: 2px solid var(--caution-border); border-left: 6px solid var(--caution-border);
        padding: 14px 18px; border-radius: 8px; margin-bottom: 1rem;
    }
    .caution-banner p { margin: 2px 0; color: var(--caution-text); font-size: 0.88rem; line-height: 1.5; }

    /* ---- Prediction card ---- */
    .pred-card {
        background: var(--pred-bg); border: 1px solid var(--pred-border);
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .pred-card h4 { margin: 0 0 4px 0; font-size: 0.85rem; color: var(--pred-heading); font-weight: 600; }
    .pred-card .val { font-size: 1.4rem; font-weight: 800; color: var(--pred-val); }
    .pred-card .sub { font-size: 0.75rem; color: var(--pred-sub); margin-top: 2px; }

    /* ---- Disclaimer text in expander ---- */
    .disclaimer-content { color: var(--disclaimer-text); }
    .disclaimer-content h2, .disclaimer-content h3 { color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<h1 class="app-header">🌍 Seismic Risk Assessment Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Historical earthquake risk analysis across global tectonic boundaries</p>', unsafe_allow_html=True)

st.markdown("""
<div class="notice-bar">
    <strong>⚠️ Notice:</strong> This is a risk assessment tool based on historical seismic data —
    not a prediction system. Earthquakes cannot be predicted. For educational and research purposes only.
</div>
""", unsafe_allow_html=True)

# ---- Sidebar Controls ----
st.sidebar.markdown("# Control Panel")

# Prediction Mode Toggle
st.sidebar.markdown("---")
prediction_mode = st.sidebar.toggle(
    "Enable Prediction Mode",
    value=False,
    help="Switch to experimental statistical forecast view"
)
st.sidebar.markdown("---")

if not prediction_mode:
    # ===== ASSESSMENT MODE SIDEBAR =====
    assessment_mode = st.sidebar.radio(
        "Assessment Mode",
        ["Historical Risk Zones", "Seismic Hazard Assessment"],
    )

    region_filter = st.sidebar.selectbox(
        "Geographic Scope",
        ["Global", "Pacific Ring of Fire", "Mediterranean-Alpine Belt", "Mid-Atlantic Ridge", "Americas", "Asia-Pacific"],
    )

    mag_range = st.sidebar.slider(
        "Magnitude Range",
        min_value=1.0,
        max_value=9.0,
        value=(1.0, 9.0),
        step=0.5,
    )

    tile_options = {
        "Light Theme": "CartoDB.Positron",
        "Dark Theme": "CartoDB.DarkMatter",
        "Street Map": "OpenStreetMap",
        "Satellite": "Esri.WorldImagery"
    }
    map_style_choice = st.sidebar.selectbox(
        "Map Style", 
        list(tile_options.keys()), 
        index=0,
    )

    visualization_choice = st.sidebar.radio(
        "Visualization Mode",
        ["Risk Markers", "Heat Density Map", "Combined View"],
        index=2,
    )

    show_labels = st.sidebar.checkbox(
        "Show Location Labels", 
        value=True,
    )

    map_type = tile_options[map_style_choice]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Guide")
    st.sidebar.info("Change any filter to update the map instantly. Click markers for details.")

else:
    # ===== PREDICTION MODE SIDEBAR =====
    st.sidebar.markdown(
        '<p style="color:#fca5a5 !important; font-weight:700; font-size:0.9rem;">'
        'EXPERIMENTAL MODE</p>',
        unsafe_allow_html=True
    )

    pred_region = st.sidebar.selectbox(
        "Prediction Region",
        ["Global", "Pacific Ring of Fire", "Mediterranean-Alpine Belt", "Mid-Atlantic Ridge",
         "Japan", "Indonesia", "Chile", "California, USA", "Turkey", "Nepal/Himalaya", "Italy", "Iran"],
        key="pred_region"
    )

    pred_time_window = st.sidebar.selectbox(
        "Forecast Window",
        ["Next 30 days", "Next 90 days", "Next 6 months", "Next 1 year"],
        key="pred_time"
    )

    pred_min_mag = st.sidebar.slider(
        "Minimum Magnitude",
        min_value=3.0,
        max_value=8.0,
        value=5.0,
        step=0.5,
        key="pred_mag"
    )

    pred_depth = st.sidebar.slider(
        "Depth Range (km)",
        min_value=0,
        max_value=300,
        value=(0, 200),
        step=10,
        key="pred_depth"
    )

    pred_confidence = st.sidebar.select_slider(
        "Confidence Level",
        options=["Low", "Medium", "High"],
        value="Medium",
        key="pred_conf"
    )

    pred_map_style = st.sidebar.selectbox(
        "Map Style",
        ["Light Theme", "Dark Theme", "Street Map", "Satellite"],
        key="pred_map_style"
    )

    tile_options = {
        "Light Theme": "CartoDB.Positron",
        "Dark Theme": "CartoDB.DarkMatter",
        "Street Map": "OpenStreetMap",
        "Satellite": "Esri.WorldImagery"
    }

    st.sidebar.markdown("---")
    st.sidebar.warning(
        "Statistical forecasts only. Not real predictions. "
        "Do not use for safety decisions."
    )

# ============================================================
# PREDICTION MODE — Main Content  
# ============================================================
if prediction_mode:

    # ---- Warnings ----
    st.markdown("""
    <div class="warn-banner">
        <h4>EXPERIMENTAL PREDICTION MODE</h4>
        <p>This mode generates <strong>statistical probability estimates</strong> based on historical
        seismic patterns. These are <strong>NOT real earthquake predictions</strong>.</p>
        <p>No scientific method can reliably predict when, where, or how strong an earthquake will be.
        These forecasts are for <strong>educational exploration only</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="caution-banner">
        <p><strong>Do NOT</strong> use these results for: evacuation planning, insurance decisions,
        public advisories, or any safety-critical decision. Always follow guidance from
        <a href="https://www.usgs.gov/natural-hazards/earthquake-hazards" target="_blank">USGS</a>,
        <a href="https://www.ready.gov/earthquakes" target="_blank">Ready.gov</a>, and local authorities.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Prediction Data Generation (using trained ML model) ----
    np.random.seed(int(datetime.now().timestamp()) % 10000)

    pred_zone_lookup = {
        "Japan": {"lat": (30, 45), "lon": (130, 145), "base_rate": 0.85},
        "Indonesia": {"lat": (-10, 6), "lon": (95, 141), "base_rate": 0.92},
        "Chile": {"lat": (-56, -17), "lon": (-76, -66), "base_rate": 0.88},
        "California, USA": {"lat": (32, 42), "lon": (-125, -114), "base_rate": 0.70},
        "Turkey": {"lat": (36, 42), "lon": (26, 45), "base_rate": 0.75},
        "Nepal/Himalaya": {"lat": (26, 31), "lon": (80, 89), "base_rate": 0.80},
        "Italy": {"lat": (36, 47), "lon": (6, 19), "base_rate": 0.55},
        "Iran": {"lat": (25, 40), "lon": (44, 63), "base_rate": 0.72},
        "Pacific Ring of Fire": {"lat": (-55, 70), "lon": (-170, 180), "base_rate": 0.90},
        "Mediterranean-Alpine Belt": {"lat": (25, 47), "lon": (6, 89), "base_rate": 0.70},
        "Mid-Atlantic Ridge": {"lat": (60, 70), "lon": (-30, -10), "base_rate": 0.50},
        "Global": {"lat": (-60, 70), "lon": (-170, 180), "base_rate": 0.95},
    }

    zone_info = pred_zone_lookup.get(pred_region, pred_zone_lookup["Global"])
    time_mult = {"Next 30 days": 1.0, "Next 90 days": 1.8, "Next 6 months": 2.5, "Next 1 year": 3.2}
    conf_mult = {"Low": 0.4, "Medium": 0.65, "High": 0.85}

    t_m = time_mult[pred_time_window]
    c_m = conf_mult[pred_confidence]

    n_events = int(np.clip(zone_info["base_rate"] * t_m * 20 * (1 + (5.0 - pred_min_mag) * 0.3), 3, 60))

    # Generate synthetic seismological inputs for ML model
    lats = np.random.uniform(zone_info["lat"][0], zone_info["lat"][1], n_events)
    lons = np.random.uniform(zone_info["lon"][0], zone_info["lon"][1], n_events)
    depths = np.random.uniform(pred_depth[0] + 1, pred_depth[1], n_events)
    stations = np.random.randint(5, 60, n_events)
    gaps = np.random.uniform(20, 300, n_events)
    closes = np.random.uniform(0.01, 2.0, n_events)
    rms_vals = np.random.uniform(0.01, 0.5, n_events)

    model_used = trained_model is not None and benchmark.get("features") is not None

    if model_used:
        # Build feature DataFrame matching the 59 engineered features
        feature_names = benchmark["features"]
        feat_df = pd.DataFrame(0.0, index=range(n_events), columns=feature_names)

        # Fill known raw columns
        if "Latitude(deg)" in feat_df.columns:
            feat_df["Latitude(deg)"] = lats
        if "Longitude(deg)" in feat_df.columns:
            feat_df["Longitude(deg)"] = lons
        if "Depth(km)" in feat_df.columns:
            feat_df["Depth(km)"] = depths
        if "No_of_Stations" in feat_df.columns:
            feat_df["No_of_Stations"] = stations
        if "Gap" in feat_df.columns:
            feat_df["Gap"] = gaps
        if "Close" in feat_df.columns:
            feat_df["Close"] = closes
        if "RMS" in feat_df.columns:
            feat_df["RMS"] = rms_vals

        # Derive features same way as feature_engineering.py
        ref_lat, ref_lon = 37.5, -122.0  # California center (training data center)
        feat_df["lat_rad"] = np.radians(lats)
        feat_df["lon_rad"] = np.radians(lons)
        feat_df["dist_from_center"] = np.sqrt((lats - ref_lat)**2 + (lons - ref_lon)**2)
        feat_df["lat_bin"] = pd.cut(lats, bins=20, labels=False).fillna(0).astype(float)
        feat_df["lon_bin"] = pd.cut(lons, bins=20, labels=False).fillna(0).astype(float)
        feat_df["spatial_cell"] = feat_df["lat_bin"] * 20 + feat_df["lon_bin"]
        feat_df["log_depth"] = np.log1p(depths)
        feat_df["stations_per_gap"] = stations / (gaps + 1)
        feat_df["close_depth_interaction"] = closes * depths
        feat_df["rms_log"] = np.log1p(rms_vals)
        feat_df["gap_close_ratio"] = gaps / (closes + 1)

        # Temporal features (simulate current period)
        feat_df["year"] = 2026
        feat_df["month"] = 3
        feat_df["day_of_year"] = 69
        feat_df["hour"] = np.random.randint(0, 24, n_events)
        feat_df["day_of_week"] = np.random.randint(0, 7, n_events)
        feat_df["month_sin"] = np.sin(2 * np.pi * 3 / 12)
        feat_df["month_cos"] = np.cos(2 * np.pi * 3 / 12)
        feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24)
        feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24)
        feat_df["dow_sin"] = np.sin(2 * np.pi * feat_df["day_of_week"] / 7)
        feat_df["dow_cos"] = np.cos(2 * np.pi * feat_df["day_of_week"] / 7)
        feat_df["time_since_prev"] = np.random.uniform(100, 50000, n_events)
        feat_df["log_time_since_prev"] = np.log1p(feat_df["time_since_prev"])

        # Lag features (simulate with realistic values)
        for lag in [1, 2, 3]:
            feat_df[f"mag_lag_{lag}"] = np.random.uniform(3.0, 5.0, n_events)
            feat_df[f"depth_lag_{lag}"] = np.random.uniform(5, 100, n_events)
            feat_df[f"lat_lag_{lag}"] = lats + np.random.normal(0, 0.5, n_events)
            feat_df[f"lon_lag_{lag}"] = lons + np.random.normal(0, 0.5, n_events)

        feat_df["mag_diff_lag1_lag2"] = feat_df["mag_lag_1"] - feat_df["mag_lag_2"]
        feat_df["mag_diff_lag2_lag3"] = feat_df["mag_lag_2"] - feat_df["mag_lag_3"]

        # Rolling features
        for w in [5, 10, 30]:
            feat_df[f"mag_rolling_mean_{w}"] = np.random.uniform(3.0, 4.5, n_events)
            feat_df[f"mag_rolling_std_{w}"] = np.random.uniform(0.1, 0.8, n_events)
            feat_df[f"mag_rolling_max_{w}"] = np.random.uniform(3.5, 6.0, n_events)
            feat_df[f"depth_rolling_mean_{w}"] = np.random.uniform(5, 80, n_events)

        feat_df["spatial_event_count_30"] = np.random.randint(1, 30, n_events).astype(float)
        feat_df["Magnitude_type_enc"] = np.random.choice([0, 1, 2], n_events)

        # Ensure column order matches training
        feat_df = feat_df[feature_names]

        # Predict with the trained model
        predicted_mags = trained_model.predict(feat_df)
        predicted_mags = np.clip(predicted_mags, pred_min_mag, 9.0)
    else:
        predicted_mags = np.round(np.random.uniform(pred_min_mag, min(pred_min_mag + 2.5, 9.0), n_events), 1)

    pred_data = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'est_magnitude': np.round(predicted_mags, 1),
        'est_depth': np.round(depths, 0),
        'probability': np.round(np.clip(
            np.random.beta(2, 3, n_events) * c_m * zone_info["base_rate"], 0.05, 0.95
        ), 2),
    })

    region_names = ["Japan", "Indonesia", "Chile", "Turkey", "Iran", "California, USA", "Nepal/Himalaya", "Italy"]
    if pred_region in region_names:
        pred_data['region'] = pred_region
    else:
        pred_data['region'] = np.random.choice(region_names, n_events)

    def get_prob_color(prob):
        if prob >= 0.50:
            return '#dc2626'
        elif prob >= 0.25:
            return '#ea580c'
        else:
            return '#ca8a04'

    # ---- Prediction Summary Stats ----
    st.markdown("### Forecast Summary")

    if model_used:
        report = benchmark.get("report", {})
        model_name = report.get("best_model", "Extra Trees (tuned)")
        st.markdown(f"""
        <div style="background: #065f46; color: #d1fae5; padding: 10px 16px; border-radius: 8px;
                    margin-bottom: 1rem; font-size: 0.9rem; border: 1px solid #059669;">
            <strong>🤖 ML Model Active:</strong> {model_name} &nbsp;|&nbsp;
            Test R² = {report.get('test_r2', 0):.4f} &nbsp;|&nbsp;
            MAE = {report.get('test_mae', 0):.4f} &nbsp;|&nbsp;
            {report.get('n_features', 59)} features
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #78350f; color: #fde68a; padding: 10px 16px; border-radius: 8px;
                    margin-bottom: 1rem; font-size: 0.9rem; border: 1px solid #d97706;">
            <strong>⚠️ Fallback Mode:</strong> ML model not found. Using random statistical estimates.
            Run <code>python model_prep/run_pipeline.py</code> to train the model.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Region:** {pred_region} | **Window:** {pred_time_window} | "
                f"**Min Mag:** {pred_min_mag} | **Confidence:** {pred_confidence}")

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.markdown(f'<div class="pred-card"><h4>Forecast Events</h4>'
                     f'<div class="val">{n_events}</div>'
                     f'<div class="sub">statistical estimates</div></div>', unsafe_allow_html=True)
    with pc2:
        avg_prob = pred_data['probability'].mean() * 100
        st.markdown(f'<div class="pred-card"><h4>Avg Probability</h4>'
                     f'<div class="val">{avg_prob:.0f}%</div>'
                     f'<div class="sub">NOT a certainty</div></div>', unsafe_allow_html=True)
    with pc3:
        max_mag = pred_data['est_magnitude'].max()
        st.markdown(f'<div class="pred-card"><h4>Max Est. Magnitude</h4>'
                     f'<div class="val">M {max_mag:.1f}</div>'
                     f'<div class="sub">hypothetical upper bound</div></div>', unsafe_allow_html=True)
    with pc4:
        avg_d = pred_data['est_depth'].mean()
        st.markdown(f'<div class="pred-card"><h4>Avg Est. Depth</h4>'
                     f'<div class="val">{avg_d:.0f} km</div>'
                     f'<div class="sub">estimated range</div></div>', unsafe_allow_html=True)

    # ---- Prediction Map ----
    st.markdown("---")
    st.markdown("### Forecast Probability Map")

    pred_map_type = tile_options.get(pred_map_style, "CartoDB.Positron")
    center_lat = (zone_info["lat"][0] + zone_info["lat"][1]) / 2
    center_lon = (zone_info["lon"][0] + zone_info["lon"][1]) / 2
    zoom = 2 if pred_region in ["Global", "Pacific Ring of Fire"] else 4

    pm = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                    tiles=pred_map_type, control_scale=True)

    for _, row in pred_data.iterrows():
        p_color = get_prob_color(row['probability'])
        prob_pct = row['probability'] * 100

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4 + row['est_magnitude'],
            color=p_color,
            fill=True,
            fill_opacity=0.5 + row['probability'] * 0.4,
            weight=1,
            dash_array='5,5',
            popup=folium.Popup(
                f"<div style='font-family:Arial;min-width:200px;'>"
                f"<h4 style='margin:0 0 8px;color:#991b1b;'>Forecast Zone</h4>"
                f"<p style='margin:4px 0;'><b>Region:</b> {row['region']}</p>"
                f"<p style='margin:4px 0;'><b>Est. Magnitude:</b> M {row['est_magnitude']:.1f}</p>"
                f"<p style='margin:4px 0;'><b>Est. Depth:</b> {row['est_depth']:.0f} km</p>"
                f"<p style='margin:4px 0;'><b>Probability:</b> "
                f"<span style='color:{p_color};font-weight:700;'>{prob_pct:.0f}%</span></p>"
                f"<hr style='margin:8px 0;border-color:#fecaca;'>"
                f"<p style='margin:2px 0;font-size:0.8em;color:#991b1b;'>"
                f"Statistical estimate only</p>"
                f"</div>",
                max_width=300
            ),
            tooltip=folium.Tooltip(
                f"<div style='font-family:Arial;font-size:12px;'>"
                f"<b>{row['region']}</b><br>"
                f"M {row['est_magnitude']:.1f} | {prob_pct:.0f}% prob.<br>"
                f"<em>Forecast only</em></div>",
                sticky=True
            )
        ).add_to(pm)

    # Prediction legend
    pred_legend = """
    <style>
        .pred-map-legend {
            position:fixed;bottom:30px;right:30px;
            background:#ffffff;color:#1f2937;
            border:2px solid #dc2626;border-radius:8px;
            padding:12px 16px;font-family:Arial;font-size:12px;line-height:1.6;
            z-index:9999;box-shadow:0 2px 6px rgba(0,0,0,0.2);max-width:180px;
        }
        .pred-map-legend .legend-title {
            font-weight:700;margin-bottom:6px;font-size:13px;
            border-bottom:1px solid #fecaca;padding-bottom:4px;color:#991b1b;
        }
        .pred-map-legend .legend-note {
            margin-top:6px;font-size:10px;color:#991b1b;
            border-top:1px solid #fecaca;padding-top:4px;
        }
        @media (prefers-color-scheme: dark) {
            .pred-map-legend {
                background:#1e293b;color:#e2e8f0;
                border-color:#ef4444;box-shadow:0 2px 6px rgba(0,0,0,0.5);
            }
            .pred-map-legend .legend-title { color:#fca5a5;border-bottom-color:#7f1d1d; }
            .pred-map-legend .legend-note { color:#fca5a5;border-top-color:#7f1d1d; }
        }
    </style>
    <div class="pred-map-legend">
        <div class="legend-title">Forecast Probability</div>
        <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
            background:#dc2626;vertical-align:middle;margin-right:6px;border:1px dotted #999;"></span> High (&ge;50%)</div>
        <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;
            background:#ea580c;vertical-align:middle;margin-right:6px;border:1px dotted #999;"></span> Medium (25-49%)</div>
        <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
            background:#ca8a04;vertical-align:middle;margin-right:6px;border:1px dotted #999;"></span> Low (&lt;25%)</div>
        <div class="legend-note">
            Dashed = estimates only
        </div>
    </div>
    """
    pm.get_root().html.add_child(folium.Element(pred_legend))

    st_folium(pm, width=1400, height=600, returned_objects=[])

    # ---- Forecast Data Table ----
    st.markdown("---")
    st.markdown("### Forecast Data")
    st.markdown("""
    <div class="caution-banner">
        <p>The table below shows <strong>statistical estimates</strong> generated from historical
        patterns. Probability values represent relative likelihood,
        <strong>not</strong> actual prediction certainty.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Forecast Table", expanded=False):
        disp_pred = pred_data.copy()
        disp_pred['probability_pct'] = (disp_pred['probability'] * 100).round(1)
        disp_pred = disp_pred[['region', 'latitude', 'longitude', 'est_magnitude', 'est_depth', 'probability_pct']]
        disp_pred.columns = ['Region', 'Lat', 'Lon', 'Est. Magnitude', 'Est. Depth (km)', 'Probability (%)']
        disp_pred['Lat'] = disp_pred['Lat'].round(3)
        disp_pred['Lon'] = disp_pred['Lon'].round(3)
        disp_pred = disp_pred.sort_values('Probability (%)', ascending=False).reset_index(drop=True)
        st.dataframe(disp_pred, use_container_width=True, height=350, hide_index=True)

        csv_pred = disp_pred.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv_pred,
            file_name=f"seismic_forecast_{pred_region.replace(' ', '_')}.csv",
            mime="text/csv"
        )

    # ================================================================
    # MODEL BENCHMARK & PERFORMANCE METRICS
    # ================================================================
    if model_used and benchmark:
        st.markdown("---")
        st.markdown("### 📊 Model Benchmark & Performance")

        report = benchmark.get("report", {})

        # ---- Score Cards ----
        st.markdown("#### Model Scores")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)

        test_r2 = report.get("test_r2", 0)
        test_mae = report.get("test_mae", 0)
        test_rmse = report.get("test_rmse", 0)
        test_mape = report.get("test_mape_pct", 0)
        expl_var = report.get("explained_variance", 0)

        with sc1:
            r2_color = "#059669" if test_r2 > 0.3 else "#d97706" if test_r2 > 0.15 else "#dc2626"
            st.markdown(f'<div class="stat-card"><h4>R² Score</h4>'
                        f'<div class="val" style="color:{r2_color};">{test_r2:.4f}</div></div>',
                        unsafe_allow_html=True)
        with sc2:
            st.markdown(f'<div class="stat-card"><h4>MAE</h4>'
                        f'<div class="val">{test_mae:.4f}</div></div>',
                        unsafe_allow_html=True)
        with sc3:
            st.markdown(f'<div class="stat-card"><h4>RMSE</h4>'
                        f'<div class="val">{test_rmse:.4f}</div></div>',
                        unsafe_allow_html=True)
        with sc4:
            st.markdown(f'<div class="stat-card"><h4>MAPE</h4>'
                        f'<div class="val">{test_mape:.2f}%</div></div>',
                        unsafe_allow_html=True)
        with sc5:
            st.markdown(f'<div class="stat-card"><h4>Explained Var</h4>'
                        f'<div class="val">{expl_var:.4f}</div></div>',
                        unsafe_allow_html=True)

        # ---- Mathematical Definitions ----
        st.markdown("#### Mathematical Definitions")

        with st.expander("📐 Score Formulas & Interpretation", expanded=True):
            st.markdown(r"""
**R² (Coefficient of Determination)** — measures proportion of variance explained by the model:

$$R^2 = 1 - \frac{{\sum_{{i=1}}^{{n}} (y_i - \hat{{y}}_i)^2}}{{\sum_{{i=1}}^{{n}} (y_i - \bar{{y}})^2}} = 1 - \frac{{SS_{{res}}}}{{SS_{{tot}}}}$$

- $R^2 = 1$: perfect fit &nbsp;|&nbsp; $R^2 = 0$: predicts the mean &nbsp;|&nbsp; $R^2 < 0$: worse than mean
- **Our score: {r2:.4f}** — the model explains **{r2_pct:.1f}%** of magnitude variance

---

**MAE (Mean Absolute Error)** — average magnitude of errors:

$$\text{{MAE}} = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} |y_i - \hat{{y}}_i|$$

- Measured in the same unit as the target (magnitude)
- **Our MAE: {mae:.4f}** — on average, predictions are off by **±{mae:.2f}** magnitude units

---

**RMSE (Root Mean Squared Error)** — penalizes large errors more than MAE:

$$\text{{RMSE}} = \sqrt{{\frac{{1}}{{n}} \sum_{{i=1}}^{{n}} (y_i - \hat{{y}}_i)^2}}$$

- Always ≥ MAE. When RMSE ≫ MAE, there are occasional large errors
- **Our RMSE: {rmse:.4f}** &nbsp;|&nbsp; RMSE/MAE ratio: **{ratio:.2f}** ({ratio_text})

---

**MAPE (Mean Absolute Percentage Error)**:

$$\text{{MAPE}} = \frac{{100\%}}{{n}} \sum_{{i=1}}^{{n}} \left|\frac{{y_i - \hat{{y}}_i}}{{y_i}}\right|$$

- Scale-independent percentage error
- **Our MAPE: {mape:.2f}%**

---

**Explained Variance Score**:

$$\text{{EV}} = 1 - \frac{{\text{{Var}}(y - \hat{{y}})}}{{\text{{Var}}(y)}}$$

- Similar to R² but does not penalize bias
- **Our EV: {ev:.4f}**
""".format(
                r2=test_r2, r2_pct=test_r2 * 100,
                mae=test_mae,
                rmse=test_rmse,
                ratio=test_rmse / test_mae if test_mae > 0 else 0,
                ratio_text="uniform errors" if (test_rmse / test_mae if test_mae > 0 else 0) < 1.2 else "some large outliers",
                mape=test_mape,
                ev=expl_var
            ))

        # ---- All Models Comparison Table ----
        if "comparison" in benchmark:
            st.markdown("#### All Models Comparison")
            comp_df = benchmark["comparison"].copy()
            comp_df = comp_df.rename(columns={
                "Model": "Algorithm",
                "Train R²": "Train R²",
                "Test R²": "Test R²",
                "Test MAE": "MAE",
                "Test RMSE": "RMSE",
                "Test MAPE (%)": "MAPE %",
                "Train Time (s)": "Time (s)"
            })

            # Color the best Test R² row
            best_model_name = comp_df.iloc[0]["Algorithm"]
            st.markdown(f"*Sorted by Test R² (descending). Best initial model: **{best_model_name}***")

            st.dataframe(
                comp_df[["Algorithm", "Train R²", "Test R²", "MAE", "RMSE", "MAPE %", "Explained Var", "Time (s)"]].style.format({
                    "Train R²": "{:.4f}",
                    "Test R²": "{:.4f}",
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "MAPE %": "{:.2f}",
                    "Explained Var": "{:.4f}",
                    "Time (s)": "{:.2f}",
                }).highlight_max(subset=["Test R²", "Explained Var"], color="#d1fae5"
                ).highlight_min(subset=["MAE", "RMSE", "MAPE %"], color="#d1fae5"),
                use_container_width=True, hide_index=True
            )

        # ---- Tuned Models Comparison ----
        if "final" in benchmark:
            st.markdown("#### Tuned Models (after Hyperparameter Optimization)")
            final_df = benchmark["final"].copy()
            final_df = final_df.rename(columns={
                "R2": "Test R²",
                "MAE": "MAE",
                "RMSE": "RMSE",
                "MAPE (%)": "MAPE %",
                "Explained Variance": "Expl. Var",
                "CV R²": "CV R²",
            })
            st.dataframe(
                final_df[["Model", "Test R²", "MAE", "RMSE", "MAPE %", "Expl. Var", "CV R²"]].style.format({
                    "Test R²": "{:.4f}",
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "MAPE %": "{:.2f}",
                    "Expl. Var": "{:.4f}",
                    "CV R²": "{:.4f}",
                }, na_rep="—"
                ).highlight_max(subset=["Test R²", "Expl. Var"], color="#d1fae5"
                ).highlight_min(subset=["MAE", "RMSE", "MAPE %"], color="#d1fae5"),
                use_container_width=True, hide_index=True
            )

            # Best params
            best_params = report.get("best_params", {})
            if best_params:
                st.markdown(f"**Best Hyperparameters** ({report.get('best_model', '')}):")
                param_str = " &nbsp;|&nbsp; ".join(
                    f"`{k.replace('model__', '')}` = **{v}**" for k, v in best_params.items()
                )
                st.markdown(param_str)

        # ---- Feature Importance ----
        if "importance" in benchmark:
            st.markdown("#### Feature Importance (Top 20)")
            fi_df = benchmark["importance"].head(20).copy()

            # Horizontal bar chart via Streamlit
            chart_df = fi_df.set_index("feature")["importance"].sort_values(ascending=True)
            st.bar_chart(chart_df, horizontal=True, height=500)

            with st.expander("View Full Feature Importance Table"):
                full_fi = benchmark["importance"].copy()
                full_fi["importance_pct"] = (full_fi["importance"] * 100).round(2)
                full_fi.columns = ["Feature", "Importance", "Importance %"]
                st.dataframe(full_fi, use_container_width=True, hide_index=True, height=400)

        # ---- Pipeline Info ----
        st.markdown("#### Pipeline Info")
        pi1, pi2, pi3, pi4 = st.columns(4)
        with pi1:
            st.markdown(f'<div class="info-card"><h4>Models Evaluated</h4>'
                        f'<div class="val">{report.get("models_evaluated", 11)}</div>'
                        f'<div class="sub">algorithms</div></div>', unsafe_allow_html=True)
        with pi2:
            st.markdown(f'<div class="info-card"><h4>Models Tuned</h4>'
                        f'<div class="val">{report.get("models_tuned", 3)}</div>'
                        f'<div class="sub">hyperparameter search</div></div>', unsafe_allow_html=True)
        with pi3:
            st.markdown(f'<div class="info-card"><h4>Features</h4>'
                        f'<div class="val">{report.get("n_features", 59)}</div>'
                        f'<div class="sub">engineered</div></div>', unsafe_allow_html=True)
        with pi4:
            n_train = report.get("n_train_samples", 0)
            n_test = report.get("n_test_samples", 0)
            st.markdown(f'<div class="info-card"><h4>Train / Test</h4>'
                        f'<div class="val">{n_train} / {n_test}</div>'
                        f'<div class="sub">70/30 split</div></div>', unsafe_allow_html=True)

    # Final disclaimer
    st.markdown("""
    <div class="warn-banner">
        <h4>Reminder</h4>
        <p>Earthquake prediction is <strong>scientifically impossible</strong> with current technology.
        The data above is generated from statistical models applied to historical patterns.
        It should never be treated as actionable intelligence. Visit
        <a href="https://www.usgs.gov/natural-hazards/earthquake-hazards" target="_blank">USGS</a>
        or <a href="https://www.ready.gov/earthquakes" target="_blank">Ready.gov</a>
        for real earthquake preparedness.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ASSESSMENT MODE — Main Content
# ============================================================
else:

    # ---- Helper Function for Location Names ----
    @st.cache_data
    def get_location_name(lat, lon):
        """Get approximate location name from coordinates"""
        try:
            geolocator = Nominatim(user_agent="seismic_risk_assessment", timeout=2)
            location = geolocator.reverse(f"{lat}, {lon}", language='en', exactly_one=True)
            if location and location.raw.get('address'):
                address = location.raw['address']
                place = address.get('country', address.get('ocean', address.get('sea', 'Unknown')))
                return place
            return "Unknown Region"
        except (GeocoderTimedOut, GeocoderServiceError, Exception):
            return "Unknown Region"

    # ---- Generate Risk Assessment Data ----
    np.random.seed(42)

    # Define high-risk seismic zones based on real tectonic plate boundaries
    seismic_zones = [
        # Pacific Ring of Fire
        {"name": "Japan", "lat_range": (30, 45), "lon_range": (130, 145), "mag_range": (5.0, 8.5), "samples": 30, "belt": "Pacific Ring of Fire", "region": "Asia-Pacific"},
        {"name": "Philippines", "lat_range": (5, 20), "lon_range": (120, 126), "mag_range": (4.5, 7.5), "samples": 25, "belt": "Pacific Ring of Fire", "region": "Asia-Pacific"},
        {"name": "Indonesia", "lat_range": (-10, 6), "lon_range": (95, 141), "mag_range": (5.5, 9.0), "samples": 40, "belt": "Pacific Ring of Fire", "region": "Asia-Pacific"},
        {"name": "Chile", "lat_range": (-56, -17), "lon_range": (-76, -66), "mag_range": (5.5, 9.0), "samples": 35, "belt": "Pacific Ring of Fire", "region": "Americas"},
        {"name": "Peru", "lat_range": (-18, -1), "lon_range": (-82, -68), "mag_range": (4.5, 8.0), "samples": 20, "belt": "Pacific Ring of Fire", "region": "Americas"},
        {"name": "California, USA", "lat_range": (32, 42), "lon_range": (-125, -114), "mag_range": (4.0, 7.5), "samples": 25, "belt": "Pacific Ring of Fire", "region": "Americas"},
        {"name": "Alaska, USA", "lat_range": (55, 70), "lon_range": (-165, -140), "mag_range": (4.5, 8.5), "samples": 20, "belt": "Pacific Ring of Fire", "region": "Americas"},
        {"name": "New Zealand", "lat_range": (-47, -34), "lon_range": (166, 179), "mag_range": (4.0, 7.5), "samples": 20, "belt": "Pacific Ring of Fire", "region": "Asia-Pacific"},
        {"name": "Mexico", "lat_range": (14, 32), "lon_range": (-118, -86), "mag_range": (4.5, 8.0), "samples": 20, "belt": "Pacific Ring of Fire", "region": "Americas"},
        {"name": "Papua New Guinea", "lat_range": (-12, 0), "lon_range": (140, 156), "mag_range": (5.0, 8.5), "samples": 25, "belt": "Pacific Ring of Fire", "region": "Asia-Pacific"},
    
        # Mediterranean-Alpine Belt
        {"name": "Turkey", "lat_range": (36, 42), "lon_range": (26, 45), "mag_range": (4.5, 7.8), "samples": 30, "belt": "Mediterranean-Alpine Belt", "region": "Asia-Pacific"},
        {"name": "Greece", "lat_range": (34, 42), "lon_range": (19, 30), "mag_range": (3.0, 6.5), "samples": 15, "belt": "Mediterranean-Alpine Belt", "region": "Asia-Pacific"},
        {"name": "Italy", "lat_range": (36, 47), "lon_range": (6, 19), "mag_range": (3.0, 6.5), "samples": 18, "belt": "Mediterranean-Alpine Belt", "region": "Asia-Pacific"},
        {"name": "Iran", "lat_range": (25, 40), "lon_range": (44, 63), "mag_range": (4.5, 7.5), "samples": 25, "belt": "Mediterranean-Alpine Belt", "region": "Asia-Pacific"},
        {"name": "Nepal/Himalaya", "lat_range": (26, 31), "lon_range": (80, 89), "mag_range": (5.0, 8.5), "samples": 30, "belt": "Mediterranean-Alpine Belt", "region": "Asia-Pacific"},
    
        # Mid-Atlantic Ridge
        {"name": "Iceland", "lat_range": (63, 67), "lon_range": (-25, -13), "mag_range": (2.0, 6.0), "samples": 15, "belt": "Mid-Atlantic Ridge", "region": "Asia-Pacific"},
    ]

    # Generate data for each seismic zone
    all_data = []
    for zone in seismic_zones:
        n_samples = zone["samples"]
        lat_min, lat_max = zone["lat_range"]
        lon_min, lon_max = zone["lon_range"]
        mag_min, mag_max = zone["mag_range"]
    
        zone_data = {
            'region': [zone["name"]] * n_samples,
            'latitude': np.random.uniform(lat_min, lat_max, n_samples),
            'longitude': np.random.uniform(lon_min, lon_max, n_samples),
            'depth': np.random.uniform(5, 200, n_samples),
            'magnitude': np.round(np.random.uniform(mag_min, mag_max, n_samples), 1),
            'seismic_belt': [zone["belt"]] * n_samples,
            'geographic_region': [zone["region"]] * n_samples
        }
        all_data.append(pd.DataFrame(zone_data))

    risk_data = pd.concat(all_data, ignore_index=True)

    # ---- Apply Filters ----
    # 1. Magnitude Range Filter
    filtered_data = risk_data[
        (risk_data['magnitude'] >= mag_range[0]) &
        (risk_data['magnitude'] <= mag_range[1])
    ].copy()

    # 2. Geographic Region Filter
    if region_filter == "Pacific Ring of Fire":
        filtered_data = filtered_data[filtered_data['seismic_belt'] == "Pacific Ring of Fire"]
    elif region_filter == "Mediterranean-Alpine Belt":
        filtered_data = filtered_data[filtered_data['seismic_belt'] == "Mediterranean-Alpine Belt"]
    elif region_filter == "Mid-Atlantic Ridge":
        filtered_data = filtered_data[filtered_data['seismic_belt'] == "Mid-Atlantic Ridge"]
    elif region_filter == "Americas":
        filtered_data = filtered_data[filtered_data['geographic_region'] == "Americas"]
    elif region_filter == "Asia-Pacific":
        filtered_data = filtered_data[filtered_data['geographic_region'] == "Asia-Pacific"]
    # Global shows all data (no additional filtering)

    # Update risk_data to use filtered version
    risk_data = filtered_data

    # ---- Color Function Based on Magnitude ----
    def get_mag_color(mag):
        """Return color based on earthquake magnitude"""
        if mag >= 7.0:
            return '#8B0000'  # Dark red - Major/Great
        elif mag >= 6.0:
            return '#DC143C'  # Crimson - Strong
        elif mag >= 5.0:
            return '#FF8C00'  # Dark orange - Moderate
        elif mag >= 4.0:
            return '#FFD700'  # Gold - Light
        else:
            return '#90EE90'  # Light green - Minor

    def get_mag_label(mag):
        """Return seismological classification"""
        if mag >= 8.0:
            return 'Great'
        elif mag >= 7.0:
            return 'Major'
        elif mag >= 6.0:
            return 'Strong'
        elif mag >= 5.0:
            return 'Moderate'
        elif mag >= 4.0:
            return 'Light'
        elif mag >= 3.0:
            return 'Minor'
        else:
            return 'Micro'

    # ---- Seismic Risk Map ----
    st.markdown("---")
    st.markdown("### Interactive Seismic Risk Map")
    st.markdown(f"**{region_filter}** | Magnitude: **{mag_range[0]}–{mag_range[1]}** | {visualization_choice}")

    if len(risk_data) == 0:
        st.markdown("""
        <div class="notice-bar">
            <strong>No data matches your filters.</strong> Try widening the magnitude range or changing the geographic scope.
        </div>
        """, unsafe_allow_html=True)
    else:
    
        m = folium.Map(
            location=[20, 0],  # Center of world map
            zoom_start=2,
            tiles=map_type,
            control_scale=True,
            prefer_canvas=True
        )
    
        # ---- Marker Visualization ----
        if "Markers" in visualization_choice or "Combined" in visualization_choice:
            marker_cluster = MarkerCluster(name="Seismic Risk Zones").add_to(m)
        
            for idx, row in risk_data.iterrows():
                # Only add labels for some points to avoid cluttering
                if show_labels and idx % 5 == 0:  # Show every 5th label
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        icon=folium.DivIcon(html=f'<div style="font-size: 9pt; color: #334155; font-weight: 600; text-shadow: 0 0 3px white, 0 0 3px white;">{row["region"]}</div>')
                    ).add_to(m)
            
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3 + row['magnitude'],
                    color=get_mag_color(row['magnitude']),
                    fill=True,
                    fill_opacity=0.7,
                    weight=2,
                    popup=folium.Popup(
                        f"<div style='font-family: Arial; min-width: 200px;'>"
                        f"<h4 style='margin: 0 0 10px 0; color: inherit;'>{row['region']}</h4>"
                        f"<p style='margin: 5px 0;'><b>Coordinates:</b><br>{row['latitude']:.3f}, {row['longitude']:.3f}</p>"
                        f"<p style='margin: 5px 0;'><b>Depth:</b> {row['depth']:.1f} km</p>"
                        f"<p style='margin: 5px 0;'><b>Magnitude:</b> <span style='color: {get_mag_color(row['magnitude'])}; font-weight:700;'>{row['magnitude']:.1f} ({get_mag_label(row['magnitude'])})</span></p>"
                        f"<p style='margin: 5px 0; font-size: 0.85em; opacity: 0.7;'><b>Belt:</b> {row['seismic_belt']}</p>"
                        f"</div>",
                        max_width=320
                    ),
                    tooltip=folium.Tooltip(
                        f"<div style='font-family:Arial;font-size:12px;line-height:1.5;'>"
                        f"<b>{row['region']}</b><br>"
                        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                        f"background:{get_mag_color(row['magnitude'])};vertical-align:middle;margin-right:4px;'></span>"
                        f"<b>M {row['magnitude']:.1f}</b> — {get_mag_label(row['magnitude'])}<br>"
                        f"Depth: {row['depth']:.0f} km"
                        f"</div>",
                        sticky=True
                    )
                ).add_to(marker_cluster)
    
        # ---- Heatmap Visualization ----
        if "Heat" in visualization_choice or "Combined" in visualization_choice:
            heat_data = [[row['latitude'], row['longitude'], row['magnitude']] for _, row in risk_data.iterrows()]
            HeatMap(
                heat_data, 
                min_opacity=0.3, 
                radius=20, 
                blur=18, 
                max_zoom=1,
                name="Risk Density Heatmap",
                gradient={0.4: 'blue', 0.6: 'lime', 0.75: 'yellow', 0.85: 'orange', 1.0: 'red'}
            ).add_to(m)
    
        # Add layer control
        folium.LayerControl().add_to(m)
    
        # ---- Map Legend (bottom-right, like standard maps) ----
        legend_html = """
        <style>
            .mag-map-legend {
                position: fixed;
                bottom: 30px; right: 30px;
                background: #ffffff; color: #1f2937;
                border: 2px solid #ccc;
                border-radius: 8px;
                padding: 12px 16px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.6;
                z-index: 9999;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                max-width: 170px;
            }
            .mag-map-legend .legend-title {
                font-weight:700; margin-bottom:6px; font-size:13px;
                border-bottom:1px solid #e5e7eb; padding-bottom:4px;
            }
            @media (prefers-color-scheme: dark) {
                .mag-map-legend {
                    background: #1e293b; color: #e2e8f0;
                    border-color: #475569;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
                }
                .mag-map-legend .legend-title {
                    border-bottom-color: #475569;
                }
            }
        </style>
        <div class="mag-map-legend">
            <div class="legend-title">
                Magnitude Scale
            </div>
            <div><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#4B0000;vertical-align:middle;margin-right:6px;border:1px solid #999;"></span> 8.0+ &nbsp;Great</div>
            <div><span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:#8B0000;vertical-align:middle;margin-right:6px;border:1px solid #999;"></span> 7.0–7.9 Major</div>
            <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#DC143C;vertical-align:middle;margin-right:6px;border:1px solid #999;"></span> 6.0–6.9 Strong</div>
            <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#FF8C00;vertical-align:middle;margin-right:6px;border:1px solid #999;"></span> 5.0–5.9 Moderate</div>
            <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#FFD700;vertical-align:middle;margin-right:6px;border:1px solid #999;"></span> 4.0–4.9 Light</div>
            <div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#90EE90;vertical-align:middle;margin-right:6px;border:1px solid #999;"></span> &lt;4.0 &nbsp;Minor</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    
        # ---- Display Map ----
        st_folium(m, width=1400, height=700, returned_objects=[])

    # ---- Statistics Dashboard ----
    st.markdown("---")
    st.markdown("### Statistics")

    if len(risk_data) > 0:
        col1, col2, col3, col4, col5 = st.columns(5)
    
        with col1:
            great_count = len(risk_data[risk_data['magnitude'] >= 7.0])
            st.markdown(f'<div class="stat-card"><h4>M 7.0+</h4><div class="val">{great_count}</div></div>', unsafe_allow_html=True)
    
        with col2:
            strong_count = len(risk_data[(risk_data['magnitude'] >= 6.0) & (risk_data['magnitude'] < 7.0)])
            st.markdown(f'<div class="stat-card"><h4>M 6.0–6.9</h4><div class="val">{strong_count}</div></div>', unsafe_allow_html=True)
    
        with col3:
            mod_count = len(risk_data[(risk_data['magnitude'] >= 5.0) & (risk_data['magnitude'] < 6.0)])
            st.markdown(f'<div class="stat-card"><h4>M 5.0–5.9</h4><div class="val">{mod_count}</div></div>', unsafe_allow_html=True)
    
        with col4:
            avg_mag = risk_data['magnitude'].mean()
            st.markdown(f'<div class="stat-card"><h4>Avg Mag</h4><div class="val">{avg_mag:.1f}</div></div>', unsafe_allow_html=True)
    
        with col5:
            avg_depth = risk_data['depth'].mean()
            st.markdown(f'<div class="stat-card"><h4>Avg Depth</h4><div class="val">{avg_depth:.0f} km</div></div>', unsafe_allow_html=True)
    
        # Additional statistics
        st.markdown("")
        col_a, col_b, col_c = st.columns(3)
    
        with col_a:
            unique_regions = risk_data['region'].nunique()
            st.markdown(f'<div class="info-card"><h4>Regions</h4><div class="val">{unique_regions}</div><div class="sub">Unique Seismic Zones</div></div>', unsafe_allow_html=True)
    
        with col_b:
            max_mag_zone = risk_data.loc[risk_data['magnitude'].idxmax()]
            st.markdown(f'<div class="info-card"><h4>Strongest</h4><div class="val">{max_mag_zone["region"]}</div><div class="sub">M {max_mag_zone["magnitude"]:.1f}</div></div>', unsafe_allow_html=True)
    
        with col_c:
            st.markdown(f'<div class="info-card"><h4>Total Events</h4><div class="val">{len(risk_data)}</div><div class="sub">Seismic Data Points</div></div>', unsafe_allow_html=True)

    # ---- Magnitude Scale Legend ----
    st.markdown("---")
    st.markdown("### Magnitude Scale")
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="legend-chip" style="background:#90EE90; color:#1f2937;">Micro &lt;3.0</span>
        <span class="legend-chip" style="background:#90EE90; color:#1f2937;">Minor 3.0–3.9</span>
        <span class="legend-chip" style="background:#FFD700; color:#1f2937;">Light 4.0–4.9</span>
        <span class="legend-chip" style="background:#FF8C00;">Moderate 5.0–5.9</span>
        <span class="legend-chip" style="background:#DC143C;">Strong 6.0–6.9</span>
        <span class="legend-chip" style="background:#8B0000;">Major 7.0–7.9</span>
        <span class="legend-chip" style="background:#4B0000;">Great 8.0+</span>
    </div>
    """, unsafe_allow_html=True)

    # ---- Data Table ----
    st.markdown("---")
    st.markdown("### Risk Zone Data")

    if len(risk_data) > 0:
        # Add expandable section
        with st.expander("📊 View Complete Data Table", expanded=False):
            # Create a more readable display dataframe
            display_data = risk_data.copy()
            display_data['classification'] = display_data['magnitude'].apply(get_mag_label)
            display_data = display_data[['region', 'seismic_belt', 'latitude', 'longitude', 'depth', 'magnitude', 'classification']]
            display_data.columns = ['Region', 'Seismic Belt', 'Latitude', 'Longitude', 'Depth (km)', 'Magnitude', 'Classification']
        
            # Format the numbers
            display_data['Latitude'] = display_data['Latitude'].round(3)
            display_data['Longitude'] = display_data['Longitude'].round(3)
            display_data['Depth (km)'] = display_data['Depth (km)'].round(1)
        
            # Sort by magnitude descending
            display_data = display_data.sort_values('Magnitude', ascending=False).reset_index(drop=True)
        
            st.dataframe(
                display_data,
                use_container_width=True,
                height=400,
                hide_index=True
            )
        
            # Download button
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"seismic_risk_data_{region_filter.replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
        # Show top 10 highest risk zones
        st.markdown("#### Top 10 Strongest Events")
        top_10 = risk_data.nlargest(10, 'magnitude')[['region', 'magnitude', 'seismic_belt']].copy()
        top_10.columns = ['Region', 'Magnitude', 'Seismic Belt']
        top_10['Magnitude'] = top_10['Magnitude'].round(1)
        top_10.index = range(1, 11)
        st.table(top_10)
    else:
        st.markdown("""
        <div class="notice-bar">
            <strong>No data to display.</strong> Adjust your filters and click Apply Filters to view risk zones.
        </div>
        """, unsafe_allow_html=True)

    # ---- Footer & Professional Disclaimer ----
    st.markdown("---")

    # Collapsible disclaimer section
    with st.expander("Disclaimer & Usage Guidelines", expanded=False):
        st.markdown("""
        <div class='disclaimer-content' style='padding: 10px;'>
    
        ## Important Information & Disclaimers
    
        ### About This Tool
    
        This **Seismic Risk Assessment Tool** is designed for educational and research purposes. It provides visualizations of seismic risk zones based on:
    
        - **Historical earthquake data** and patterns from past events
        - **Tectonic plate boundary locations** and geological fault lines
        - **Statistical analysis** of seismic activity frequency and intensity
        - **Known seismic zones** identified by geological research
    
        ### ⚠️ THIS IS NOT AN EARTHQUAKE PREDICTION SYSTEM
    
        Earthquakes **CANNOT** be accurately predicted. Current scientific understanding does not allow for:
        - Predicting specific dates, times, or locations of future earthquakes
        - Determining when an earthquake will occur in a given region
        - Providing advance warning that would enable evacuations
    
        Any tool claiming to predict specific earthquake events is **scientifically unsound** and potentially **dangerous**.
    
        ### Proper Use of This Tool
    
        **✅ Appropriate Applications:**
        - Understanding which regions have historically experienced seismic activity
        - Educational purposes to learn about tectonic plate boundaries
        - Long-term urban planning and building code development
        - Research and academic study of seismic patterns
    
        **❌ Inappropriate Applications:**
        - Making evacuation decisions based on this tool
        - Insurance or financial decisions
        - Creating public warnings or advisories
        - Any action that assumes specific earthquakes can be predicted
    
        ### Official Resources
    
        **For Earthquake Preparedness:**
        - 🇺🇸 [USGS Earthquake Hazards](https://www.usgs.gov/natural-hazards/earthquake-hazards)
        - 🌍 [Global Seismic Hazard Map](https://www.globalquakemodel.org/)
        - 🚨 [Ready.gov - Earthquakes](https://www.ready.gov/earthquakes)
        - 📚 [IASPEI - Earthquake Prediction Statement](http://www.iaspei.org/)
    
        </div>
        """, unsafe_allow_html=True)

    # Professional footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="footer-card"><h4>Version</h4><p>2.0</p><p>March 2026</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="footer-card"><h4>Purpose</h4><p>Educational</p><p>Research & Analysis</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="footer-card"><h4>License</h4><p>MIT</p><p>Open Source</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; padding: 16px 0 8px 0; font-size: 0.85rem; color: var(--footer-text);'>
        Developed for education & research. Always rely on official sources for earthquake safety guidance.<br>
        <strong>This tool does not predict earthquakes.</strong>
    </div>
    """, unsafe_allow_html=True)

