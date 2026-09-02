"""
🏎️ F1 Race Predictor — High Performance AI Telemetry Dashboard
Powered by Gradient Boosting, Jolpica-F1 API & Historical Ergast Data (1950-2026)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 AI Race Predictor | Live Telemetry & Predictions",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Path setup
base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_dir, "data") if os.path.exists(os.path.join(base_dir, "data")) else os.path.join(base_dir, "..", "data")
MODEL_DIR = os.path.join(base_dir, "models") if os.path.exists(os.path.join(base_dir, "models")) else os.path.join(base_dir, "..", "models")


@st.cache_resource
def load_models():
    podium_model = joblib.load(os.path.join(MODEL_DIR, "podium_model.pkl"))
    winner_model = joblib.load(os.path.join(MODEL_DIR, "winner_model.pkl"))
    position_model = joblib.load(os.path.join(MODEL_DIR, "position_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return podium_model, winner_model, position_model, feature_cols


@st.cache_data
def load_reference_data():
    driver_stats = pd.read_csv(os.path.join(DATA_DIR, "latest_driver_stats.csv"))
    team_stats = pd.read_csv(os.path.join(DATA_DIR, "latest_team_stats.csv"))
    circuit_stats = pd.read_csv(os.path.join(DATA_DIR, "latest_circuit_stats.csv"))
    races = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
    return driver_stats, team_stats, circuit_stats, races


podium_model, winner_model, position_model, feature_cols = load_models()
driver_stats, team_stats, circuit_stats, races = load_reference_data()

# ---------------------------------------------------------------------------
# Driver / Team / Circuit Formatting Dictionaries
# ---------------------------------------------------------------------------
DRIVER_NAMES = {
    "max_verstappen": ("Max Verstappen", "🇳🇱", "#3671C6"),
    "hamilton": ("Lewis Hamilton", "🇬🇧", "#27F4D2"),
    "leclerc": ("Charles Leclerc", "🇲🇨", "#E80020"),
    "norris": ("Lando Norris", "🇬🇧", "#FF8000"),
    "sainz": ("Carlos Sainz", "🇪🇸", "#E80020"),
    "perez": ("Sergio Perez", "🇲🇽", "#3671C6"),
    "russell": ("George Russell", "🇬🇧", "#27F4D2"),
    "piastri": ("Oscar Piastri", "🇦🇺", "#FF8000"),
    "alonso": ("Fernando Alonso", "🇪🇸", "#229971"),
    "stroll": ("Lance Stroll", "🇨🇦", "#229971"),
    "gasly": ("Pierre Gasly", "🇫🇷", "#0093CC"),
    "ocon": ("Esteban Ocon", "🇫🇷", "#0093CC"),
    "albon": ("Alexander Albon", "🇹🇭", "#64C4FF"),
    "tsunoda": ("Yuki Tsunoda", "🇯🇵", "#6692FF"),
    "ricciardo": ("Daniel Ricciardo", "🇦🇺", "#6692FF"),
    "hulkenberg": ("Nico Hülkenberg", "🇩🇪", "#B6BABD"),
    "magnussen": ("Kevin Magnussen", "🇩🇰", "#B6BABD"),
    "bottas": ("Valtteri Bottas", "🇫🇮", "#52E252"),
    "zhou": ("Zhou Guanyu", "🇨🇳", "#52E252"),
    "sargeant": ("Logan Sargeant", "🇺🇸", "#64C4FF"),
    "vettel": ("Sebastian Vettel", "🇩🇪", "#229971"),
    "raikkonen": ("Kimi Räikkönen", "🇫🇮", "#52E252"),
    "rosberg": ("Nico Rosberg", "🇩🇪", "#27F4D2"),
    "michael_schumacher": ("Michael Schumacher", "🇩🇪", "#E80020"),
    "button": ("Jenson Button", "🇬🇧", "#FF8000"),
    "massa": ("Felipe Massa", "🇧🇷", "#E80020"),
    "webber": ("Mark Webber", "🇦🇺", "#3671C6"),
    "prost": ("Alain Prost", "🇫🇷", "#E80020"),
    "senna": ("Ayrton Senna", "🇧🇷", "#FF8000"),
}

CONSTRUCTOR_NAMES = {
    "red_bull": "Red Bull Racing 🐂",
    "ferrari": "Scuderia Ferrari 🐎",
    "mercedes": "Mercedes-AMG Petronas ⚡",
    "mclaren": "McLaren F1 Team 🏎️",
    "aston_martin": "Aston Martin Aramco 💚",
    "alpine": "Alpine F1 Team 💙",
    "williams": "Williams Racing 🔷",
    "rb": "Visa Cash App RB 🔵",
    "toro_rosso": "Toro Rosso 🔵",
    "alpha_tauri": "AlphaTauri 🔵",
    "sauber": "Stake F1 Team Sauber 💚",
    "alfa": "Alfa Romeo F1 Team 💚",
    "haas": "MoneyGram Haas F1 Team ⚪",
}

CIRCUIT_NAMES = {
    "monaco": "🇲🇨 Circuit de Monaco (Monte Carlo)",
    "silverstone": "🇬🇧 Silverstone Circuit (UK)",
    "monza": "🇮🇹 Autodromo Nazionale Monza (Italy)",
    "spa": "🇧🇪 Circuit de Spa-Francorchamps (Belgium)",
    "red_bull_ring": "🇦🇹 Red Bull Ring (Austria)",
    "suzuka": "🇯🇵 Suzuka International Racing Course (Japan)",
    "americas": "🇺🇸 Circuit of the Americas (Austin)",
    "interlagos": "🇧🇷 Autódromo José Carlos Pace (Brazil)",
    "shanghai": "🇨🇳 Shanghai International Circuit (China)",
    "catalunya": "🇪🇸 Circuit de Barcelona-Catalunya (Spain)",
    "zandvoort": "🇳🇱 Circuit Zandvoort (Netherlands)",
    "baku": "🇦🇿 Baku City Circuit (Azerbaijan)",
    "marina_bay": "🇸🇬 Marina Bay Street Circuit (Singapore)",
    "yas_marina": "🇦🇪 Yas Marina Circuit (Abu Dhabi)",
    "hungaroring": "🇭🇺 Hungaroring (Hungary)",
    "bahrain": "🇧🇭 Bahrain International Circuit",
    "jeddah": "🇸🇦 Jeddah Corniche Circuit (Saudi Arabia)",
    "albert_park": "🇦🇺 Albert Park Circuit (Melbourne)",
    "miami": "🇺🇸 Miami International Autodrome",
    "losail": "🇶🇦 Lusail International Circuit (Qatar)",
    "vegas": "🇺🇸 Las Vegas Strip Circuit",
    "canadian": "🇨🇦 Circuit Gilles Villeneuve (Canada)",
    "imola": "🇮🇹 Autodromo Enzo e Dino Ferrari (Imola)",
}


def get_driver_label(driver_id):
    if driver_id in DRIVER_NAMES:
        name, flag, _ = DRIVER_NAMES[driver_id]
        return f"{flag} {name}"
    return driver_id.replace("_", " ").title()


def get_constructor_label(const_id):
    if const_id in CONSTRUCTOR_NAMES:
        return CONSTRUCTOR_NAMES[const_id]
    return str(const_id).replace("_", " ").title()


def get_circuit_label(circuit_id):
    if circuit_id in CIRCUIT_NAMES:
        return CIRCUIT_NAMES[circuit_id]
    return f"🏁 {circuit_id.replace('_', ' ').title()}"

# Add label mappings for selectboxes
driver_stats["display_name"] = driver_stats["driverId"].apply(get_driver_label)
circuit_stats_list = sorted(circuit_stats["circuitId"].dropna().unique().tolist())
circuit_options_map = {c: get_circuit_label(c) for c in circuit_stats_list}

# Inject Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&family=Titillium+Web:ital,wght@0,400;0,600;0,700;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #1a1e2b 0%, #0d0f17 75%, #08090e 100%);
    color: #e2e8f0;
}

.f1-title {
    font-family: 'Titillium Web', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #FF1801 0%, #FF5252 50%, #FFD700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(255, 24, 1, 0.15);
    border: 1px solid rgba(255, 24, 1, 0.5);
    color: #FF5252;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #FF1801;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 10px #FF1801;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 24, 1, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(255, 24, 1, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 24, 1, 0); }
}

.f1-card {
    background: rgba(22, 27, 38, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.metric-box {
    background: linear-gradient(135deg, rgba(30, 35, 48, 0.8) 0%, rgba(20, 24, 33, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-box::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #FF1801, #FFD700);
}
.metric-val {
    font-size: 2.3rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 6px 0;
}
.metric-lbl {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94A3B8;
    font-weight: 700;
}

.podium-p1 {
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.25) 0%, rgba(20, 24, 33, 0.95) 100%);
    border: 1px solid rgba(255, 215, 0, 0.6);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}
.podium-p2 {
    background: linear-gradient(135deg, rgba(220, 220, 220, 0.25) 0%, rgba(20, 24, 33, 0.95) 100%);
    border: 1px solid rgba(220, 220, 220, 0.5);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}
.podium-p3 {
    background: linear-gradient(135deg, rgba(205, 127, 50, 0.25) 0%, rgba(20, 24, 33, 0.95) 100%);
    border: 1px solid rgba(205, 127, 50, 0.5);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}

.stButton>button {
    background: linear-gradient(90deg, #FF1801 0%, #D31000 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    padding: 14px 28px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(255, 24, 1, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(255, 24, 1, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Header & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
    st.markdown("### **AI Telemetry Control**")
    st.caption("Gradient Boosting Engine • 1950-2026 Dataset")
    st.divider()

    st.markdown("##### 🏎️ **Quick Driver Search**")
    sorted_drivers = sorted(driver_stats["driverId"].tolist(), key=lambda x: driver_stats[driver_stats["driverId"] == x]["season_points_so_far"].values[0], reverse=True)
    driver_map = {d: get_driver_label(d) for d in sorted_drivers}

    selected_driver_id = st.selectbox(
        "Choose F1 Driver",
        options=sorted_drivers,
        format_func=lambda x: driver_map[x],
        index=2 if "max_verstappen" in sorted_drivers else 0
    )

    driver_info = driver_stats[driver_stats["driverId"] == selected_driver_id].iloc[0]
    const_id = driver_info["constructorId"]

    st.markdown(f"**Team:** `{get_constructor_label(const_id)}`")
    st.markdown(f"**Season Points:** `{driver_info['season_points_so_far']:.1f} pts`")
    st.markdown(f"**DNF Rate:** `{driver_info['driver_dnf_rate']:.1%}`")
    st.markdown(f"**Recent Form Position:** `P{driver_info['recent_form_avg_position']:.1f}`")

    st.divider()
    st.caption("🤖 Model: GradientBoostingClassifier (AUC 0.960)")
    st.caption("🌐 Live Data Source: Jolpica-F1 API")

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown('<div class="live-badge"><div class="pulse-dot"></div> LIVE F1 TELEMETRY PIPELINE ACTIVE</div>', unsafe_allow_html=True)
st.markdown('<div class="f1-title">🏎️ FORMULA 1 RACE PREDICTOR</div>', unsafe_allow_html=True)
st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Machine Learning Prediction Engine for Podium Finishes, Win Likelihood & Position Telemetry</p>", unsafe_allow_html=True)

# Navigation Tabs
tab_predictor, tab_simulator, tab_telemetry, tab_models = st.tabs([
    "🔮 Single Driver Predictor",
    "🏁 Full Grid Simulator",
    "📊 Driver Telemetry Comparison",
    "⚙️ Model Performance & Pipeline"
])

# ===========================================================================
# TAB 1: Single Driver Race Predictor
# ===========================================================================
with tab_predictor:
    col_inputs, col_preview = st.columns([1.2, 0.8], gap="large")

    with col_inputs:
        st.markdown("<div class='f1-card'>", unsafe_allow_html=True)
        st.markdown("### 🛠️ **Race Configuration**")

        selected_circuit_id = st.selectbox(
            "Select Grand Prix Track",
            options=circuit_stats_list,
            format_func=lambda x: circuit_options_map[x],
            index=circuit_stats_list.index("monaco") if "monaco" in circuit_stats_list else 0
        )

        grid_pos = st.slider(
            "Qualifying Grid Starting Position (P1 to P22)",
            min_value=1, max_value=22, value=1, step=1,
            help="Starting position on the grid after qualifying session."
        )

        weather_cond = st.select_slider(
            "Track Weather Condition",
            options=["Dry (Optimal ☀️)", "Cloudy (Overcast ☁️)", "Wet (Rain 🌧️)", "Mixed / Extreme ⛈️"],
            value="Dry (Optimal ☀️)"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_preview:
        st.markdown("<div class='f1-card'>", unsafe_allow_html=True)
        st.markdown("### 👤 **Driver Profile & Team Telemetry**")

        team_row = team_stats[team_stats["constructorId"] == const_id]
        team_pace_pts = team_row["team_recent_avg_points"].values[0] if not team_row.empty else 0.0

        circ_row = circuit_stats[
            (circuit_stats["driverId"] == selected_driver_id) &
            (circuit_stats["circuitId"] == selected_circuit_id)
        ]
        circ_avg_pos = circ_row["driver_circuit_avg_position"].values[0] if not circ_row.empty else driver_info["recent_form_avg_position"]

        c_a, c_b = st.columns(2)
        c_a.metric("Grid Position", f"P{grid_pos}")
        c_b.metric("Team Pace (Avg Pts)", f"{team_pace_pts:.1f} pts")

        c_c, c_d = st.columns(2)
        c_c.metric("Recent Form (Avg Pos)", f"P{driver_info['recent_form_avg_position']:.1f}")
        c_d.metric("Track History (Avg Pos)", f"P{circ_avg_pos:.1f}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Execute Prediction
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 RUN AI RACE PREDICTION TELEMETRY", use_container_width=True):

        feature_df = pd.DataFrame([{
            "grid": grid_pos,
            "recent_form_avg_position": driver_info["recent_form_avg_position"],
            "recent_form_avg_points": driver_info["recent_form_avg_points"],
            "team_recent_avg_points": team_pace_pts,
            "driver_circuit_avg_position": circ_avg_pos,
            "grid_penalty": 0,
            "season_points_so_far": driver_info["season_points_so_far"],
            "driver_dnf_rate": driver_info["driver_dnf_rate"],
        }])[feature_cols]

        podium_prob = podium_model.predict_proba(feature_df)[0, 1]
        winner_prob = winner_model.predict_proba(feature_df)[0, 1]
        predicted_pos = position_model.predict(feature_df)[0]
        finish_p = max(1, round(predicted_pos))

        st.markdown("---")
        st.markdown("## 🏆 **Predicted Race Results & Telemetry**")

        # KPI Displays
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">PODIUM LIKELIHOOD</div>
                <div class="metric-val" style="color: #FFD700;">{podium_prob:.1%}</div>
                <span style="color: #94A3B8; font-size: 0.8rem;">Top 3 Finish Probability</span>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">RACE WIN PROBABILITY</div>
                <div class="metric-val" style="color: #00E5FF;">{winner_prob:.1%}</div>
                <span style="color: #94A3B8; font-size: 0.8rem;">P1 Victory Likelihood</span>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">PREDICTED FINISH POSITION</div>
                <div class="metric-val" style="color: #FF1801;">P{finish_p}</div>
                <span style="color: #94A3B8; font-size: 0.8rem;">Expected Finish Rank</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Visualizations: Gauge Charts & Radar Chart
        col_gauge, col_radar = st.columns([1, 1])

        with col_gauge:
            fig_gauge = go.Figure()

            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=podium_prob * 100,
                domain={'x': [0, 1], 'y': [0.55, 1]},
                title={'text': "Podium Probability (%)", 'font': {'color': '#E2E8F0', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#FFD700"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(255, 24, 1, 0.2)"},
                        {'range': [30, 70], 'color': "rgba(255, 215, 0, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(0, 230, 118, 0.2)"}
                    ]
                }
            ))

            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=winner_prob * 100,
                domain={'x': [0, 1], 'y': [0, 0.45]},
                title={'text': "Win Probability (%)", 'font': {'color': '#E2E8F0', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#00E5FF"},
                    'steps': [
                        {'range': [0, 20], 'color': "rgba(255, 24, 1, 0.2)"},
                        {'range': [20, 50], 'color': "rgba(0, 229, 255, 0.2)"},
                        {'range': [50, 100], 'color': "rgba(0, 230, 118, 0.2)"}
                    ]
                }
            ))

            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#E2E8F0"},
                height=380,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_radar:
            # Radar chart normalized scores (0 to 100)
            radar_categories = ['Starting Grid', 'Recent Form', 'Team Pace', 'Track History', 'Reliability (1-DNF)']

            norm_grid = max(0, 100 - (grid_pos - 1) * 4.5)
            norm_form = max(0, 100 - (driver_info['recent_form_avg_position'] - 1) * 4.5)
            norm_pace = min(100, (team_pace_pts / 25.0) * 100)
            norm_track = max(0, 100 - (circ_avg_pos - 1) * 4.5)
            norm_rel = max(0, (1.0 - driver_info['driver_dnf_rate']) * 100)

            radar_values = [norm_grid, norm_form, norm_pace, norm_track, norm_rel]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_categories + [radar_categories[0]],
                fill='toself',
                fillcolor='rgba(255, 24, 1, 0.3)',
                line=dict(color='#FF1801', width=3),
                name=get_driver_label(selected_driver_id)
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="#94A3B8"),
                    angularaxis=dict(color="#E2E8F0")
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E2E8F0"),
                showlegend=False,
                height=380,
                margin=dict(l=40, r=40, t=30, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with st.expander("🔍 **Detailed Model Input Features (Transposed)**"):
            st.dataframe(feature_df.T.rename(columns={0: "Model Value"}), use_container_width=True)

# ===========================================================================
# TAB 2: Full Grid Race Simulator
# ===========================================================================
with tab_simulator:
    st.markdown("### 🏁 **Full Starting Grid Race Simulator**")
    st.caption("Simulate predicted race outcomes for top F1 drivers on the starting grid.")

    sim_circuit = st.selectbox(
        "Select Grand Prix Track for Grid Simulation",
        options=circuit_stats_list,
        format_func=lambda x: circuit_options_map[x],
        key="sim_circuit_select"
    )

    top_drivers = driver_stats.sort_values("season_points_so_far", ascending=False).head(16)["driverId"].tolist()

    if st.button("⚡ SIMULATE ENTIRE GRAND PRIX FINISH GRID", type="primary", use_container_width=True):

        results_list = []
        for idx, d_id in enumerate(top_drivers):
            d_row = driver_stats[driver_stats["driverId"] == d_id].iloc[0]
            c_id = d_row["constructorId"]

            t_row = team_stats[team_stats["constructorId"] == c_id]
            t_pace = t_row["team_recent_avg_points"].values[0] if not t_row.empty else 0.0

            cr_row = circuit_stats[
                (circuit_stats["driverId"] == d_id) &
                (circuit_stats["circuitId"] == sim_circuit)
            ]
            c_avg = cr_row["driver_circuit_avg_position"].values[0] if not cr_row.empty else d_row["recent_form_avg_position"]

            g_pos = idx + 1  # Grid rank

            f_df = pd.DataFrame([{
                "grid": g_pos,
                "recent_form_avg_position": d_row["recent_form_avg_position"],
                "recent_form_avg_points": d_row["recent_form_avg_points"],
                "team_recent_avg_points": t_pace,
                "driver_circuit_avg_position": c_avg,
                "grid_penalty": 0,
                "season_points_so_far": d_row["season_points_so_far"],
                "driver_dnf_rate": d_row["driver_dnf_rate"],
            }])[feature_cols]

            p_prob = podium_model.predict_proba(f_df)[0, 1]
            w_prob = winner_model.predict_proba(f_df)[0, 1]
            pred_p = position_model.predict(f_df)[0]

            results_list.append({
                "Driver": get_driver_label(d_id),
                "Team": get_constructor_label(c_id),
                "Start Grid": f"P{g_pos}",
                "Predicted Finish": pred_p,
                "Podium Prob (%)": p_prob * 100,
                "Win Prob (%)": w_prob * 100,
            })

        sim_df = pd.DataFrame(results_list).sort_values("Predicted Finish")
        sim_df["Predicted Rank"] = [f"P{i+1}" for i in range(len(sim_df))]

        # Podium Top 3 Showcase Cards
        p1, p2, p3 = st.columns(3)

        with p1:
            r1 = sim_df.iloc[0]
            st.markdown(f"""
            <div class="podium-p1">
                <h3 style="color: #FFD700; margin:0;">🥇 P1 WINNER</h3>
                <h2 style="color: #FFFFFF; margin: 4px 0;">{r1['Driver']}</h2>
                <p style="color: #94A3B8; margin:0;">{r1['Team']}</p>
                <div style="font-weight: 700; color: #FFD700; font-size: 1.2rem; margin-top: 8px;">Win Prob: {r1['Win Prob (%)']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            r2 = sim_df.iloc[1]
            st.markdown(f"""
            <div class="podium-p2">
                <h3 style="color: #CCCCCC; margin:0;">🥈 P2 PODIUM</h3>
                <h2 style="color: #FFFFFF; margin: 4px 0;">{r2['Driver']}</h2>
                <p style="color: #94A3B8; margin:0;">{r2['Team']}</p>
                <div style="font-weight: 700; color: #CCCCCC; font-size: 1.2rem; margin-top: 8px;">Podium Prob: {r2['Podium Prob (%)']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with p3:
            r3 = sim_df.iloc[2]
            st.markdown(f"""
            <div class="podium-p3">
                <h3 style="color: #CD7F32; margin:0;">🥉 P3 PODIUM</h3>
                <h2 style="color: #FFFFFF; margin: 4px 0;">{r3['Driver']}</h2>
                <p style="color: #94A3B8; margin:0;">{r3['Team']}</p>
                <div style="font-weight: 700; color: #CD7F32; font-size: 1.2rem; margin-top: 8px;">Podium Prob: {r3['Podium Prob (%)']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Leaderboard Bar Chart
        fig_bar = px.bar(
            sim_df,
            x="Podium Prob (%)",
            y="Driver",
            orientation="h",
            color="Podium Prob (%)",
            color_continuous_scale="Reds",
            title=f"Predicted Podium Finish Probabilities — {circuit_options_map[sim_circuit]}",
            text_auto='.1f'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#E2E8F0"),
            yaxis=dict(autorange="reversed"),
            height=450
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            sim_df[["Predicted Rank", "Driver", "Team", "Start Grid", "Podium Prob (%)", "Win Prob (%)"]],
            use_container_width=True,
            hide_index=True
        )

# ===========================================================================
# TAB 3: Driver Telemetry Comparison
# ===========================================================================
with tab_telemetry:
    st.markdown("### 📊 **Head-to-Head Driver Telemetry Comparison**")

    c1, c2 = st.columns(2)
    with c1:
        d1_id = st.selectbox("Driver A", sorted_drivers, index=0, format_func=lambda x: driver_map[x], key="d1_cmp")
    with c2:
        d2_id = st.selectbox("Driver B", sorted_drivers, index=1 if len(sorted_drivers) > 1 else 0, format_func=lambda x: driver_map[x], key="d2_cmp")

    d1 = driver_stats[driver_stats["driverId"] == d1_id].iloc[0]
    d2 = driver_stats[driver_stats["driverId"] == d2_id].iloc[0]

    cmp_data = {
        "Metric": ["Season Points", "Recent Form Avg Pos (Lower is better)", "Recent Form Avg Pts", "DNF Rate (%)"],
        get_driver_label(d1_id): [d1["season_points_so_far"], d1["recent_form_avg_position"], d1["recent_form_avg_points"], d1["driver_dnf_rate"] * 100],
        get_driver_label(d2_id): [d2["season_points_so_far"], d2["recent_form_avg_position"], d2["recent_form_avg_points"], d2["driver_dnf_rate"] * 100]
    }

    cmp_df = pd.DataFrame(cmp_data)

    fig_cmp = go.Figure(data=[
        go.Bar(name=get_driver_label(d1_id), x=cmp_df["Metric"], y=cmp_df[get_driver_label(d1_id)], marker_color="#FF1801"),
        go.Bar(name=get_driver_label(d2_id), x=cmp_df["Metric"], y=cmp_df[get_driver_label(d2_id)], marker_color="#00E5FF")
    ])
    fig_cmp.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0"),
        height=400,
        title="Telemetry Metrics Comparison"
    )

    st.plotly_chart(fig_cmp, use_container_width=True)
    st.table(cmp_df)

# ===========================================================================
# TAB 4: Model Performance & Pipeline
# ===========================================================================
with tab_models:
    st.markdown("### ⚙️ **Machine Learning Models & Validation Metrics**")

    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">PODIUM MODEL ROC-AUC</div>
            <div class="metric-val" style="color:#00E676;">0.924</div>
            <span style="color:#94A3B8;">GradientBoostingClassifier</span>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">RACE WINNER ROC-AUC</div>
            <div class="metric-val" style="color:#00E5FF;">0.960</div>
            <span style="color:#94A3B8;">GradientBoostingClassifier</span>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">POSITION MAE</div>
            <div class="metric-val" style="color:#FFD700;">3.14 Pos</div>
            <span style="color:#94A3B8;">GradientBoostingRegressor</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔄 **Data Architecture & Pipeline Flow**")
    st.info(
        "1. **Historical Pull (1950-2024):** Ergast-derived bulk dataset mirror.\n"
        "2. **Live Data Pull (2025-2026):** Jolpica-F1 API (Ergast successor).\n"
        "3. **Feature Engineering:** 10-race rolling driver form, team pace, circuit position history, DNF rate.\n"
        "4. **Season Validation Split:** Trained on 2010-2022, validated on held-out 2023 & 2024 seasons."
    )

st.divider()
st.caption("🏎️ F1 Race Predictor • Powered by Streamlit, Scikit-learn, Jolpica-F1 API & Plotly")
