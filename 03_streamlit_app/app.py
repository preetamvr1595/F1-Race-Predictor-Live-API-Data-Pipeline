"""
F1 Race Predictor — Streamlit App
Preetham's Data Science Mastery — Scikit-learn Project

Run locally with:
    streamlit run app.py

Requires the models/ and data/ folders from notebooks 01 and 02
(run those first, or use the pre-trained models shipped alongside this app).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


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
# Header
# ---------------------------------------------------------------------------
st.title("🏎️ F1 Race Predictor")
st.caption(
    "Trained on 15 seasons of real F1 data (2010-2024) via the Jolpica-F1 API "
    "and a historical Ergast-derived dataset. Gradient Boosting models predict "
    "podium finish, race win probability, and exact finishing position."
)

st.info(
    "**A design note:** the original mockup for this app included sliders for "
    "things like 'Top Speed' and 'Acceleration' as direct inputs. In reality "
    "those are *outcomes* of car performance, not something a user would set "
    "arbitrarily — so this app derives them from real historical driver and "
    "team form instead of letting you dial in an arbitrary top speed.",
    icon="💡",
)

col1, col2 = st.columns(2)

# ---------------------------------------------------------------------------
# LEFT COLUMN — Track and Race Conditions
# ---------------------------------------------------------------------------
with col1:
    st.subheader("Track and Race Conditions")

    circuit_options = sorted(circuit_stats["circuitId"].dropna().unique().tolist())
    selected_circuit = st.selectbox("Select Track", circuit_options, index=0)

    weather = st.selectbox("Weather Conditions", ["Dry", "Cloudy", "Wet", "Mixed"])

    st.caption(
        "Weather is shown for context — the current model version doesn't yet "
        "have historical weather data joined in (a known next-step improvement, "
        "noted in the README)."
    )

# ---------------------------------------------------------------------------
# RIGHT COLUMN — Driver and Car Parameters
# ---------------------------------------------------------------------------
with col2:
    st.subheader("Driver and Car Parameters")

    driver_options = sorted(driver_stats["driverId"].dropna().unique().tolist())
    selected_driver = st.selectbox("Select Driver", driver_options, index=0)

    grid_position = st.number_input(
        "Grid Position (starting position after qualifying)",
        min_value=1, max_value=22, value=10, step=1
    )

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.divider()

if st.button("🔮 Predict Race Outcome", type="primary", use_container_width=True):

    driver_row = driver_stats[driver_stats["driverId"] == selected_driver]
    if driver_row.empty:
        st.error("No historical data found for this driver.")
    else:
        driver_row = driver_row.iloc[0]
        constructor_id = driver_row["constructorId"]

        team_row = team_stats[team_stats["constructorId"] == constructor_id]
        team_pace = team_row["team_recent_avg_points"].values[0] if not team_row.empty else 0.0

        circuit_row = circuit_stats[
            (circuit_stats["driverId"] == selected_driver)
            & (circuit_stats["circuitId"] == selected_circuit)
        ]
        circuit_history = (
            circuit_row["driver_circuit_avg_position"].values[0]
            if not circuit_row.empty
            else driver_row["recent_form_avg_position"]
        )

        # Build the feature row in the exact order the model expects
        features = pd.DataFrame([{
            "grid": grid_position,
            "recent_form_avg_position": driver_row["recent_form_avg_position"],
            "recent_form_avg_points": driver_row["recent_form_avg_points"],
            "team_recent_avg_points": team_pace,
            "driver_circuit_avg_position": circuit_history,
            "grid_penalty": 0,  # no penalty info at prediction time -> assume none
            "season_points_so_far": driver_row["season_points_so_far"],
            "driver_dnf_rate": driver_row["driver_dnf_rate"],
        }])[feature_cols]

        podium_prob = podium_model.predict_proba(features)[0, 1]
        winner_prob = winner_model.predict_proba(features)[0, 1]
        predicted_position = position_model.predict(features)[0]

        st.subheader("Prediction Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Podium Probability", f"{podium_prob:.1%}")
        m2.metric("Win Probability", f"{winner_prob:.1%}")
        m3.metric("Predicted Finish", f"P{round(predicted_position)}")

        st.progress(min(float(podium_prob), 1.0), text="Podium likelihood")

        with st.expander("See the features that drove this prediction"):
            st.dataframe(features.T.rename(columns={0: "value"}))

        st.caption(
            "Reminder: this predicts based on recent form, team pace, grid "
            "position, and circuit history only — it has no visibility into "
            "race-day randomness (safety cars, weather changes, strategy calls) "
            "that often decides real F1 results."
        )

st.divider()
st.caption(
    "Built with real F1 data from the Jolpica-F1 API (live) and a historical "
    "1950-2024 Ergast-derived dataset. Part of Preetham's Data Science Mastery repo."
)
