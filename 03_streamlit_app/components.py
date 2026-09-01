"""
UI Component Helpers for Streamlit Dashboard.
"""

import streamlit as st

def render_header():
    st.title("🏎️ F1 Race Predictor")
    st.caption("Powered by Gradient Boosting Models & Live Jolpica-F1 API Data")

def render_metrics(podium_prob: float, winner_prob: float, predicted_pos: float):
    c1, c2, c3 = st.columns(3)
    c1.metric("Podium Probability", f"{podium_prob:.1%}")
    c2.metric("Win Probability", f"{winner_prob:.1%}")
    c3.metric("Predicted Finish", f"P{round(predicted_pos)}")
