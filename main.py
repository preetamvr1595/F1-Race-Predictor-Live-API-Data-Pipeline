"""
🏎️ F1 AI Telemetry & Race Predictor — FastAPI Backend Service
High-performance REST API serving Scikit-Learn Gradient Boosting Models
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI(title="F1 AI Telemetry & Race Predictor API", version="2.0.0")

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") if os.path.exists(os.path.join(BASE_DIR, "data")) else os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "models") if os.path.exists(os.path.join(BASE_DIR, "models")) else os.path.join(BASE_DIR, "..", "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

# Load Scikit-learn Models & Data
podium_model = joblib.load(os.path.join(MODEL_DIR, "podium_model.pkl"))
winner_model = joblib.load(os.path.join(MODEL_DIR, "winner_model.pkl"))
position_model = joblib.load(os.path.join(MODEL_DIR, "position_model.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

driver_stats = pd.read_csv(os.path.join(DATA_DIR, "latest_driver_stats.csv"))
team_stats = pd.read_csv(os.path.join(DATA_DIR, "latest_team_stats.csv"))
circuit_stats = pd.read_csv(os.path.join(DATA_DIR, "latest_circuit_stats.csv"))
races = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))

# Driver Metadata Mapping
DRIVER_META = {
    "max_verstappen": {"name": "Max Verstappen", "flag": "🇳🇱", "team": "Red Bull Racing", "color": "#3671C6"},
    "hamilton": {"name": "Lewis Hamilton", "flag": "🇬🇧", "team": "Mercedes-AMG Petronas", "color": "#27F4D2"},
    "leclerc": {"name": "Charles Leclerc", "flag": "🇲🇨", "team": "Scuderia Ferrari", "color": "#E80020"},
    "norris": {"name": "Lando Norris", "flag": "🇬🇧", "team": "McLaren F1 Team", "color": "#FF8000"},
    "sainz": {"name": "Carlos Sainz", "flag": "🇪🇸", "team": "Scuderia Ferrari", "color": "#E80020"},
    "perez": {"name": "Sergio Perez", "flag": "🇲🇽", "team": "Red Bull Racing", "color": "#3671C6"},
    "russell": {"name": "George Russell", "flag": "🇬🇧", "team": "Mercedes-AMG Petronas", "color": "#27F4D2"},
    "piastri": {"name": "Oscar Piastri", "flag": "🇦🇺", "team": "McLaren F1 Team", "color": "#FF8000"},
    "alonso": {"name": "Fernando Alonso", "flag": "🇪🇸", "team": "Aston Martin Aramco", "color": "#229971"},
    "stroll": {"name": "Lance Stroll", "flag": "🇨🇦", "team": "Aston Martin Aramco", "color": "#229971"},
    "gasly": {"name": "Pierre Gasly", "flag": "🇫🇷", "team": "Alpine F1 Team", "color": "#0093CC"},
    "ocon": {"name": "Esteban Ocon", "flag": "🇫🇷", "team": "Alpine F1 Team", "color": "#0093CC"},
    "albon": {"name": "Alexander Albon", "flag": "🇹🇭", "team": "Williams Racing", "color": "#64C4FF"},
    "tsunoda": {"name": "Yuki Tsunoda", "flag": "🇯🇵", "team": "Visa Cash App RB", "color": "#6692FF"},
    "ricciardo": {"name": "Daniel Ricciardo", "flag": "🇦🇺", "team": "Visa Cash App RB", "color": "#6692FF"},
    "hulkenberg": {"name": "Nico Hülkenberg", "flag": "🇩🇪", "team": "Haas F1 Team", "color": "#B6BABD"},
    "magnussen": {"name": "Kevin Magnussen", "flag": "🇩🇰", "team": "Haas F1 Team", "color": "#B6BABD"},
    "bottas": {"name": "Valtteri Bottas", "flag": "🇫🇮", "team": "Stake F1 Team Sauber", "color": "#52E252"},
    "zhou": {"name": "Zhou Guanyu", "flag": "🇨🇳", "team": "Stake F1 Team Sauber", "color": "#52E252"},
    "sargeant": {"name": "Logan Sargeant", "flag": "🇺🇸", "team": "Williams Racing", "color": "#64C4FF"},
    "vettel": {"name": "Sebastian Vettel", "flag": "🇩🇪", "team": "Aston Martin Aramco", "color": "#229971"},
    "raikkonen": {"name": "Kimi Räikkönen", "flag": "🇫🇮", "team": "Alfa Romeo F1 Team", "color": "#52E252"},
}

CIRCUIT_META = {
    "monaco": {"name": "Circuit de Monaco", "location": "Monte Carlo 🇲🇨", "laps": 78},
    "silverstone": {"name": "Silverstone Circuit", "location": "Great Britain 🇬🇧", "laps": 52},
    "monza": {"name": "Autodromo Nazionale Monza", "location": "Italy 🇮🇹", "laps": 53},
    "spa": {"name": "Circuit de Spa-Francorchamps", "location": "Belgium 🇧🇪", "laps": 44},
    "red_bull_ring": {"name": "Red Bull Ring", "location": "Austria 🇦🇹", "laps": 71},
    "suzuka": {"name": "Suzuka International Course", "location": "Japan 🇯🇵", "laps": 53},
    "americas": {"name": "Circuit of the Americas", "location": "Austin 🇺🇸", "laps": 56},
    "interlagos": {"name": "Autódromo José Carlos Pace", "location": "Brazil 🇧🇷", "laps": 71},
    "shanghai": {"name": "Shanghai International Circuit", "location": "China 🇨🇳", "laps": 56},
    "catalunya": {"name": "Circuit de Barcelona-Catalunya", "location": "Spain 🇪🇸", "laps": 66},
    "zandvoort": {"name": "Circuit Zandvoort", "location": "Netherlands 🇳🇱", "laps": 72},
    "baku": {"name": "Baku City Circuit", "location": "Azerbaijan 🇦🇿", "laps": 51},
    "marina_bay": {"name": "Marina Bay Street Circuit", "location": "Singapore 🇸🇬", "laps": 62},
    "yas_marina": {"name": "Yas Marina Circuit", "location": "Abu Dhabi 🇦🇪", "laps": 58},
    "hungaroring": {"name": "Hungaroring", "location": "Hungary 🇭🇺", "laps": 70},
    "bahrain": {"name": "Bahrain International Circuit", "location": "Bahrain 🇧🇭", "laps": 57},
    "jeddah": {"name": "Jeddah Corniche Circuit", "location": "Saudi Arabia 🇸🇦", "laps": 50},
    "albert_park": {"name": "Albert Park Circuit", "location": "Australia 🇦🇺", "laps": 58},
    "miami": {"name": "Miami International Autodrome", "location": "USA 🇺🇸", "laps": 57},
    "losail": {"name": "Lusail International Circuit", "location": "Qatar 🇶🇦", "laps": 57},
    "vegas": {"name": "Las Vegas Strip Circuit", "location": "USA 🇺🇸", "laps": 50},
}


class PredictRequest(BaseModel):
    driver_id: str
    circuit_id: str
    grid_position: int


class SimulateGridRequest(BaseModel):
    circuit_id: str


@app.get("/api/drivers")
def get_drivers():
    drivers_data = []
    for _, row in driver_stats.iterrows():
        d_id = row["driverId"]
        meta = DRIVER_META.get(d_id, {
            "name": d_id.replace("_", " ").title(),
            "flag": "🏎️",
            "team": str(row["constructorId"]).replace("_", " ").title(),
            "color": "#FF1801"
        })
        drivers_data.append({
            "id": d_id,
            "name": meta["name"],
            "flag": meta["flag"],
            "team": meta["team"],
            "color": meta["color"],
            "constructor_id": row["constructorId"],
            "recent_form_position": float(row["recent_form_avg_position"]),
            "recent_form_points": float(row["recent_form_avg_points"]),
            "season_points": float(row["season_points_so_far"]),
            "dnf_rate": float(row["driver_dnf_rate"])
        })
    drivers_data.sort(key=lambda x: x["season_points"], reverse=True)
    return drivers_data


@app.get("/api/circuits")
def get_circuits():
    circuits_data = []
    unique_circuits = sorted(circuit_stats["circuitId"].dropna().unique().tolist())
    for c_id in unique_circuits:
        meta = CIRCUIT_META.get(c_id, {
            "name": c_id.replace("_", " ").title(),
            "location": "Grand Prix Track 🏁",
            "laps": 55
        })
        circuits_data.append({
            "id": c_id,
            "name": meta["name"],
            "location": meta["location"],
            "laps": meta["laps"]
        })
    return circuits_data


@app.post("/api/predict")
def predict_race(req: PredictRequest):
    driver_rows = driver_stats[driver_stats["driverId"] == req.driver_id]
    if driver_rows.empty:
        raise HTTPException(status_code=404, detail="Driver not found")

    driver_row = driver_rows.iloc[0]
    c_id = driver_row["constructorId"]

    t_row = team_stats[team_stats["constructorId"] == c_id]
    team_pace = float(t_row["team_recent_avg_points"].values[0]) if not t_row.empty else 0.0

    cr_row = circuit_stats[
        (circuit_stats["driverId"] == req.driver_id) &
        (circuit_stats["circuitId"] == req.circuit_id)
    ]
    circ_history = float(cr_row["driver_circuit_avg_position"].values[0]) if not cr_row.empty else float(driver_row["recent_form_avg_position"])

    feature_dict = {
        "grid": req.grid_position,
        "recent_form_avg_position": float(driver_row["recent_form_avg_position"]),
        "recent_form_avg_points": float(driver_row["recent_form_avg_points"]),
        "team_recent_avg_points": team_pace,
        "driver_circuit_avg_position": circ_history,
        "grid_penalty": 0,
        "season_points_so_far": float(driver_row["season_points_so_far"]),
        "driver_dnf_rate": float(driver_row["driver_dnf_rate"]),
    }

    feature_df = pd.DataFrame([feature_dict])[feature_cols]

    podium_prob = float(podium_model.predict_proba(feature_df)[0, 1])
    winner_prob = float(winner_model.predict_proba(feature_df)[0, 1])
    pred_pos = float(position_model.predict(feature_df)[0])
    finish_p = max(1, round(pred_pos))

    # Calculate telemetry radar scores (0 to 100)
    norm_grid = max(0, round(100 - (req.grid_position - 1) * 4.5, 1))
    norm_form = max(0, round(100 - (driver_row['recent_form_avg_position'] - 1) * 4.5, 1))
    norm_pace = min(100, round((team_pace / 25.0) * 100, 1))
    norm_track = max(0, round(100 - (circ_history - 1) * 4.5, 1))
    norm_rel = max(0, round((1.0 - driver_row['driver_dnf_rate']) * 100, 1))

    return {
        "driver_id": req.driver_id,
        "circuit_id": req.circuit_id,
        "grid_position": req.grid_position,
        "podium_probability": podium_prob,
        "winner_probability": winner_prob,
        "predicted_position": finish_p,
        "raw_position": pred_pos,
        "radar_scores": {
            "grid": norm_grid,
            "form": norm_form,
            "team_pace": norm_pace,
            "track_history": norm_track,
            "reliability": norm_rel
        },
        "features": feature_dict
    }


@app.post("/api/simulate-grid")
def simulate_grid(req: SimulateGridRequest):
    top_drivers = driver_stats.sort_values("season_points_so_far", ascending=False).head(16)["driverId"].tolist()

    grid_results = []
    for idx, d_id in enumerate(top_drivers):
        d_row = driver_stats[driver_stats["driverId"] == d_id].iloc[0]
        c_id = d_row["constructorId"]

        t_row = team_stats[team_stats["constructorId"] == c_id]
        t_pace = float(t_row["team_recent_avg_points"].values[0]) if not t_row.empty else 0.0

        cr_row = circuit_stats[
            (circuit_stats["driverId"] == d_id) &
            (circuit_stats["circuitId"] == req.circuit_id)
        ]
        c_avg = float(cr_row["driver_circuit_avg_position"].values[0]) if not cr_row.empty else float(d_row["recent_form_avg_position"])

        g_pos = idx + 1

        f_df = pd.DataFrame([{
            "grid": g_pos,
            "recent_form_avg_position": float(d_row["recent_form_avg_position"]),
            "recent_form_avg_points": float(d_row["recent_form_avg_points"]),
            "team_recent_avg_points": t_pace,
            "driver_circuit_avg_position": c_avg,
            "grid_penalty": 0,
            "season_points_so_far": float(d_row["season_points_so_far"]),
            "driver_dnf_rate": float(d_row["driver_dnf_rate"]),
        }])[feature_cols]

        p_prob = float(podium_model.predict_proba(f_df)[0, 1])
        w_prob = float(winner_model.predict_proba(f_df)[0, 1])
        pred_p = float(position_model.predict(f_df)[0])

        meta = DRIVER_META.get(d_id, {
            "name": d_id.replace("_", " ").title(),
            "flag": "🏎️",
            "team": str(c_id).replace("_", " ").title(),
            "color": "#FF1801"
        })

        grid_results.append({
            "driver_id": d_id,
            "driver_name": meta["name"],
            "flag": meta["flag"],
            "team": meta["team"],
            "color": meta["color"],
            "start_grid": g_pos,
            "predicted_finish_raw": pred_p,
            "podium_probability": p_prob,
            "winner_probability": w_prob
        })

    grid_results.sort(key=lambda x: x["predicted_finish_raw"])
    for idx, item in enumerate(grid_results):
        item["predicted_rank"] = idx + 1

    return {
        "circuit_id": req.circuit_id,
        "results": grid_results
    }


# Serve static files and fallback to index.html
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
