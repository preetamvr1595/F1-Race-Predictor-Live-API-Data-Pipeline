# F1 Race Predictor

An end-to-end machine learning project: predict Formula 1 podium finishes, race winners, and exact finishing positions using Gradient Boosting, trained on real historical and live F1 data.

## What's in here

| File | What it does |
|---|---|
| `01_live_api_and_data_pipeline.ipynb` | Pulls live 2026 season data from the Jolpica-F1 API (Ergast's maintained successor), loads 75 years of historical race data (1950-2024), merges everything into one clean dataset |
| `02_feature_engineering_and_models.ipynb` | Builds real predictive features (rolling form, team pace, circuit history, DNF rate), trains 3 Gradient Boosting models with proper season-based validation |
| `03_streamlit_app/app.py` | Interactive prediction app — pick a driver, track, and grid position, get podium/win probability and predicted finish |
| `data/` | Historical CSVs (1950-2024) + live 2026 snapshot + engineered reference data for the app |
| `models/` | The 3 trained models (`.pkl`), ready to load without retraining |

## Data sources

- **Historical (1950-2024):** Ergast-derived CSV mirror ([muharsyad/formula-one-datasets](https://github.com/muharsyad/formula-one-datasets)) — the classic Ergast API shut down at the end of 2024, so bulk historical pulls now go through community mirrors rather than the dead live endpoint.
- **Live (2025-2026):** [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1) — the community-maintained, Ergast-compatible successor. Free, no auth, rate-limited to 200 requests/hour.

## Model results (2023-2024 held out, not seen during training)

| Model | Metric | Score |
|---|---|---|
| Podium classifier | Accuracy / ROC-AUC | 89.4% / 0.924 |
| Race winner classifier | Accuracy / ROC-AUC | 95.7% / 0.960 |
| Exact position regressor | MAE | 3.14 positions (vs. 3.60 naive baseline) |

The podium model was validated on 2023 and 2024 **separately** to confirm it isn't just fitting one lucky season (AUC 0.915 and 0.930 respectively).

## Honest limitations

- No weather, tire strategy, or safety-car data — the randomness that often decides real races isn't in this feature set.
- The original app mockup had sliders for "Top Speed" and "Acceleration" as user inputs — those are actually *outputs* of car performance, not knobs a user should set arbitrarily, so the app derives them from real historical data instead.
- `lap_times.csv` is included in `data/` but not yet used by any model — a natural next extension (stint pace, tire degradation modeling).

## Running the app

```bash
pip install streamlit scikit-learn pandas numpy joblib
cd 03_streamlit_app
streamlit run app.py
```

## Push to your repo

```powershell
cd C:\Users\praveen\Data-Science-Mastery
mkdir 06-Scikit-learn
# move the whole 06-Scikit-learn folder contents in
git add .
git commit -m "Add F1 Race Predictor: live API + historical data, 3 Gradient Boosting models, Streamlit app"
git push
```
