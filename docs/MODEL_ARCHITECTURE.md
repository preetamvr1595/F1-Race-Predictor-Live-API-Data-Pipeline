# F1 Race Predictor — Model Architecture & Hyperparameters

## Overview

The prediction engine consists of three trained Scikit-learn models:

1. **Podium Classifier (`podium_model.pkl`)**
   - **Model Type:** `GradientBoostingClassifier`
   - **Target:** `target_podium` (1 if position <= 3 else 0)
   - **Performance:** 89.4% Accuracy, 0.924 ROC-AUC on 2023-2024 test split.

2. **Race Winner Classifier (`winner_model.pkl`)**
   - **Model Type:** `GradientBoostingClassifier`
   - **Target:** `target_winner` (1 if position == 1 else 0)
   - **Performance:** 95.7% Accuracy, 0.960 ROC-AUC.

3. **Position Regressor (`position_model.pkl`)**
   - **Model Type:** `GradientBoostingRegressor`
   - **Target:** `target_position` (Continuous position 1-20)
   - **Performance:** MAE = 3.14 positions (vs. 3.60 naive baseline).

## Input Feature Vector (8 Features)
1. `grid`: Qualifying / starting grid position.
2. `recent_form_avg_position`: Rolling 5-race mean finishing position.
3. `recent_form_avg_points`: Rolling 5-race mean points scored.
4. `team_recent_avg_points`: Constructor rolling points average.
5. `driver_circuit_avg_position`: Historical average finish position at the selected circuit.
6. `grid_penalty`: Grid position penalty offset.
7. `season_points_so_far`: Cumulative season points accumulated before race.
8. `driver_dnf_rate`: Driver historical DNF percentage.
