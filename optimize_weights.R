library(reticulate)
py_install("optuna")

# Define the relevant features for scaling
feature_cols <- c("SHOT_DIST", "CLOSE_DEF_DIST", "DRIBBLES", "TOUCH_TIME")

# Pre-scale the full dataset (excluding test set if necessary)
scaled_nba_shots <- scale_features(nba_shots %>% filter(set != 'test'), feature_cols)
scaled_shots_data <- scaled_nba_shots$data


py_run_string("
import optuna
import numpy as np

def objective(trial):
    # Define parameters to optimize
    prior_strength = trial.suggest_float('prior_strength', 10, 10000, log=True)
    m1 = trial.suggest_float('m1', 0, 2)
    m2 = trial.suggest_float('m2', 0, 2)
    m3 = trial.suggest_float('m3', 0, 2)
    m4 = trial.suggest_float('m4', 0, 2)

    # Call R function (using reticulate)
    r_predict_shots = r.predict_shots
    error = r_predict_shots(
        r.scaled_shots_data,
        prior_strength,
        [m1, m2, m3, m4]
    )

    return error  # Optuna minimizes this

study = optuna.create_study(direction='minimize')
study.optimize(objective, timeout=64800)  # Run for 12 hours

# Store best parameters
best_params = study.best_params
")


