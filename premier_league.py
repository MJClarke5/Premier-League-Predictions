# Converted from Premier League.Rmd -> Python (pandas + scikit-learn + xgboost)

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier, Booster
import joblib

# --- Loading Data ---
df = pd.read_csv(r"C:\Users\mattc\git-folder\Premier-League-Predictions\england-premier-league-matches-2025-to-2026-stats.csv")

# --- Utility: cumulative (rolling) mean per team like rollapplyr width = seq_along(...) ---
def cumulative_mean_by_team(data: pd.DataFrame, team_col: str, stat_col: str, new_col: str):
    s = data.groupby(data[team_col])[stat_col].apply(lambda x: x.expanding().mean())
    data[new_col] = s.reset_index(level=0, drop=True)
    return data

# Apply cumulative averages (matches ordered as in CSV)
df = cumulative_mean_by_team(df, "home_team_name", "home_team_shots", "HT_avgShots")
df = cumulative_mean_by_team(df, "away_team_name", "away_team_shots", "AT_avgShots")
df = cumulative_mean_by_team(df, "home_team_name", "home_team_shots_on_target", "HT_avgTarget")
df = cumulative_mean_by_team(df, "away_team_name", "away_team_shots_on_target", "AT_avgTarget")
df = cumulative_mean_by_team(df, "home_team_name", "home_team_possession", "HT_Possess")
df = cumulative_mean_by_team(df, "away_team_name", "away_team_possession", "AT_Possess")

# --- Betting Targets ---
# Note: original R used thresholds like >= 2.5, 3.5, 4.5
df["Over2"] = (df["total_goal_count"] >= 2.5).astype(int)   # becomes 0/1
df["Over3"] = (df["total_goal_count"] >= 3.5).astype(int)
df["Over4"] = (df["total_goal_count"] >= 4.5).astype(int)

# If you prefer categorical labels:
df["Over2"] = df["Over2"].astype("category")
df["Over3"] = df["Over3"].astype("category")
df["Over4"] = df["Over4"].astype("category")

# --- Odds - Under (auxiliary calculations) ---
# These mirror the R code (for quick analysis; not profit calculation)
if "odds_ft_over35" in df.columns:
    df["PercentOver35"] = 1.0 / df["odds_ft_over35"].replace(0, np.nan)
    df["PercentUnder35"] = 1.0 - df["PercentOver35"]
    df["OddsUnder35"] = 1.0 / df["PercentUnder35"].replace(0, np.nan)

if "odds_ft_over25" in df.columns:
    df["PercentOver25"] = 1.0 / df["odds_ft_over25"].replace(0, np.nan)
    df["PercentUnder25"] = 1.0 - df["PercentOver25"]
    df["OddsUnder25"] = 1.0 / df["PercentUnder25"].replace(0, np.nan)

# --- Preprocessing / "recipes" ---
# Map R recipe features to Python lists — adjust names if your CSV columns differ.
features_over3 = [
    "odds_ft_home_team_win", "odds_ft_away_team_win", "odds_ft_draw",
    "home_ppg", "away_ppg", "Home.Team.Pre.Match.xG", "Away.Team.Pre.Match.xG",
    "average_goals_per_match_pre_match"
]

features_over2 = features_over3 + ["HT_avgTarget", "AT_avgTarget"]

# choose recipe for Over3 (as in original knn_workflow for Over3)
target_col = "Over3"
features = [f for f in features_over3 if f in df.columns]  # keep only existing columns

# Identify numeric and categorical predictors
numeric_features = [c for c in features if df[c].dtype.kind in "biufc"]
categorical_features = [c for c in features if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
    ],
    remainder="drop"
)

# --- Train/Test split (row-based same as R indices) ---
# R used 1:110 for train and 111:120 for test (1-based). Translate to 0-based iloc:
train_df = df.iloc[0:110].copy()
test_df = df.iloc[110:120].copy()

X_train = train_df[features].copy()
y_train = train_df[target_col].astype(int).copy()
X_test = test_df[features].copy()
y_test = test_df[target_col].astype(int).copy()

# --- KNN: hyperparameter tuning for odd K values 1..15 ---
knn = KNeighborsClassifier()
pipe_knn = Pipeline([("pre", preprocessor), ("knn", knn)])

param_grid = {"knn__n_neighbors": [k for k in range(1, 16) if k % 2 == 1]}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
grid_knn = GridSearchCV(pipe_knn, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
grid_knn.fit(X_train, y_train)

best_knn = grid_knn.best_estimator_
print("Best KNN params:", grid_knn.best_params_, "Best CV acc:", grid_knn.best_score_)

# Predict probabilities and classes on test set
probs_knn = best_knn.predict_proba(X_test)[:, 1]  # probability of class 1
threshold = 0.7  # same idea used in R (0.7 or 0.8)
preds_knn = (probs_knn > threshold).astype(int)

test_results_knn = test_df[["Game.Week", "home_team_name", "away_team_name"]].copy()
test_results_knn["prob_1"] = probs_knn
test_results_knn["pred_class"] = preds_knn
print(test_results_knn[[ "home_team_name", "away_team_name", "Game.Week", "prob_1", "pred_class"]])

# Save a CSV of KNN predictions for a given week (example week=12)
pred_all_knn = best_knn.predict(preprocessor.transform(df[features])) if False else None
# safer to use pipeline.predict which includes preprocessing:
pred_all_knn = best_knn.predict(df[features])
df_pred_knn = df.assign(PredictedClass_knn = pred_all_knn)
df_pred_knn[df_pred_knn["Game.Week"] == 12][["Game.Week","home_team_name","away_team_name","PredictedClass_knn"]].to_csv(
    r"C:\Users\mattc\git-folder\Premier-League-Predictions\predictions_knn_week12.csv", index=False
)

# --- XGBoost: tuning and training ---
# Use the Over2 recipe as example (prem_data_receipe.1 in R)
features_xgb = [f for f in features_over2 if f in df.columns]
X_train_xgb = train_df[features_xgb]
y_train_xgb = train_df["Over2"].astype(int)
X_test_xgb = test_df[features_xgb]
y_test_xgb = test_df["Over2"].astype(int)

# Preprocess for XGB: scale numeric and one-hot categorical (same preprocessor can be reused)
preprocessor_xgb = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), [c for c in features_xgb if df[c].dtype.kind in "biufc"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), [c for c in features_xgb if c not in (c for c in features_xgb if df[c].dtype.kind in "biufc")]),
    ],
    remainder="drop"
)

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=1000)

pipe_xgb = Pipeline([("pre", preprocessor_xgb), ("xgb", xgb_clf)])

# Parameter search space (smaller randomized search like R)
param_dist = {
    "xgb__max_depth": [2, 3, 4, 5, 6],
    "xgb__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "xgb__min_child_weight": [1, 3, 5, 7],
    "xgb__gamma": [0, 0.1, 0.5, 1],
    "xgb__subsample": [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.6, 0.8, 1.0],
}

rand_xgb = RandomizedSearchCV(
    pipe_xgb, param_distributions=param_dist, n_iter=30, scoring="roc_auc",
    cv=cv, random_state=123, n_jobs=-1
)

rand_xgb.fit(X_train_xgb, y_train_xgb)
print("Best XGB params:", rand_xgb.best_params_, "Best CV roc_auc:", rand_xgb.best_score_)

best_xgb = rand_xgb.best_estimator_

# Predictions
probs_xgb = best_xgb.predict_proba(X_test_xgb)[:, 1]
preds_xgb = best_xgb.predict(X_test_xgb)

results_xgb = test_df[["Game.Week", "home_team_name", "away_team_name"]].copy()
results_xgb["prob_1"] = probs_xgb
results_xgb["pred_class"] = preds_xgb
print(results_xgb[["home_team_name","away_team_name","Game.Week","prob_1","pred_class"]])

# Feature importance (XGBoost native)
# To get feature names after preprocessing, transform a dataframe through preprocessor
try:
    X_train_trans = best_xgb.named_steps["pre"].transform(X_train_xgb)
    # reconstruct column names: numeric features + one-hot encoder feature names
    num_cols = best_xgb.named_steps["pre"].transformers_[0][2]
    ohe = best_xgb.named_steps["pre"].transformers_[1][1]
    cat_cols = best_xgb.named_steps["pre"].transformers_[1][2]
    ohe_feature_names = []
    if hasattr(ohe, "get_feature_names_out"):
        ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feat_names = list(num_cols) + ohe_feature_names
    booster = best_xgb.named_steps["xgb"].get_booster()
    # map feature importances to names
    imp = booster.get_score(importance_type="gain")
    imp_df = pd.DataFrame([
        {"feature": k.replace("f", ""), "importance": v} for k, v in imp.items()
    ])
    # Note: mapping numeric f indices to actual names is non-trivial after ColumnTransformer; skip rigorous mapping here
    print("Top XGBoost feature importances (raw):", imp_df.sort_values("importance", ascending=False).head(15))
except Exception as e:
    print("Feature importance extraction skipped or failed:", str(e))

# Save models
joblib.dump(best_knn, r"C:\Users\mattc\git-folder\Premier-League-Predictions\best_knn.pkl")
joblib.dump(best_xgb, r"C:\Users\mattc\git-folder\Premier-League-Predictions\best_xgb.pkl")

# Save example predictions CSV
results_xgb.to_csv(r"C:\Users\mattc\git-folder\Premier-League-Predictions\predictions_xgb_test.csv", index=False)

print("Conversion complete. Adjust feature lists if column names differ from the original R