"""
SSDS water level (tma_mdpl) forecasting — v2 pipeline.

Upgrades over v1:
  - Leakage-safe missing-value imputation (per-station ffill/bfill + median fallback)
  - ~20 new engineered features (rainfall accumulation, wind vectors, soil/pressure
    trends, station target-encoding, calendar/season flags, rate-of-change features)
  - 10-model baseline comparison on a time-based holdout, top 3 kept
  - Short GridSearchCV (TimeSeriesSplit) tuning of the best model
  - Vectorized (per-timestep, all-stations-at-once) recursive forecasting for speed
  - Whole script budgeted to run in well under 15 minutes

Run: python build_model_v2.py
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

T0 = time.time()
def elapsed():
    return f"[{time.time()-T0:6.1f}s]"

ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

# ============================================================================
# 1. LOAD
# ============================================================================
train = pd.read_csv(ROOT / "train.csv", parse_dates=["datetime"])
test_raw = pd.read_csv(ROOT / "test.csv")
sample_sub = pd.read_csv(ROOT / "sample_submission.csv")
env = pd.read_csv(ROOT / "data_pendukung" / "data_lingkungan.csv", parse_dates=["datetime"])
coord = pd.read_csv(ROOT / "data_pendukung" / "koordinat_pos.csv")

split_id = test_raw["id"].str.split(" - ", n=1, expand=True)
test = pd.DataFrame({
    "id": test_raw["id"],
    "datetime": pd.to_datetime(split_id[0]),
    "nama_pos": split_id[1],
})
print(f"{elapsed()} loaded train={train.shape} test={test.shape} env={env.shape}")

# ============================================================================
# 2. ENVIRONMENTAL FEATURE WINDOWS (shared across train+test target timeline)
# ============================================================================
raw_num_cols = [c for c in env.columns if c not in ("datetime", "nama_pos", "landcover_name")]
SM_COLS = ["soil_moisture_0_7cm", "soil_moisture_7_28cm", "soil_moisture_28_100cm", "soil_moisture_100_255cm"]

env = env.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)
env["soil_moisture_avg"] = env[SM_COLS].mean(axis=1)
env["wind_u"] = env["wind_speed_kmh"] * np.cos(np.deg2rad(env["wind_direction_deg"]))
env["wind_v"] = env["wind_speed_kmh"] * np.sin(np.deg2rad(env["wind_direction_deg"]))
env["temp_dew_spread"] = env["temperature_c"] - env["dew_point_c"]
env["mjo_interaction"] = env["mjo_amplitude"] * env["mjo_active"]
env["humidity_temp_interaction"] = env["humidity_pct"] * env["temperature_c"] / 100.0

MEAN_COLS = raw_num_cols + ["soil_moisture_avg", "wind_u", "wind_v", "temp_dew_spread",
                            "mjo_interaction", "humidity_temp_interaction"]
SUM_COLS = ["rainfall_mm", "rainfall_openmeteo_mm"]

def build_env_features(env_df, target_df):
    """6h-window aggregation + multi-day rainfall accumulation, vectorized across
    all stations at once via groupby().rolling() + a single by-group merge_asof
    (avoids an O(stations x rows) Python loop)."""
    e = env_df.sort_values(["nama_pos", "datetime"]).set_index("datetime")

    roll6 = e.groupby("nama_pos", observed=True)[MEAN_COLS].rolling("6h", min_periods=1).mean()
    rain_sum6 = e.groupby("nama_pos", observed=True)[SUM_COLS].rolling("6h", min_periods=1).sum()
    roll6 = roll6.droplevel(0)
    rain_sum6 = rain_sum6.droplevel(0)
    roll6["nama_pos"] = e["nama_pos"].values
    roll6["rainfall_mm"] = rain_sum6["rainfall_mm"].values
    roll6["rainfall_openmeteo_mm"] = rain_sum6["rainfall_openmeteo_mm"].values

    for days, label in [(1, "1d"), (3, "3d"), (7, "7d"), (14, "14d")]:
        rs = e.groupby("nama_pos", observed=True)["rainfall_mm"].rolling(f"{days}d", min_periods=1).sum()
        roll6[f"rainfall_sum_{label}"] = rs.droplevel(0).values

    roll6 = roll6.reset_index().sort_values(["datetime"]).reset_index(drop=True)

    tgt = target_df.drop_duplicates().sort_values(["datetime"]).reset_index(drop=True)
    out = pd.merge_asof(tgt, roll6, on="datetime", by="nama_pos",
                         direction="nearest", tolerance=pd.Timedelta("1h"))
    return out

combined_targets = pd.concat([
    train[["nama_pos", "datetime"]],
    test[["nama_pos", "datetime"]],
], ignore_index=True).drop_duplicates()

print(f"{elapsed()} aggregating environmental windows for {len(combined_targets)} target timestamps...")
env_features = build_env_features(env, combined_targets)
print(f"{elapsed()} env window aggregation done -> {env_features.shape}")

# per-station forward/backward fill for any residual gaps (leakage-safe: only uses
# each station's own timeline, not other stations or the target)
env_features = env_features.sort_values(["nama_pos", "datetime"])
fill_cols = [c for c in env_features.columns if c not in ("nama_pos", "datetime")]
env_features[fill_cols] = env_features.groupby("nama_pos")[fill_cols].transform(lambda s: s.ffill().bfill())
env_features[fill_cols] = env_features[fill_cols].fillna(env_features[fill_cols].median())

# pressure / soil-moisture trend (~1 day back, 3 steps at 3 obs/day) computed on the
# actual observation grid, per station, in chronological order
env_features = env_features.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)
env_features["pressure_trend"] = env_features.groupby("nama_pos")["surface_pressure_hpa"].transform(
    lambda s: s - s.shift(3))
env_features["soil_moisture_trend"] = env_features.groupby("nama_pos")["soil_moisture_avg"].transform(
    lambda s: s - s.shift(3))
env_features[["pressure_trend", "soil_moisture_trend"]] = env_features[["pressure_trend", "soil_moisture_trend"]].fillna(0)

lc = env.sort_values("datetime").groupby("nama_pos")["landcover_name"].last()

train = train.merge(env_features, on=["nama_pos", "datetime"], how="left")
test = test.merge(env_features, on=["nama_pos", "datetime"], how="left")
train["landcover_name"] = train["nama_pos"].map(lc)
test["landcover_name"] = test["nama_pos"].map(lc)
train = train.merge(coord, on="nama_pos", how="left")
test = test.merge(coord, on="nama_pos", how="left")
print(f"{elapsed()} merged env features onto train/test")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
def add_calendar_features(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_wet_season"] = df["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df

train = add_calendar_features(train)
test = add_calendar_features(test)

# station target-encoding (computed from train only -> static lookup applied to test)
station_stats = train.groupby("nama_pos")["tma_mdpl"].agg(["mean", "std"]).rename(
    columns={"mean": "station_mean_tma", "std": "station_std_tma"})
train = train.merge(station_stats, on="nama_pos", how="left")
test = test.merge(station_stats, on="nama_pos", how="left")

cat_cols = ["nama_pos", "landcover_name"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))

# 3 obs/day (06/12/18) -> ~1 day = 3 steps, ~1 week = 21 steps
LAGS = [1, 2, 3, 6, 9, 21]
ROLL_WINDOWS = [3, 21]

train = train.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)

def add_lag_features(df, lags, rolls, target_col="tma_mdpl"):
    df = df.copy()
    grp = df.groupby("nama_pos", observed=True)[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)
    shifted = grp.shift(1)
    for w in rolls:
        df[f"roll_mean_{w}"] = shifted.groupby(df["nama_pos"], observed=True).transform(
            lambda s: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = shifted.groupby(df["nama_pos"], observed=True).transform(
            lambda s: s.rolling(w, min_periods=1).std())
    df["tma_diff_1"] = df["lag_1"] - df["lag_2"]
    df["tma_diff_1d"] = df["lag_1"] - df["lag_3"]
    df["tma_diff_7d"] = df["lag_1"] - df["lag_21"]
    return df

train = add_lag_features(train, LAGS, ROLL_WINDOWS)

new_engineered = (
    [f"rainfall_sum_{l}" for l in ["1d", "3d", "7d", "14d"]]
    + ["wind_u", "wind_v", "temp_dew_spread", "soil_moisture_avg", "soil_moisture_trend",
       "pressure_trend", "mjo_interaction", "humidity_temp_interaction",
       "day_of_week", "is_weekend", "is_wet_season",
       "station_mean_tma", "station_std_tma",
       "tma_diff_1", "tma_diff_1d", "tma_diff_7d"]
)
print(f"{elapsed()} feature engineering done — {len(new_engineered)} new features added: {new_engineered}")

feature_cols = (
    ["nama_pos", "landcover_name", "latitude", "longitude",
     "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    + MEAN_COLS + ["rainfall_mm", "rainfall_openmeteo_mm"]
    + [f"rainfall_sum_{l}" for l in ["1d", "3d", "7d", "14d"]]
    + ["pressure_trend", "soil_moisture_trend"]
    + ["day_of_week", "is_weekend", "is_wet_season", "station_mean_tma", "station_std_tma"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS] + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["tma_diff_1", "tma_diff_1d", "tma_diff_7d"]
)
feature_cols = [c for c in dict.fromkeys(feature_cols) if c in train.columns]
target_col = "tma_mdpl"
num_feature_cols = [c for c in feature_cols if c not in cat_cols]
print(f"{elapsed()} total feature count: {len(feature_cols)}")

# ============================================================================
# 4. TIME-BASED SPLIT
# ============================================================================
cutoff = train["datetime"].max() - pd.Timedelta(days=60)
tr = train[train["datetime"] <= cutoff].dropna(subset=[f"lag_{max(LAGS)}"])
va = train[train["datetime"] > cutoff]
print(f"{elapsed()} train rows={len(tr)}  val rows={len(va)}  cutoff={cutoff.date()}")

# stratified-by-station subsample for the 10-model speed comparison
tr_sub = tr.groupby("nama_pos", observed=True, group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), 600), random_state=RANDOM_STATE))
print(f"{elapsed()} comparison subsample: {len(tr_sub)} rows (from {len(tr)})")

# ============================================================================
# 5. PREPROCESSING PIPELINES
# ============================================================================
pre_tree = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", SimpleImputer(strategy="median"), num_feature_cols),
])
pre_linear = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_feature_cols),
])

MODELS = {
    "LinearRegression": (pre_linear, LinearRegression()),
    "Ridge": (pre_linear, Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    "Lasso": (pre_linear, Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=1000, tol=1e-2)),
    "KNN": (pre_linear, KNeighborsRegressor(n_neighbors=15, n_jobs=4)),
    "DecisionTree": (pre_tree, DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
    "RandomForest": (pre_tree, RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=4, random_state=RANDOM_STATE)),
    "ExtraTrees": (pre_tree, ExtraTreesRegressor(n_estimators=60, max_depth=12, n_jobs=4, random_state=RANDOM_STATE)),
    "GradientBoosting": (pre_tree, GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)),
    "LightGBM": (pre_tree, lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, n_jobs=4, random_state=RANDOM_STATE, verbosity=-1)),
    "XGBoost": (pre_tree, xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method="hist", n_jobs=4, random_state=RANDOM_STATE)),
}

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
results = []
X_tr_sub, y_tr_sub = tr_sub[feature_cols], tr_sub[target_col]
X_va, y_va = va[feature_cols], va[target_col]

for name, (pre, model) in MODELS.items():
    t1 = time.time()
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_tr_sub, y_tr_sub)
    pred = pipe.predict(X_va)
    rmse = float(np.sqrt(np.mean((y_va.values - pred) ** 2)))
    mae = float(np.mean(np.abs(y_va.values - pred)))
    dt = time.time() - t1
    results.append({"model": name, "rmse": rmse, "mae": mae, "fit_predict_s": dt})
    print(f"{elapsed()}   {name:<18} RMSE={rmse:7.3f}  MAE={mae:7.3f}  ({dt:5.1f}s)")

results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
results_df.to_csv(OUT / "model_comparison.csv", index=False)
top3 = results_df.head(3)["model"].tolist()
print(f"{elapsed()} MODEL COMPARISON DONE — top 3: {top3}")
print(results_df.to_string(index=False))

# ============================================================================
# 7. GRID-SEARCH TUNING (best model, small grid, TimeSeriesSplit, on subsample)
# ============================================================================
PARAM_GRIDS = {
    "LightGBM": {"model__num_leaves": [31, 63], "model__learning_rate": [0.05, 0.08]},
    "XGBoost": {"model__max_depth": [4, 6], "model__learning_rate": [0.05, 0.08]},
    "RandomForest": {"model__max_depth": [10, 16], "model__min_samples_leaf": [1, 4]},
    "ExtraTrees": {"model__max_depth": [10, 16], "model__min_samples_leaf": [1, 4]},
    "GradientBoosting": {"model__max_depth": [2, 3], "model__learning_rate": [0.05, 0.1]},
    "DecisionTree": {"model__max_depth": [6, 12], "model__min_samples_leaf": [1, 4]},
    "Ridge": {"model__alpha": [0.1, 1.0, 10.0]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1]},
    "KNN": {"model__n_neighbors": [5, 15, 30]},
    "LinearRegression": {},
}

# tuning runs on a smaller sub-subsample than the 10-model comparison, purely to
# keep GridSearchCV's fit count cheap under the time budget
tune_sub = tr_sub.groupby("nama_pos", observed=True, group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), 250), random_state=RANDOM_STATE))
X_tune, y_tune = tune_sub[feature_cols], tune_sub[target_col]
print(f"{elapsed()} tuning subsample: {len(tune_sub)} rows")

# tune all top-3 candidates (not just #1) since the #1 model by RMSE may have no
# meaningful hyperparameters to search — pick whichever tuned model validates best
tuned_candidates = []
for name in top3:
    pre, model = MODELS[name]
    grid = PARAM_GRIDS.get(name, {})
    t2 = time.time()
    if grid:
        pipe = Pipeline([("pre", pre), ("model", model)])
        tscv = TimeSeriesSplit(n_splits=2)
        gs = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=1)
        gs.fit(X_tune, y_tune)
        # refit the tuned hyperparameters on the full comparison subsample
        candidate_pipe = Pipeline([("pre", pre), ("model", gs.best_estimator_.named_steps["model"])])
        candidate_pipe.fit(X_tr_sub, y_tr_sub)
        print(f"{elapsed()} grid search on {name}: best_params={gs.best_params_}  "
              f"cv_rmse={-gs.best_score_:.4f}  ({time.time()-t2:.1f}s)")
    else:
        candidate_pipe = Pipeline([("pre", pre), ("model", model)])
        candidate_pipe.fit(X_tr_sub, y_tr_sub)
        print(f"{elapsed()} no grid for {name} (no tunable params), using defaults")

    pred = candidate_pipe.predict(X_va)
    rmse = float(np.sqrt(np.mean((y_va.values - pred) ** 2)))
    mae = float(np.mean(np.abs(y_va.values - pred)))
    tuned_candidates.append({"name": name, "pipe": candidate_pipe, "rmse": rmse, "mae": mae})
    print(f"{elapsed()}   tuned {name} on validation: RMSE={rmse:.4f}  MAE={mae:.4f}")

best_candidate = min(tuned_candidates, key=lambda c: c["rmse"])
best_name = best_candidate["name"]
tuned_pipe = best_candidate["pipe"]
tuned_val_rmse, tuned_val_mae = best_candidate["rmse"], best_candidate["mae"]
print(f"{elapsed()} SELECTED: {best_name}  (val RMSE={tuned_val_rmse:.4f}, MAE={tuned_val_mae:.4f})")

tuned_summary = pd.DataFrame([{"model": c["name"], "tuned_rmse": c["rmse"], "tuned_mae": c["mae"]} for c in tuned_candidates])
tuned_summary.to_csv(OUT / "tuning_summary.csv", index=False)

# ============================================================================
# 8. FINAL FIT ON FULL TRAIN DATA
# ============================================================================
train_full = train.dropna(subset=[f"lag_{max(LAGS)}"])
final_pipe = tuned_pipe
final_pipe.fit(train_full[feature_cols], train_full[target_col])
joblib.dump(final_pipe, OUT / "final_model.pkl")
print(f"{elapsed()} final model ({best_name}, tuned) fit on {len(train_full)} rows and saved")

# ============================================================================
# 9. VECTORIZED RECURSIVE FORECAST (batch across stations per timestep)
# ============================================================================
history = {pos: g.sort_values("datetime")[["datetime", target_col]].values.tolist()
           for pos, g in train[["nama_pos", "datetime", target_col]].groupby("nama_pos", observed=True)}

test = test.sort_values(["datetime", "nama_pos"]).reset_index(drop=True)
non_lag_static_cols = [c for c in feature_cols if c not in
                       [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
                       + [f"roll_std_{w}" for w in ROLL_WINDOWS] + ["tma_diff_1", "tma_diff_1d", "tma_diff_7d"]]

unique_times = sorted(test["datetime"].unique())
preds = np.empty(len(test))
test_by_time = {t: idx.tolist() for t, idx in test.groupby("datetime").groups.items()}

t3 = time.time()
for t_ in unique_times:
    idxs = test_by_time[t_]
    rows = test.loc[idxs]
    feat_rows = []
    for _, row in rows.iterrows():
        pos = row["nama_pos"]
        vals = [v for _, v in history[pos]]
        feat = {c: row[c] for c in non_lag_static_cols}
        n = len(vals)
        for lag in LAGS:
            feat[f"lag_{lag}"] = vals[n - lag] if n >= lag else np.nan
        for w in ROLL_WINDOWS:
            window_vals = vals[max(0, n - w):n]
            feat[f"roll_mean_{w}"] = np.mean(window_vals) if window_vals else np.nan
            feat[f"roll_std_{w}"] = np.std(window_vals, ddof=1) if len(window_vals) > 1 else np.nan
        feat["tma_diff_1"] = feat["lag_1"] - feat["lag_2"] if n >= 2 else np.nan
        feat["tma_diff_1d"] = feat["lag_1"] - feat["lag_3"] if n >= 3 else np.nan
        feat["tma_diff_7d"] = feat["lag_1"] - feat["lag_21"] if n >= 21 else np.nan
        feat_rows.append(feat)

    X_step = pd.DataFrame(feat_rows)[feature_cols]
    for c in cat_cols:
        X_step[c] = X_step[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
    step_pred = final_pipe.predict(X_step)

    for local_i, (idx, pos) in enumerate(zip(idxs, rows["nama_pos"])):
        preds[test.index.get_loc(idx)] = step_pred[local_i]
        history[pos].append((t_, step_pred[local_i]))

print(f"{elapsed()} recursive forecast over {len(unique_times)} timesteps done ({time.time()-t3:.1f}s)")

test["tma_mdpl"] = preds
submission = sample_sub[["id"]].merge(test[["id", "tma_mdpl"]], on="id", how="left")
submission.to_csv(OUT / "submission.csv", index=False)
print(f"{elapsed()} submission saved — missing={submission['tma_mdpl'].isna().sum()}")
print(submission.head())

print(f"\n{elapsed()} TOTAL PIPELINE TIME: {time.time()-T0:.1f}s ({(time.time()-T0)/60:.2f} min)")
