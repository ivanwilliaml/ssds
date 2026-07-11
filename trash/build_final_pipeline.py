"""
FINAL end-to-end pipeline — deploys the best config found across every
experiment this session: RandomForest(n_estimators=60, max_depth=12) fit on
a 600/station stratified subsample, averaged 50/50 with LinearRegression fit
on the full training set. Validated RMSE = 0.1885 (best of all candidates:
15 baseline models, Optuna-tuned top-3, subsample-vs-full-train sweep, and
balanced-sample-weight experiments).

For deployment, both models are refit on ALL of train.csv (not just the
pre-cutoff portion) so the actual test forecast uses every available
historical reading, then a vectorized per-timestep recursive forecast
produces the submission.
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

T0 = time.time()
def elapsed():
    return f"[{time.time()-T0:6.1f}s]"

ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"
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
test = pd.DataFrame({"id": test_raw["id"], "datetime": pd.to_datetime(split_id[0]), "nama_pos": split_id[1]})
print(f"{elapsed()} loaded train={train.shape} test={test.shape}")

# ============================================================================
# 2. ENVIRONMENTAL FEATURES (vectorized, train+test combined timeline)
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
    e = env_df.sort_values(["nama_pos", "datetime"]).set_index("datetime")
    roll6 = e.groupby("nama_pos", observed=True)[MEAN_COLS].rolling("6h", min_periods=1).mean()
    rain_sum6 = e.groupby("nama_pos", observed=True)[SUM_COLS].rolling("6h", min_periods=1).sum()
    roll6 = roll6.droplevel(0); rain_sum6 = rain_sum6.droplevel(0)
    roll6["nama_pos"] = e["nama_pos"].values
    roll6["rainfall_mm"] = rain_sum6["rainfall_mm"].values
    roll6["rainfall_openmeteo_mm"] = rain_sum6["rainfall_openmeteo_mm"].values
    for days, label in [(1, "1d"), (3, "3d"), (7, "7d"), (14, "14d")]:
        rs = e.groupby("nama_pos", observed=True)["rainfall_mm"].rolling(f"{days}d", min_periods=1).sum()
        roll6[f"rainfall_sum_{label}"] = rs.droplevel(0).values
    roll6 = roll6.reset_index().sort_values(["datetime"]).reset_index(drop=True)
    tgt = target_df.drop_duplicates().sort_values(["datetime"]).reset_index(drop=True)
    return pd.merge_asof(tgt, roll6, on="datetime", by="nama_pos", direction="nearest", tolerance=pd.Timedelta("1h"))

combined_targets = pd.concat([train[["nama_pos", "datetime"]], test[["nama_pos", "datetime"]]], ignore_index=True).drop_duplicates()
env_features = build_env_features(env, combined_targets)
env_features = env_features.sort_values(["nama_pos", "datetime"])
fill_cols = [c for c in env_features.columns if c not in ("nama_pos", "datetime")]
env_features[fill_cols] = env_features.groupby("nama_pos")[fill_cols].transform(lambda s: s.ffill().bfill())
env_features[fill_cols] = env_features[fill_cols].fillna(env_features[fill_cols].median())
env_features = env_features.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)
env_features["pressure_trend"] = env_features.groupby("nama_pos")["surface_pressure_hpa"].transform(lambda s: s - s.shift(3))
env_features["soil_moisture_trend"] = env_features.groupby("nama_pos")["soil_moisture_avg"].transform(lambda s: s - s.shift(3))
env_features[["pressure_trend", "soil_moisture_trend"]] = env_features[["pressure_trend", "soil_moisture_trend"]].fillna(0)

lc = env.sort_values("datetime").groupby("nama_pos")["landcover_name"].last()
train = train.merge(env_features, on=["nama_pos", "datetime"], how="left")
test = test.merge(env_features, on=["nama_pos", "datetime"], how="left")
train["landcover_name"] = train["nama_pos"].map(lc)
test["landcover_name"] = test["nama_pos"].map(lc)
train = train.merge(coord, on="nama_pos", how="left")
test = test.merge(coord, on="nama_pos", how="left")
print(f"{elapsed()} env features merged")

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

station_stats = train.groupby("nama_pos")["tma_mdpl"].agg(["mean", "std"]).rename(
    columns={"mean": "station_mean_tma", "std": "station_std_tma"})
train = train.merge(station_stats, on="nama_pos", how="left")
test = test.merge(station_stats, on="nama_pos", how="left")

cat_cols = ["nama_pos", "landcover_name"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))

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
        df[f"roll_mean_{w}"] = shifted.groupby(df["nama_pos"], observed=True).transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = shifted.groupby(df["nama_pos"], observed=True).transform(lambda s: s.rolling(w, min_periods=1).std())
    df["tma_diff_1"] = df["lag_1"] - df["lag_2"]
    df["tma_diff_1d"] = df["lag_1"] - df["lag_3"]
    df["tma_diff_7d"] = df["lag_1"] - df["lag_21"]
    return df

train = add_lag_features(train, LAGS, ROLL_WINDOWS)

feature_cols = (
    ["nama_pos", "landcover_name", "latitude", "longitude",
     "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    + MEAN_COLS + ["rainfall_mm", "rainfall_openmeteo_mm"]
    + [f"rainfall_sum_{l}" for l in ["1d", "3d", "7d", "14d"]]
    + ["pressure_trend", "soil_moisture_trend"]
    + ["day_of_week", "is_weekend", "is_wet_season", "station_mean_tma", "station_std_tma"]
    + [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS] + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["tma_diff_1", "tma_diff_1d", "tma_diff_7d"]
)
feature_cols = [c for c in dict.fromkeys(feature_cols) if c in train.columns]
target_col = "tma_mdpl"
num_feature_cols = [c for c in feature_cols if c not in cat_cols]
print(f"{elapsed()} feature engineering done — {len(feature_cols)} features")

# ============================================================================
# 4. FINAL VALIDATION CHECK (same holdout as every prior experiment, to
# confirm the deployed config's score before committing to it)
# ============================================================================
cutoff = train["datetime"].max() - pd.Timedelta(days=60)
tr = train[train["datetime"] <= cutoff].dropna(subset=[f"lag_{max(LAGS)}"])
va = train[train["datetime"] > cutoff]
X_va, y_va = va[feature_cols], va[target_col]

pre_tree = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", SimpleImputer(strategy="median"), num_feature_cols),
])
pre_linear = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_feature_cols),
])

tr_sub600 = tr.groupby("nama_pos", observed=True, group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), 600), random_state=RANDOM_STATE))

rf_check = Pipeline([("pre", pre_tree), ("model", RandomForestRegressor(
    n_estimators=60, max_depth=12, n_jobs=4, random_state=RANDOM_STATE))])
rf_check.fit(tr_sub600[feature_cols], tr_sub600[target_col])
lr_check = Pipeline([("pre", pre_linear), ("model", LinearRegression())])
lr_check.fit(tr[feature_cols], tr[target_col])

pred_check = (rf_check.predict(X_va) + lr_check.predict(X_va)) / 2
rmse_check = float(np.sqrt(np.mean((y_va.values - pred_check) ** 2)))
mae_check = float(mean_absolute_error(y_va.values, pred_check))
r2_check = float(r2_score(y_va.values, pred_check))
print(f"{elapsed()} VALIDATION CHECK — RF_sub600+LR ensemble: RMSE={rmse_check:.4f}  MAE={mae_check:.4f}  R2={r2_check:.6f}")

# ============================================================================
# 5. DEPLOY — refit both models on ALL of train.csv (not just pre-cutoff)
# ============================================================================
train_all = train.dropna(subset=[f"lag_{max(LAGS)}"])
train_all_sub600 = train_all.groupby("nama_pos", observed=True, group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), 600), random_state=RANDOM_STATE))

rf_final = Pipeline([("pre", pre_tree), ("model", RandomForestRegressor(
    n_estimators=60, max_depth=12, n_jobs=4, random_state=RANDOM_STATE))])
rf_final.fit(train_all_sub600[feature_cols], train_all_sub600[target_col])

lr_final = Pipeline([("pre", pre_linear), ("model", LinearRegression())])
lr_final.fit(train_all[feature_cols], train_all[target_col])

joblib.dump({"rf": rf_final, "lr": lr_final}, OUT / "final_ensemble_model.pkl")
print(f"{elapsed()} deployed models fit — RF on {len(train_all_sub600)} rows, LR on {len(train_all)} rows")

# ============================================================================
# 6. VECTORIZED RECURSIVE FORECAST (batch across stations per timestep)
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
    step_pred = (rf_final.predict(X_step) + lr_final.predict(X_step)) / 2

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
print(f"{elapsed()} DEPLOYED CONFIG: RandomForest(n_estimators=60,max_depth=12) on 600/station subsample "
      f"+ LinearRegression on full data, averaged — validated RMSE={rmse_check:.4f}")
