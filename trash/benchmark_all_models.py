"""
Consolidated benchmark across all non-deep-learning models from the user's
preference list, plus the original 10-model comparison set, scored uniformly
with R2, RMSE, MAE, MAPE on the same chronological holdout used throughout
this project (last 60 days of train).

Deep-learning / transformer models (TFT, TiDE, N-HiTS, N-BEATS, PatchTST,
TimeXer, Informer, Autoformer, FEDformer, DLinear, iTransformer, LSTM/GRU)
are explicitly OUT OF SCOPE per user decision (CPU-only torch + demonstrated
environment instability made a fair 11-model DL comparison impractical here).
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                               HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

T0 = time.time()
def elapsed():
    return f"[{time.time()-T0:6.1f}s]"

ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"
RANDOM_STATE = 42

# ============================================================================
# 1. REBUILD FEATURES (same pipeline as build_model_v2.py)
# ============================================================================
train = pd.read_csv(ROOT / "train.csv", parse_dates=["datetime"])
test_raw = pd.read_csv(ROOT / "test.csv")
env = pd.read_csv(ROOT / "data_pendukung" / "data_lingkungan.csv", parse_dates=["datetime"])
coord = pd.read_csv(ROOT / "data_pendukung" / "koordinat_pos.csv")

split_id = test_raw["id"].str.split(" - ", n=1, expand=True)
test = pd.DataFrame({"id": test_raw["id"], "datetime": pd.to_datetime(split_id[0]), "nama_pos": split_id[1]})

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
train["landcover_name"] = train["nama_pos"].map(lc)
train = train.merge(coord, on="nama_pos", how="left")
print(f"{elapsed()} features rebuilt: {train.shape}")

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
station_stats = train.groupby("nama_pos")["tma_mdpl"].agg(["mean", "std"]).rename(
    columns={"mean": "station_mean_tma", "std": "station_std_tma"})
train = train.merge(station_stats, on="nama_pos", how="left")

cat_cols = ["nama_pos", "landcover_name"]
for c in cat_cols:
    train[c] = train[c].astype("category")

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

cutoff = train["datetime"].max() - pd.Timedelta(days=60)
tr = train[train["datetime"] <= cutoff].dropna(subset=[f"lag_{max(LAGS)}"])
va = train[train["datetime"] > cutoff]
tr_sub = tr.groupby("nama_pos", observed=True, group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), 600), random_state=RANDOM_STATE))
print(f"{elapsed()} tr_sub={len(tr_sub)}  va={len(va)}")

X_tr_sub, y_tr_sub = tr_sub[feature_cols], tr_sub[target_col]
X_va, y_va = va[feature_cols], va[target_col]

# ============================================================================
# 2. TABULAR MODELS
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
    "ElasticNet": (pre_linear, ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=1000, tol=1e-2)),
    "KNN": (pre_linear, KNeighborsRegressor(n_neighbors=15, n_jobs=4)),
    "DecisionTree": (pre_tree, DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
    "RandomForest": (pre_tree, RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=4, random_state=RANDOM_STATE)),
    "ExtraTrees": (pre_tree, ExtraTreesRegressor(n_estimators=60, max_depth=12, n_jobs=4, random_state=RANDOM_STATE)),
    "GradientBoosting": (pre_tree, GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)),
    "HistGradientBoosting": (pre_tree, HistGradientBoostingRegressor(max_iter=200, max_depth=8, learning_rate=0.08, random_state=RANDOM_STATE)),
    "LightGBM": (pre_tree, lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, n_jobs=4, random_state=RANDOM_STATE, verbosity=-1)),
    "XGBoost": (pre_tree, xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method="hist", n_jobs=4, random_state=RANDOM_STATE)),
    "CatBoost": (pre_tree, cb.CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8, random_state=RANDOM_STATE, verbose=False, thread_count=4)),
}

def safe_mape(y_true, y_pred, eps=1e-2):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

results = []
for name, (pre, model) in MODELS.items():
    t1 = time.time()
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_tr_sub, y_tr_sub)
    pred = pipe.predict(X_va)
    rmse = float(np.sqrt(np.mean((y_va.values - pred) ** 2)))
    mae = float(mean_absolute_error(y_va.values, pred))
    r2 = float(r2_score(y_va.values, pred))
    mape = safe_mape(y_va.values, pred)
    dt = time.time() - t1
    results.append({"model": name, "type": "tabular", "r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "fit_predict_s": dt})
    print(f"{elapsed()}   {name:<22} R2={r2:6.3f}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  MAPE={mape:7.3f}  ({dt:5.1f}s)")

pd.DataFrame(results).to_csv(OUT / "benchmark_tabular.csv", index=False)
print(f"{elapsed()} tabular models done")

# ============================================================================
# 3. SARIMAX (per-station univariate, pooled metrics)
# ============================================================================
from statsmodels.tsa.statespace.sarimax import SARIMAX

t1 = time.time()
sarimax_true, sarimax_pred = [], []
for pos, g in train.groupby("nama_pos", observed=True):
    g = g.sort_values("datetime")
    g_tr = g[g["datetime"] <= cutoff]
    g_va = g[g["datetime"] > cutoff]
    if len(g_va) == 0 or len(g_tr) < 30:
        continue
    try:
        mod = SARIMAX(g_tr[target_col].values, order=(2, 1, 1), seasonal_order=(1, 0, 1, 3),
                       enforce_stationarity=False, enforce_invertibility=False)
        fit = mod.fit(disp=False, maxiter=50)
        fc = fit.forecast(steps=len(g_va))
        sarimax_true.extend(g_va[target_col].values.tolist())
        sarimax_pred.extend(np.asarray(fc).tolist())
    except Exception as e:
        print(f"{elapsed()}   SARIMAX failed for {pos}: {e}")

sarimax_true = np.array(sarimax_true)
sarimax_pred = np.array(sarimax_pred)
rmse = float(np.sqrt(np.mean((sarimax_true - sarimax_pred) ** 2)))
mae = float(mean_absolute_error(sarimax_true, sarimax_pred))
r2 = float(r2_score(sarimax_true, sarimax_pred))
mape = safe_mape(sarimax_true, sarimax_pred)
dt = time.time() - t1
results.append({"model": "SARIMAX", "type": "statistical", "r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "fit_predict_s": dt})
print(f"{elapsed()}   {'SARIMAX':<22} R2={r2:6.3f}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  MAPE={mape:7.3f}  ({dt:5.1f}s, 30 stations)")

pd.DataFrame(results).to_csv(OUT / "benchmark_all.csv", index=False)
print(f"\n{elapsed()} TOTAL BENCHMARK TIME: {time.time()-T0:.1f}s ({(time.time()-T0)/60:.2f} min)")
print(pd.DataFrame(results).sort_values("rmse").to_string(index=False))
