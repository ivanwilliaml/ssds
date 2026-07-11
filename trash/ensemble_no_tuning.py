"""
Same A/B/C ensemble comparison as optuna_ensemble.py, but with the ORIGINAL
untuned hyperparameters (the ones from benchmark_all_models.py / build_model_v2.py's
early default) fit directly on the full training set. Answers: is Optuna tuning
actually worth it, or does the untuned default already win?
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

T0 = time.time()
def elapsed():
    return f"[{time.time()-T0:6.1f}s]"

ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"
RANDOM_STATE = 42

# ============================================================================
# 1. REBUILD FEATURES (identical to optuna_ensemble.py / benchmark_all_models.py)
# ============================================================================
train = pd.read_csv(ROOT / "train.csv", parse_dates=["datetime"])
env = pd.read_csv(ROOT / "data_pendukung" / "data_lingkungan.csv", parse_dates=["datetime"])
coord = pd.read_csv(ROOT / "data_pendukung" / "koordinat_pos.csv")

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

env_features = build_env_features(env, train[["nama_pos", "datetime"]])
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
print(f"{elapsed()} tr(full)={len(tr)}  va={len(va)}")

X_tr_full, y_tr_full = tr[feature_cols], tr[target_col]
X_va, y_va = va[feature_cols], va[target_col]

pre_tree = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", SimpleImputer(strategy="median"), num_feature_cols),
])
pre_linear = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_feature_cols),
])

def rmse_of(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def safe_mape(y_true, y_pred, eps=1e-2):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

# ============================================================================
# 2. UNTUNED / DEFAULT MODELS — same params used in the original 15-model benchmark
# ============================================================================
model_a = Pipeline([("pre", pre_tree), ("model", RandomForestRegressor(
    n_estimators=60, max_depth=12, n_jobs=4, random_state=RANDOM_STATE))])
model_b = Pipeline([("pre", pre_tree), ("model", DecisionTreeRegressor(
    max_depth=10, random_state=RANDOM_STATE))])
model_c = Pipeline([("pre", pre_linear), ("model", LinearRegression())])

t1 = time.time()
model_a.fit(X_tr_full, y_tr_full)
pred_a = model_a.predict(X_va)
print(f"{elapsed()} [A untuned RandomForest] fit on full train ({time.time()-t1:.1f}s)")

t1 = time.time()
model_b.fit(X_tr_full, y_tr_full)
pred_b = model_b.predict(X_va)
print(f"{elapsed()} [B untuned DecisionTree] fit on full train ({time.time()-t1:.1f}s)")

t1 = time.time()
model_c.fit(X_tr_full, y_tr_full)
pred_c = model_c.predict(X_va)
print(f"{elapsed()} [C untuned LinearRegression] fit on full train ({time.time()-t1:.1f}s)")

# ============================================================================
# 3. INDIVIDUAL + PAIRWISE + TRIPLE ENSEMBLE COMPARISON
# ============================================================================
def score(name, pred):
    y = y_va.values
    return {
        "combo": name,
        "r2": float(r2_score(y, pred)),
        "rmse": rmse_of(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": safe_mape(y, pred),
    }

combos = {
    "A (RandomForest)": pred_a,
    "B (DecisionTree)": pred_b,
    "C (LinearRegression)": pred_c,
    "AB": (pred_a + pred_b) / 2,
    "BC": (pred_b + pred_c) / 2,
    "AC": (pred_a + pred_c) / 2,
    "ABC (bonus)": (pred_a + pred_b + pred_c) / 3,
}

results = [score(name, pred) for name, pred in combos.items()]
results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
results_df.to_csv(OUT / "ensemble_comparison_notuning.csv", index=False)
print(f"\n{elapsed()} RESULTS (no tuning, defaults):")
print(results_df.to_string(index=False))
print(f"\n{elapsed()} TOTAL TIME: {time.time()-T0:.1f}s ({(time.time()-T0)/60:.2f} min)")
