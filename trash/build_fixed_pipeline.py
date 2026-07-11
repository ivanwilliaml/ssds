"""
FIXED end-to-end pipeline — eliminates the two root causes of the 0.19->1.89 gap:

  1. NO autoregressive target lags  -> zero recursive error accumulation.
     The model predicts every test timestamp DIRECTLY and independently, so the
     error at step 726 is no worse than at step 1.
  2. Non-recursive SEASONAL signal instead of lag_1:
       - seasonal_lag_1y  : the station's actual tma ~365 days before the target
                            time (for the ENTIRE test window, t-365d lies inside
                            train, so it is fully observed, never predicted).
       - doy_climatology  : station x day-of-year historical mean tma (seasonal
                            profile), fully computable from train.
     Both replace the poisonous autoregressive lag with information that is real
     and available at prediction time.

Validation is measured the SAME way the submission is produced (direct, no teacher
forcing), so the reported RMSE is honest and should track the leaderboard.
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
def el(): return f"[{time.time()-T0:6.1f}s]"

ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"
RS = 42

# ---------- load ----------
train = pd.read_csv(ROOT / "train.csv", parse_dates=["datetime"])
test_raw = pd.read_csv(ROOT / "test.csv")
sample_sub = pd.read_csv(ROOT / "sample_submission.csv")
env = pd.read_csv(ROOT / "data_pendukung" / "data_lingkungan.csv", parse_dates=["datetime"])
coord = pd.read_csv(ROOT / "data_pendukung" / "koordinat_pos.csv")
sid = test_raw["id"].str.split(" - ", n=1, expand=True)
test = pd.DataFrame({"id": test_raw["id"], "datetime": pd.to_datetime(sid[0]), "nama_pos": sid[1]})
print(f"{el()} loaded train={train.shape} test={test.shape}")

# ---------- exogenous env features (vectorized, train+test) ----------
raw_num_cols = [c for c in env.columns if c not in ("datetime", "nama_pos", "landcover_name")]
SM = ["soil_moisture_0_7cm", "soil_moisture_7_28cm", "soil_moisture_28_100cm", "soil_moisture_100_255cm"]
env = env.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)
env["soil_moisture_avg"] = env[SM].mean(axis=1)
env["wind_u"] = env["wind_speed_kmh"] * np.cos(np.deg2rad(env["wind_direction_deg"]))
env["wind_v"] = env["wind_speed_kmh"] * np.sin(np.deg2rad(env["wind_direction_deg"]))
env["temp_dew_spread"] = env["temperature_c"] - env["dew_point_c"]
env["mjo_interaction"] = env["mjo_amplitude"] * env["mjo_active"]
env["humidity_temp_interaction"] = env["humidity_pct"] * env["temperature_c"] / 100.0
MEAN_COLS = raw_num_cols + ["soil_moisture_avg", "wind_u", "wind_v", "temp_dew_spread", "mjo_interaction", "humidity_temp_interaction"]
SUM_COLS = ["rainfall_mm", "rainfall_openmeteo_mm"]

def build_env_features(env_df, target_df):
    e = env_df.sort_values(["nama_pos", "datetime"]).set_index("datetime")
    roll6 = e.groupby("nama_pos", observed=True)[MEAN_COLS].rolling("6h", min_periods=1).mean()
    rs6 = e.groupby("nama_pos", observed=True)[SUM_COLS].rolling("6h", min_periods=1).sum()
    roll6 = roll6.droplevel(0); rs6 = rs6.droplevel(0)
    roll6["nama_pos"] = e["nama_pos"].values
    roll6["rainfall_mm"] = rs6["rainfall_mm"].values
    roll6["rainfall_openmeteo_mm"] = rs6["rainfall_openmeteo_mm"].values
    for d, lab in [(1, "1d"), (3, "3d"), (7, "7d"), (14, "14d")]:
        r = e.groupby("nama_pos", observed=True)["rainfall_mm"].rolling(f"{d}d", min_periods=1).sum()
        roll6[f"rainfall_sum_{lab}"] = r.droplevel(0).values
    roll6 = roll6.reset_index().sort_values(["datetime"]).reset_index(drop=True)
    tgt = target_df.drop_duplicates().sort_values(["datetime"]).reset_index(drop=True)
    return pd.merge_asof(tgt, roll6, on="datetime", by="nama_pos", direction="nearest", tolerance=pd.Timedelta("1h"))

combined = pd.concat([train[["nama_pos", "datetime"]], test[["nama_pos", "datetime"]]], ignore_index=True).drop_duplicates()
ef = build_env_features(env, combined)
ef = ef.sort_values(["nama_pos", "datetime"])
fc = [c for c in ef.columns if c not in ("nama_pos", "datetime")]
ef[fc] = ef.groupby("nama_pos")[fc].transform(lambda s: s.ffill().bfill())
ef[fc] = ef[fc].fillna(ef[fc].median())
ef = ef.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)
ef["pressure_trend"] = ef.groupby("nama_pos")["surface_pressure_hpa"].transform(lambda s: s - s.shift(3))
ef["soil_moisture_trend"] = ef.groupby("nama_pos")["soil_moisture_avg"].transform(lambda s: s - s.shift(3))
ef[["pressure_trend", "soil_moisture_trend"]] = ef[["pressure_trend", "soil_moisture_trend"]].fillna(0)

lc = env.sort_values("datetime").groupby("nama_pos")["landcover_name"].last()
train = train.merge(ef, on=["nama_pos", "datetime"], how="left")
test = test.merge(ef, on=["nama_pos", "datetime"], how="left")
train["landcover_name"] = train["nama_pos"].map(lc); test["landcover_name"] = test["nama_pos"].map(lc)
train = train.merge(coord, on="nama_pos", how="left"); test = test.merge(coord, on="nama_pos", how="left")

def cal(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour; df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear; df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_wet_season"] = df["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24); df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12); df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["doy_sin"] = np.sin(2*np.pi*df["day_of_year"]/365.25); df["doy_cos"] = np.cos(2*np.pi*df["day_of_year"]/365.25)
    return df
train = cal(train); test = cal(test)
print(f"{el()} exogenous features built")

# ---------- NON-RECURSIVE seasonal features ----------
def add_seasonal(target_df, source_df):
    """seasonal_lag_1y (nearest actual tma ~365d before) + doy_climatology
    (station x day-of-year mean). source_df is the observed-tma table to look
    backward into; for a target at time t, t-365d always precedes the target,
    so this never leaks the target's own value."""
    out = target_df.copy()
    # seasonal_lag_1y via merge_asof on shifted time
    src = source_df[["nama_pos", "datetime", "tma_mdpl"]].sort_values("datetime").rename(columns={"tma_mdpl": "seasonal_lag_1y"})
    tmp = out[["nama_pos", "datetime"]].copy()
    tmp["seek"] = tmp["datetime"] - pd.Timedelta(days=365)
    tmp = tmp.sort_values("seek")
    merged = pd.merge_asof(tmp, src, left_on="seek", right_on="datetime", by="nama_pos",
                           direction="nearest", tolerance=pd.Timedelta(days=4), suffixes=("", "_src"))
    merged = merged.set_index(tmp.index)  # keep alignment
    out["seasonal_lag_1y"] = merged["seasonal_lag_1y"].reindex(out.index).values if False else merged.sort_index()["seasonal_lag_1y"].values
    # doy climatology
    clim = source_df.groupby(["nama_pos", source_df["datetime"].dt.dayofyear])["tma_mdpl"].mean()
    clim.index.names = ["nama_pos", "doy"]
    clim = clim.reset_index().rename(columns={"tma_mdpl": "doy_climatology"})
    out = out.merge(clim, left_on=["nama_pos", "day_of_year"], right_on=["nama_pos", "doy"], how="left").drop(columns=["doy"])
    # fallbacks
    stn_mean = source_df.groupby("nama_pos")["tma_mdpl"].mean()
    fallback = out["nama_pos"].astype(str).map(stn_mean).astype(float)
    out["doy_climatology"] = out["doy_climatology"].astype(float).fillna(fallback)
    out["seasonal_lag_1y"] = out["seasonal_lag_1y"].astype(float).fillna(out["doy_climatology"])
    return out

# station target stats (mean/std) — computed from a given source
def add_station_stats(target_df, source_df):
    ss = source_df.groupby("nama_pos")["tma_mdpl"].agg(["mean", "std"]).rename(columns={"mean": "station_mean_tma", "std": "station_std_tma"})
    return target_df.merge(ss, on="nama_pos", how="left")

feature_cols = (["nama_pos", "landcover_name", "latitude", "longitude",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    + MEAN_COLS + ["rainfall_mm", "rainfall_openmeteo_mm"]
    + [f"rainfall_sum_{l}" for l in ["1d", "3d", "7d", "14d"]]
    + ["pressure_trend", "soil_moisture_trend", "day_of_week", "is_weekend", "is_wet_season",
       "station_mean_tma", "station_std_tma", "seasonal_lag_1y", "doy_climatology"])
feature_cols = [c for c in dict.fromkeys(feature_cols)]
cat_cols = ["nama_pos", "landcover_name"]
TGT = "tma_mdpl"

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b): return float(np.mean(np.abs((a - b) / np.maximum(np.abs(a), 1e-2))))

def make_models():
    ncols = [c for c in feature_cols if c not in cat_cols]
    pre_tree = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
                                  ("num", SimpleImputer(strategy="median"), ncols)])
    pre_lin = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                 ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), ncols)])
    ridge = Pipeline([("pre", pre_lin), ("m", Ridge(alpha=1.0, random_state=RS))])
    hgb = Pipeline([("pre", pre_tree), ("m", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=None, l2_regularization=1.0, random_state=RS))])
    return ridge, hgb

# ============================================================
# VALIDATION (honest, direct) on the 60-day holdout
# ============================================================
for c in cat_cols:
    train[c] = train[c].astype("category"); test[c] = test[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))

cutoff = train["datetime"].max() - pd.Timedelta(days=60)
tr_raw = train[train["datetime"] <= cutoff].copy()
va_raw = train[train["datetime"] > cutoff].copy()

tr_v = add_station_stats(add_seasonal(tr_raw, tr_raw), tr_raw)
va_v = add_station_stats(add_seasonal(va_raw, tr_raw), tr_raw)  # seasonal/stats sourced from tr only -> no leakage
for c in cat_cols:
    tr_v[c] = tr_v[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
    va_v[c] = va_v[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))

ridge, hgb = make_models()
ridge.fit(tr_v[feature_cols], tr_v[TGT]); hgb.fit(tr_v[feature_cols], tr_v[TGT])
pr_r = ridge.predict(va_v[feature_cols]); pr_h = hgb.predict(va_v[feature_cols])
y = va_v[TGT].values
print(f"{el()} [VAL] Ridge          RMSE={rmse(y, pr_r):.4f}  MAE={mean_absolute_error(y, pr_r):.4f}")
print(f"{el()} [VAL] HistGB         RMSE={rmse(y, pr_h):.4f}  MAE={mean_absolute_error(y, pr_h):.4f}")
best_rmse, best_w = 1e9, 0.5
for w in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
    r = rmse(y, w*pr_r + (1-w)*pr_h)
    if r < best_rmse: best_rmse, best_w = r, w
    print(f"{el()} [VAL] ens w_ridge={w:.1f}   RMSE={r:.4f}")
print(f"{el()} [VAL] BEST ensemble  RMSE={best_rmse:.4f}  (w_ridge={best_w})  <-- honest, direct, no accumulation")

# ============================================================
# DEPLOY: refit on ALL train (seasonal/stats sourced from all train), predict test DIRECTLY
# ============================================================
tr_all = add_station_stats(add_seasonal(train, train), train)
test_all = add_station_stats(add_seasonal(test, train), train)
for c in cat_cols:
    tr_all[c] = tr_all[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
    test_all[c] = test_all[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))

ridge_f, hgb_f = make_models()
ridge_f.fit(tr_all[feature_cols], tr_all[TGT]); hgb_f.fit(tr_all[feature_cols], tr_all[TGT])
joblib.dump({"ridge": ridge_f, "hgb": hgb_f, "w_ridge": best_w, "features": feature_cols}, OUT / "fixed_model.pkl")

test_pred = best_w * ridge_f.predict(test_all[feature_cols]) + (1-best_w) * hgb_f.predict(test_all[feature_cols])
test_all["tma_mdpl"] = test_pred
submission = sample_sub[["id"]].merge(test_all[["id", "tma_mdpl"]], on="id", how="left")
submission.to_csv(OUT / "submission_fixed.csv", index=False)
print(f"{el()} submission_fixed.csv saved — missing={submission['tma_mdpl'].isna().sum()}  rows={len(submission)}")
print(submission.head())
print(f"\n{el()} DONE ({time.time()-T0:.1f}s) — honest validation RMSE={best_rmse:.4f} (vs old teacher-forced 0.1885 / recursive 0.5926)")
