"""
DIAGNOSIS + FIX for the validation(0.19) vs leaderboard(1.89) gap.

Hypothesis: the 0.1885 validation RMSE was measured with TEACHER FORCING
(lag features = true historical tma_mdpl), but the test submission is generated
by RECURSIVE forecasting (lag features = the model's own prior predictions).
Over 726 steps the autoregressive error compounds, so real test RMSE >> 0.19.

This script measures, on the SAME 60-day holdout:
  [A] teacher-forced RMSE of the lag-based ensemble       (the optimistic 0.19)
  [B] recursive RMSE of the same lag-based ensemble       (honest; should ~match 1.89)
  [C] direct RMSE of a NO-TARGET-LAG model                (honest; no accumulation)
      -> candidate fix, because for a model without target lags, recursive == direct
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
def el(): return f"[{time.time()-T0:6.1f}s]"

ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"
RS = 42

# ---------- feature build (train only; identical recipe) ----------
train = pd.read_csv(ROOT / "train.csv", parse_dates=["datetime"])
env = pd.read_csv(ROOT / "data_pendukung" / "data_lingkungan.csv", parse_dates=["datetime"])
coord = pd.read_csv(ROOT / "data_pendukung" / "koordinat_pos.csv")

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

ef = build_env_features(env, train[["nama_pos", "datetime"]])
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
train["landcover_name"] = train["nama_pos"].map(lc)
train = train.merge(coord, on="nama_pos", how="left")

def cal(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour; df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear; df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_wet_season"] = df["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    for c, p in [("hour", 24), ("month", 12), ("day_of_year", 365.25)]:
        df[f"{c}_sin"] = np.sin(2*np.pi*df[c]/p); df[f"{c}_cos"] = np.cos(2*np.pi*df[c]/p)
    return df.rename(columns={"day_of_year_sin": "doy_sin", "day_of_year_cos": "doy_cos"})

train = cal(train)
ss = train.groupby("nama_pos")["tma_mdpl"].agg(["mean", "std"]).rename(columns={"mean": "station_mean_tma", "std": "station_std_tma"})
train = train.merge(ss, on="nama_pos", how="left")
cat_cols = ["nama_pos", "landcover_name"]
for c in cat_cols: train[c] = train[c].astype("category")

LAGS = [1, 2, 3, 6, 9, 21]; ROLLW = [3, 21]
train = train.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)
def add_lags(df):
    df = df.copy(); g = df.groupby("nama_pos", observed=True)["tma_mdpl"]
    for l in LAGS: df[f"lag_{l}"] = g.shift(l)
    sh = g.shift(1)
    for w in ROLLW:
        df[f"roll_mean_{w}"] = sh.groupby(df["nama_pos"], observed=True).transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = sh.groupby(df["nama_pos"], observed=True).transform(lambda s: s.rolling(w, min_periods=1).std())
    df["tma_diff_1"] = df["lag_1"] - df["lag_2"]; df["tma_diff_1d"] = df["lag_1"] - df["lag_3"]; df["tma_diff_7d"] = df["lag_1"] - df["lag_21"]
    return df
train = add_lags(train)

exog_cols = (["nama_pos", "landcover_name", "latitude", "longitude",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    + MEAN_COLS + ["rainfall_mm", "rainfall_openmeteo_mm"]
    + [f"rainfall_sum_{l}" for l in ["1d", "3d", "7d", "14d"]]
    + ["pressure_trend", "soil_moisture_trend", "day_of_week", "is_weekend", "is_wet_season",
       "station_mean_tma", "station_std_tma"])
exog_cols = [c for c in dict.fromkeys(exog_cols) if c in train.columns]
lag_cols = [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLLW] + [f"roll_std_{w}" for w in ROLLW] + ["tma_diff_1", "tma_diff_1d", "tma_diff_7d"]
full_cols = exog_cols + lag_cols
tgt = "tma_mdpl"

cutoff = train["datetime"].max() - pd.Timedelta(days=60)
tr = train[train["datetime"] <= cutoff].dropna(subset=[f"lag_{max(LAGS)}"])
va = train[train["datetime"] > cutoff]
print(f"{el()} tr={len(tr)} va={len(va)}")

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b): return float(np.mean(np.abs((a - b) / np.maximum(np.abs(a), 1e-2))))

def make_pre(cols):
    ncols = [c for c in cols if c not in cat_cols]
    tree = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
                              ("num", SimpleImputer(strategy="median"), ncols)])
    lin = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                             ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), ncols)])
    return tree, lin

# ============================================================
# [A] + [B]: lag-based ensemble (current best), teacher-forced vs recursive
# ============================================================
pre_tree_full, pre_lin_full = make_pre(full_cols)
tr_sub = tr.groupby("nama_pos", observed=True, group_keys=False).apply(lambda g: g.sample(n=min(len(g), 600), random_state=RS))
rf = Pipeline([("pre", pre_tree_full), ("m", RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=4, random_state=RS))]).fit(tr_sub[full_cols], tr_sub[tgt])
lr = Pipeline([("pre", pre_lin_full), ("m", LinearRegression())]).fit(tr[full_cols], tr[tgt])

# [A] teacher forced: use the real precomputed lag features in va
pred_tf = (rf.predict(va[full_cols]) + lr.predict(va[full_cols])) / 2
print(f"{el()} [A] TEACHER-FORCED  RMSE={rmse(va[tgt].values, pred_tf):.4f}  (the optimistic number we reported)")

# [B] recursive: rebuild lags from predictions, chronological per station
hist = {p: g.sort_values("datetime")[["datetime", tgt]].values.tolist()
        for p, g in train[train["datetime"] <= cutoff][["nama_pos", "datetime", tgt]].groupby("nama_pos", observed=True)}
va_sorted = va.sort_values(["datetime", "nama_pos"]).reset_index(drop=True)
non_lag = [c for c in full_cols if c not in lag_cols]
times = sorted(va_sorted["datetime"].unique())
by_time = {t: idx.tolist() for t, idx in va_sorted.groupby("datetime").groups.items()}
rec_pred = np.empty(len(va_sorted)); rec_true = va_sorted[tgt].values
for t_ in times:
    idxs = by_time[t_]; rows = va_sorted.loc[idxs]; frs = []
    for _, row in rows.iterrows():
        p = row["nama_pos"]; vals = [v for _, v in hist[p]]; n = len(vals)
        f = {c: row[c] for c in non_lag}
        for l in LAGS: f[f"lag_{l}"] = vals[n-l] if n >= l else np.nan
        for w in ROLLW:
            wv = vals[max(0, n-w):n]
            f[f"roll_mean_{w}"] = np.mean(wv) if wv else np.nan
            f[f"roll_std_{w}"] = np.std(wv, ddof=1) if len(wv) > 1 else np.nan
        f["tma_diff_1"] = f["lag_1"]-f["lag_2"] if n >= 2 else np.nan
        f["tma_diff_1d"] = f["lag_1"]-f["lag_3"] if n >= 3 else np.nan
        f["tma_diff_7d"] = f["lag_1"]-f["lag_21"] if n >= 21 else np.nan
        frs.append(f)
    Xs = pd.DataFrame(frs)[full_cols]
    for c in cat_cols: Xs[c] = Xs[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
    sp = (rf.predict(Xs) + lr.predict(Xs)) / 2
    for li, (idx, p) in enumerate(zip(idxs, rows["nama_pos"])):
        rec_pred[va_sorted.index.get_loc(idx)] = sp[li]; hist[p].append((t_, sp[li]))
print(f"{el()} [B] RECURSIVE       RMSE={rmse(rec_true, rec_pred):.4f}  (honest — should resemble the 1.89 leaderboard)")

# ============================================================
# [C]: NO-TARGET-LAG model (direct; recursive==direct so honest by construction)
# ============================================================
pre_tree_ex, pre_lin_ex = make_pre(exog_cols)
tr_all = train[train["datetime"] <= cutoff]  # no lag dropna needed
rf_ex = Pipeline([("pre", pre_tree_ex), ("m", RandomForestRegressor(n_estimators=200, max_depth=16, n_jobs=4, random_state=RS))]).fit(tr_all[exog_cols], tr_all[tgt])
hgb_ex = Pipeline([("pre", pre_tree_ex), ("m", HistGradientBoostingRegressor(max_iter=400, max_depth=None, learning_rate=0.05, random_state=RS))]).fit(tr_all[exog_cols], tr_all[tgt])
ridge_ex = Pipeline([("pre", pre_lin_ex), ("m", Ridge(alpha=1.0, random_state=RS))]).fit(tr_all[exog_cols], tr_all[tgt])
for name, mdl in [("RF", rf_ex), ("HistGB", hgb_ex), ("Ridge", ridge_ex)]:
    pr = mdl.predict(va[exog_cols])
    print(f"{el()} [C] no-lag {name:<7} DIRECT RMSE={rmse(va[tgt].values, pr):.4f}  MAE={mean_absolute_error(va[tgt].values, pr):.4f}  MAPE={mape(va[tgt].values, pr):.4f}")
pr_ens = (rf_ex.predict(va[exog_cols]) + hgb_ex.predict(va[exog_cols]) + ridge_ex.predict(va[exog_cols])) / 3
print(f"{el()} [C] no-lag ENSEMBLE DIRECT RMSE={rmse(va[tgt].values, pr_ens):.4f}")

print(f"\n{el()} DONE ({time.time()-T0:.1f}s)")
