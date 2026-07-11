"""
End-to-end model for SSDS water level (tma_mdpl) prediction.

Pipeline:
1. Load train/test/env/coord data.
2. Merge hourly environmental features onto the 6-hourly (06/12/18) observation grid
   by aggregating the environmental data over the preceding window.
3. Feature engineering: calendar features, station encoding, lag/rolling features of
   tma_mdpl built from train history.
4. Time-based validation split to estimate generalization error.
5. Train a LightGBM regressor on the full train set.
6. Recursively predict the test period (chronological, per station) since test dates
   are in the future relative to train but env/weather data already covers them.
7. Write submission.csv in the same format as sample_submission.csv.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
train = pd.read_csv(ROOT / "train.csv", parse_dates=["datetime"])
test_raw = pd.read_csv(ROOT / "test.csv")
sample_sub = pd.read_csv(ROOT / "sample_submission.csv")
env = pd.read_csv(ROOT / "data_pendukung" / "data_lingkungan.csv", parse_dates=["datetime"])
coord = pd.read_csv(ROOT / "data_pendukung" / "koordinat_pos.csv")

# split test id "YYYY-mm-dd HH:MM:SS - nama_pos" -> datetime, nama_pos
split_id = test_raw["id"].str.split(" - ", n=1, expand=True)
test = pd.DataFrame({
    "id": test_raw["id"],
    "datetime": pd.to_datetime(split_id[0]),
    "nama_pos": split_id[1],
})

print(f"train: {train.shape}, test: {test.shape}, env: {env.shape}")

# ----------------------------------------------------------------------------
# 2. Aggregate hourly env data into the 6-hour windows ending at each obs time
#    (06:00 -> [prev 18:00+1h .. 06:00], i.e. preceding 6 hours window incl. endpoint)
# ----------------------------------------------------------------------------
env = env.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)

num_cols = [c for c in env.columns if c not in ("datetime", "nama_pos", "landcover_name")]

def build_env_windows(env_df, target_times):
    """For each (nama_pos, datetime) in target_times, aggregate env_df over the
    preceding 6-hour window (mean/max/sum where relevant)."""
    env_df = env_df.set_index("datetime")
    results = []
    for pos, grp_times in target_times.groupby("nama_pos"):
        sub = env_df[env_df["nama_pos"] == pos]
        sub_num = sub[num_cols]
        for dt in grp_times["datetime"].unique():
            window = sub_num.loc[(sub_num.index > dt - pd.Timedelta(hours=6)) & (sub_num.index <= dt)]
            if window.empty:
                agg = pd.Series({c: np.nan for c in num_cols})
            else:
                agg = window.mean()
                if "rainfall_mm" in window:
                    agg["rainfall_mm"] = window["rainfall_mm"].sum()
                if "rainfall_openmeteo_mm" in window:
                    agg["rainfall_openmeteo_mm"] = window["rainfall_openmeteo_mm"].sum()
            agg["nama_pos"] = pos
            agg["datetime"] = dt
            results.append(agg)
    out = pd.DataFrame(results)
    return out

# This is efficient because merge_asof groups by station too; use it instead of python loop.
def build_env_windows_fast(env_df, target_times):
    frames = []
    env_df = env_df.sort_values("datetime")
    for pos in target_times["nama_pos"].unique():
        e = env_df[env_df["nama_pos"] == pos].sort_values("datetime").set_index("datetime")
        t = sorted(target_times.loc[target_times["nama_pos"] == pos, "datetime"].unique())
        rolled = e[num_cols].rolling("6h", min_periods=1).agg("mean")
        rain_sum = e[["rainfall_mm", "rainfall_openmeteo_mm"]].rolling("6h", min_periods=1).sum()
        rolled["rainfall_mm"] = rain_sum["rainfall_mm"]
        rolled["rainfall_openmeteo_mm"] = rain_sum["rainfall_openmeteo_mm"]
        sel = rolled.reindex(t, method="nearest", tolerance=pd.Timedelta("1h"))
        sel["nama_pos"] = pos
        sel["datetime"] = sel.index
        frames.append(sel.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)

print("Aggregating environmental features for train timestamps...")
train_env = build_env_windows_fast(env, train[["nama_pos", "datetime"]])
print("Aggregating environmental features for test timestamps...")
test_env = build_env_windows_fast(env, test[["nama_pos", "datetime"]])

train = train.merge(train_env, on=["nama_pos", "datetime"], how="left")
test = test.merge(test_env, on=["nama_pos", "datetime"], how="left")

# static landcover_name (take latest known per station)
lc = env.sort_values("datetime").groupby("nama_pos")["landcover_name"].last()
train["landcover_name"] = train["nama_pos"].map(lc)
test["landcover_name"] = test["nama_pos"].map(lc)

# coordinates
train = train.merge(coord, on="nama_pos", how="left")
test = test.merge(coord, on="nama_pos", how="left")

# ----------------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------------
def add_calendar_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df

train = add_calendar_features(train)
test = add_calendar_features(test)

cat_cols = ["nama_pos", "landcover_name"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))

# lag/rolling features of tma_mdpl, computed per-station in chronological order
train = train.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)

LAGS = [1, 2, 3, 4, 8, 28]  # steps of 6h: 1=6h,4=1d,28=7d
ROLL_WINDOWS = [4, 28]  # 1 day, 7 days

def add_lag_features(df, lags, rolls, target_col="tma_mdpl"):
    df = df.copy()
    grp = df.groupby("nama_pos")[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)
    for w in rolls:
        shifted = grp.shift(1)
        df[f"roll_mean_{w}"] = shifted.groupby(df["nama_pos"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = shifted.groupby(df["nama_pos"]).transform(lambda s: s.rolling(w, min_periods=1).std())
    return df

train = add_lag_features(train, LAGS, ROLL_WINDOWS)

feature_cols = (
    ["nama_pos", "landcover_name", "latitude", "longitude",
     "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    + num_cols
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
)
feature_cols = [c for c in feature_cols if c in train.columns]
target_col = "tma_mdpl"

# ----------------------------------------------------------------------------
# 4. Time-based validation split (last 60 days of train as validation)
# ----------------------------------------------------------------------------
cutoff = train["datetime"].max() - pd.Timedelta(days=60)
tr = train[train["datetime"] <= cutoff]
va = train[train["datetime"] > cutoff]
print(f"Train rows: {len(tr)}, Val rows: {len(va)} (cutoff={cutoff})")

lgb_params = dict(
    objective="regression",
    metric="rmse",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbosity=-1,
)

model_val = lgb.LGBMRegressor(**lgb_params)
model_val.fit(
    tr[feature_cols], tr[target_col],
    eval_set=[(va[feature_cols], va[target_col])],
    eval_metric="rmse",
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
)
val_pred = model_val.predict(va[feature_cols])
rmse = np.sqrt(np.mean((va[target_col].values - val_pred) ** 2))
mae = np.mean(np.abs(va[target_col].values - val_pred))
print(f"Validation RMSE: {rmse:.4f}  MAE: {mae:.4f}  (best_iter={model_val.best_iteration_})")

# ----------------------------------------------------------------------------
# 5. Train final model on all train data
# ----------------------------------------------------------------------------
final_params = dict(lgb_params)
final_params["n_estimators"] = model_val.best_iteration_ or lgb_params["n_estimators"]
model = lgb.LGBMRegressor(**final_params)
model.fit(train[feature_cols], train[target_col], categorical_feature=cat_cols)

import joblib
joblib.dump(model, OUT / "lgb_model.pkl")
print(f"Model saved to {OUT / 'lgb_model.pkl'}")

# ----------------------------------------------------------------------------
# 6. Recursive prediction over the test period (per station, chronological)
# ----------------------------------------------------------------------------
history = train[["nama_pos", "datetime", target_col]].copy()

test = test.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)
test["pred"] = np.nan

for pos in test["nama_pos"].cat.categories if hasattr(test["nama_pos"], "cat") else test["nama_pos"].unique():
    hist_pos = history[history["nama_pos"] == pos].sort_values("datetime")
    hist_vals = list(hist_pos[target_col].values)
    hist_times = list(hist_pos["datetime"].values)

    test_idx = test.index[test["nama_pos"] == pos]
    test_pos = test.loc[test_idx].sort_values("datetime")

    for idx, row in test_pos.iterrows():
        # build lag features from history (train history + previously predicted test steps)
        feat = {c: row[c] for c in feature_cols if c not in
                [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS] + [f"roll_std_{w}" for w in ROLL_WINDOWS]}
        n = len(hist_vals)
        for lag in LAGS:
            feat[f"lag_{lag}"] = hist_vals[n - lag] if n >= lag else np.nan
        for w in ROLL_WINDOWS:
            window_vals = hist_vals[max(0, n - w):n]
            feat[f"roll_mean_{w}"] = np.mean(window_vals) if window_vals else np.nan
            feat[f"roll_std_{w}"] = np.std(window_vals, ddof=1) if len(window_vals) > 1 else np.nan

        X = pd.DataFrame([feat])[feature_cols]
        for c in cat_cols:
            X[c] = X[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
        pred = model.predict(X)[0]

        test.loc[idx, "pred"] = pred
        hist_vals.append(pred)
        hist_times.append(row["datetime"])

# ----------------------------------------------------------------------------
# 7. Write submission
# ----------------------------------------------------------------------------
submission = test[["id", "pred"]].rename(columns={"pred": "tma_mdpl"})
submission = sample_sub[["id"]].merge(submission, on="id", how="left")
submission.to_csv(OUT / "submission.csv", index=False)
print(f"Submission saved to {OUT / 'submission.csv'}")
print(submission.head())
print("Missing predictions:", submission["tma_mdpl"].isna().sum())
