"""Prophet per-station benchmark, appended to benchmark_all.csv."""
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

from prophet import Prophet

T0 = time.time()
def elapsed():
    return f"[{time.time()-T0:6.1f}s]"

ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"

train = pd.read_csv(ROOT / "train.csv", parse_dates=["datetime"])
cutoff = train["datetime"].max() - pd.Timedelta(days=60)

def safe_mape(y_true, y_pred, eps=1e-2):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

true_vals, pred_vals = [], []
stations = sorted(train["nama_pos"].unique())
for i, pos in enumerate(stations):
    g = train[train["nama_pos"] == pos].sort_values("datetime")
    g_tr = g[g["datetime"] <= cutoff]
    g_va = g[g["datetime"] > cutoff]
    if len(g_va) == 0 or len(g_tr) < 30:
        continue
    dfp = g_tr[["datetime", "tma_mdpl"]].rename(columns={"datetime": "ds", "tma_mdpl": "y"})
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
                changepoint_prior_scale=0.1)
    m.fit(dfp)
    future = g_va[["datetime"]].rename(columns={"datetime": "ds"})
    fc = m.predict(future)
    true_vals.extend(g_va["tma_mdpl"].values.tolist())
    pred_vals.extend(fc["yhat"].values.tolist())
    print(f"{elapsed()}   [{i+1}/{len(stations)}] {pos} done")

true_vals = np.array(true_vals)
pred_vals = np.array(pred_vals)
rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
mae = float(mean_absolute_error(true_vals, pred_vals))
r2 = float(r2_score(true_vals, pred_vals))
mape = safe_mape(true_vals, pred_vals)
dt = time.time() - T0
print(f"{elapsed()}   {'Prophet':<22} R2={r2:6.3f}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  MAPE={mape:7.3f}  ({dt:5.1f}s, {len(stations)} stations)")

all_df = pd.read_csv(OUT / "benchmark_all.csv")
all_df = all_df[all_df["model"] != "Prophet"]
new_row = pd.DataFrame([{"model": "Prophet", "type": "statistical", "r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "fit_predict_s": dt}])
all_df = pd.concat([all_df, new_row], ignore_index=True)
all_df.to_csv(OUT / "benchmark_all.csv", index=False)
print(f"{elapsed()} TOTAL PROPHET TIME: {dt:.1f}s ({dt/60:.2f} min)")
