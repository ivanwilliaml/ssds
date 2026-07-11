"""
Per-station tau grid search vs the single global tau. Also runs the
Wonogiri Dam / Jurug / Peren / Bojonegoro-Kali Kethek plateau-detection audit
(regulated / dam-controlled dynamics diagnostic) referenced in the writeup.
"""
import pandas as pd, numpy as np, warnings, time
from sklearn.metrics import mean_squared_error
import feature_lib as fl
warnings.filterwarnings('ignore')
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

RAW, TEST_REAL = fl.load_raw()
dl, dl_feat, static = fl.load_exog()
def rmse(a, b): return np.sqrt(mean_squared_error(a, b))

CUT = pd.Timestamp('2024-09-18 18:00:00'); VA_END = pd.Timestamp('2025-05-18 18:00:00')
train_only = fl.clean_dataframe(RAW[RAW.datetime <= CUT].copy())
val_only = RAW[(RAW.datetime > CUT) & (RAW.datetime <= VA_END)].copy()
tr_feat = fl.build_features(train_only, train_only, dl_feat, static)
va_feat = fl.build_features(train_only, val_only[['datetime','nama_pos']].copy(), dl_feat, static)
va_feat = va_feat.merge(val_only[['datetime','nama_pos','tma_mdpl']].rename(columns={'tma_mdpl':'y_true'}), on=['datetime','nama_pos'], how='left')
tr_feat[fl.FEATURE_COLS_NUM] = tr_feat[fl.FEATURE_COLS_NUM].fillna(0)
va_feat[fl.FEATURE_COLS_NUM] = va_feat[fl.FEATURE_COLS_NUM].fillna(0)
y_tr = tr_feat['tma_mdpl'].values

ridge, histgb = fl.make_pipelines()
ridge.fit(tr_feat[fl.XCOLS], y_tr); histgb.fit(tr_feat[fl.XCOLS], y_tr)
pred_r = ridge.predict(va_feat[fl.XCOLS]); pred_h = histgb.predict(va_feat[fl.XCOLS])
pred_direct = 0.8*pred_r + 0.2*pred_h
va_feat['pred_direct'] = pred_direct

# --- per-station error breakdown (which stations dominate error) ---
h_global = va_feat.horizon_days.values
w_global = np.exp(-h_global/20)
pred_glob = w_global*va_feat.last_known.values + (1-w_global)*va_feat.pred_direct.values
va_feat['pred_glob'] = pred_glob
va_feat['err2'] = (va_feat.y_true - va_feat.pred_glob)**2
per_stn = va_feat.groupby('nama_pos')['err2'].agg(['mean','count'])
per_stn['rmse'] = np.sqrt(per_stn['mean'])
per_stn['share'] = per_stn['mean']*per_stn['count']/va_feat.err2.sum()
per_stn['level'] = va_feat.groupby('nama_pos').y_true.mean()
log("Top-12 stations by squared-error share (global tau=20 baseline):")
print(per_stn.sort_values('share', ascending=False).head(12).to_string())

# --- per-station tau grid search ---
TAU_GRID = [5,10,15,20,25,30,40,50,60,75,90,120,150,180,240,300,1e9]
results = {}
for stn, g in va_feat.groupby('nama_pos'):
    hh = g.horizon_days.values; lk = g.last_known.values; pdv = g.pred_direct.values; yt = g.y_true.values
    best = (None, 1e9)
    for tau in TAU_GRID:
        pred = np.exp(-hh/tau)*lk + (1-np.exp(-hh/tau))*pdv
        s = rmse(yt, pred)
        if s < best[1]: best = (tau, s)
    results[stn] = best

log(f"GLOBAL tau=20 RMSE: {rmse(va_feat.y_true.values, pred_glob):.4f}")
va_feat = va_feat.reset_index(drop=True)
pred_perstn = np.zeros(len(va_feat))
for stn, (tau, s) in results.items():
    mask = (va_feat.nama_pos == stn).values
    hh = va_feat.loc[mask, 'horizon_days'].values
    w = np.exp(-hh/tau)
    pred_perstn[mask] = w*va_feat.loc[mask,'last_known'].values + (1-w)*va_feat.loc[mask,'pred_direct'].values
log(f"PER-STATION tau RMSE: {rmse(va_feat.y_true.values, pred_perstn):.4f}")
print("\nper-station best tau:")
for stn, (tau, s) in sorted(results.items(), key=lambda kv: -kv[1][1]):
    print(f"  {stn:28s} best_tau={tau:>6} rmse={s:.4f}")

# --- regulated/dam-station plateau audit ---
log("\nPlateau (flat-run) audit for the 4 top error-contributor stations:")
clean = train_only
for stn in ['Jurug', 'Peren', 'Wonogiri Dam', 'Bojonegoro - Kali Kethek']:
    s = clean[clean.nama_pos == stn].sort_values('datetime').reset_index(drop=True)
    v = s['tma_mdpl'].values
    same = np.isclose(v[1:], v[:-1], atol=0.01)
    runs = []; cur = 1
    for x in same:
        if x: cur += 1
        else: runs.append(cur); cur = 1
    runs.append(cur); runs = np.array(runs)
    decimals_ok = not np.allclose(v, np.round(v, 1))  # full precision retained -> not sensor rounding
    print(f"  {stn:28s} n_plateaus(>=3 steps)={ (runs>=3).sum():3d}  max_plateau_steps={runs.max():3d}  full_precision={decimals_ok}")
log("DONE per_station_tau_test.py")
