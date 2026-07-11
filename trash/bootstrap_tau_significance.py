"""
Review point #2: per-station tau (30 independent argmins over a 17-value
grid, each fit to only ~650 points) risks fitting sampling noise rather than
a genuine station-specific persistence characteristic. Bootstrap each
station's chosen tau against the global tau to see which stations show a
real, lopsided advantage vs which are indistinguishable from noise.
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
tr_feat = fl.build_features(train_only, train_only, dl_feat, static, self_fit=True)
va_feat = fl.build_features(train_only, val_only[['datetime','nama_pos']].copy(), dl_feat, static)
va_feat = va_feat.merge(val_only[['datetime','nama_pos','tma_mdpl']].rename(columns={'tma_mdpl':'y_true'}), on=['datetime','nama_pos'], how='left')
tr_feat[fl.FEATURE_COLS_NUM] = tr_feat[fl.FEATURE_COLS_NUM].fillna(0)
va_feat[fl.FEATURE_COLS_NUM] = va_feat[fl.FEATURE_COLS_NUM].fillna(0)
y_tr = tr_feat['tma_mdpl'].values

import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

ridge, histgb = fl.make_pipelines()
ridge.fit(tr_feat[fl.XCOLS], y_tr); histgb.fit(tr_feat[fl.XCOLS], y_tr)
lgb_pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), fl.CAT_COLS)], remainder='passthrough')
Xtr_lgb = lgb_pre.fit_transform(tr_feat[fl.XCOLS]); Xva_lgb = lgb_pre.transform(va_feat[fl.XCOLS])
lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=0, verbosity=-1)
lgbm.fit(Xtr_lgb, y_tr)
pred_r = ridge.predict(va_feat[fl.XCOLS]); pred_l = lgbm.predict(Xva_lgb)
# ensemble weights confirmed via multi_fold_validate_v3.py's FOLD2 selection: (ridge=0.4, histgb=0.0, lgbm=0.6)
pred_direct = 0.4*pred_r + 0.6*pred_l
va_feat['pred_direct'] = pred_direct

TAU_GRID = [5,10,15,20,25,30,40,50,60,75,90,120,150,180,240,300,1e9]
GLOBAL_TAU = 20
results = {}
for stn, g in va_feat.groupby('nama_pos'):
    hh = g.horizon_days.values; lk = g.last_known.values; pdv = g.pred_direct.values; yt = g.y_true.values
    best = (None, 1e9)
    for tau in TAU_GRID:
        pred = np.exp(-hh/tau)*lk + (1-np.exp(-hh/tau))*pdv
        s = rmse(yt, pred)
        if s < best[1]: best = (tau, s)
    results[stn] = best
log("per-station best tau computed")

rng = np.random.default_rng(0)
N_BOOT = 300
rows = []
for stn, (tau_star, s_star) in results.items():
    g = va_feat[va_feat.nama_pos == stn]
    hh, lk, pdv, yt = g.horizon_days.values, g.last_known.values, g.pred_direct.values, g.y_true.values
    n = len(g)
    wins = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        pred_star = np.exp(-hh[idx]/tau_star)*lk[idx] + (1-np.exp(-hh[idx]/tau_star))*pdv[idx]
        pred_glob = np.exp(-hh[idx]/GLOBAL_TAU)*lk[idx] + (1-np.exp(-hh[idx]/GLOBAL_TAU))*pdv[idx]
        wins += rmse(yt[idx], pred_star) < rmse(yt[idx], pred_glob)
    p_beats_global = wins / N_BOOT
    rows.append((stn, tau_star, s_star, p_beats_global))

res = pd.DataFrame(rows, columns=['station','tau_star','rmse_star','p_beats_global']).sort_values('p_beats_global', ascending=False)
print(res.to_string(index=False))

# decision rule: keep per-station tau only where the bootstrap win-rate is
# lopsided (>=0.70 -- i.e. tau_star beats the global tau in at least 70% of
# resamples); otherwise fall back to the global tau for that station.
CONFIDENT_THRESHOLD = 0.70
robust_map = {}
for _, row in res.iterrows():
    if row['p_beats_global'] >= CONFIDENT_THRESHOLD:
        robust_map[row['station']] = int(row['tau_star']) if row['tau_star'] != 1e9 else 100000
n_kept = len(robust_map)
log(f"\n{n_kept}/{len(res)} stations pass the p>={CONFIDENT_THRESHOLD} bar -- these get a station-specific tau;"
    f" the remaining {len(res)-n_kept} fall back to the global tau (avoids fitting sampling noise).")
print("\nROBUST per-station tau map (bootstrap-confirmed):")
for stn, tau in sorted(robust_map.items()):
    print(f"  '{stn}': {tau},")

res.to_csv(r'D:/Lomba/ssds/model/bootstrap_tau_results.csv', index=False)
log("saved bootstrap_tau_results.csv")
