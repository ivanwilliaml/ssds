"""
Test point #4 from review: reparametrize target as anomaly from climatology
  y' = tma_mdpl - doy_climatology
train models on y', add doy_climatology back at prediction time.
Compared against the leakage-fixed absolute-level baseline on FOLD2 (the
climate-matched fold), using the SAME train/val split and features.
"""
import pandas as pd, numpy as np, warnings, time
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
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
va_feat = va_feat.merge(val_only[['datetime','nama_pos','tma_mdpl']].rename(columns={'tma_mdpl':'y_true'}),
                         on=['datetime','nama_pos'], how='left')
tr_feat[fl.FEATURE_COLS_NUM] = tr_feat[fl.FEATURE_COLS_NUM].fillna(0)
va_feat[fl.FEATURE_COLS_NUM] = va_feat[fl.FEATURE_COLS_NUM].fillna(0)
log(f"tr_feat={tr_feat.shape} va_feat={va_feat.shape}")

# ---- baseline: absolute level target ----
y_tr_level = tr_feat['tma_mdpl'].values
y_true = va_feat['y_true'].values

ridge_l, histgb_l = fl.make_pipelines()
ridge_l.fit(tr_feat[fl.XCOLS], y_tr_level)
histgb_l.fit(tr_feat[fl.XCOLS], y_tr_level)
lgb_pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), fl.CAT_COLS)], remainder='passthrough')
Xtr_lgb = lgb_pre.fit_transform(tr_feat[fl.XCOLS])
Xva_lgb = lgb_pre.transform(va_feat[fl.XCOLS])
lgbm_l = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=0, verbosity=-1)
lgbm_l.fit(Xtr_lgb, y_tr_level)

pred_r_l = ridge_l.predict(va_feat[fl.XCOLS])
pred_h_l = histgb_l.predict(va_feat[fl.XCOLS])
pred_l_l = lgbm_l.predict(Xva_lgb)
pred_level = 0.8*pred_r_l + 0.2*pred_h_l  # weights from multi_fold_validate_v2
log(f"[LEVEL target] Ridge={rmse(y_true,pred_r_l):.4f} HistGB={rmse(y_true,pred_h_l):.4f} LGBM={rmse(y_true,pred_l_l):.4f} ensemble(0.8/0.2/0)={rmse(y_true,pred_level):.4f}")

# ---- anomaly target: y' = tma_mdpl - doy_climatology ----
y_tr_anom = tr_feat['tma_mdpl'].values - tr_feat['doy_climatology'].values

ridge_a, histgb_a = fl.make_pipelines()
ridge_a.fit(tr_feat[fl.XCOLS], y_tr_anom)
histgb_a.fit(tr_feat[fl.XCOLS], y_tr_anom)
lgbm_a = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=0, verbosity=-1)
lgbm_a.fit(Xtr_lgb, y_tr_anom)

pred_r_a = ridge_a.predict(va_feat[fl.XCOLS]) + va_feat['doy_climatology'].values
pred_h_a = histgb_a.predict(va_feat[fl.XCOLS]) + va_feat['doy_climatology'].values
pred_l_a = lgbm_a.predict(Xva_lgb) + va_feat['doy_climatology'].values
pred_anom = 0.8*pred_r_a + 0.2*pred_h_a
log(f"[ANOMALY target] Ridge={rmse(y_true,pred_r_a):.4f} HistGB={rmse(y_true,pred_h_a):.4f} LGBM={rmse(y_true,pred_l_a):.4f} ensemble(0.8/0.2/0)={rmse(y_true,pred_anom):.4f}")

# re-optimize ensemble weights for anomaly variant too (may differ from level)
best = (None, 1e9)
for wr in np.arange(0,1.05,0.1):
    for wh in np.arange(0,1.05-wr,0.1):
        wl = 1-wr-wh
        if wl < -1e-9: continue
        s = rmse(y_true, wr*pred_r_a + wh*pred_h_a + wl*pred_l_a)
        if s < best[1]: best=((wr,wh,wl), s)
log(f"[ANOMALY target] best re-tuned weights={best[0]} RMSE={best[1]:.4f}")

# blend with persistence like before, using best tau grid
h = va_feat['horizon_days'].values
last_known = va_feat['last_known'].values
TAU_GRID = [5,10,15,20,25,30,40,50,60,75,90,120,150,180,240,300]
def blend(anchor, tau):
    w = np.exp(-h/tau)
    return w*last_known + (1-w)*anchor

wr,wh,wl = best[0]
pred_anom_best = wr*pred_r_a + wh*pred_h_a + wl*pred_l_a
tau_scores = {tau: rmse(y_true, blend(pred_anom_best, tau)) for tau in TAU_GRID}
best_tau = min(tau_scores, key=tau_scores.get)
log(f"[ANOMALY target] tau grid: " + ", ".join(f"{t}={s:.4f}" for t,s in tau_scores.items()))
log(f"[ANOMALY target] BEST blend tau={best_tau} RMSE={tau_scores[best_tau]:.4f}")

log("\n=== FINAL COMPARISON (same fold, same features, only target parametrization differs) ===")
log(f"LEVEL   target best: direct=({rmse(y_true,pred_level):.4f})")
log(f"ANOMALY target best: direct=({best[1]:.4f})  blend={tau_scores[best_tau]:.4f}")
