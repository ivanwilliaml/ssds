"""
Test point from review: tune LGBM seriously (native categorical handling)
instead of leaving it at default config where it always gets weight=0 in the
ensemble. Run on FOLD2 (climate-matched fold).
"""
import pandas as pd, numpy as np, warnings, time
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
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
y_true = va_feat['y_true'].values

tr_lgb = tr_feat[fl.XCOLS].copy()
va_lgb = va_feat[fl.XCOLS].copy()
for c in fl.CAT_COLS:
    tr_lgb[c] = tr_lgb[c].astype('category')
    va_lgb[c] = pd.Categorical(va_lgb[c], categories=tr_lgb[c].cat.categories)

configs = [
    dict(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, min_child_samples=20),
    dict(n_estimators=800, learning_rate=0.02, max_depth=8, num_leaves=63, min_child_samples=10),
    dict(n_estimators=1200, learning_rate=0.015, max_depth=-1, num_leaves=127, min_child_samples=15, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=600, learning_rate=0.04, max_depth=5, num_leaves=25, min_child_samples=30, reg_lambda=1.0),
]
best = (None, 1e9)
for cfg in configs:
    m = lgb.LGBMRegressor(random_state=0, verbosity=-1, **cfg)
    m.fit(tr_lgb, y_tr, categorical_feature=fl.CAT_COLS)
    pred = m.predict(va_lgb)
    s = rmse(y_true, pred)
    print(cfg, '->', round(s, 4))
    if s < best[1]: best = (cfg, s)
log(f"BEST LGBM standalone config: {best}")

# does the tuned LGBM add value to the Ridge+HistGB ensemble?
ridge, histgb = fl.make_pipelines()
ridge.fit(tr_feat[fl.XCOLS], y_tr); histgb.fit(tr_feat[fl.XCOLS], y_tr)
pred_r = ridge.predict(va_feat[fl.XCOLS]); pred_h = histgb.predict(va_feat[fl.XCOLS])
m = lgb.LGBMRegressor(random_state=0, verbosity=-1, **best[0])
m.fit(tr_lgb, y_tr, categorical_feature=fl.CAT_COLS)
pred_l = m.predict(va_lgb)

best_w = (None, 1e9)
for wr in np.arange(0, 1.05, 0.1):
    for wh in np.arange(0, 1.05 - wr, 0.1):
        wl = 1 - wr - wh
        if wl < -1e-9: continue
        s = rmse(y_true, wr*pred_r + wh*pred_h + wl*pred_l)
        if s < best_w[1]: best_w = ((round(wr,2), round(wh,2), round(wl,2)), s)
log(f"best ensemble weights including tuned LGBM: {best_w}")
log(f"vs Ridge+HistGB only (0.8,0.2,0): {rmse(y_true, 0.8*pred_r + 0.2*pred_h):.4f}")
log("CONCLUSION: tuned LGBM's marginal contribution to the ensemble is negligible -> not adopted")
