"""
v7 final submission -- incorporates all review fixes:
  - LOYO climatology for training rows (review #4)
  - fold-aware upstream-lag map (review #5)
  - ensemble weights re-selected under the fixed pipeline via
    multi_fold_validate_v3.py's FOLD2 selection pass: Ridge=0.4, HistGB=0.0,
    LGBM=0.6 (LGBM now earns real weight, unlike v6 where it was excluded --
    the earlier self-referential climatology leak in the v5/v6 training
    features was apparently absorbed differently by tree vs linear models)
  - per-station tau restricted to bootstrap-confirmed stations only (p>=0.70
    win-rate vs the global tau across 300 resamples, review #2); the 4
    stations that failed the bar (Badegan, Peren, Ngadipiro, Karangnongko)
    fall back to the global tau=15 instead of their noise-fit optimum
"""
import pandas as pd, numpy as np, warnings, time
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import feature_lib as fl
warnings.filterwarnings('ignore')
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

GLOBAL_TAU = 15  # from multi_fold_validate_v3.py FOLD2 selection
ROBUST_STATION_TAU = {  # bootstrap-confirmed (p>=0.70), see bootstrap_tau_significance.py
    'Arjowinangun - Pacitan': 25, 'Babat': 75, 'Bengkelolor': 10,
    'Boboh Kali Lamong': 15, 'Bojonegoro - Kali Kethek': 15, 'Brangkal': 40,
    'Cepu': 5, 'Colo Weir': 10, 'Floodway Bridge C': 5, 'Gunungsari': 15,
    'Jarum': 15, 'Jurug': 10, 'Kajangan': 5, 'Kali Anyar - Kreteg Abang': 150,
    'Kali Pepe - PTPN': 90, 'Kali Pepe - Tugu Boto': 40, 'Karanggeneng': 25,
    'Kedungupit': 10, 'Ketonggo': 5, 'Lorog': 30, 'Napel': 5,
    'Ngrembang': 150, 'Sekayu': 15, 'Serenan': 10, 'Sumberrejo': 30,
    'Wonogiri Dam': 25,
    # Badegan, Peren, Ngadipiro, Karangnongko deliberately omitted -> GLOBAL_TAU fallback
}
WEIGHTS = (0.4, 0.0, 0.6)  # (ridge, histgb, lgbm)

tr = pd.read_parquet(r'D:/Lomba/ssds/model/tr_feat_v7.parquet')
te = pd.read_parquet(r'D:/Lomba/ssds/model/te_feat_v7.parquet')
log(f"loaded tr={tr.shape} te={te.shape}")

y_tr = tr['tma_mdpl'].values
ridge, histgb = fl.make_pipelines()
ridge.fit(tr[fl.XCOLS], y_tr)
lgb_pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), fl.CAT_COLS)], remainder='passthrough')
Xtr_lgb = lgb_pre.fit_transform(tr[fl.XCOLS])
Xte_lgb = lgb_pre.transform(te[fl.XCOLS])
lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=0, verbosity=-1)
lgbm.fit(Xtr_lgb, y_tr)
log("models fit (Ridge, LGBM -- HistGB weight is 0 this round, skip fitting it)")

pred_r = ridge.predict(te[fl.XCOLS])
pred_l = lgbm.predict(Xte_lgb)
pred_direct = WEIGHTS[0]*pred_r + WEIGHTS[2]*pred_l

tau_arr = te['nama_pos'].map(ROBUST_STATION_TAU).fillna(GLOBAL_TAU).values
w = np.exp(-te['horizon_days'].values / tau_arr)
final_pred = w*te['last_known'].values + (1-w)*pred_direct
final_pred = np.clip(final_pred, 0, None)

sub = pd.DataFrame({
    'id': te['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') + ' - ' + te['nama_pos'],
    'tma_mdpl': final_pred
})
te_raw_order = pd.read_csv(fl.TEST_CSV)
sub = te_raw_order[['id']].merge(sub, on='id', how='left')
assert sub['tma_mdpl'].isna().sum() == 0
sub.to_csv(r'D:/Lomba/ssds/model/submission_v7.csv', index=False)
log(f"saved submission_v7.csv rows={len(sub)}")
print(f"pred stats: min={final_pred.min():.3f} max={final_pred.max():.3f} mean={final_pred.mean():.3f}")
log("DONE")
