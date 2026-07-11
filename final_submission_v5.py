"""
v5 final submission. Config sourced from multi_fold_validate_v2.py's PRIMARY
fold (FOLD2 2024-25, chosen because its nino_34 climate regime is nearly
identical to the real test period, gap=0.005): weights(r,h,l)=(0.8,0.2,0.0),
tau=20 -- both picked via an explicit unified grid search + argmin, not
hardcoded from an ad-hoc experiment (fixes review point #1).
"""
import pandas as pd, numpy as np, warnings, time
import feature_lib as fl
warnings.filterwarnings('ignore')
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

tr = pd.read_parquet(r'D:/Lomba/ssds/model/tr_feat_v5.parquet')
te = pd.read_parquet(r'D:/Lomba/ssds/model/te_feat_v5.parquet')
log(f"loaded tr={tr.shape} te={te.shape}")

y_tr = tr['tma_mdpl'].values
ridge, histgb = fl.make_pipelines()
ridge.fit(tr[fl.XCOLS], y_tr)
histgb.fit(tr[fl.XCOLS], y_tr)
log("models fit")

pred_r = ridge.predict(te[fl.XCOLS])
pred_h = histgb.predict(te[fl.XCOLS])
WEIGHTS = (0.8, 0.2, 0.0)  # (ridge, histgb, lgbm) from multi_fold_validate_v2 PRIMARY fold
TAU = 20                    # from same source, unified grid argmin
pred_direct = WEIGHTS[0]*pred_r + WEIGHTS[1]*pred_h

w = np.exp(-te['horizon_days'].values/TAU)
final_pred = w*te['last_known'].values + (1-w)*pred_direct
final_pred = np.clip(final_pred, 0, None)

sub = pd.DataFrame({
    'id': te['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') + ' - ' + te['nama_pos'],
    'tma_mdpl': final_pred
})
te_raw_order = pd.read_csv(fl.TEST_CSV)
sub = te_raw_order[['id']].merge(sub, on='id', how='left')
assert sub['tma_mdpl'].isna().sum() == 0
sub.to_csv(r'D:/Lomba/ssds/model/submission_v5.csv', index=False)
log(f"saved submission_v5.csv rows={len(sub)}")
print(sub.head())
print(f"pred stats: min={final_pred.min():.3f} max={final_pred.max():.3f} mean={final_pred.mean():.3f}")
log("DONE")
