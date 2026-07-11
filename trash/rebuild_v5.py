"""
v5: builds tr_feat/te_feat using the consolidated, leakage-fixed feature_lib.
Differences vs v4:
  - spike cleaning applied only to the pre-cutoff slice (train itself here,
    since this is the final full-data build, so it's equivalent -- but the
    shared function is now fold-safe for backtesting too)
  - adds empirically-derived upstream_lag_value feature
  - uses feature_lib to guarantee train/val/test feature parity (no drift
    between copy-pasted script versions)
"""
import pandas as pd, numpy as np, warnings, time
import feature_lib as fl
warnings.filterwarnings('ignore')
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

RAW, TEST = fl.load_raw()
dl, dl_feat, static = fl.load_exog()
log(f"loaded train={RAW.shape} test={TEST.shape}")

train_clean = fl.clean_dataframe(RAW)
n_cleaned = (train_clean['tma_mdpl'].values != RAW.sort_values(['nama_pos','datetime'])['tma_mdpl'].values).sum()
log(f"cleaned ~{n_cleaned} points (spike/negative)")

tr_feat = fl.build_features(train_clean, train_clean, dl_feat, static, self_fit=True)
te_feat = fl.build_features(train_clean, TEST, dl_feat, static)
tr_feat[fl.FEATURE_COLS_NUM] = tr_feat[fl.FEATURE_COLS_NUM].fillna(0)
te_feat[fl.FEATURE_COLS_NUM] = te_feat[fl.FEATURE_COLS_NUM].fillna(0)
log(f"features built: tr_feat={tr_feat.shape} te_feat={te_feat.shape}")

tr_feat.to_parquet(r'D:/Lomba/ssds/model/tr_feat_v5.parquet')
te_feat.to_parquet(r'D:/Lomba/ssds/model/te_feat_v5.parquet')
log("saved tr_feat_v5.parquet / te_feat_v5.parquet")
