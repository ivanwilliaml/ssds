"""
v7: builds tr_feat/te_feat using feature_lib after the review-driven fixes:
  - self_fit=True for training rows (leave-one-year-out climatology, review #4)
  - fold-aware upstream-lag map, recomputed fresh from the full train set here
    (review #5 -- no longer a constant frozen from one earlier cutoff)
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
log("spike cleaning applied")

tr_feat = fl.build_features(train_clean, train_clean, dl_feat, static, self_fit=True)
te_feat = fl.build_features(train_clean, TEST, dl_feat, static)
tr_feat[fl.FEATURE_COLS_NUM] = tr_feat[fl.FEATURE_COLS_NUM].fillna(0)
te_feat[fl.FEATURE_COLS_NUM] = te_feat[fl.FEATURE_COLS_NUM].fillna(0)
log(f"features built: tr_feat={tr_feat.shape} te_feat={te_feat.shape}")

tr_feat.to_parquet(r'D:/Lomba/ssds/model/tr_feat_v7.parquet')
te_feat.to_parquet(r'D:/Lomba/ssds/model/te_feat_v7.parquet')
log("saved tr_feat_v7.parquet / te_feat_v7.parquet")
