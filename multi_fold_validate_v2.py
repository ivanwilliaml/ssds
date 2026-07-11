"""
Multi-fold season-matched backtest v2 -- fixes vs v1 (per code review):
  1. tau is now selected via a SINGLE unified grid + programmatic argmin,
     instead of copy-pasted magic numbers that drifted out of sync with the
     grid actually searched (v1 hardcoded tau=20 downstream, which was never
     in either grid tested).
  2. Climate-regime alignment (nino_34, rainfall) is ACTUALLY computed per
     fold and compared to the real test period -- v1's docstring claimed this
     check existed but the code never computed it.
  3. Spike cleaning is applied per-fold to train_only ONLY (not to the full
     RAW series before the CUT split), so the median/MAD reference used for
     spike detection cannot see data from after the fold's cutoff.
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

TAU_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 180, 240, 300]

REAL_TEST_START = TEST_REAL.datetime.min()
REAL_TEST_END = TEST_REAL.datetime.max()
real_regime = fl.climate_regime(dl, REAL_TEST_START, REAL_TEST_END)
log(f"REAL TEST regime {REAL_TEST_START.date()}..{REAL_TEST_END.date()}: {real_regime}")


def run_fold(cut_str, va_end_str, label='', select_tau=True, fixed_tau=None):
    CUT = pd.Timestamp(cut_str); VA_END = pd.Timestamp(va_end_str)
    train_only_raw = RAW[RAW.datetime <= CUT].copy()
    val_only = RAW[(RAW.datetime > CUT) & (RAW.datetime <= VA_END)].copy()
    # FIX #3: clean only the pre-cutoff slice, never the full RAW series
    train_only = fl.clean_dataframe(train_only_raw)
    log(f"[{label}] CUT={CUT} VA_END={VA_END} train_only={len(train_only)} val_only={len(val_only)}")

    # FIX #2: actually compute climate regime for this fold and compare
    fold_regime = fl.climate_regime(dl, CUT + pd.Timedelta(days=1), VA_END)
    nino_gap = abs(fold_regime['nino34_mean'] - real_regime['nino34_mean'])
    rain_gap_pct = 100 * abs(fold_regime['rainfall_mean'] - real_regime['rainfall_mean']) / real_regime['rainfall_mean']
    log(f"[{label}] fold climate regime: {fold_regime}  | nino34_gap={nino_gap:.3f}  rainfall_gap={rain_gap_pct:.1f}%")

    real_counts = TEST_REAL.groupby('nama_pos').size()
    fold_counts = val_only.groupby('nama_pos').size()
    coverage = (fold_counts / real_counts.reindex(fold_counts.index)).describe()
    log(f"[{label}] row-coverage vs real test: min={coverage['min']:.2f} mean={coverage['mean']:.2f} max={coverage['max']:.2f}")

    tr_feat = fl.build_features(train_only, train_only, dl_feat, static)
    va_feat = fl.build_features(train_only, val_only[['datetime','nama_pos']].copy(), dl_feat, static)
    va_feat = va_feat.merge(val_only[['datetime','nama_pos','tma_mdpl']].rename(columns={'tma_mdpl':'y_true'}),
                             on=['datetime','nama_pos'], how='left')
    assert va_feat['y_true'].isna().sum() == 0

    y_tr = tr_feat['tma_mdpl'].values
    tr_feat[fl.FEATURE_COLS_NUM] = tr_feat[fl.FEATURE_COLS_NUM].fillna(0)
    va_feat[fl.FEATURE_COLS_NUM] = va_feat[fl.FEATURE_COLS_NUM].fillna(0)

    ridge, histgb = fl.make_pipelines()
    ridge.fit(tr_feat[fl.XCOLS], y_tr)
    histgb.fit(tr_feat[fl.XCOLS], y_tr)
    lgb_pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), fl.CAT_COLS)], remainder='passthrough')
    Xtr_lgb = lgb_pre.fit_transform(tr_feat[fl.XCOLS])
    Xva_lgb = lgb_pre.transform(va_feat[fl.XCOLS])
    lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=0, verbosity=-1)
    lgbm.fit(Xtr_lgb, y_tr)

    pred_r = ridge.predict(va_feat[fl.XCOLS])
    pred_h = histgb.predict(va_feat[fl.XCOLS])
    pred_l = lgbm.predict(Xva_lgb)
    y_true = va_feat['y_true'].values

    best = (None, 1e9)
    for wr in np.arange(0, 1.05, 0.2):
        for wh in np.arange(0, 1.05 - wr, 0.2):
            wl = 1 - wr - wh
            if wl < -1e-9: continue
            s = rmse(y_true, wr*pred_r + wh*pred_h + wl*pred_l)
            if s < best[1]: best = ((wr, wh, wl), s)
    (wr, wh, wl), direct_rmse = best
    pred_direct = wr*pred_r + wh*pred_h + wl*pred_l
    log(f"[{label}] ensemble weights(r,h,l)={best[0]} direct RMSE={direct_rmse:.4f}")

    h = va_feat['horizon_days'].values
    last_known = va_feat['last_known'].values

    def blend(tau):
        w = np.exp(-h / tau)
        return w * last_known + (1 - w) * pred_direct

    if select_tau:
        tau_scores = {tau: rmse(y_true, blend(tau)) for tau in TAU_GRID}
        best_tau = min(tau_scores, key=tau_scores.get)
        log(f"[{label}] tau grid: " + ", ".join(f"{t}={s:.4f}" for t, s in tau_scores.items()))
    else:
        best_tau = fixed_tau
    pred_blend = blend(best_tau)
    blend_rmse = rmse(y_true, pred_blend)
    log(f"[{label}] BEST tau={best_tau} blend RMSE={blend_rmse:.4f}  (persistence={rmse(y_true,last_known):.4f})")

    return {'label': label, 'weights': best[0], 'direct_rmse': direct_rmse,
            'best_tau': best_tau, 'blend_rmse': blend_rmse,
            'nino_gap': nino_gap, 'rainfall_gap_pct': rain_gap_pct}


results = []
results.append(run_fold('2023-09-18 18:00:00', '2024-05-18 18:00:00', label='FOLD1 2023-24'))
results.append(run_fold('2024-09-18 18:00:00', '2025-05-18 18:00:00', label='FOLD2 2024-25'))

print("\n===== SUMMARY =====")
for r in results:
    print(r)

# fold selected as the primary tau/weights source for final submission =
# the one with the smallest climate-regime gap to the real test period
primary = min(results, key=lambda r: r['nino_gap'])
print(f"\nPRIMARY FOLD (closest climate regime to real test): {primary['label']}")
print(f"  -> use weights={primary['weights']} tau={primary['best_tau']} for final submission")

log("DONE multi_fold_validate_v2.py")
