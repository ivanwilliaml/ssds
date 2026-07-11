"""
Multi-fold season-matched backtest v3 -- fixes review point #1: v2's run_fold
selected (wr,wh,wl) AND tau by minimizing RMSE directly against the SAME
va_feat['y_true'] that was then reported as the fold's score. Validation set
and tuning set were identical (~19.5k points), so FOLD2's reported 1.4433 and
the resulting "primary config" (tau=20, weights=0.8/0.2/0) both carry a
selection bias from that double duty -- probably small given the coarse grid
(~21 weight combos x 16 tau values), but worth measuring directly rather than
assuming "probably small" is good enough.

Fix: run_fold now takes select_weights/fixed_weights mirroring the existing
select_tau/fixed_tau. This script selects the config on FOLD2 (as before),
then re-scores FOLD1 with that FROZEN config (no re-selection) to check
whether the config is stable across folds or curve-fit to FOLD2's noise.
FOLD1's own climate mismatch (El Nino, nino_34 gap=1.8) means it isn't a
clean unbiased estimate either -- but a frozen-config cross-fold score that
is NOT wildly worse than FOLD1's own self-tuned score is reassuring; a score
that blows up says the "primary" config overfit FOLD2.

Also fixes review point #5: derive_upstream_map(train_only) is now called
fresh inside feature_lib.build_features for every fold (see feature_lib.py),
so the upstream-lag map used here is always fold-specific, never a constant
frozen from a different fold's cutoff.
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
WEIGHT_STEP = 0.2

REAL_TEST_START = TEST_REAL.datetime.min()
REAL_TEST_END = TEST_REAL.datetime.max()
real_regime = fl.climate_regime(dl, REAL_TEST_START, REAL_TEST_END)
log(f"REAL TEST regime {REAL_TEST_START.date()}..{REAL_TEST_END.date()}: {real_regime}")


def run_fold(cut_str, va_end_str, label='',
             select_tau=True, fixed_tau=None,
             select_weights=True, fixed_weights=None):
    CUT = pd.Timestamp(cut_str); VA_END = pd.Timestamp(va_end_str)
    train_only_raw = RAW[RAW.datetime <= CUT].copy()
    val_only = RAW[(RAW.datetime > CUT) & (RAW.datetime <= VA_END)].copy()
    train_only = fl.clean_dataframe(train_only_raw)
    log(f"[{label}] CUT={CUT} VA_END={VA_END} train_only={len(train_only)} val_only={len(val_only)}")

    fold_regime = fl.climate_regime(dl, CUT + pd.Timedelta(days=1), VA_END)
    nino_gap = abs(fold_regime['nino34_mean'] - real_regime['nino34_mean'])
    rain_gap_pct = 100 * abs(fold_regime['rainfall_mean'] - real_regime['rainfall_mean']) / real_regime['rainfall_mean']
    log(f"[{label}] fold climate regime: {fold_regime}  | nino34_gap={nino_gap:.3f}  rainfall_gap={rain_gap_pct:.1f}%")

    real_counts = TEST_REAL.groupby('nama_pos').size()
    fold_counts = val_only.groupby('nama_pos').size()
    coverage = (fold_counts / real_counts.reindex(fold_counts.index)).describe()
    log(f"[{label}] row-coverage vs real test: min={coverage['min']:.2f} mean={coverage['mean']:.2f} max={coverage['max']:.2f}")

    tr_feat = fl.build_features(train_only, train_only, dl_feat, static, self_fit=True)
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

    if select_weights:
        best = (None, 1e9)
        for wr in np.arange(0, 1.05, WEIGHT_STEP):
            for wh in np.arange(0, 1.05 - wr, WEIGHT_STEP):
                wl = 1 - wr - wh
                if wl < -1e-9: continue
                s = rmse(y_true, wr*pred_r + wh*pred_h + wl*pred_l)
                if s < best[1]: best = ((wr, wh, wl), s)
        weights, direct_rmse = best
    else:
        weights = fixed_weights
        wr, wh, wl = weights
        direct_rmse = rmse(y_true, wr*pred_r + wh*pred_h + wl*pred_l)
    wr, wh, wl = weights
    pred_direct = wr*pred_r + wh*pred_h + wl*pred_l
    log(f"[{label}] ensemble weights(r,h,l)={weights}  direct RMSE={direct_rmse:.4f}  (select_weights={select_weights})")

    h = va_feat['horizon_days'].values
    last_known = va_feat['last_known'].values

    def blend(tau):
        w = np.exp(-h / tau)
        return w * last_known + (1 - w) * pred_direct

    if select_tau:
        tau_scores = {tau: rmse(y_true, blend(tau)) for tau in TAU_GRID}
        best_tau = min(tau_scores, key=tau_scores.get)
    else:
        best_tau = fixed_tau
    pred_blend = blend(best_tau)
    blend_rmse = rmse(y_true, pred_blend)
    log(f"[{label}] tau={best_tau} blend RMSE={blend_rmse:.4f}  (persistence={rmse(y_true,last_known):.4f})  (select_tau={select_tau})")

    return {'label': label, 'weights': weights, 'direct_rmse': direct_rmse,
            'best_tau': best_tau, 'blend_rmse': blend_rmse,
            'nino_gap': nino_gap, 'rainfall_gap_pct': rain_gap_pct,
            'va_feat': va_feat, 'pred_direct': pred_direct}


# ---- step 1: select config on FOLD2 (climate-matched, as before) ----
log("=== SELECTION PASS (both folds self-tune, same as v2) ===")
r2_select = run_fold('2024-09-18 18:00:00', '2025-05-18 18:00:00', label='FOLD2 select')
r1_select = run_fold('2023-09-18 18:00:00', '2024-05-18 18:00:00', label='FOLD1 select')

# ---- step 2: cross-fold stability check -- score FOLD1 with FOLD2's frozen config ----
log("\n=== CROSS-FOLD STABILITY CHECK (review point #1) ===")
r1_frozen = run_fold('2023-09-18 18:00:00', '2024-05-18 18:00:00',
                      label='FOLD1 scored w/ FOLD2 config',
                      select_tau=False, fixed_tau=r2_select['best_tau'],
                      select_weights=False, fixed_weights=r2_select['weights'])

print("\n===== SUMMARY =====")
print(f"FOLD2 self-tuned (this fold picked the config):      RMSE={r2_select['blend_rmse']:.4f}  tau={r2_select['best_tau']}  weights={r2_select['weights']}")
print(f"FOLD1 self-tuned (its own best-case, different config): RMSE={r1_select['blend_rmse']:.4f}  tau={r1_select['best_tau']}  weights={r1_select['weights']}")
print(f"FOLD1 scored w/ FOLD2's FROZEN config (no re-selection): RMSE={r1_frozen['blend_rmse']:.4f}")
gap = r1_frozen['blend_rmse'] - r1_select['blend_rmse']
print(f"\nFrozen-config penalty on FOLD1 vs its own self-tuned best: {gap:+.4f}")
print("(a large positive gap here would mean the FOLD2-selected config is curve-fit to FOLD2's")
print(" noise rather than a genuinely stable choice; FOLD1's own climate mismatch to the real")
print(" test period means this isn't a clean unbiased estimate of the final config's real-test")
print(" RMSE either, but it tells us whether the config is at least directionally stable.)")

print(f"\nPRIMARY FOLD (closest climate regime to real test): FOLD2")
print(f"  -> use weights={r2_select['weights']} tau={r2_select['best_tau']} for final submission")

log("DONE multi_fold_validate_v3.py")
