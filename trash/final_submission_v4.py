import pandas as pd, numpy as np, warnings, time
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
warnings.filterwarnings('ignore')
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

tr = pd.read_parquet(r'D:/Lomba/ssds/model/tr_feat_v4.parquet')
te = pd.read_parquet(r'D:/Lomba/ssds/model/te_feat_v4.parquet')
log(f"loaded tr={tr.shape} te={te.shape}")

FEATURE_COLS_NUM = ['rainfall_mm_roll24h_sum','rainfall_mm_roll72h_sum','rainfall_mm_roll168h_sum',
    'temperature_c_roll24h_mean','humidity_pct_roll24h_mean','soil_moisture_0_7cm_roll24h_mean',
    'soil_moisture_28_100cm_roll24h_mean','surface_pressure_hpa_roll24h_mean','pressure_msl_hpa_roll24h_mean',
    'rainfall_mm','humidity_pct','dew_point_c','cloud_cover_pct','temperature_c','wind_speed_kmh',
    'soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm','soil_moisture_100_255cm',
    'surface_pressure_hpa','pressure_msl_hpa','rmm1','rmm2','mjo_phase','mjo_amplitude','mjo_active','nino_34',
    'hour_sin','hour_cos','month_sin','month_cos','doy_sin','doy_cos','is_wet_season',
    'latitude','longitude','built_surface_m2',
    'station_mean','station_std','station_median','doy_climatology','seasonal_lag_1y']
CAT_COLS = ['nama_pos','landcover_class']
Xcols = CAT_COLS + FEATURE_COLS_NUM

tr[FEATURE_COLS_NUM] = tr[FEATURE_COLS_NUM].fillna(0)
te[FEATURE_COLS_NUM] = te[FEATURE_COLS_NUM].fillna(0)
y_tr = tr['tma_mdpl'].values

pre_tree = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS)], remainder='passthrough')
pre_lin = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS),
                              ('num', StandardScaler(), FEATURE_COLS_NUM)], remainder='drop')
ridge = Pipeline([('pre', pre_lin), ('model', Ridge(alpha=5.0))])
histgb = Pipeline([('pre', pre_tree), ('model', HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400, random_state=0))])
lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=0, verbosity=-1)

ridge.fit(tr[Xcols], y_tr)
histgb.fit(tr[Xcols], y_tr)
lgb_pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS)], remainder='passthrough')
Xtr_lgb = lgb_pre.fit_transform(tr[Xcols])
Xte_lgb = lgb_pre.transform(te[Xcols])
lgbm.fit(Xtr_lgb, y_tr)
log("models fit")

pred_r = ridge.predict(te[Xcols])
pred_h = histgb.predict(te[Xcols])
pred_l = lgbm.predict(Xte_lgb)
# weights from season-matched backtest: (r,h,l)=(0.8,0.2,0.0)
pred_direct = 0.8*pred_r + 0.2*pred_h

tau = 20
w = np.exp(-te['horizon_days'].values/tau)
final_pred = w*te['last_known'].values + (1-w)*pred_direct
final_pred = np.clip(final_pred, 0, None)  # TMA physically non-negative

sub = pd.DataFrame({
    'id': te['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') + ' - ' + te['nama_pos'],
    'tma_mdpl': final_pred
})
te_raw_order = pd.read_csv(r'D:/Lomba/ssds/test.csv')
sub = te_raw_order[['id']].merge(sub, on='id', how='left')
assert sub['tma_mdpl'].isna().sum() == 0
sub.to_csv(r'D:/Lomba/ssds/model/submission_v4.csv', index=False)
log(f"saved submission_v4.csv rows={len(sub)} missing=0")
print(sub.head())
print(f"pred stats: min={final_pred.min():.3f} max={final_pred.max():.3f} mean={final_pred.mean():.3f}")
log("DONE")
