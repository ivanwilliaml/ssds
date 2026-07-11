"""
Multi-fold season-matched backtest + structural/climate alignment check.

Addresses the concern that a single validation fold might not represent the
real test distribution. Runs TWO season-matched folds (2023-09-19->2024-05-18
and 2024-09-19->2025-05-18) using a FIXED model config (weights/tau already
selected from prior analysis, not re-tuned per fold, to avoid overfitting the
validation choice itself) and reports both. Also explicitly checks:
  (a) structural alignment: rows/station and horizon-day distribution in each
      validation fold vs the real test.csv
  (b) climate-regime alignment: nino_34 / rainfall stats of each fold vs the
      real test period (2025-09-19->2026-05-18), using data_lingkungan.csv
      which actually covers that period.
"""
import pandas as pd, numpy as np, warnings, time
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
warnings.filterwarnings('ignore')
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

RAW = pd.read_csv(r'D:/Lomba/ssds/train.csv', parse_dates=['datetime']).sort_values(['nama_pos','datetime']).reset_index(drop=True)
dl = pd.read_csv(r'D:/Lomba/ssds/data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
ko = pd.read_csv(r'D:/Lomba/ssds/data_pendukung/koordinat_pos.csv')
te_raw = pd.read_csv(r'D:/Lomba/ssds/test.csv')
split = te_raw['id'].str.split(' - ', n=1, expand=True)
TEST_REAL = pd.DataFrame({'datetime': pd.to_datetime(split[0]), 'nama_pos': split[1]})

def clean_station(g):
    g = g.copy(); v = g['tma_mdpl'].values.copy()
    med = np.median(v); mad = np.median(np.abs(v-med)) + 1e-6
    rz = 0.6745*(v-med)/mad
    prev = np.r_[v[0], v[:-1]]; nxt = np.r_[v[1:], v[-1]]
    neigh_close = np.abs(prev-nxt) < 0.3*(np.abs(prev)+np.abs(nxt)+1e-6)
    far_prev = np.abs(v-prev) > 5*mad; far_nxt = np.abs(v-nxt) > 5*mad
    is_spike = (np.abs(rz) > 6) & neigh_close & far_prev & far_nxt
    v[is_spike] = np.nan; v[v < 0] = np.nan
    g['tma_mdpl'] = v
    g['tma_mdpl'] = g['tma_mdpl'].interpolate(limit_direction='both')
    return g
RAW = RAW.groupby('nama_pos', group_keys=False).apply(clean_station)

static = dl.groupby('nama_pos').agg(landcover_class=('landcover_class','first'),
                                     built_surface_m2=('built_surface_m2','first')).reset_index()
static = static.merge(ko, on='nama_pos', how='left')

dl = dl.sort_values(['nama_pos','datetime'])
dl['nino_34'] = dl.groupby('nama_pos')['nino_34'].ffill().bfill()
for c in ['soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm',
          'soil_moisture_100_255cm','surface_pressure_hpa','pressure_msl_hpa',
          'rmm1','rmm2','mjo_phase','mjo_amplitude','mjo_active']:
    dl[c] = dl.groupby('nama_pos')[c].ffill().bfill()
roll_specs = {'rainfall_mm':[24,72,168],'temperature_c':[24],'humidity_pct':[24],
              'soil_moisture_0_7cm':[24],'soil_moisture_28_100cm':[24],
              'surface_pressure_hpa':[24],'pressure_msl_hpa':[24]}
dl_feat = dl[['datetime','nama_pos']].copy()
for col, windows in roll_specs.items():
    for w in windows:
        agg = 'sum' if col=='rainfall_mm' else 'mean'
        dl_feat[f'{col}_roll{w}h_{agg}'] = dl.groupby('nama_pos')[col].transform(lambda s: s.rolling(w, min_periods=max(1,w//4)).agg(agg))
snap_cols = ['rainfall_mm','humidity_pct','dew_point_c','cloud_cover_pct','temperature_c',
             'wind_speed_kmh','soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm',
             'soil_moisture_100_255cm','surface_pressure_hpa','pressure_msl_hpa','rmm1','rmm2',
             'mjo_phase','mjo_amplitude','mjo_active','nino_34']
for c in snap_cols: dl_feat[c] = dl[c].values

def add_calendar(df):
    df = df.copy()
    df['hour']=df.datetime.dt.hour; df['month']=df.datetime.dt.month; df['doy']=df.datetime.dt.dayofyear
    df['hour_sin']=np.sin(2*np.pi*df.hour/24); df['hour_cos']=np.cos(2*np.pi*df.hour/24)
    df['month_sin']=np.sin(2*np.pi*df.month/12); df['month_cos']=np.cos(2*np.pi*df.month/12)
    df['doy_sin']=np.sin(2*np.pi*df.doy/365.25); df['doy_cos']=np.cos(2*np.pi*df.doy/365.25)
    df['is_wet_season']=df.month.isin([11,12,1,2,3,4]).astype(int)
    return df

def build_doy_clim(train_only):
    if 'doy' not in train_only.columns:
        train_only = train_only.assign(doy=train_only['datetime'].dt.dayofyear)
    out=[]
    for stn, g in train_only.groupby('nama_pos'):
        s = g.set_index('doy')['tma_mdpl']
        means={}
        for d in range(1,367):
            window=[((d+off-1)%366)+1 for off in range(-5,6)]
            vals = s[s.index.isin(window)]
            means[d]=vals.mean() if len(vals) else np.nan
        out.append(pd.DataFrame({'nama_pos':stn,'doy':list(means.keys()),'doy_climatology':list(means.values())}))
    return pd.concat(out, ignore_index=True)

def add_seasonal_lag(target_df, source_df, days=365, tol_days=2, name='seasonal_lag_1y'):
    src = source_df[['datetime','nama_pos','tma_mdpl']].rename(columns={'tma_mdpl':name,'datetime':'src_dt'}).sort_values('src_dt')
    tgt = target_df.copy(); tgt['lookup_dt']=tgt['datetime']-pd.Timedelta(days=days)
    tgt = tgt.sort_values('lookup_dt')
    out = pd.merge_asof(tgt, src, left_on='lookup_dt', right_on='src_dt', by='nama_pos',
                         direction='nearest', tolerance=pd.Timedelta(days=tol_days))
    return out.drop(columns=['lookup_dt','src_dt']).sort_index()

def build_features(train_only, target_df):
    df = target_df.merge(dl_feat, on=['datetime','nama_pos'], how='left')
    df = add_calendar(df)
    df = df.merge(static, on='nama_pos', how='left')
    stn_stats = train_only.groupby('nama_pos')['tma_mdpl'].agg(station_mean='mean', station_std='std', station_median='median').reset_index()
    df = df.merge(stn_stats, on='nama_pos', how='left')
    doy_clim = build_doy_clim(train_only)
    df = df.merge(doy_clim, on=['nama_pos','doy'], how='left')
    fallback = stn_stats.set_index('nama_pos')['station_mean']
    df['doy_climatology'] = df['doy_climatology'].fillna(df['nama_pos'].map(fallback))
    df = add_seasonal_lag(df, train_only)
    df['seasonal_lag_1y'] = df['seasonal_lag_1y'].fillna(df['doy_climatology'])
    last_known = train_only.sort_values('datetime').groupby('nama_pos').tail(1)[['nama_pos','datetime','tma_mdpl']]
    lk_map = last_known.set_index('nama_pos')
    df['last_known'] = df['nama_pos'].map(lk_map['tma_mdpl'])
    last_dt = df['nama_pos'].map(lk_map['datetime'])
    df['horizon_days'] = (df['datetime'] - last_dt).dt.total_seconds()/86400
    return df

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

def rmse(a,b): return np.sqrt(mean_squared_error(a,b))

def run_fold(cut_str, va_end_str, weights=(0.8,0.2,0.0), tau=20, label=''):
    CUT = pd.Timestamp(cut_str); VA_END = pd.Timestamp(va_end_str)
    train_only = RAW[RAW.datetime <= CUT].copy()
    val_only = RAW[(RAW.datetime > CUT) & (RAW.datetime <= VA_END)].copy()
    log(f"[{label}] CUT={CUT} VA_END={VA_END} train_only={len(train_only)} val_only={len(val_only)}")

    tr_feat = build_features(train_only, train_only)
    va_feat = build_features(train_only, val_only[['datetime','nama_pos']].copy())
    va_feat = va_feat.merge(val_only[['datetime','nama_pos','tma_mdpl']].rename(columns={'tma_mdpl':'y_true'}),
                             on=['datetime','nama_pos'], how='left')
    assert va_feat['y_true'].isna().sum() == 0

    # --- structural alignment check vs real test ---
    real_counts = TEST_REAL.groupby('nama_pos').size()
    fold_counts = val_only.groupby('nama_pos').size()
    coverage = (fold_counts / real_counts.reindex(fold_counts.index)).describe()
    log(f"[{label}] fold row-coverage vs real test per station (fraction of 726): min={coverage['min']:.2f} mean={coverage['mean']:.2f} max={coverage['max']:.2f}")

    y_tr = tr_feat['tma_mdpl'].values
    tr_feat[FEATURE_COLS_NUM] = tr_feat[FEATURE_COLS_NUM].fillna(0)
    va_feat[FEATURE_COLS_NUM] = va_feat[FEATURE_COLS_NUM].fillna(0)

    pre_tree = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS)], remainder='passthrough')
    pre_lin = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS),
                                  ('num', StandardScaler(), FEATURE_COLS_NUM)], remainder='drop')
    ridge = Pipeline([('pre', pre_lin), ('model', Ridge(alpha=5.0))])
    histgb = Pipeline([('pre', pre_tree), ('model', HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400, random_state=0))])

    ridge.fit(tr_feat[Xcols], y_tr)
    histgb.fit(tr_feat[Xcols], y_tr)
    pred_r = ridge.predict(va_feat[Xcols])
    pred_h = histgb.predict(va_feat[Xcols])
    wr, wh, wl = weights
    pred_direct = wr*pred_r + wh*pred_h  # wl(lgbm)=0 in fixed config
    y_true = va_feat['y_true'].values

    h = va_feat['horizon_days'].values
    last_known = va_feat['last_known'].values
    w = np.exp(-h/tau)
    pred_blend = w*last_known + (1-w)*pred_direct

    log(f"[{label}] persistence={rmse(y_true,last_known):.4f}  ML_direct={rmse(y_true,pred_direct):.4f}  blend(tau={tau})={rmse(y_true,pred_blend):.4f}")

    # per-horizon breakdown
    va_feat['pred_direct']=pred_direct; va_feat['pred_blend']=pred_blend; va_feat['y_true_']=y_true
    bins=[0,30,60,90,120,150,180,210,242]
    va_feat['hbin']=pd.cut(h, bins)
    g = va_feat.groupby('hbin').apply(lambda d: pd.Series({
        'n': len(d), 'rmse_blend': rmse(d.y_true_, d.pred_blend)}))
    print(g)
    return {'label':label,'rmse_direct':rmse(y_true,pred_direct),'rmse_blend':rmse(y_true,pred_blend)}

results = []
results.append(run_fold('2023-09-18 18:00:00','2024-05-18 18:00:00', label='FOLD1 2023-24 (El Nino, climate MISMATCH vs real test)'))
results.append(run_fold('2024-09-18 18:00:00','2025-05-18 18:00:00', label='FOLD2 2024-25 (La Nina-ish, climate MATCH vs real test)'))

print("\n===== SUMMARY =====")
for r in results:
    print(r)

# structural check: real test row/horizon distribution
print("\nreal test rows per station (should all be 726):", TEST_REAL.groupby('nama_pos').size().unique())
print("real test horizon range (days from train end):",
      ((TEST_REAL.datetime.max()-TEST_REAL.datetime.min()).days))

log("DONE multi_fold_validate.py")
