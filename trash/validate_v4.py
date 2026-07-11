"""
Season-matched, leakage-free backtest for v4 pipeline.
Mimics real test: CUT = train_end - 365 days (so validation horizon spans the
same wet-season-crossing 242-day window the real test spans), train on data
<= CUT, evaluate on data in (CUT, CUT+242d].
All station-level statistics (mean/std, doy_climatology, seasonal_lag_1y) are
computed using ONLY data <= CUT to avoid leakage.
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

# ---- clean spikes once (uses only local neighbor info, causal-safe: a spike
#      detector using immediate neighbors is standard hydrological QC and does
#      not leak future distributional info) ----
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
    """train_only: cleaned rows with datetime<=CUT (used for all stats, no leakage).
       target_df: rows to build features FOR (can be train_only itself or validation rows)."""
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

def make_pipelines():
    pre_tree = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS)], remainder='passthrough')
    pre_lin = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS),
                                  ('num', StandardScaler(), FEATURE_COLS_NUM)], remainder='drop')
    ridge = Pipeline([('pre', pre_lin), ('model', Ridge(alpha=5.0))])
    histgb = Pipeline([('pre', pre_tree), ('model', HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400, random_state=0))])
    lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=0, verbosity=-1)
    return ridge, histgb, lgbm

CUT = RAW.datetime.max() - pd.Timedelta(days=365)
VA_END = CUT + pd.Timedelta(days=242)
train_only = RAW[RAW.datetime <= CUT].copy()
val_only = RAW[(RAW.datetime > CUT) & (RAW.datetime <= VA_END)].copy()
log(f"CUT={CUT} VA_END={VA_END} train_only={len(train_only)} val_only={len(val_only)}")

tr_feat = build_features(train_only, train_only)
va_feat = build_features(train_only, val_only[['datetime','nama_pos']].copy())
va_feat = va_feat.merge(val_only[['datetime','nama_pos','tma_mdpl']].rename(columns={'tma_mdpl':'y_true'}),
                         on=['datetime','nama_pos'], how='left')
assert va_feat['y_true'].isna().sum() == 0, "y_true merge produced NaN - key mismatch"
y_tr = tr_feat['tma_mdpl'].values
log(f"features built. tr_feat={tr_feat.shape} va_feat={va_feat.shape}")

Xcols = CAT_COLS + FEATURE_COLS_NUM
tr_feat[FEATURE_COLS_NUM] = tr_feat[FEATURE_COLS_NUM].fillna(0)
va_feat[FEATURE_COLS_NUM] = va_feat[FEATURE_COLS_NUM].fillna(0)
ridge, histgb, lgbm = make_pipelines()
ridge.fit(tr_feat[Xcols], y_tr)
histgb.fit(tr_feat[Xcols], y_tr)
lgb_pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS)], remainder='passthrough')
Xtr_lgb = lgb_pre.fit_transform(tr_feat[Xcols])
Xva_lgb = lgb_pre.transform(va_feat[Xcols])
lgbm.fit(Xtr_lgb, y_tr)

pred_r = ridge.predict(va_feat[Xcols])
pred_h = histgb.predict(va_feat[Xcols])
pred_l = lgbm.predict(Xva_lgb)
y_true = va_feat['y_true'].values

def rmse(a,b): return np.sqrt(mean_squared_error(a,b))
log(f"Ridge={rmse(y_true,pred_r):.4f}  HistGB={rmse(y_true,pred_h):.4f}  LGBM={rmse(y_true,pred_l):.4f}")

best = (None, 1e9)
for wr in np.arange(0,1.05,0.2):
    for wh in np.arange(0,1.05-wr,0.2):
        wl = 1-wr-wh
        if wl < -1e-9: continue
        pred = wr*pred_r + wh*pred_h + wl*pred_l
        s = rmse(y_true, pred)
        if s < best[1]: best = ((wr,wh,wl), s)
log(f"BEST direct ensemble RMSE={best[1]:.4f} weights(r,h,l)={best[0]}")
wr,wh,wl = best[0]
pred_direct = wr*pred_r + wh*pred_h + wl*pred_l

h = va_feat['horizon_days'].values
last_known = va_feat['last_known'].values
seasonal_1y = va_feat['seasonal_lag_1y'].values
doy_clim = va_feat['doy_climatology'].values

log(f"pure persistence RMSE={rmse(y_true, last_known):.4f}")
log(f"seasonal_lag_1y only RMSE={rmse(y_true, seasonal_1y):.4f}")
log(f"doy_climatology only RMSE={rmse(y_true, doy_clim):.4f}")
log(f"ML direct only RMSE={rmse(y_true, pred_direct):.4f}")

def blend(anchor, tau):
    w = np.exp(-h/tau)
    return w*last_known + (1-w)*anchor

for tau in [30,45,60,90,120,150,180,240,300]:
    for name, anchor in [('ML_direct', pred_direct), ('seasonal_1y', seasonal_1y), ('doy_clim', doy_clim)]:
        s = rmse(y_true, blend(anchor, tau))
        print(f"  tau={tau:4d} anchor={name:12s} RMSE={s:.4f}")

# 3-way anchor blend: combine seasonal_1y + doy_clim + ML_direct as the far-horizon anchor itself
best3 = (None, 1e9)
for tau in [90,120,150,180,240]:
    for a in np.arange(0,1.05,0.2):       # weight on seasonal_1y
        for b in np.arange(0,1.05-a,0.2): # weight on doy_clim
            c = 1-a-b                      # weight on ML_direct
            if c < -1e-9: continue
            anchor = a*seasonal_1y + b*doy_clim + c*pred_direct
            s = rmse(y_true, blend(anchor, tau))
            if s < best3[1]: best3 = ((tau,a,b,c), s)
log(f"BEST 3-way anchor blend RMSE={best3[1]:.4f} (tau,w_seas,w_doy,w_ml)={best3[0]}")

# per-station breakdown at best config
tau, a, b, c = best3[0]
anchor = a*seasonal_1y + b*doy_clim + c*pred_direct
final_pred = blend(anchor, tau)
va_feat['pred'] = final_pred
va_feat['err2'] = (va_feat['y_true']-va_feat['pred'])**2
per_stn = va_feat.groupby('nama_pos')['err2'].agg(['mean','count'])
per_stn['rmse'] = np.sqrt(per_stn['mean'])
per_stn['share'] = per_stn['mean']*per_stn['count']/ (va_feat['err2'].sum())
per_stn = per_stn.sort_values('share', ascending=False)
print("\nTOP-10 stations by error share (best config):")
print(per_stn.head(10)[['rmse','share']].to_string())

log("DONE validate_v4.py")
