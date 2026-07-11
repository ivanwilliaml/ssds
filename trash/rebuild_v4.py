"""
SSDS 2026 - Full rebuild v4
Pipeline: clean -> feature engineer -> season-matched backtest -> final model+submission
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

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}")

# ============ 1. LOAD ============
tr = pd.read_csv(r'D:/Lomba/ssds/train.csv', parse_dates=['datetime'])
te_raw = pd.read_csv(r'D:/Lomba/ssds/test.csv')
split = te_raw['id'].str.split(' - ', n=1, expand=True)
te = pd.DataFrame({'datetime': pd.to_datetime(split[0]), 'nama_pos': split[1]})
dl = pd.read_csv(r'D:/Lomba/ssds/data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
ko = pd.read_csv(r'D:/Lomba/ssds/data_pendukung/koordinat_pos.csv')
log(f"loaded train={tr.shape} test={te.shape} dl={dl.shape}")

# ============ 2. CLEAN TARGET (outlier spikes / negative values) ============
# Sensor glitch signature: single-timestep spike where value jumps far from both
# neighbors and reverts immediately (confirmed via EDA: Napel 2023-04-10, etc.)
tr = tr.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def clean_station(g):
    g = g.copy()
    v = g['tma_mdpl'].values.copy()
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    # robust z-score
    rz = 0.6745 * (v - med) / mad
    prev = np.r_[v[0], v[:-1]]
    nxt = np.r_[v[1:], v[-1]]
    # a point is a spike if it's far (robust z) from BOTH neighbors while neighbors
    # are close to each other (isolated single-point glitch, not a real sustained rise)
    neigh_close = np.abs(prev - nxt) < 0.3 * (np.abs(prev) + np.abs(nxt) + 1e-6)
    far_prev = np.abs(v - prev) > 5 * mad
    far_nxt = np.abs(v - nxt) > 5 * mad
    is_spike = (np.abs(rz) > 6) & neigh_close & far_prev & far_nxt
    v[is_spike] = np.nan
    # negative / physically implausible (TMA should be >= 0)
    v[v < 0] = np.nan
    g['tma_mdpl'] = v
    g['tma_mdpl'] = g['tma_mdpl'].interpolate(limit_direction='both')
    g['is_cleaned'] = is_spike | (g['tma_mdpl'].values < 0)
    return g

tr = tr.groupby('nama_pos', group_keys=False).apply(clean_station)
n_cleaned = tr['is_cleaned'].sum()
log(f"cleaned {n_cleaned} spike/negative points out of {len(tr)} ({100*n_cleaned/len(tr):.2f}%)")
tr = tr.drop(columns=['is_cleaned'])

# ============ 3. STATION STATIC ATTRIBUTES ============
static = dl.groupby('nama_pos').agg(
    landcover_class=('landcover_class', 'first'),
    built_surface_m2=('built_surface_m2', 'first'),
).reset_index()
static = static.merge(ko, on='nama_pos', how='left')
log(f"static attrs shape={static.shape}")

# ============ 4. EXOGENOUS FEATURES aggregated to 6h obs times ============
# dl is hourly; TMA obs at 06/12/18. For each obs, use env data up to AND INCLUDING
# that hour (no future leakage) with rolling windows capturing recent conditions.
dl = dl.sort_values(['nama_pos', 'datetime'])
dl['nino_34'] = dl.groupby('nama_pos')['nino_34'].ffill().bfill()
for c in ['soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm',
          'soil_moisture_100_255cm','surface_pressure_hpa','pressure_msl_hpa',
          'rmm1','rmm2','mjo_phase','mjo_amplitude','mjo_active']:
    dl[c] = dl.groupby('nama_pos')[c].ffill().bfill()

roll_specs = {
    'rainfall_mm': [24, 72, 168],       # 1d,3d,7d accum (sum)
    'temperature_c': [24],
    'humidity_pct': [24],
    'soil_moisture_0_7cm': [24],
    'soil_moisture_28_100cm': [24],
    'surface_pressure_hpa': [24],
    'pressure_msl_hpa': [24],
}
dl_feat = dl[['datetime', 'nama_pos']].copy()
for col, windows in roll_specs.items():
    for w in windows:
        agg = 'sum' if col == 'rainfall_mm' else 'mean'
        dl_feat[f'{col}_roll{w}h_{agg}'] = (
            dl.groupby('nama_pos')[col]
              .transform(lambda s: s.rolling(w, min_periods=max(1, w//4)).agg(agg))
        )
# snapshot (instantaneous) exogenous values at the hour itself
snap_cols = ['rainfall_mm','humidity_pct','dew_point_c','cloud_cover_pct','temperature_c',
             'wind_speed_kmh','soil_moisture_0_7cm','soil_moisture_7_28cm',
             'soil_moisture_28_100cm','soil_moisture_100_255cm','surface_pressure_hpa',
             'pressure_msl_hpa','rmm1','rmm2','mjo_phase','mjo_amplitude','mjo_active','nino_34']
for c in snap_cols:
    dl_feat[c] = dl[c].values
log(f"dl_feat built shape={dl_feat.shape}")

def merge_exog(df):
    out = df.merge(dl_feat, on=['datetime', 'nama_pos'], how='left')
    return out

tr = merge_exog(tr)
te = merge_exog(te)
log(f"after exog merge: train={tr.shape} test={te.shape} test_na_exog={te[snap_cols].isna().sum().sum()}")

# ============ 5. CALENDAR FEATURES ============
def add_calendar(df):
    df = df.copy()
    df['hour'] = df.datetime.dt.hour
    df['month'] = df.datetime.dt.month
    df['doy'] = df.datetime.dt.dayofyear
    df['hour_sin'] = np.sin(2*np.pi*df.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*df.hour/24)
    df['month_sin'] = np.sin(2*np.pi*df.month/12)
    df['month_cos'] = np.cos(2*np.pi*df.month/12)
    df['doy_sin'] = np.sin(2*np.pi*df.doy/365.25)
    df['doy_cos'] = np.cos(2*np.pi*df.doy/365.25)
    df['is_wet_season'] = df.month.isin([11,12,1,2,3,4]).astype(int)
    return df

tr = add_calendar(tr)
te = add_calendar(te)

# ============ 6. STATION ATTRIBUTES + CLIMATOLOGY (computed on CLEANED train only) ============
tr = tr.merge(static, on='nama_pos', how='left')
te = te.merge(static, on='nama_pos', how='left')

stn_stats = tr.groupby('nama_pos')['tma_mdpl'].agg(station_mean='mean', station_std='std', station_median='median').reset_index()
tr = tr.merge(stn_stats, on='nama_pos', how='left')
te = te.merge(stn_stats, on='nama_pos', how='left')

# day-of-year climatology per station on CLEANED data (smoothed with +-3 day window)
doy_clim_list = []
for stn, g in tr.groupby('nama_pos'):
    s = g.set_index('doy')['tma_mdpl']
    means = {}
    for d in range(1, 367):
        window = [((d + off - 1) % 366) + 1 for off in range(-5, 6)]
        vals = s[s.index.isin(window)]
        means[d] = vals.mean() if len(vals) else np.nan
    doy_clim_list.append(pd.DataFrame({'nama_pos': stn, 'doy': list(means.keys()), 'doy_climatology': list(means.values())}))
doy_clim = pd.concat(doy_clim_list, ignore_index=True)
tr = tr.merge(doy_clim, on=['nama_pos', 'doy'], how='left')
te = te.merge(doy_clim, on=['nama_pos', 'doy'], how='left')
fallback = stn_stats.set_index('nama_pos')['station_mean']
tr['doy_climatology'] = tr['doy_climatology'].fillna(tr['nama_pos'].map(fallback))
te['doy_climatology'] = te['doy_climatology'].fillna(te['nama_pos'].map(fallback))
log("doy_climatology built (cleaned, smoothed +-5d)")

# seasonal_lag_1y: value from ~365 days prior, via merge_asof on CLEANED series (tolerance 2 days)
def add_seasonal_lag(target_df, source_df, days=365, tol_days=2, name='seasonal_lag_1y'):
    src = source_df[['datetime', 'nama_pos', 'tma_mdpl']].copy()
    src = src.rename(columns={'tma_mdpl': name, 'datetime': 'src_dt'})
    src = src.sort_values('src_dt')
    tgt = target_df.copy()
    tgt['lookup_dt'] = tgt['datetime'] - pd.Timedelta(days=days)
    tgt = tgt.sort_values('lookup_dt')
    out = pd.merge_asof(tgt, src.sort_values('src_dt'), left_on='lookup_dt', right_on='src_dt',
                         by='nama_pos', direction='nearest', tolerance=pd.Timedelta(days=tol_days))
    out = out.drop(columns=['lookup_dt', 'src_dt'])
    return out.sort_index()

tr = add_seasonal_lag(tr, tr)
te = add_seasonal_lag(te, tr)  # test lag must come from train (no leakage), source=cleaned train
tr['seasonal_lag_1y'] = tr['seasonal_lag_1y'].fillna(tr['doy_climatology'])
te['seasonal_lag_1y'] = te['seasonal_lag_1y'].fillna(te['doy_climatology'])
log("seasonal_lag_1y rebuilt on cleaned data")

# ============ 7. LAST-KNOWN (persistence) value & horizon from train end (for blending) ============
last_known = tr.sort_values('datetime').groupby('nama_pos').tail(1)[['nama_pos', 'datetime', 'tma_mdpl']]
last_known = last_known.rename(columns={'datetime': 'last_dt', 'tma_mdpl': 'last_known'})
last_known_map = last_known.set_index('nama_pos')

def add_persistence(df, origin_map):
    df = df.copy()
    df['last_known'] = df['nama_pos'].map(origin_map['last_known'])
    df['last_dt'] = df['nama_pos'].map(origin_map['last_dt'])
    df['horizon_days'] = (df['datetime'] - df['last_dt']).dt.total_seconds() / 86400
    return df.drop(columns=['last_dt'])

te = add_persistence(te, last_known_map)
log(f"persistence anchor added, horizon range: {te.horizon_days.min():.1f} to {te.horizon_days.max():.1f}")

train_end_global = tr.datetime.max()
log(f"FEATURE ENGINEERING DONE. train={tr.shape} test={te.shape}")

tr.to_parquet(r'D:/Lomba/ssds/model/tr_feat_v4.parquet')
te.to_parquet(r'D:/Lomba/ssds/model/te_feat_v4.parquet')
log("saved tr_feat_v4.parquet / te_feat_v4.parquet")
