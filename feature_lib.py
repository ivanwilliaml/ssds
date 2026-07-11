"""
Shared feature-engineering library for the SSDS 2026 v4 pipeline.
Consolidates logic previously duplicated across rebuild_v4.py, validate_v4.py,
and multi_fold_validate.py (flagged in review as risk of drift between copies).

Key fix vs earlier version: clean_station() must be called ONLY on the
train-side slice of a given fold (data <= CUT), never on the full RAW series
before splitting, otherwise the median/MAD reference used for spike detection
is contaminated by future data relative to that fold.
"""
import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TRAIN_CSV = r'D:/Lomba/ssds/train.csv'
TEST_CSV = r'D:/Lomba/ssds/test.csv'
DL_CSV = r'D:/Lomba/ssds/data_pendukung/data_lingkungan.csv'
KO_CSV = r'D:/Lomba/ssds/data_pendukung/koordinat_pos.csv'

FEATURE_COLS_NUM = ['rainfall_mm_roll24h_sum','rainfall_mm_roll72h_sum','rainfall_mm_roll168h_sum',
    'temperature_c_roll24h_mean','humidity_pct_roll24h_mean','soil_moisture_0_7cm_roll24h_mean',
    'soil_moisture_28_100cm_roll24h_mean','surface_pressure_hpa_roll24h_mean','pressure_msl_hpa_roll24h_mean',
    'rainfall_mm','humidity_pct','dew_point_c','cloud_cover_pct','temperature_c','wind_speed_kmh',
    'soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm','soil_moisture_100_255cm',
    'surface_pressure_hpa','pressure_msl_hpa','rmm1','rmm2','mjo_phase','mjo_amplitude','mjo_active','nino_34',
    'hour_sin','hour_cos','month_sin','month_cos','doy_sin','doy_cos','is_wet_season',
    'latitude','longitude','built_surface_m2',
    'station_mean','station_std','station_median','doy_climatology','seasonal_lag_1y','upstream_lag_value']
CAT_COLS = ['nama_pos','landcover_class']
XCOLS = CAT_COLS + FEATURE_COLS_NUM

ROLL_SPECS = {'rainfall_mm':[24,72,168],'temperature_c':[24],'humidity_pct':[24],
              'soil_moisture_0_7cm':[24],'soil_moisture_28_100cm':[24],
              'surface_pressure_hpa':[24],'pressure_msl_hpa':[24]}
SNAP_COLS = ['rainfall_mm','humidity_pct','dew_point_c','cloud_cover_pct','temperature_c',
             'wind_speed_kmh','soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm',
             'soil_moisture_100_255cm','surface_pressure_hpa','pressure_msl_hpa','rmm1','rmm2',
             'mjo_phase','mjo_amplitude','mjo_active','nino_34']


def load_raw():
    tr = pd.read_csv(TRAIN_CSV, parse_dates=['datetime']).sort_values(['nama_pos','datetime']).reset_index(drop=True)
    te_raw = pd.read_csv(TEST_CSV)
    split = te_raw['id'].str.split(' - ', n=1, expand=True)
    te = pd.DataFrame({'datetime': pd.to_datetime(split[0]), 'nama_pos': split[1]})
    return tr, te


def load_exog():
    dl = pd.read_csv(DL_CSV, parse_dates=['datetime']).sort_values(['nama_pos','datetime'])
    ko = pd.read_csv(KO_CSV)
    static = dl.groupby('nama_pos').agg(landcover_class=('landcover_class','first'),
                                         built_surface_m2=('built_surface_m2','first')).reset_index()
    static = static.merge(ko, on='nama_pos', how='left')

    dl['nino_34'] = dl.groupby('nama_pos')['nino_34'].ffill().bfill()
    for c in ['soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm',
              'soil_moisture_100_255cm','surface_pressure_hpa','pressure_msl_hpa',
              'rmm1','rmm2','mjo_phase','mjo_amplitude','mjo_active']:
        dl[c] = dl.groupby('nama_pos')[c].ffill().bfill()

    dl_feat = dl[['datetime','nama_pos']].copy()
    for col, windows in ROLL_SPECS.items():
        for w in windows:
            agg = 'sum' if col == 'rainfall_mm' else 'mean'
            dl_feat[f'{col}_roll{w}h_{agg}'] = dl.groupby('nama_pos')[col].transform(
                lambda s: s.rolling(w, min_periods=max(1, w // 4)).agg(agg))
    for c in SNAP_COLS:
        dl_feat[c] = dl[c].values
    return dl, dl_feat, static


def clean_station(g, mad_z=6.0, mad_mult=5.0, neigh_frac=0.3):
    """Detect & interpolate isolated single-timestep sensor spikes.
    A point qualifies as a spike if: (a) it's far from BOTH immediate
    neighbors (in MAD units) AND (b) the two neighbors are themselves close
    to each other (rules out a genuine sustained rise/fall)."""
    g = g.copy()
    v = g['tma_mdpl'].values.copy()
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    rz = 0.6745 * (v - med) / mad
    prev = np.r_[v[0], v[:-1]]
    nxt = np.r_[v[1:], v[-1]]
    neigh_close = np.abs(prev - nxt) < neigh_frac * (np.abs(prev) + np.abs(nxt) + 1e-6)
    far_prev = np.abs(v - prev) > mad_mult * mad
    far_nxt = np.abs(v - nxt) > mad_mult * mad
    is_spike = (np.abs(rz) > mad_z) & neigh_close & far_prev & far_nxt
    v[is_spike] = np.nan
    v[v < 0] = np.nan
    g['tma_mdpl'] = v
    g['tma_mdpl'] = g['tma_mdpl'].interpolate(limit_direction='both')
    return g


def clean_dataframe(df):
    """Apply clean_station per station. Caller MUST ensure df is already the
    correct train-only slice for the fold being evaluated (no future leakage)."""
    return df.groupby('nama_pos', group_keys=False).apply(clean_station)


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


def build_doy_clim(train_only):
    if 'doy' not in train_only.columns:
        train_only = train_only.assign(doy=train_only['datetime'].dt.dayofyear)
    out = []
    for stn, g in train_only.groupby('nama_pos'):
        s = g.set_index('doy')['tma_mdpl']
        means = {}
        for d in range(1, 367):
            window = [((d + off - 1) % 366) + 1 for off in range(-5, 6)]
            vals = s[s.index.isin(window)]
            means[d] = vals.mean() if len(vals) else np.nan
        out.append(pd.DataFrame({'nama_pos': stn, 'doy': list(means.keys()), 'doy_climatology': list(means.values())}))
    return pd.concat(out, ignore_index=True)


def add_seasonal_lag(target_df, source_df, days=365, tol_days=2, name='seasonal_lag_1y'):
    src = source_df[['datetime','nama_pos','tma_mdpl']].rename(columns={'tma_mdpl': name, 'datetime': 'src_dt'}).sort_values('src_dt')
    tgt = target_df.copy()
    tgt['lookup_dt'] = tgt['datetime'] - pd.Timedelta(days=days)
    tgt = tgt.sort_values('lookup_dt')
    out = pd.merge_asof(tgt, src, left_on='lookup_dt', right_on='src_dt', by='nama_pos',
                         direction='nearest', tolerance=pd.Timedelta(days=tol_days))
    return out.drop(columns=['lookup_dt', 'src_dt']).sort_index()


def build_features(train_only, target_df, dl_feat, static):
    """train_only: CLEANED rows with datetime<=CUT, used for all fold statistics.
       target_df: rows to build features for (train_only itself, or validation/test rows)."""
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
    df = add_upstream_lag(df, train_only)
    df['upstream_lag_value'] = df['upstream_lag_value'].fillna(df['doy_climatology'])
    last_known = train_only.sort_values('datetime').groupby('nama_pos').tail(1)[['nama_pos','datetime','tma_mdpl']]
    lk_map = last_known.set_index('nama_pos')
    df['last_known'] = df['nama_pos'].map(lk_map['tma_mdpl'])
    last_dt = df['nama_pos'].map(lk_map['datetime'])
    df['horizon_days'] = (df['datetime'] - last_dt).dt.total_seconds() / 86400
    return df


def make_pipelines(ridge_alpha=5.0, histgb_depth=6, histgb_lr=0.05, histgb_iter=400):
    pre_tree = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS)], remainder='passthrough')
    pre_lin = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS),
                                  ('num', StandardScaler(), FEATURE_COLS_NUM)], remainder='drop')
    ridge = Pipeline([('pre', pre_lin), ('model', Ridge(alpha=ridge_alpha))])
    histgb = Pipeline([('pre', pre_tree), ('model', HistGradientBoostingRegressor(
        max_depth=histgb_depth, learning_rate=histgb_lr, max_iter=histgb_iter, random_state=0))])
    return ridge, histgb


# Empirically-derived upstream lead-lag map (see upstream_lag_test.py):
# station -> (upstream predictor station, lag in 6h-steps). Built from REAL
# HydroRIVERS topology (upstream_shapefile_test.py): each station snapped to
# its nearest river segment, DIST_DN_KM (distance to river mouth) on the same
# MAIN_RIV gives genuine upstream/downstream order, capped at 100km gap.
# The lag itself is then found empirically via cross-correlation restricted
# to lag>=0 (physically causal direction only, since direction is already
# fixed by the topology -- unlike the earlier blind pairwise-correlation
# version, lag=0 here is trustworthy: it means fast travel time within one
# 6h sampling step, not spurious shared-weather correlation).
# NOTE: two shapefile-topology-derived variants were tested and NEITHER
# improved on this smaller empirical map (see upstream_shapefile_test.py /
# upstream_shapefile_map.csv for the full 23-pair candidate set derived from
# real HydroRIVERS upstream/downstream ordering):
#   - full replacement (23 pairs): FOLD2 RMSE 1.4433 -> 1.4707 (WORSE, likely
#     HistGB overfitting on the extra columns within a fold-sized train set)
#   - hybrid (add shapefile pairs only for Jurug/Peren, the top error
#     contributors with no entry here): FOLD2 RMSE 1.4433 -> 1.4442
#     (statistically negligible, within noise)
# Kept as-is: whatever signal exists in the shapefile-informed pairs appears
# to already be captured by other features (rolling exogenous windows,
# seasonal_lag_1y, doy_climatology).
UPSTREAM_MAP = {
    'Bojonegoro - Kali Kethek': ('Cepu', 1),
    'Karanggeneng': ('Sumberrejo', 1),
    'Boboh Kali Lamong': ('Bengkelolor', 1),
    'Wonogiri Dam': ('Karanggeneng', 12),
    'Floodway Bridge C': ('Bojonegoro - Kali Kethek', 2),
    'Kali Anyar - Kreteg Abang': ('Wonogiri Dam', 9),
}


def add_upstream_lag(target_df, source_df):
    """For stations with a known empirical upstream predictor, add the
    predictor's value `lag_steps*6h` before each target timestamp (merge_asof,
    nearest within 3h tolerance). Falls back to NaN (caller should fillna)
    for stations without a qualifying upstream pair."""
    df = target_df.copy()
    df['upstream_lag_value'] = np.nan
    for stn, (pred_stn, lag_steps) in UPSTREAM_MAP.items():
        mask = df['nama_pos'] == stn
        if not mask.any():
            continue
        src = source_df[source_df['nama_pos'] == pred_stn][['datetime', 'tma_mdpl']].rename(
            columns={'tma_mdpl': 'upstream_val', 'datetime': 'src_dt'}).sort_values('src_dt')
        sub = df.loc[mask, ['datetime']].copy()
        sub['lookup_dt'] = sub['datetime'] - pd.Timedelta(hours=6 * lag_steps)
        sub = sub.sort_values('lookup_dt')
        merged = pd.merge_asof(sub, src, left_on='lookup_dt', right_on='src_dt',
                                direction='nearest', tolerance=pd.Timedelta(hours=3))
        df.loc[mask, 'upstream_lag_value'] = merged.sort_index()['upstream_val'].values
    return df


def climate_regime(dl, start, end):
    """Return mean nino_34 and mean rainfall_mm for a datetime window, used to
    check whether a validation fold's climate regime resembles the real test
    period (both computed from the SAME data_lingkungan.csv source)."""
    mask = (dl['datetime'] >= pd.Timestamp(start)) & (dl['datetime'] <= pd.Timestamp(end))
    sub = dl.loc[mask]
    return {
        'nino34_mean': sub['nino_34'].mean(),
        'rainfall_mean': sub['rainfall_mm'].mean(),
        'rainfall_total_per_station': sub['rainfall_mm'].sum() / sub['nama_pos'].nunique(),
    }
