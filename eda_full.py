import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
t0=time.time()
pd.set_option('display.width',160)

tr = pd.read_csv(r'D:/Lomba/ssds/train.csv', parse_dates=['datetime'])
te_raw = pd.read_csv(r'D:/Lomba/ssds/test.csv')
dl = pd.read_csv(r'D:/Lomba/ssds/data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
ko = pd.read_csv(r'D:/Lomba/ssds/data_pendukung/koordinat_pos.csv')

# split test id into datetime/nama_pos
split = te_raw['id'].str.split(' - ', n=1, expand=True)
te = pd.DataFrame({'datetime': pd.to_datetime(split[0]), 'nama_pos': split[1], 'id': te_raw['id']})

def sec(n): print(f"\n===== EDA {n} =====")

# 1. Basic shape/schema sanity
sec(1)
print('train', tr.shape, 'test', te.shape, 'stations_train', tr.nama_pos.nunique(), 'stations_test', te.nama_pos.nunique())
print('station set equal:', set(tr.nama_pos)==set(te.nama_pos))
print('train dtypes:', tr.dtypes.to_dict())

# 2. Missing values
sec(2)
print('train NA:\n', tr.isna().sum())
print('dl NA:\n', dl.isna().sum()[dl.isna().sum()>0])

# 3. Duplicates
sec(3)
print('train dup rows', tr.duplicated().sum(), 'dup key', tr.duplicated(['datetime','nama_pos']).sum())
print('test dup key', te.duplicated(['datetime','nama_pos']).sum())
print('dl dup key', dl.duplicated(['datetime','nama_pos']).sum())

# 4. Target distribution overall + per-station range/scale heterogeneity
sec(4)
print(tr.tma_mdpl.describe())
neg = tr[tr.tma_mdpl<=0]
print('non-positive rows:\n', neg)
stn_stats = tr.groupby('nama_pos').tma_mdpl.agg(['min','max','mean','std','count']).sort_values('mean')
print(stn_stats.to_string())

# 5. Temporal coverage / gaps per station (train)
sec(5)
expected = pd.date_range(tr.datetime.min(), tr.datetime.max(), freq='6h')  # not exact due to 06/12/18 pattern but gives gap sense
g = tr.groupby('nama_pos').datetime.agg(['min','max','count'])
g['expected_3xday'] = ((g['max']-g['min']).dt.days+1)*3
g['missing_frac'] = 1 - g['count']/g['expected_3xday']
print(g.sort_values('missing_frac', ascending=False).to_string())

# 6. Test set structure: horizon length, gap from train end
sec(6)
train_end = tr.datetime.max()
print('train_end', train_end, 'test_start', te.datetime.min(), 'test_end', te.datetime.max())
print('gap days (test_start - train_end):', (te.datetime.min()-train_end).days)
print('horizon days (test_end-test_start):', (te.datetime.max()-te.datetime.min()).days)
te_counts = te.groupby('nama_pos').datetime.count()
print('rows per station in test (should be uniform):', te_counts.unique())

# 7. Seasonal cycle of target (monthly mean) -> is there a wet/dry season signal test will span
sec(7)
tr['month']=tr.datetime.dt.month
mon = tr.groupby('month').tma_mdpl.mean()
print('monthly mean tma (all stations pooled, mind scale diff):\n', mon)
test_months = sorted(te.datetime.dt.month.unique())
print('months covered by TEST:', test_months, ' -> spans wet season (Nov-Apr) peak fully')

# 8. Per-station day-of-year climatology check: does day-262(Sep19)->day138(May18) span exist in train history at all stations
sec(8)
tr['doy']=tr.datetime.dt.dayofyear
te['doy']=te.datetime.dt.dayofyear
print('train doy range per year available - years:', tr.datetime.dt.year.unique())
# does train contain a FULL prior wet season analogous to test's wet season (Sep James Y-1 to May Y)?
print('train max date used to have prior-year-same-doy target available for early test rows (doy 262):')
mask = (tr.datetime.dt.year==2024) & (tr.doy==262)
print(tr[mask][['nama_pos','datetime','tma_mdpl']].head())

# 9. Outlier scan via z-score per station (train)
sec(9)
def zscan(g):
    z = (g.tma_mdpl - g.tma_mdpl.mean())/g.tma_mdpl.std()
    return (z.abs()>4).sum()
outl = tr.groupby('nama_pos').apply(zscan)
print('stations with |z|>4 outlier count (train):\n', outl[outl>0].sort_values(ascending=False))

# 10. Sudden jump/spike detection (diff between consecutive obs per station)
sec(10)
tr_sorted = tr.sort_values(['nama_pos','datetime'])
tr_sorted['diff'] = tr_sorted.groupby('nama_pos').tma_mdpl.diff()
big_jump = tr_sorted.reindex(tr_sorted['diff'].abs().sort_values(ascending=False).index).head(15)
print(big_jump[['nama_pos','datetime','tma_mdpl','diff']])

# 11. Correlation of exogenous features with tma (merged on nearest hour) - sample a few stations
sec(11)
dl6 = dl[dl.datetime.dt.hour.isin([6,12,18])]
merged = tr.merge(dl6, on=['datetime','nama_pos'], how='left')
num_cols = ['rainfall_mm','humidity_pct','dew_point_c','cloud_cover_pct','temperature_c',
            'soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm','soil_moisture_100_255cm',
            'surface_pressure_hpa','pressure_msl_hpa','nino_34','rmm1','rmm2','mjo_amplitude']
corrs = merged[num_cols+['tma_mdpl']].corr()['tma_mdpl'].drop('tma_mdpl').sort_values(key=abs, ascending=False)
print('corr(exog, tma) pooled all stations (mind Simpson paradox across scale):\n', corrs)

# 12. Per-station correlation with rainfall/soil moisture (captures true local hydrology signal)
sec(12)
def corr_local(g):
    if g['rainfall_mm'].notna().sum()<50: return np.nan
    return g['rainfall_mm'].corr(g['tma_mdpl'])
rc = merged.groupby('nama_pos').apply(corr_local).sort_values()
print('per-station corr(rainfall, tma):\n', rc)

# 13. landcover / built_surface static per station - does it vary over time (should be near-static -> use as station attribute)
sec(13)
lc = dl.groupby('nama_pos')['landcover_class'].nunique()
print('landcover_class nunique per station (1=static):\n', lc.value_counts())
bs = dl.groupby('nama_pos')['built_surface_m2'].std()
print('built_surface_m2 std per station (near 0 = static):\n', bs.describe())

# 14. nino_34 / MJO resolution (monthly/daily) - check update frequency, useful for feature freq handling
sec(14)
print('unique nino_34 values per station (should repeat monthly):', dl.groupby('nama_pos').nino_34.apply(lambda s: s.diff().ne(0).sum()).mean())
print('unique mjo_phase transitions per day approx:', dl.groupby(dl.datetime.dt.date).mjo_phase.nunique().mean())

# 15. Autocorrelation of tma_mdpl at daily lag vs weekly vs yearly (single representative station: Jurug - high err share)
sec(15)
jurug = tr[tr.nama_pos=='Jurug'].sort_values('datetime').set_index('datetime').tma_mdpl.asfreq('6h' if False else None)
s = tr[tr.nama_pos=='Jurug'].sort_values('datetime')[['datetime','tma_mdpl']].set_index('datetime')['tma_mdpl']
for lag_days in [1,3,7,30,90,180,365]:
    lag_steps = lag_days*3  # approx since 3 obs/day, assumes no gaps -- rough
    if lag_steps < len(s):
        ac = s.autocorr(lag=lag_steps)
        print(f'Jurug autocorr at ~{lag_days}d lag (steps={lag_steps}):', round(ac,4))

print(f"\nDONE eda_full.py in {time.time()-t0:.1f}s")
