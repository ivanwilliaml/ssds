"""
Review point #3: backtest row-coverage vs real test was checked (min 0.88 in
FOLD2) but never tested for WHETHER the missing rows are informative -- i.e.
whether they're missing-not-at-random (sensor dropouts during extreme
rainfall/level events), which would mean the backtest quietly skips some of
the hardest rows while the real test (uniform 726 rows/station, zero
missingness by construction of sample_submission) gets no such pass.
"""
import pandas as pd, numpy as np, warnings, time
import feature_lib as fl
warnings.filterwarnings('ignore')
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

RAW, TEST_REAL = fl.load_raw()
dl, dl_feat, static = fl.load_exog()

days = pd.date_range(RAW.datetime.min().normalize(), RAW.datetime.max().normalize(), freq='D')
expected_dt = pd.DatetimeIndex(sorted(d + pd.Timedelta(hours=h) for d in days for h in (6, 12, 18)))
grid = pd.MultiIndex.from_product([RAW.nama_pos.unique(), expected_dt], names=['nama_pos', 'datetime']).to_frame(index=False)
obs = grid.merge(RAW[['nama_pos', 'datetime', 'tma_mdpl']], on=['nama_pos', 'datetime'], how='left')
obs['missing'] = obs['tma_mdpl'].isna()
log(f"expected grid rows={len(grid)}  observed={grid.shape[0]-obs.missing.sum()}  missing={obs.missing.sum()} ({100*obs.missing.mean():.2f}%)")

obs = obs.merge(dl_feat[['nama_pos', 'datetime', 'rainfall_mm_roll24h_sum']], on=['nama_pos', 'datetime'], how='left')
print("\nrainfall_mm_roll24h_sum, missing vs observed rows (pooled all stations):")
print(obs.groupby('missing')['rainfall_mm_roll24h_sum'].describe().to_string())

log("\nper-station missingness rate + mean rainfall on missing vs observed rows:")
per_stn = obs.groupby(['nama_pos', 'missing'])['rainfall_mm_roll24h_sum'].mean().unstack('missing')
per_stn.columns = ['rain_mean_observed', 'rain_mean_missing']
per_stn['missing_rate_pct'] = 100 * obs.groupby('nama_pos')['missing'].mean()
per_stn['rain_ratio_missing_vs_observed'] = per_stn['rain_mean_missing'] / per_stn['rain_mean_observed']
per_stn = per_stn.sort_values('missing_rate_pct', ascending=False)
print(per_stn.head(15).to_string())

# focus on the stations flagged in EDA as having the most missing data
log("\nfocus stations (highest missingness from EDA #5):")
focus = ['Floodway Bridge C', 'Bojonegoro - Kali Kethek', 'Gunungsari']
print(per_stn.loc[per_stn.index.intersection(focus)].to_string())

# is missingness itself correlated with rainfall intensity, pooled?
from scipy import stats
rain_obs = obs.loc[~obs.missing, 'rainfall_mm_roll24h_sum'].dropna()
rain_miss = obs.loc[obs.missing, 'rainfall_mm_roll24h_sum'].dropna()
u, p = stats.mannwhitneyu(rain_miss, rain_obs, alternative='greater')
log(f"\nMann-Whitney U test (H1: missing rows have HIGHER rainfall than observed rows): p-value={p:.4g}")
log(f"median rainfall: missing={rain_miss.median():.3f}  observed={rain_obs.median():.3f}")
if p < 0.05 and rain_miss.median() > rain_obs.median():
    log("CONCLUSION: missingness IS skewed toward wetter conditions -- backtest RMSE is")
    log("  likely an underestimate of real-test difficulty for high-missingness stations.")
else:
    log("CONCLUSION: no strong evidence missingness is rainfall-driven at the pooled level.")
