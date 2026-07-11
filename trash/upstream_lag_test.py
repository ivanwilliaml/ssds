"""
Test point #5 from review (lighter version): instead of parsing the
HydroRIVERS shapefile for exact upstream/downstream topology, empirically
find lead-lag relationships between station pairs via cross-correlation of
the cleaned tma_mdpl series. If station B's level at time t correlates more
strongly with station A's level at time t-k (k>0) than with A at t itself,
A is a likely upstream predictor of B with travel-time k.
"""
import pandas as pd, numpy as np, warnings, time
import feature_lib as fl
warnings.filterwarnings('ignore')
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

RAW, TEST_REAL = fl.load_raw()
CUT = pd.Timestamp('2024-09-18 18:00:00')
train_only = fl.clean_dataframe(RAW[RAW.datetime <= CUT].copy())

# pivot to wide: rows=datetime, cols=station, on the common 3x/day grid
wide = train_only.pivot_table(index='datetime', columns='nama_pos', values='tma_mdpl')
wide = wide.sort_index()
log(f"wide shape={wide.shape} (timestamps x stations)")

stations = wide.columns.tolist()
best_lead = {}  # station -> (best_predictor_station, lag_steps, corr)
for stn in stations:
    y = wide[stn]
    best = (None, 0, -2)
    for other in stations:
        if other == stn: continue
        x = wide[other]
        for lag in [0,1,2,3,6,9,12]:  # steps of ~6h -> up to 3 days
            xs = x.shift(lag)
            c = xs.corr(y)
            if pd.notna(c) and c > best[2]:
                best = (other, lag, c)
    best_lead[stn] = best

res = pd.DataFrame(best_lead, index=['best_predictor','lag_steps','corr']).T
res['lag_hours'] = res['lag_steps']*6
res = res.sort_values('corr', ascending=False)
print(res.to_string())
res.to_csv(r'D:/Lomba/ssds/model/upstream_lag_map.csv')
log("saved upstream_lag_map.csv")

# how many stations have a meaningfully strong (>0.5) AND lagged (lag>0) predictor?
strong_lagged = res[(res['corr']>0.5) & (res['lag_steps']>0)]
print(f"\nstations with corr>0.5 AND lag>0 (genuine upstream signal candidate): {len(strong_lagged)}/{len(res)}")
print(strong_lagged.to_string())
