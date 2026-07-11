"""
Proper upstream-downstream feature using real HydroRIVERS topology instead of
blind cross-correlation. Each station is snapped to its nearest river segment;
DIST_DN_KM (distance to river mouth) on the SAME MAIN_RIV gives a genuine
upstream/downstream ordering -- larger DIST_DN_KM = further upstream.
For each station, the nearest upstream neighbor (smallest DIST_DN_KM gap,
capped at max_gap_km) becomes a candidate predictor; the actual lag (in 6h
steps) is then found empirically via cross-correlation restricted to lag>=0
(physically causal direction only -- this avoids the earlier blind approach
picking up same-time weather-driven correlation as "signal").
"""
import pandas as pd, numpy as np, warnings, time
import feature_lib as fl
warnings.filterwarnings('ignore')
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}")

snap = pd.read_csv(r'D:/Lomba/ssds/model/station_river_snap.csv')
snap = snap[['nama_pos','MAIN_RIV','DIST_DN_KM']].drop_duplicates('nama_pos')

RAW, TEST_REAL = fl.load_raw()
CUT = pd.Timestamp('2024-09-18 18:00:00')
train_only = fl.clean_dataframe(RAW[RAW.datetime <= CUT].copy())
wide = train_only.pivot_table(index='datetime', columns='nama_pos', values='tma_mdpl').sort_index()

# build candidate upstream neighbor per station (same MAIN_RIV, smallest DIST_DN_KM gap upward)
snap_idx = snap.set_index('nama_pos')
upstream_map = {}
for stn in snap['nama_pos']:
    if stn not in wide.columns: continue
    riv, dist = snap_idx.loc[stn, ['MAIN_RIV','DIST_DN_KM']]
    same_riv = snap[(snap.MAIN_RIV==riv) & (snap.nama_pos!=stn) & (snap.DIST_DN_KM>dist)]
    if len(same_riv)==0: continue
    same_riv = same_riv.assign(gap=same_riv.DIST_DN_KM-dist).sort_values('gap')
    cand = same_riv.iloc[0]
    if cand['gap'] > 100:  # cap: don't use a predictor >100km upstream
        continue
    upstream_map[stn] = (cand['nama_pos'], cand['gap'])

log(f"candidate upstream pairs found: {len(upstream_map)}/30")
for stn,(pred,gap) in sorted(upstream_map.items(), key=lambda kv: kv[1][1]):
    print(f"  {stn:28s} <- {pred:28s} (gap={gap:.1f} km)")

# now find best empirical lag (>=0 only) for each pair
final_map = {}
for stn,(pred,gap) in upstream_map.items():
    if stn not in wide.columns or pred not in wide.columns: continue
    y = wide[stn]
    best=(0,-2)
    for lag in [0,1,2,3,4,6,9,12,18,24]:
        c = wide[pred].shift(lag).corr(y)
        if pd.notna(c) and c>best[1]: best=(lag,c)
    if best[1] > 0.3:  # keep only meaningfully correlated pairs
        final_map[stn] = (pred, best[0], best[1])

print(f"\nfinal shapefile-informed upstream map ({len(final_map)} stations):")
for stn,(pred,lag,c) in sorted(final_map.items(), key=lambda kv: -kv[1][2]):
    print(f"  {stn:28s} <- {pred:28s} lag={lag*6:>3}h corr={c:.3f}")

pd.DataFrame([(k,v[0],v[1],v[2]) for k,v in final_map.items()],
             columns=['station','upstream_predictor','lag_steps','corr']).to_csv(
    r'D:/Lomba/ssds/model/upstream_shapefile_map.csv', index=False)
log("saved upstream_shapefile_map.csv")
