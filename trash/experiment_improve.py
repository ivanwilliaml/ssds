"""
Push far-horizon RMSE from ~1.59 toward 1.4-1.5 with TWO non-recursive levers:
  A. richer seasonal features (seasonal_lag_2y, doy climatology mean+std)
  B. horizon-aware PERSISTENCE BLEND at prediction time:
        pred = w(h)*last_known_station + (1-w(h))*direct_pred,  w(h)=exp(-h/tau)
     last_known = the station's last observed value before the forecast origin
     (a single value known at origin; applied only as a post-hoc blend -> clean).

Everything is measured on the SAME far-horizon backtest (train < 2025-01-01,
validate on the ~260-day forward window) that reproduced the 1.85/1.89 leaderboard.
"""
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import lightgbm as lgb
warnings.filterwarnings("ignore")
T0=time.time()
def el(): return f"[{time.time()-T0:6.1f}s]"
ROOT=Path(r"D:\Lomba\ssds"); RS=42

train=pd.read_csv(ROOT/"train.csv",parse_dates=["datetime"])
env=pd.read_csv(ROOT/"data_pendukung"/"data_lingkungan.csv",parse_dates=["datetime"])
coord=pd.read_csv(ROOT/"data_pendukung"/"koordinat_pos.csv")
raw_num=[c for c in env.columns if c not in ("datetime","nama_pos","landcover_name")]
SM=["soil_moisture_0_7cm","soil_moisture_7_28cm","soil_moisture_28_100cm","soil_moisture_100_255cm"]
env=env.sort_values(["nama_pos","datetime"]).reset_index(drop=True)
env["soil_moisture_avg"]=env[SM].mean(axis=1)
env["wind_u"]=env["wind_speed_kmh"]*np.cos(np.deg2rad(env["wind_direction_deg"]))
env["wind_v"]=env["wind_speed_kmh"]*np.sin(np.deg2rad(env["wind_direction_deg"]))
env["temp_dew_spread"]=env["temperature_c"]-env["dew_point_c"]
env["mjo_interaction"]=env["mjo_amplitude"]*env["mjo_active"]
env["humidity_temp_interaction"]=env["humidity_pct"]*env["temperature_c"]/100.0
MEAN=raw_num+["soil_moisture_avg","wind_u","wind_v","temp_dew_spread","mjo_interaction","humidity_temp_interaction"]
SUMC=["rainfall_mm","rainfall_openmeteo_mm"]
def envfeat(env_df,tg):
    e=env_df.sort_values(["nama_pos","datetime"]).set_index("datetime")
    r6=e.groupby("nama_pos",observed=True)[MEAN].rolling("6h",min_periods=1).mean()
    s6=e.groupby("nama_pos",observed=True)[SUMC].rolling("6h",min_periods=1).sum()
    r6=r6.droplevel(0);s6=s6.droplevel(0);r6["nama_pos"]=e["nama_pos"].values
    r6["rainfall_mm"]=s6["rainfall_mm"].values;r6["rainfall_openmeteo_mm"]=s6["rainfall_openmeteo_mm"].values
    for d,l in [(1,"1d"),(3,"3d"),(7,"7d"),(14,"14d")]:
        r=e.groupby("nama_pos",observed=True)["rainfall_mm"].rolling(f"{d}d",min_periods=1).sum()
        r6[f"rainfall_sum_{l}"]=r.droplevel(0).values
    r6=r6.reset_index().sort_values(["datetime"]).reset_index(drop=True)
    t=tg.drop_duplicates().sort_values(["datetime"]).reset_index(drop=True)
    return pd.merge_asof(t,r6,on="datetime",by="nama_pos",direction="nearest",tolerance=pd.Timedelta("1h"))
ef=envfeat(env,train[["nama_pos","datetime"]]).sort_values(["nama_pos","datetime"])
fc=[c for c in ef.columns if c not in ("nama_pos","datetime")]
ef[fc]=ef.groupby("nama_pos")[fc].transform(lambda s:s.ffill().bfill()); ef[fc]=ef[fc].fillna(ef[fc].median())
ef=ef.sort_values(["nama_pos","datetime"]).reset_index(drop=True)
ef["pressure_trend"]=ef.groupby("nama_pos")["surface_pressure_hpa"].transform(lambda s:s-s.shift(3))
ef["soil_moisture_trend"]=ef.groupby("nama_pos")["soil_moisture_avg"].transform(lambda s:s-s.shift(3))
ef[["pressure_trend","soil_moisture_trend"]]=ef[["pressure_trend","soil_moisture_trend"]].fillna(0)
lc=env.sort_values("datetime").groupby("nama_pos")["landcover_name"].last()
train=train.merge(ef,on=["nama_pos","datetime"],how="left")
train["landcover_name"]=train["nama_pos"].map(lc); train=train.merge(coord,on="nama_pos",how="left")
def cal(df):
    df=df.copy();df["hour"]=df["datetime"].dt.hour;df["month"]=df["datetime"].dt.month
    df["day_of_year"]=df["datetime"].dt.dayofyear;df["day_of_week"]=df["datetime"].dt.dayofweek
    df["is_weekend"]=(df["day_of_week"]>=5).astype(int);df["is_wet_season"]=df["month"].isin([11,12,1,2,3,4]).astype(int)
    df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24);df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"]=np.sin(2*np.pi*df["month"]/12);df["month_cos"]=np.cos(2*np.pi*df["month"]/12)
    df["doy_sin"]=np.sin(2*np.pi*df["day_of_year"]/365.25);df["doy_cos"]=np.cos(2*np.pi*df["day_of_year"]/365.25)
    return df
train=cal(train)
cat=["nama_pos","landcover_name"];TGT="tma_mdpl"
for c in cat: train[c]=train[c].astype("category")

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def seasonal(target_df,source_df):
    out=target_df.copy()
    src=source_df[["nama_pos","datetime","tma_mdpl"]].sort_values("datetime")
    for yrs,name in [(365,"seasonal_lag_1y")]:
        s=src.rename(columns={"tma_mdpl":name})
        tmp=out[["nama_pos","datetime"]].copy();tmp["seek"]=tmp["datetime"]-pd.Timedelta(days=yrs);tmp=tmp.sort_values("seek")
        m=pd.merge_asof(tmp,s,left_on="seek",right_on="datetime",by="nama_pos",direction="nearest",tolerance=pd.Timedelta(days=6),suffixes=("","_s"))
        out[name]=m.sort_index()[name].values
    clim=source_df.groupby(["nama_pos",source_df["datetime"].dt.dayofyear])["tma_mdpl"].agg(["mean","std"])
    clim.index.names=["nama_pos","doy"];clim=clim.reset_index().rename(columns={"mean":"doy_clim_mean","std":"doy_clim_std"})
    out=out.merge(clim,left_on=["nama_pos","day_of_year"],right_on=["nama_pos","doy"],how="left").drop(columns=["doy"])
    sm=source_df.groupby("nama_pos")["tma_mdpl"].agg(["mean","std"]).rename(columns={"mean":"station_mean_tma","std":"station_std_tma"})
    out=out.merge(sm,on="nama_pos",how="left")
    fbm=out["nama_pos"].astype(str).map(sm["station_mean_tma"]).astype(float)
    out["doy_clim_mean"]=out["doy_clim_mean"].astype(float).fillna(fbm)
    out["doy_clim_std"]=out["doy_clim_std"].astype(float).fillna(0.0)
    out["seasonal_lag_1y"]=out["seasonal_lag_1y"].astype(float).fillna(out["doy_clim_mean"])
    for c in cat: out[c]=out[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
    return out

exog=(["nama_pos","landcover_name","latitude","longitude","hour_sin","hour_cos","month_sin","month_cos","doy_sin","doy_cos"]
    +MEAN+["rainfall_mm","rainfall_openmeteo_mm"]+[f"rainfall_sum_{l}" for l in ["1d","3d","7d","14d"]]
    +["pressure_trend","soil_moisture_trend","day_of_week","is_weekend","is_wet_season"])
feat=list(dict.fromkeys(exog+["station_mean_tma","station_std_tma","seasonal_lag_1y","doy_clim_mean","doy_clim_std"]))
nc=[c for c in feat if c not in cat]

CUT=pd.Timestamp("2025-01-01")
tr=train[train["datetime"]<CUT]; va=train[train["datetime"]>=CUT]
print(f"{el()} far-horizon: tr={len(tr)} va={len(va)} ({(va['datetime'].max()-va['datetime'].min()).days}d)")
src=tr
tr_n=seasonal(tr,src); va_n=seasonal(va,src)

pre_t=ColumnTransformer([("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),cat),("num",SimpleImputer(strategy="median"),nc)])
pre_l=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),nc)])
ridge=Pipeline([("pre",pre_l),("m",Ridge(alpha=1.0,random_state=RS))]).fit(tr_n[feat],tr_n[TGT])
hgb=Pipeline([("pre",pre_t),("m",HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,l2_regularization=1.0,random_state=RS))]).fit(tr_n[feat],tr_n[TGT])
lgbm=Pipeline([("pre",pre_t),("m",lgb.LGBMRegressor(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,colsample_bytree=0.8,n_jobs=4,random_state=RS,verbosity=-1))]).fit(tr_n[feat],tr_n[TGT])
y=va_n[TGT].values
pr=ridge.predict(va_n[feat]);ph=hgb.predict(va_n[feat]);pl=lgbm.predict(va_n[feat])
print(f"{el()} Ridge={rmse(y,pr):.4f}  HistGB={rmse(y,ph):.4f}  LGBM={rmse(y,pl):.4f}")
# best static ensemble weights (grid over 3-simplex, coarse)
best=(1e9,None)
for wr in np.arange(0,1.01,0.2):
    for wh in np.arange(0,1.01-wr+1e-9,0.2):
        wl=1-wr-wh
        if wl<-1e-9: continue
        r=rmse(y,wr*pr+wh*ph+wl*pl)
        if r<best[0]: best=(r,(round(wr,2),round(wh,2),round(wl,2)))
direct=best[1][0]*pr+best[1][1]*ph+best[1][2]*pl
print(f"{el()} BEST direct ensemble RMSE={best[0]:.4f}  weights(r,h,l)={best[1]}")

# ---- persistence blend ----
last_known=tr.sort_values("datetime").groupby("nama_pos")["tma_mdpl"].last()
va_sorted=va_n.copy()
lk=va_sorted["nama_pos"].astype(str).map(last_known).astype(float).values
h=(va_sorted["datetime"]-CUT).dt.total_seconds().values/86400.0  # days since origin
print(f"{el()} horizon days: min={h.min():.0f} max={h.max():.0f}")
print(f"{el()}   pure persistence (w=1) RMSE={rmse(y,lk):.4f}")
best_blend=(best[0],None)
for tau in [30,45,60,90,120,150,180,240,300]:
    w=np.exp(-h/tau)
    blended=w*lk+(1-w)*direct
    r=rmse(y,blended)
    tag=" <--" if r<best_blend[0] else ""
    if r<best_blend[0]: best_blend=(r,tau)
    print(f"{el()}   persistence blend tau={tau:>3}  RMSE={r:.4f}{tag}")
print(f"\n{el()} BEST OVERALL far-horizon RMSE={best_blend[1] and best_blend[0]:.4f}  (tau={best_blend[1]}, direct-only={best[0]:.4f})")
print(f"{el()} DONE ({time.time()-T0:.1f}s)")

# save chosen recipe for deployment
import json
json.dump({"ens_weights":best[1],"tau":best_blend[1],"features":feat}, open(ROOT/"model"/"improve_recipe.json","w"))
