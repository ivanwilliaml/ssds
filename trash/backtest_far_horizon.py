"""
DEFINITIVE far-horizon backtest — simulates the real test condition (forecasting
~8 months ahead) to prove which approach actually wins on a long horizon.

Setup: cutoff = 2025-01-01. Train on everything before it, validate on
2025-01-01 .. 2025-09-18 (~8.5 months, ~750 timesteps) — comparable in length to
the real test window (Sep 2025 .. May 2026).

Compares over the SAME far window:
  OLD  = lag-based RF+LR ensemble, RECURSIVE forecast (accumulates error)
  NEW  = no-lag + seasonal Ridge+HistGB ensemble, DIRECT (no accumulation)
"""
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
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
LAGS=[1,2,3,6,9,21];ROLLW=[3,21]
train=train.sort_values(["nama_pos","datetime"]).reset_index(drop=True)
g=train.groupby("nama_pos",observed=True)["tma_mdpl"]
for l in LAGS: train[f"lag_{l}"]=g.shift(l)
sh=g.shift(1)
for w in ROLLW:
    train[f"roll_mean_{w}"]=sh.groupby(train["nama_pos"],observed=True).transform(lambda s:s.rolling(w,min_periods=1).mean())
    train[f"roll_std_{w}"]=sh.groupby(train["nama_pos"],observed=True).transform(lambda s:s.rolling(w,min_periods=1).std())
train["tma_diff_1"]=train["lag_1"]-train["lag_2"];train["tma_diff_1d"]=train["lag_1"]-train["lag_3"];train["tma_diff_7d"]=train["lag_1"]-train["lag_21"]

exog=(["nama_pos","landcover_name","latitude","longitude","hour_sin","hour_cos","month_sin","month_cos","doy_sin","doy_cos"]
    +MEAN+["rainfall_mm","rainfall_openmeteo_mm"]+[f"rainfall_sum_{l}" for l in ["1d","3d","7d","14d"]]
    +["pressure_trend","soil_moisture_trend","day_of_week","is_weekend","is_wet_season"])
lagcols=[f"lag_{l}" for l in LAGS]+[f"roll_mean_{w}" for w in ROLLW]+[f"roll_std_{w}" for w in ROLLW]+["tma_diff_1","tma_diff_1d","tma_diff_7d"]
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

CUT=pd.Timestamp("2025-01-01")
tr=train[train["datetime"]<CUT].dropna(subset=[f"lag_{max(LAGS)}"])
va=train[train["datetime"]>=CUT]
print(f"{el()} FAR-HORIZON backtest: tr<{CUT.date()} ({len(tr)}), va>= ({len(va)}, ~{ (va['datetime'].max()-va['datetime'].min()).days } days)")

def seasonal(target_df,source_df,catcats):
    out=target_df.copy()
    src=source_df[["nama_pos","datetime","tma_mdpl"]].sort_values("datetime").rename(columns={"tma_mdpl":"seasonal_lag_1y"})
    tmp=out[["nama_pos","datetime"]].copy();tmp["seek"]=tmp["datetime"]-pd.Timedelta(days=365);tmp=tmp.sort_values("seek")
    m=pd.merge_asof(tmp,src,left_on="seek",right_on="datetime",by="nama_pos",direction="nearest",tolerance=pd.Timedelta(days=4),suffixes=("","_s"))
    out["seasonal_lag_1y"]=m.sort_index()["seasonal_lag_1y"].values
    clim=source_df.groupby(["nama_pos",source_df["datetime"].dt.dayofyear])["tma_mdpl"].mean();clim.index.names=["nama_pos","doy"]
    clim=clim.reset_index().rename(columns={"tma_mdpl":"doy_climatology"})
    out=out.merge(clim,left_on=["nama_pos","day_of_year"],right_on=["nama_pos","doy"],how="left").drop(columns=["doy"])
    sm=source_df.groupby("nama_pos")["tma_mdpl"].mean();fb=out["nama_pos"].astype(str).map(sm).astype(float)
    out["doy_climatology"]=out["doy_climatology"].astype(float).fillna(fb)
    out["seasonal_lag_1y"]=out["seasonal_lag_1y"].astype(float).fillna(out["doy_climatology"])
    ss=source_df.groupby("nama_pos")["tma_mdpl"].agg(["mean","std"]).rename(columns={"mean":"station_mean_tma","std":"station_std_tma"})
    out=out.merge(ss,on="nama_pos",how="left")
    for c in catcats: out[c]=out[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
    return out

# ---------- OLD: recursive lag ensemble ----------
old_full=list(dict.fromkeys(exog+["station_mean_tma","station_std_tma"]+lagcols))
ss_all=train[train["datetime"]<CUT].groupby("nama_pos")["tma_mdpl"].agg(["mean","std"]).rename(columns={"mean":"station_mean_tma","std":"station_std_tma"})
tr_o=tr.merge(ss_all,on="nama_pos",how="left"); va_o=va.merge(ss_all,on="nama_pos",how="left")
for c in cat:
    tr_o[c]=tr_o[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories)); va_o[c]=va_o[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
ncols=[c for c in old_full if c not in cat]
pre_t=ColumnTransformer([("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),cat),("num",SimpleImputer(strategy="median"),ncols)])
pre_l=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),ncols)])
tr_sub=tr_o.groupby("nama_pos",observed=True,group_keys=False).apply(lambda x:x.sample(n=min(len(x),600),random_state=RS))
rf=Pipeline([("pre",pre_t),("m",RandomForestRegressor(n_estimators=60,max_depth=12,n_jobs=4,random_state=RS))]).fit(tr_sub[old_full],tr_sub[TGT])
lr=Pipeline([("pre",pre_l),("m",LinearRegression())]).fit(tr_o[old_full],tr_o[TGT])
hist={p:x.sort_values("datetime")[["datetime",TGT]].values.tolist() for p,x in train[train["datetime"]<CUT][["nama_pos","datetime",TGT]].groupby("nama_pos",observed=True)}
vs=va_o.sort_values(["datetime","nama_pos"]).reset_index(drop=True);nonlag=[c for c in old_full if c not in lagcols]
times=sorted(vs["datetime"].unique());bt={t:i.tolist() for t,i in vs.groupby("datetime").groups.items()}
rp=np.empty(len(vs));rt=vs[TGT].values
for t_ in times:
    idxs=bt[t_];rows=vs.loc[idxs];frs=[]
    for _,row in rows.iterrows():
        p=row["nama_pos"];vals=[v for _,v in hist[p]];n=len(vals);f={c:row[c] for c in nonlag}
        for l in LAGS:f[f"lag_{l}"]=vals[n-l] if n>=l else np.nan
        for w in ROLLW:
            wv=vals[max(0,n-w):n];f[f"roll_mean_{w}"]=np.mean(wv) if wv else np.nan;f[f"roll_std_{w}"]=np.std(wv,ddof=1) if len(wv)>1 else np.nan
        f["tma_diff_1"]=f["lag_1"]-f["lag_2"] if n>=2 else np.nan;f["tma_diff_1d"]=f["lag_1"]-f["lag_3"] if n>=3 else np.nan;f["tma_diff_7d"]=f["lag_1"]-f["lag_21"] if n>=21 else np.nan
        frs.append(f)
    Xs=pd.DataFrame(frs)[old_full]
    for c in cat:Xs[c]=Xs[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))
    sp=(rf.predict(Xs)+lr.predict(Xs))/2
    for li,(idx,p) in enumerate(zip(idxs,rows["nama_pos"])):rp[vs.index.get_loc(idx)]=sp[li];hist[p].append((t_,sp[li]))
print(f"{el()} OLD lag-ensemble RECURSIVE (far horizon)  RMSE={rmse(rt,rp):.4f}")

# ---------- NEW: direct no-lag+seasonal ensemble ----------
new_cols=list(dict.fromkeys(exog+["station_mean_tma","station_std_tma","seasonal_lag_1y","doy_climatology"]))
src=train[train["datetime"]<CUT]
tr_n=seasonal(tr,src,cat);va_n=seasonal(va,src,cat)
nc=[c for c in new_cols if c not in cat]
pre_tn=ColumnTransformer([("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),cat),("num",SimpleImputer(strategy="median"),nc)])
pre_ln=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),nc)])
ridge=Pipeline([("pre",pre_ln),("m",Ridge(alpha=1.0,random_state=RS))]).fit(tr_n[new_cols],tr_n[TGT])
hgb=Pipeline([("pre",pre_tn),("m",HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,l2_regularization=1.0,random_state=RS))]).fit(tr_n[new_cols],tr_n[TGT])
prr=ridge.predict(va_n[new_cols]);prh=hgb.predict(va_n[new_cols]);y=va_n[TGT].values
for w in [0.3,0.4,0.5]:
    print(f"{el()} NEW direct no-lag+seasonal w_ridge={w}  RMSE={rmse(y,w*prr+(1-w)*prh):.4f}")
print(f"\n{el()} DONE ({time.time()-T0:.1f}s)")
