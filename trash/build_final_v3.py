"""
FINAL v3 pipeline — direct (no-lag) seasonal model + horizon-aware PERSISTENCE BLEND.
Far-horizon backtest (the honest analog of the real test window) improved from:
    OLD recursive lag-ensemble ............ 1.85  (≈ leaderboard 1.89)
    v2 direct no-lag+seasonal ............. 1.59
    v3 + persistence blend (this) ......... 1.37-1.35 backtest
Prediction for each test row:
    pred = w(h)*last_known_station + (1-w(h))*direct,   w(h)=exp(-h/TAU)
  - last_known_station = station's last observed tma at the forecast origin
    (last training timestamp) — known at prediction time, applied as a post-hoc
    damped-persistence blend (standard, leakage-free).
  - direct = 0.8*Ridge + 0.2*HistGB on exogenous + calendar + station stats +
    seasonal_lag_1y + doy climatology(mean,std). No autoregressive target lags,
    so ZERO recursive error accumulation over the 726-step horizon.
TAU=150 chosen deliberately below the backtest optimum (240) to hedge the fact
that the real test spans the Dec-Feb wet-season peak (rising levels) whereas the
backtest window did not — keeping more weight on the seasonal-aware direct model.
"""
import time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
warnings.filterwarnings("ignore")
T0=time.time()
def el(): return f"[{time.time()-T0:6.1f}s]"
ROOT=Path(r"D:\Lomba\ssds"); OUT=ROOT/"model"; RS=42
TAU=150.0; W_RIDGE=0.8; W_HGB=0.2

train=pd.read_csv(ROOT/"train.csv",parse_dates=["datetime"])
test_raw=pd.read_csv(ROOT/"test.csv"); sample_sub=pd.read_csv(ROOT/"sample_submission.csv")
env=pd.read_csv(ROOT/"data_pendukung"/"data_lingkungan.csv",parse_dates=["datetime"])
coord=pd.read_csv(ROOT/"data_pendukung"/"koordinat_pos.csv")
sid=test_raw["id"].str.split(" - ",n=1,expand=True)
test=pd.DataFrame({"id":test_raw["id"],"datetime":pd.to_datetime(sid[0]),"nama_pos":sid[1]})
print(f"{el()} loaded train={train.shape} test={test.shape}")

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
comb=pd.concat([train[["nama_pos","datetime"]],test[["nama_pos","datetime"]]],ignore_index=True).drop_duplicates()
ef=envfeat(env,comb).sort_values(["nama_pos","datetime"])
fc=[c for c in ef.columns if c not in ("nama_pos","datetime")]
ef[fc]=ef.groupby("nama_pos")[fc].transform(lambda s:s.ffill().bfill()); ef[fc]=ef[fc].fillna(ef[fc].median())
ef=ef.sort_values(["nama_pos","datetime"]).reset_index(drop=True)
ef["pressure_trend"]=ef.groupby("nama_pos")["surface_pressure_hpa"].transform(lambda s:s-s.shift(3))
ef["soil_moisture_trend"]=ef.groupby("nama_pos")["soil_moisture_avg"].transform(lambda s:s-s.shift(3))
ef[["pressure_trend","soil_moisture_trend"]]=ef[["pressure_trend","soil_moisture_trend"]].fillna(0)
lc=env.sort_values("datetime").groupby("nama_pos")["landcover_name"].last()
train=train.merge(ef,on=["nama_pos","datetime"],how="left"); test=test.merge(ef,on=["nama_pos","datetime"],how="left")
train["landcover_name"]=train["nama_pos"].map(lc); test["landcover_name"]=test["nama_pos"].map(lc)
train=train.merge(coord,on="nama_pos",how="left"); test=test.merge(coord,on="nama_pos",how="left")
def cal(df):
    df=df.copy();df["hour"]=df["datetime"].dt.hour;df["month"]=df["datetime"].dt.month
    df["day_of_year"]=df["datetime"].dt.dayofyear;df["day_of_week"]=df["datetime"].dt.dayofweek
    df["is_weekend"]=(df["day_of_week"]>=5).astype(int);df["is_wet_season"]=df["month"].isin([11,12,1,2,3,4]).astype(int)
    df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24);df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"]=np.sin(2*np.pi*df["month"]/12);df["month_cos"]=np.cos(2*np.pi*df["month"]/12)
    df["doy_sin"]=np.sin(2*np.pi*df["day_of_year"]/365.25);df["doy_cos"]=np.cos(2*np.pi*df["day_of_year"]/365.25)
    return df
train=cal(train); test=cal(test)
cat=["nama_pos","landcover_name"]; TGT="tma_mdpl"
for c in cat:
    train[c]=train[c].astype("category"); test[c]=test[c].astype(pd.CategoricalDtype(categories=train[c].cat.categories))

def seasonal(target_df,source_df):
    out=target_df.copy()
    s=source_df[["nama_pos","datetime","tma_mdpl"]].sort_values("datetime").rename(columns={"tma_mdpl":"seasonal_lag_1y"})
    tmp=out[["nama_pos","datetime"]].copy();tmp["seek"]=tmp["datetime"]-pd.Timedelta(days=365);tmp=tmp.sort_values("seek")
    m=pd.merge_asof(tmp,s,left_on="seek",right_on="datetime",by="nama_pos",direction="nearest",tolerance=pd.Timedelta(days=6),suffixes=("","_s"))
    out["seasonal_lag_1y"]=m.sort_index()["seasonal_lag_1y"].values
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
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

# ---- honest far-horizon self-check (cut 2025-01-01) before deploy ----
CUT=pd.Timestamp("2025-01-01")
trh=train[train["datetime"]<CUT]; vah=train[train["datetime"]>=CUT]
tr_h=seasonal(trh,trh); va_h=seasonal(vah,trh)
pre_t=ColumnTransformer([("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),cat),("num",SimpleImputer(strategy="median"),nc)])
pre_l=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),nc)])
rg=Pipeline([("pre",pre_l),("m",Ridge(alpha=1.0,random_state=RS))]).fit(tr_h[feat],tr_h[TGT])
hg=Pipeline([("pre",pre_t),("m",HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,l2_regularization=1.0,random_state=RS))]).fit(tr_h[feat],tr_h[TGT])
direct_h=W_RIDGE*rg.predict(va_h[feat])+W_HGB*hg.predict(va_h[feat])
lk_map=trh.sort_values("datetime").groupby("nama_pos")["tma_mdpl"].last()
lk_h=va_h["nama_pos"].astype(str).map(lk_map).astype(float).values
hh=(va_h["datetime"]-CUT).dt.total_seconds().values/86400.0
wgt=np.exp(-hh/TAU); blend_h=wgt*lk_h+(1-wgt)*direct_h
print(f"{el()} [SELF-CHECK far-horizon] direct={rmse(va_h[TGT].values,direct_h):.4f}  blend(tau={TAU:.0f})={rmse(va_h[TGT].values,blend_h):.4f}")

# ---- deploy on ALL train ----
tr_all=seasonal(train,train); test_all=seasonal(test,train)
ridge=Pipeline([("pre",pre_l),("m",Ridge(alpha=1.0,random_state=RS))]).fit(tr_all[feat],tr_all[TGT])
hgb=Pipeline([("pre",pre_t),("m",HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,l2_regularization=1.0,random_state=RS))]).fit(tr_all[feat],tr_all[TGT])
joblib.dump({"ridge":ridge,"hgb":hgb,"tau":TAU,"w_ridge":W_RIDGE,"w_hgb":W_HGB,"features":feat},OUT/"final_v3_model.pkl")

origin=train["datetime"].max()
last_known=train.sort_values("datetime").groupby("nama_pos")["tma_mdpl"].last()
direct_test=W_RIDGE*ridge.predict(test_all[feat])+W_HGB*hgb.predict(test_all[feat])
lk_test=test_all["nama_pos"].astype(str).map(last_known).astype(float).values
h_test=(test_all["datetime"]-origin).dt.total_seconds().values/86400.0
w_test=np.exp(-h_test/TAU)
test_all["tma_mdpl"]=w_test*lk_test+(1-w_test)*direct_test
print(f"{el()} origin={origin}  horizon days: min={h_test.min():.0f} max={h_test.max():.0f}")

submission=sample_sub[["id"]].merge(test_all[["id","tma_mdpl"]],on="id",how="left")
submission.to_csv(OUT/"submission_v3.csv",index=False)
print(f"{el()} submission_v3.csv saved — missing={submission['tma_mdpl'].isna().sum()} rows={len(submission)}")
print(submission.head())
print(f"\n{el()} DONE ({time.time()-T0:.1f}s)  TAU={TAU:.0f}  direct=0.8*Ridge+0.2*HistGB")
