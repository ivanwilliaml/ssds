import time,warnings;from pathlib import Path;import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
warnings.filterwarnings("ignore");T0=time.time()
def el():return f"[{time.time()-T0:5.1f}s]"
exec(open("experiment_season.py").read().split("CUT=pd.Timestamp")[0])  # reuse feature build + seasonal()
CUT=pd.Timestamp("2024-09-18 18:00:00");VA_END=CUT+pd.Timedelta(days=242)
tr=train[train["datetime"]<=CUT];va=train[(train["datetime"]>CUT)&(train["datetime"]<=VA_END)]
# features WITHOUT seasonal_lag_1y
feat2=[c for c in feat if c!="seasonal_lag_1y"];nc2=[c for c in feat2 if c not in cat]
tr_n=seasonal(tr,tr);va_n=seasonal(va,tr)
pt=ColumnTransformer([("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),cat),("num",SimpleImputer(strategy="median"),nc2)])
pl=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),nc2)])
rg=Pipeline([("pre",pl),("m",Ridge(alpha=1.0))]).fit(tr_n[feat2],tr_n[TGT])
hg=Pipeline([("pre",pt),("m",HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,l2_regularization=1.0,random_state=42))]).fit(tr_n[feat2],tr_n[TGT])
y=va_n[TGT].values;direct=0.8*rg.predict(va_n[feat2])+0.2*hg.predict(va_n[feat2])
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
lk=tr.sort_values("datetime").groupby("nama_pos")["tma_mdpl"].last()
lkv=va_n["nama_pos"].astype(str).map(lk).astype(float).values
h=(va_n["datetime"]-CUT).dt.total_seconds().values/86400.0;w=np.exp(-h/120)
blend=w*lkv+(1-w)*direct
print(f"{el()} direct(no seasonal_lag_1y)={rmse(y,direct):.4f}  blend tau120={rmse(y,blend):.4f}")
# per-station RMSE of the blend + station level
d=pd.DataFrame({"pos":va_n["nama_pos"].astype(str).values,"y":y,"p":blend})
g=d.groupby("pos").apply(lambda x:pd.Series({"rmse":rmse(x.y.values,x.p.values),"mean_level":x.y.mean(),"n":len(x)})).sort_values("rmse",ascending=False)
print(f"{el()} pooled RMSE={rmse(y,blend):.4f}  | sum of squared err by station share:")
d["se"]=(d.y-d.p)**2;share=d.groupby("pos")["se"].sum().sort_values(ascending=False)/((d.y-d.p)**2).sum()
print("TOP-10 stations driving squared error:")
for pos in share.head(10).index:
    print(f"   {pos:<28} rmse={g.loc[pos,'rmse']:7.3f}  level~{g.loc[pos,'mean_level']:7.2f}  err_share={share[pos]*100:5.1f}%")
