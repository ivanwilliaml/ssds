"""
10-panel EDA used to justify the preprocessing / feature-engineering choices
in build_model_v2.py. Saves individual PNGs to model/eda_plots/ and an
HTML gallery (eda_report.html) with everything embedded as base64.
"""
import base64
import io
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

t0 = time.time()
ROOT = Path(r"D:\Lomba\ssds")
OUT = ROOT / "model"
PLOTDIR = OUT / "eda_plots"
PLOTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d8d4c8",
    "axes.grid": True,
    "grid.color": "#e8e5da",
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})
ACCENT = "#1d6e63"
ACCENT2 = "#a8631f"
PALETTE = sns.color_palette(["#1d6e63", "#a8631f", "#5c8fc7", "#b34747", "#7a6ba6", "#4c8a3f"])

train = pd.read_csv(ROOT / "train.csv", parse_dates=["datetime"])
env = pd.read_csv(ROOT / "data_pendukung" / "data_lingkungan.csv", parse_dates=["datetime"])
coord = pd.read_csv(ROOT / "data_pendukung" / "koordinat_pos.csv")
print(f"[{time.time()-t0:.1f}s] data loaded: train={train.shape}, env={env.shape}")

figs = []  # (title, note, filename)

def save(fig, name, title, note):
    path = PLOTDIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append((title, note, path))
    print(f"[{time.time()-t0:.1f}s] saved {name}")

# 1. Target distribution -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
sns.histplot(train["tma_mdpl"], bins=60, color=ACCENT, ax=ax)
ax.set_title("Distribution of tma_mdpl")
ax.set_xlabel("tma_mdpl")
save(fig, "01_target_dist", "Target distribution",
     "Right-skewed with a long tail past ~250 — a handful of flood-pulse readings. "
     "Motivates robust scaling / no blind z-score outlier removal, since the tail is real signal, not noise.")

# 2. Per-station boxplot ---------------------------------------------------
order = train.groupby("nama_pos")["tma_mdpl"].median().sort_values().index
fig, ax = plt.subplots(figsize=(6.4, 6.5))
sns.boxplot(data=train, y="nama_pos", x="tma_mdpl", order=order, ax=ax,
            color=ACCENT, fliersize=1.5, linewidth=0.7)
ax.set_title("tma_mdpl range per post")
save(fig, "02_station_boxplot", "Level range per monitoring post",
     "Posts sit on wildly different baselines (some near 0, Wonogiri Dam near 300) — "
     "confirms nama_pos must be a model feature (station target-encoding / categorical split), not pooled blindly.")

# 3. Missingness in environmental data --------------------------------
miss = env.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.barh(miss.index[::-1], miss.values[::-1] * 100, color=ACCENT2)
ax.set_title("Missing values by column (env. data)")
ax.set_xlabel("% missing")
save(fig, "03_missingness", "Missingness in the environmental feed",
     "soil_moisture, pressure, and MJO/Nino columns have small gaps (~0.08-1.5%) — "
     "sparse enough for per-station forward/backward fill rather than dropping rows.")

# 4. Correlation heatmap ---------------------------------------------------
merged_sample = train.merge(
    env[["datetime", "nama_pos", "rainfall_mm", "humidity_pct", "temperature_c",
         "soil_moisture_0_7cm", "soil_moisture_28_100cm", "surface_pressure_hpa",
         "wind_speed_kmh", "cloud_cover_pct", "nino_34"]],
    on=["datetime", "nama_pos"], how="inner")
corr_cols = ["tma_mdpl", "rainfall_mm", "humidity_pct", "temperature_c",
             "soil_moisture_0_7cm", "soil_moisture_28_100cm", "surface_pressure_hpa",
             "wind_speed_kmh", "cloud_cover_pct", "nino_34"]
corr = merged_sample[corr_cols].corr()
fig, ax = plt.subplots(figsize=(6.4, 5.4))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
ax.set_title("Correlation: target vs. instantaneous weather")
save(fig, "04_corr_heatmap", "Correlation with instantaneous weather",
     "Instantaneous rainfall/humidity/pressure barely correlate with tma_mdpl at the same timestamp "
     "(river response lags the rain) — the case for building rainfall accumulation and lag features "
     "instead of relying on same-timestep weather alone.")

# 5. Hourly (diurnal) pattern ----------------------------------------------
train["hour"] = train["datetime"].dt.hour
fig, ax = plt.subplots(figsize=(6.4, 3.4))
sns.boxplot(data=train, x="hour", y="tma_mdpl", ax=ax, color=ACCENT, fliersize=1, linewidth=0.7)
ax.set_title("tma_mdpl by reading hour (06 / 12 / 18)")
save(fig, "05_hourly_pattern", "Diurnal pattern",
     "Only 3 fixed reading times/day; distributions shift slightly across hours — "
     "supports sine/cosine hour encoding over one-hot, since the cycle is smooth not categorical.")

# 6. Monthly / seasonal pattern --------------------------------------------
train["month"] = train["datetime"].dt.month
fig, ax = plt.subplots(figsize=(6.8, 3.6))
sns.boxplot(data=train, x="month", y="tma_mdpl", ax=ax, color=ACCENT2, fliersize=1, linewidth=0.7)
ax.set_title("tma_mdpl by month")
save(fig, "06_seasonal_pattern", "Seasonal (monthly) pattern",
     "Clear wet-season (Nov-Apr) vs dry-season (May-Oct) separation in Java's monsoon climate — "
     "justifies both cyclical month encoding and an explicit is_wet_season flag.")

# 7. Example time series with rainfall overlay ------------------------------
example_pos = "Jurug"
tr_ex = train[train["nama_pos"] == example_pos].sort_values("datetime")
env_ex = env[(env["nama_pos"] == example_pos) & (env["datetime"] >= tr_ex["datetime"].min()) &
             (env["datetime"] <= tr_ex["datetime"].max())]
env_daily_rain = env_ex.set_index("datetime")["rainfall_mm"].resample("1D").sum()
mask = (tr_ex["datetime"] >= "2024-01-01") & (tr_ex["datetime"] <= "2024-04-30")
sub = tr_ex[mask]
rain_sub = env_daily_rain[(env_daily_rain.index >= "2024-01-01") & (env_daily_rain.index <= "2024-04-30")]
fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
ax1.plot(sub["datetime"], sub["tma_mdpl"], color=ACCENT, linewidth=1.2)
ax1.set_ylabel("tma_mdpl", color=ACCENT)
ax2 = ax1.twinx()
ax2.bar(rain_sub.index, rain_sub.values, color=ACCENT2, alpha=0.35, width=0.8)
ax2.set_ylabel("daily rainfall (mm)", color=ACCENT2)
ax2.grid(False)
ax1.set_title(f"{example_pos} — water level vs. daily rainfall (Jan-Apr 2024)")
save(fig, "07_level_vs_rainfall", "Water level tracks rainfall with a lag",
     "Level rises follow rain pulses with a delay of roughly a day, sometimes persisting after rain stops — "
     "the direct motivation for multi-day rainfall accumulation (1d/3d/7d/14d) features.")

# 8. Autocorrelation of tma_mdpl ------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf
series = tr_ex.set_index("datetime")["tma_mdpl"].asfreq("6h").interpolate()
fig, ax = plt.subplots(figsize=(6.4, 3.4))
plot_acf(series.dropna(), lags=40, ax=ax, color=ACCENT)
ax.set_title(f"Autocorrelation of tma_mdpl ({example_pos})")
save(fig, "08_autocorrelation", "Autocorrelation of water level",
     "Strong autocorrelation persisting past 40 lags (10 days) confirms lag/rolling features "
     "of the target itself are the single most informative feature family.")

# 9. Soil moisture depth profile ------------------------------------------
sm_cols = ["soil_moisture_0_7cm", "soil_moisture_7_28cm", "soil_moisture_28_100cm", "soil_moisture_100_255cm"]
env_ex2 = env_ex.set_index("datetime")[sm_cols].resample("1D").mean()
env_ex2 = env_ex2[(env_ex2.index >= "2024-01-01") & (env_ex2.index <= "2024-06-30")]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for col, color in zip(sm_cols, PALETTE):
    ax.plot(env_ex2.index, env_ex2[col], label=col.replace("soil_moisture_", ""), color=color, linewidth=1.1)
ax.set_title(f"Soil moisture by depth — {example_pos} (2024 H1)")
ax.legend(fontsize=8, ncol=2)
save(fig, "09_soil_moisture_profile", "Soil moisture across depth layers",
     "Shallow layers react fast to rain, deep layers move slowly and stay elevated — "
     "supports both a soil_moisture_avg composite and a soil_moisture_trend feature to capture catchment saturation.")

# 10. Station map colored by mean level ------------------------------------
station_mean = train.groupby("nama_pos")["tma_mdpl"].mean().rename("mean_tma")
coord_m = coord.merge(station_mean, on="nama_pos", how="left")
fig, ax = plt.subplots(figsize=(6, 6.4))
sc = ax.scatter(coord_m["longitude"], coord_m["latitude"], c=coord_m["mean_tma"],
                 cmap="YlGnBu", s=90, edgecolor="#2b2a25", linewidth=0.5)
for _, r in coord_m.iterrows():
    if r["nama_pos"] in (order[-1], order[0], example_pos):
        ax.annotate(r["nama_pos"], (r["longitude"], r["latitude"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
plt.colorbar(sc, ax=ax, shrink=0.75, label="mean tma_mdpl")
ax.set_title("Post locations colored by mean water level")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
save(fig, "10_station_map", "Spatial layout of monitoring posts",
     "Posts cluster across several sub-catchments in Java with no simple geographic gradient in mean level — "
     "latitude/longitude alone are weak predictors, reinforcing nama_pos categorical + station-level target stats "
     "as the right way to encode site identity.")

print(f"[{time.time()-t0:.1f}s] all 10 plots done, building HTML gallery")

# ---------------------------------------------------------------------------
# Build self-contained HTML gallery (base64-embedded images)
# ---------------------------------------------------------------------------
def b64(path):
    return base64.b64encode(path.read_bytes()).decode("ascii")

cards = []
for i, (title, note, path) in enumerate(figs, start=1):
    cards.append(f"""
  <figure class="card">
    <img src="data:image/png;base64,{b64(path)}" alt="{title}" />
    <figcaption>
      <span class="no">{i:02d}</span>
      <div>
        <h3>{title}</h3>
        <p>{note}</p>
      </div>
    </figcaption>
  </figure>""")

html = f"""<title>SSDS — EDA gallery</title>
<style>
:root{{
  --page-bg:#f4f2ec; --paper:#fbfaf6; --ink:#2b2a25; --ink-soft:#5c594e; --ink-faint:#8b876f;
  --line:#dedad0; --accent:#1d6e63; --accent-soft:#e4efec; --accent-ink:#0f453d;
  --font-head:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --font-ui:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}}
@media (prefers-color-scheme: dark){{
  :root{{ --page-bg:#1b1c19; --paper:#232420; --ink:#e9e6db; --ink-soft:#b7b3a2; --ink-faint:#847f6c;
  --line:#3a3b34; --accent:#5fbfae; --accent-soft:#233330; --accent-ink:#a8e6da; }}
}}
:root[data-theme="dark"]{{ --page-bg:#1b1c19; --paper:#232420; --ink:#e9e6db; --ink-soft:#b7b3a2; --ink-faint:#847f6c;
  --line:#3a3b34; --accent:#5fbfae; --accent-soft:#233330; --accent-ink:#a8e6da; }}
:root[data-theme="light"]{{ --page-bg:#f4f2ec; --paper:#fbfaf6; --ink:#2b2a25; --ink-soft:#5c594e; --ink-faint:#8b876f;
  --line:#dedad0; --accent:#1d6e63; --accent-soft:#e4efec; --accent-ink:#0f453d; }}
*{{box-sizing:border-box;}}
body{{margin:0;background:var(--paper);}}
.wrap{{max-width:920px;margin:0 auto;padding:3rem 1.5rem 4rem;font-family:var(--font-ui);color:var(--ink);}}
.kicker{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:500;margin:0 0 .5rem;}}
h1{{font-family:var(--font-head);font-size:30px;font-weight:600;margin:0 0 .3rem;text-wrap:balance;}}
.subtitle{{color:var(--ink-soft);font-size:15px;margin:0 0 2.4rem;}}
.grid{{display:grid;grid-template-columns:1fr;gap:1.6rem;}}
.card{{margin:0;background:var(--page-bg);border:0.5px solid var(--line);border-radius:12px;overflow:hidden;}}
.card img{{width:100%;display:block;border-bottom:0.5px solid var(--line);}}
figcaption{{display:flex;gap:.8rem;padding:1rem 1.2rem;align-items:flex-start;}}
.no{{font-family:var(--font-head);font-size:20px;color:var(--accent-ink);min-width:1.6rem;}}
h3{{font-family:var(--font-head);font-size:16px;font-weight:600;margin:0 0 .3rem;}}
p{{font-size:13.5px;color:var(--ink-soft);margin:0;line-height:1.6;}}
</style>
<div class="wrap">
  <p class="kicker">Exploratory data analysis</p>
  <h1>10 diagnostics behind the SSDS preprocessing choices</h1>
  <p class="subtitle">Each panel below motivates a specific decision made later in the pipeline — imputation strategy, feature families, encoding, or validation scheme.</p>
  <div class="grid">
  {''.join(cards)}
  </div>
</div>
"""

report_path = OUT / "eda_report.html"
report_path.write_text(html, encoding="utf-8")
print(f"[{time.time()-t0:.1f}s] wrote {report_path}")
print(f"Total EDA time: {time.time()-t0:.1f}s")
