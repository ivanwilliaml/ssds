"""Builds the full 20-model-preference benchmark report (15 tested + 5 skipped-as-DL
groups clarified as 12 individual models), reusing the established design tokens."""
import base64
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(r"D:\Lomba\ssds\model")
df = pd.read_csv(OUT / "benchmark_all.csv").sort_values("rmse").reset_index(drop=True)

# ---- chart ----
plt.rcParams.update({"font.size": 10.5})
colors = ["#1d6e63" if t == "tabular" else "#a8631f" for t in df["type"]]
fig, ax = plt.subplots(figsize=(7.4, 5.2))
bars = ax.barh(df["model"][::-1], df["rmse"][::-1], color=colors[::-1])
for b, v in zip(bars, df["rmse"][::-1]):
    ax.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, color="#5c594e")
ax.set_xlabel("validation RMSE (lower is better)")
ax.set_title("15 models tested — teal = tabular ML, amber = statistical time series")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "full_benchmark_chart.png", dpi=140)
chart_b64 = base64.b64encode((OUT / "full_benchmark_chart.png").read_bytes()).decode("ascii")

def fmt(x):
    return f"{x:.4f}"

rows = "".join(
    f"""<tr class="{'top' if i < 3 else ''}">
        <td>{i+1}</td><td>{r['model']}</td><td>{r['type']}</td>
        <td>{fmt(r['r2'])}</td><td>{fmt(r['rmse'])}</td><td>{fmt(r['mae'])}</td><td>{fmt(r['mape'])}</td>
        <td>{r['fit_predict_s']:.1f}s</td>
      </tr>"""
    for i, r in df.iterrows()
)

skipped = [
    ("TFT (Temporal Fusion Transformer)", "Deep Learning"),
    ("TiDE", "Deep Learning"),
    ("N-HiTS", "Deep Learning"),
    ("N-BEATS", "Deep Learning"),
    ("PatchTST", "Transformer"),
    ("TimeXer", "Transformer"),
    ("Informer", "Transformer"),
    ("Autoformer", "Transformer"),
    ("FEDformer", "Transformer"),
    ("DLinear", "Linear Deep Model"),
    ("iTransformer", "Transformer"),
    ("LSTM / GRU", "Recurrent Neural Network"),
]
skipped_rows = "".join(f"<tr><td>{name}</td><td>{cat}</td></tr>" for name, cat in skipped)

html = f"""<title>SSDS — 20-model preference benchmark</title>
<style>
:root{{
  --page-bg:#f4f2ec; --paper:#fbfaf6; --ink:#2b2a25; --ink-soft:#5c594e; --ink-faint:#8b876f;
  --line:#dedad0; --accent:#1d6e63; --accent-soft:#e4efec; --accent-ink:#0f453d;
  --amber:#a8631f; --amber-soft:#f3e6d4;
  --font-head:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --font-body:"Charter","Georgia",serif;
  --font-ui:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --font-mono:"SFMono-Regular","Consolas","Liberation Mono",Menlo,monospace;
}}
@media (prefers-color-scheme: dark){{
  :root{{ --page-bg:#1b1c19; --paper:#232420; --ink:#e9e6db; --ink-soft:#b7b3a2; --ink-faint:#847f6c;
  --line:#3a3b34; --accent:#5fbfae; --accent-soft:#233330; --accent-ink:#a8e6da; --amber:#d9a35c; --amber-soft:#332a1c; }}
}}
:root[data-theme="dark"]{{ --page-bg:#1b1c19; --paper:#232420; --ink:#e9e6db; --ink-soft:#b7b3a2; --ink-faint:#847f6c;
  --line:#3a3b34; --accent:#5fbfae; --accent-soft:#233330; --accent-ink:#a8e6da; --amber:#d9a35c; --amber-soft:#332a1c; }}
:root[data-theme="light"]{{ --page-bg:#f4f2ec; --paper:#fbfaf6; --ink:#2b2a25; --ink-soft:#5c594e; --ink-faint:#8b876f;
  --line:#dedad0; --accent:#1d6e63; --accent-soft:#e4efec; --accent-ink:#0f453d; --amber:#a8631f; --amber-soft:#f3e6d4; }}
*{{box-sizing:border-box;}}
body{{margin:0;}}
.doc{{background:var(--paper);color:var(--ink);font-family:var(--font-body);max-width:840px;margin:0 auto;
  padding:3.5rem 3.5rem 4.5rem;line-height:1.65;}}
@media (max-width:600px){{ .doc{{padding:2rem 1.25rem 3rem;}} }}
.kicker{{font-family:var(--font-ui);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);
  font-weight:500;margin:0 0 .6rem;}}
h1{{font-family:var(--font-head);font-size:32px;font-weight:600;margin:0 0 .3rem;text-wrap:balance;}}
.subtitle{{font-family:var(--font-ui);font-size:15px;color:var(--ink-soft);margin:0 0 2.4rem;}}
h2{{font-family:var(--font-head);font-size:20px;font-weight:600;margin:0 0 .8rem;}}
section{{margin:0 0 2.6rem;}}
p{{margin:0 0 .9rem;}}
img.chart{{width:100%;border-radius:10px;border:0.5px solid var(--line);margin-bottom:.6rem;}}
table{{width:100%;border-collapse:collapse;font-family:var(--font-ui);font-size:13px;margin-top:.6rem;}}
th{{text-align:left;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-faint);
  font-weight:500;padding:.4rem .5rem;border-bottom:0.5px solid var(--line);}}
td{{padding:.45rem .5rem;border-bottom:0.5px solid var(--line);color:var(--ink-soft);font-variant-numeric:tabular-nums;}}
tr.top td{{color:var(--ink);font-weight:500;}}
tr.top{{background:var(--accent-soft);}}
.callout{{background:var(--amber-soft);border-left:3px solid var(--amber);border-radius:0 8px 8px 0;
  padding:.85rem 1.1rem;font-family:var(--font-ui);font-size:13.5px;color:var(--ink-soft);margin-top:1rem;}}
.callout strong{{color:var(--ink);font-weight:600;}}
code.inline{{font-family:var(--font-mono);font-size:13px;background:var(--accent-soft);padding:.1rem .35rem;
  border-radius:4px;color:var(--accent-ink);}}
footer{{font-family:var(--font-ui);font-size:11.5px;color:var(--ink-faint);margin-top:2.6rem;padding-top:1.2rem;
  border-top:0.5px solid var(--line);}}
</style>
<article class="doc">
  <p class="kicker">Full model-preference benchmark</p>
  <h1>15 of 20 preferred models tested on the SSDS holdout</h1>
  <p class="subtitle">Same chronological validation window (last 60 days of train) used throughout this project</p>

  <section>
    <h2>1. Results — all 15 tested models</h2>
    <p>Tabular models (LightGBM, CatBoost, XGBoost, RandomForest, etc.) are fit row-wise across all 30 stations
    jointly using the 64-feature engineered dataset. SARIMAX and Prophet are fundamentally different: fit
    per-station as univariate time series (30 independent models each), then pooled for one overall score —
    their much longer <code class="inline">fit_predict_s</code> reflects 30 model fits, not one.</p>
    <img class="chart" src="data:image/png;base64,{chart_b64}" alt="Bar chart comparing RMSE across 15 models" />
    <table>
      <tr><th>#</th><th>Model</th><th>Type</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>Time</th></tr>
      {rows}
    </table>
    <div class="callout"><strong>MAPE caveat:</strong> several stations have tma_mdpl readings near zero, so MAPE
    is computed with a small epsilon floor (0.01) in the denominator to avoid division blow-ups — treat MAPE here
    as directionally useful, not a precise percentage.</div>
  </section>

  <section>
    <h2>2. Not tested — 12 deep-learning / transformer models</h2>
    <p>Per your decision, these were skipped rather than run under-trained: this machine's PyTorch is CPU-only
    (no GPU), and this session already showed CPU-bound sklearn ensembles running 5-10&times; slower than
    expected with repeated multi-minute stalls. Training 12 global deep architectures across 30 stations with
    enough epochs to be a fair comparison against the tabular models would likely take well over an hour with
    real risk of hangs — a token run with tiny epoch counts would produce numbers that look precise but aren't
    meaningfully comparable.</p>
    <table>
      <tr><th>Model</th><th>Category</th></tr>
      {skipped_rows}
    </table>
  </section>

  <section>
    <h2>3. Takeaway</h2>
    <p>RandomForest remains the strongest model on this feature set (RMSE 0.189, R² 0.99998), consistent with
    the earlier tuning result. Among the models newly added in this round, HistGradientBoosting is a notable
    middle-of-pack performer (RMSE 0.341) that trains in under a second — worth keeping in mind as a fast
    alternative if the recursive forecast needs to run at higher frequency. SARIMAX and Prophet, despite being
    purpose-built for time series, underperform the top tabular models here because they can't use the
    exogenous rainfall/soil-moisture/pressure features that clearly matter for this problem (see the EDA
    correlation panel).</p>
  </section>

  <footer>benchmark_all_models.py + benchmark_prophet.py &middot; D:\\Lomba\\ssds\\model &middot; tabular+SARIMAX: 73.2s, Prophet: 29.1s</footer>
</article>
"""

(OUT / "full_benchmark_report.html").write_text(html, encoding="utf-8")
print("wrote full_benchmark_report.html")
