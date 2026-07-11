"""Builds the v2 results report (model comparison + tuning + final metrics) as a
self-contained HTML artifact, reusing the same design tokens as the methodology
and EDA reports."""
import base64
from pathlib import Path
import pandas as pd

OUT = Path(r"D:\Lomba\ssds\model")

mc = pd.read_csv(OUT / "model_comparison.csv").sort_values("rmse").reset_index(drop=True)
ts = pd.read_csv(OUT / "tuning_summary.csv")
chart_b64 = base64.b64encode((OUT / "model_comparison_chart.png").read_bytes()).decode("ascii")

top3_names = set(ts["model"])
selected = ts.sort_values("tuned_rmse").iloc[0]

def fmt(x):
    return f"{x:.4f}"

mc_rows = "".join(
    f"""<tr class="{'top' if r['model'] in top3_names else ''}">
        <td>{i+1}</td><td>{r['model']}</td><td>{fmt(r['rmse'])}</td><td>{fmt(r['mae'])}</td><td>{r['fit_predict_s']:.1f}s</td>
      </tr>"""
    for i, r in mc.iterrows()
)

ts_rows = "".join(
    f"""<tr class="{'selected' if r['model'] == selected['model'] else ''}">
        <td>{r['model']}</td><td>{fmt(r['tuned_rmse'])}</td><td>{fmt(r['tuned_mae'])}</td>
      </tr>"""
    for _, r in ts.iterrows()
)

html = f"""<title>SSDS v2 — model comparison &amp; results</title>
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
.doc{{background:var(--paper);color:var(--ink);font-family:var(--font-body);max-width:800px;margin:0 auto;
  padding:3.5rem 3.5rem 4.5rem;line-height:1.65;}}
@media (max-width:600px){{ .doc{{padding:2rem 1.25rem 3rem;}} }}
.kicker{{font-family:var(--font-ui);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);
  font-weight:500;margin:0 0 .6rem;}}
h1{{font-family:var(--font-head);font-size:32px;font-weight:600;margin:0 0 .3rem;text-wrap:balance;}}
.subtitle{{font-family:var(--font-ui);font-size:15px;color:var(--ink-soft);margin:0 0 2.4rem;}}
h2{{font-family:var(--font-head);font-size:20px;font-weight:600;margin:0 0 .8rem;}}
section{{margin:0 0 2.6rem;}}
p{{margin:0 0 .9rem;}}
.stats{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line);
  border:0.5px solid var(--line);border-radius:10px;overflow:hidden;margin:0 0 2.6rem;}}
.stat{{background:var(--paper);padding:1rem 1.1rem;font-family:var(--font-ui);}}
.stat .num{{font-family:var(--font-head);font-size:24px;font-variant-numeric:tabular-nums;color:var(--accent-ink);display:block;}}
.stat .label{{font-size:11px;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.06em;margin-top:2px;}}
img.chart{{width:100%;border-radius:10px;border:0.5px solid var(--line);}}
table{{width:100%;border-collapse:collapse;font-family:var(--font-ui);font-size:13.5px;margin-top:.6rem;}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-faint);
  font-weight:500;padding:.4rem .6rem;border-bottom:0.5px solid var(--line);}}
td{{padding:.5rem .6rem;border-bottom:0.5px solid var(--line);color:var(--ink-soft);font-variant-numeric:tabular-nums;}}
tr.top td, tr.selected td{{color:var(--ink);font-weight:500;}}
tr.top{{background:var(--accent-soft);}}
tr.selected{{background:var(--accent-soft);}}
.callout{{background:var(--amber-soft);border-left:3px solid var(--amber);border-radius:0 8px 8px 0;
  padding:.85rem 1.1rem;font-family:var(--font-ui);font-size:13.5px;color:var(--ink-soft);margin-top:1rem;}}
.callout strong{{color:var(--ink);font-weight:600;}}
code.inline{{font-family:var(--font-mono);font-size:13px;background:var(--accent-soft);padding:.1rem .35rem;
  border-radius:4px;color:var(--accent-ink);}}
footer{{font-family:var(--font-ui);font-size:11.5px;color:var(--ink-faint);margin-top:2.6rem;padding-top:1.2rem;
  border-top:0.5px solid var(--line);}}
</style>
<article class="doc">
  <p class="kicker">Results — v2 pipeline</p>
  <h1>10-model comparison, tuning, and final selection</h1>
  <p class="subtitle">build_model_v2.py — validation on the last 60 days of train, held out chronologically</p>

  <div class="stats">
    <div class="stat"><span class="num">{selected['model']}</span><span class="label">selected model</span></div>
    <div class="stat"><span class="num">{fmt(selected['tuned_rmse'])}</span><span class="label">tuned val. RMSE</span></div>
    <div class="stat"><span class="num">64</span><span class="label">total features (20 new)</span></div>
  </div>

  <section>
    <h2>1. Ten-model baseline comparison</h2>
    <p>All ten models were fit on the same station-stratified subsample of the training set and scored on
    the identical chronological holdout, so the ranking below reflects the model family alone, not
    differences in data. Tree-based models use ordinal-encoded station/land-cover categories; linear and
    KNN models use one-hot encoding with standardized numeric features.</p>
    <img class="chart" src="data:image/png;base64,{chart_b64}" alt="Bar chart of RMSE for 10 models" />
    <table>
      <tr><th>#</th><th>Model</th><th>RMSE</th><th>MAE</th><th>Fit+predict</th></tr>
      {mc_rows}
    </table>
    <div class="callout"><strong>Why RandomForest won:</strong> tma_mdpl is dominated by its own recent lag values,
    and the relationship between those lags and the next reading is mostly linear but with station-specific
    kinks (dam gates, weirs, floodways) that a shallow tree ensemble captures better than a single global
    linear coefficient. Deeper, more complex boosted models (GradientBoosting, LightGBM, XGBoost) overfit the
    comparatively small subsample used for this stage.</div>
  </section>

  <section>
    <h2>2. Grid-search tuning of the top 3</h2>
    <p>Rather than tuning only the #1 model by raw RMSE, all three finalists were grid-searched (small grids,
    2-fold <code class="inline">TimeSeriesSplit</code>) since a model with no meaningful hyperparameters
    (LinearRegression) can't be "tuned" in a way that would change the outcome. Each tuned candidate was then
    re-validated on the same holdout to pick the final model.</p>
    <table>
      <tr><th>Model</th><th>Tuned RMSE</th><th>Tuned MAE</th></tr>
      {ts_rows}
    </table>
  </section>

  <section>
    <h2>3. Final pipeline</h2>
    <p>The selected model — tuned RandomForest — is refit on the <em>entire</em> training set (all 30 stations,
    83,766 rows after dropping the initial rows without full lag history) and used for the vectorized recursive
    forecast over the 726 future timesteps, exactly as in the single-model v1 pipeline.</p>
  </section>

  <footer>build_model_v2.py · D:\\Lomba\\ssds\\model · total pipeline runtime: 101s (1.68 min), well under the 15-minute budget</footer>
</article>
"""

(OUT / "results_report.html").write_text(html, encoding="utf-8")
print("wrote results_report.html")
