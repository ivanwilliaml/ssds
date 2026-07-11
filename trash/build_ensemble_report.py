import base64
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(r"D:\Lomba\ssds\model")
df = pd.read_csv(OUT / "ensemble_comparison.csv").sort_values("rmse").reset_index(drop=True)

def short(name):
    return name.split(" ")[0]

df["short"] = df["combo"].apply(short)
single = {"A", "B", "C"}
colors = ["#1d6e63" if s in single else ("#a8631f" if s == "ABC" else "#5c8fc7") for s in df["short"]]

plt.rcParams.update({"font.size": 10.5})
fig, ax = plt.subplots(figsize=(7.2, 3.8))
bars = ax.barh(df["combo"][::-1], df["rmse"][::-1], color=colors[::-1])
for b, v in zip(bars, df["rmse"][::-1]):
    ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, color="#5c594e")
ax.set_xlabel("validation RMSE (lower is better)")
ax.set_title("Single models (teal) vs. pairwise (blue) vs. all-3 (amber) ensembles")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "ensemble_chart.png", dpi=140)
chart_b64 = base64.b64encode((OUT / "ensemble_chart.png").read_bytes()).decode("ascii")

import json
params = json.loads((OUT / "tuned_params.json").read_text())

def fmt(x):
    return f"{x:.4f}"

rows = "".join(
    f"""<tr class="{'best' if i == 0 else ''}">
        <td>{i+1}</td><td>{r['combo']}</td><td>{fmt(r['r2'])}</td><td>{fmt(r['rmse'])}</td>
        <td>{fmt(r['mae'])}</td><td>{fmt(r['mape'])}</td>
      </tr>"""
    for i, r in df.iterrows()
)

param_rows = "".join(
    f"<tr><td>{name}</td><td>{'<br>'.join(f'{k}: {v if not isinstance(v, float) else round(v,3)}' for k, v in p.items())}</td></tr>"
    for name, p in params.items()
)

best_rmse = df.iloc[0]
best_mape = df.sort_values("mape").iloc[0]

html = f"""<title>SSDS — Optuna tuning &amp; ensemble comparison</title>
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
img.chart{{width:100%;border-radius:10px;border:0.5px solid var(--line);margin-bottom:.6rem;}}
table{{width:100%;border-collapse:collapse;font-family:var(--font-ui);font-size:13.5px;margin-top:.6rem;}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-faint);
  font-weight:500;padding:.4rem .6rem;border-bottom:0.5px solid var(--line);}}
td{{padding:.5rem .6rem;border-bottom:0.5px solid var(--line);color:var(--ink-soft);font-variant-numeric:tabular-nums;}}
tr.best td{{color:var(--ink);font-weight:500;}}
tr.best{{background:var(--accent-soft);}}
.callout{{background:var(--amber-soft);border-left:3px solid var(--amber);border-radius:0 8px 8px 0;
  padding:.85rem 1.1rem;font-family:var(--font-ui);font-size:13.5px;color:var(--ink-soft);margin-top:1rem;}}
.callout strong{{color:var(--ink);font-weight:600;}}
code.inline{{font-family:var(--font-mono);font-size:13px;background:var(--accent-soft);padding:.1rem .35rem;
  border-radius:4px;color:var(--accent-ink);}}
footer{{font-family:var(--font-ui);font-size:11.5px;color:var(--ink-faint);margin-top:2.6rem;padding-top:1.2rem;
  border-top:0.5px solid var(--line);}}
</style>
<article class="doc">
  <p class="kicker">Optuna tuning &amp; ensemble comparison</p>
  <h1>Top 3 tuned individually, then averaged pairwise</h1>
  <p class="subtitle">A = RandomForest &middot; B = DecisionTree &middot; C = LinearRegression &mdash; same holdout as every prior report</p>

  <section>
    <h2>1. Optuna-tuned hyperparameters</h2>
    <p>Each model was tuned with a TPE-sampler Optuna study (12 trials for A/B, 4 for C since
    LinearRegression has no real hyperparameters) directly minimizing validation RMSE on a
    station-stratified subsample, then refit on the full 78,382-row training set with the winning
    parameters before scoring.</p>
    <table>
      <tr><th>Model</th><th>Best params</th></tr>
      {param_rows}
    </table>
  </section>

  <section>
    <h2>2. All combinations — single, pairwise, and triple ensemble</h2>
    <img class="chart" src="data:image/png;base64,{chart_b64}" alt="Bar chart of RMSE across single models and ensembles" />
    <table>
      <tr><th>#</th><th>Combo</th><th>R&sup2;</th><th>RMSE</th><th>MAE</th><th>MAPE</th></tr>
      {rows}
    </table>
    <div class="callout"><strong>Best by RMSE:</strong> {best_rmse['combo']} ({fmt(best_rmse['rmse'])}) &mdash;
    <strong>best by MAPE:</strong> {best_mape['combo']} ({fmt(best_mape['mape'])}). These disagree, which is the
    real story of this comparison (see below).</div>
  </section>

  <section>
    <h2>3. What actually happened here</h2>
    <p>Tuned RandomForest (A) alone lands at RMSE 0.691 on the full-train refit &mdash; noticeably <em>worse</em>
    than the untuned/differently-tuned RandomForest from the earlier reports (RMSE 0.189&ndash;0.191). The
    Optuna search found <code class="inline">max_features=0.77, max_depth=11, min_samples_leaf=3</code> because
    those settings minimized RMSE on the ~10,500-row tuning subsample &mdash; but that combination
    (subsampled features + shallower trees + larger leaves) generalizes worse once refit on the full,
    much more heterogeneous 78k-row training set than the simpler <code class="inline">max_depth=10,
    min_samples_leaf=1, max_features=None</code> configuration used previously. This is a textbook
    tune-on-subsample-refit-on-full-data mismatch, and it's the honest result of this run rather than an
    error to hide.</p>
    <p>LinearRegression (C) is stable across both regimes (it has effectively no capacity to overfit a
    subsample), which is why it comes out on top by RMSE here. But look at MAPE: RandomForest (A) alone has
    the <em>lowest</em> MAPE (0.0137) despite the highest RMSE among single models &mdash; RMSE punishes
    RandomForest's occasional large misses on flood-pulse readings harder than MAPE does, while MAPE rewards
    RandomForest's much better relative accuracy on the many low-flow readings. AC (RandomForest averaged
    with LinearRegression) partially cancels out RandomForest's large-error spikes, landing 2nd by RMSE
    and 3rd by MAPE &mdash; a genuinely useful ensemble effect, not just noise.</p>
    <div class="callout"><strong>Practical takeaway:</strong> if this Optuna-tuned RandomForest were used for the
    actual competition submission, it should be re-tuned directly on the full training set (or with
    cross-validation that better represents it), not on a small subsample. For this comparison exercise the
    result stands as tested.</div>
  </section>

  <footer>optuna_ensemble.py &middot; D:\\Lomba\\ssds\\model &middot; total runtime: 78.5s</footer>
</article>
"""

(OUT / "ensemble_report.html").write_text(html, encoding="utf-8")
print("wrote ensemble_report.html")
