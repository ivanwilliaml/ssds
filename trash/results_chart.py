import base64
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(r"D:\Lomba\ssds\model")
mc = pd.read_csv(OUT / "model_comparison.csv").sort_values("rmse")

top3 = {"RandomForest", "DecisionTree", "LinearRegression"}
colors = ["#1d6e63" if m in top3 else "#c9c4b3" for m in mc["model"]]

plt.rcParams.update({"font.size": 10.5})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.barh(mc["model"][::-1], mc["rmse"][::-1], color=colors[::-1])
for b, v in zip(bars, mc["rmse"][::-1]):
    ax.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, color="#5c594e")
ax.set_xlabel("validation RMSE (lower is better)")
ax.set_title("10-model baseline comparison — top 3 in teal")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "model_comparison_chart.png", dpi=140)
print("chart saved")
