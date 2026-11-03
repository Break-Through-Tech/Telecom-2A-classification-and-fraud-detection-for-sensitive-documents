"""Heatmap + markdown table from noise_experiment.json (train noise on rows, test noise on columns)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(f"{HERE}/noise_experiment.json"))["results"]
rows = list(res)
cols = list(next(iter(res.values())))
for metric in ("accuracy", "roc_auc"):
    m = np.array([[res[r][c][metric] for c in cols] for r in rows])
    if metric == "accuracy":
        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = ax.imshow(m, vmin=0.5, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)), rows, fontsize=8)
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", color="w" if m[i, j] < 0.8 else "k", fontsize=8)
        ax.set(xlabel="test fake type", ylabel="train fake type", title="accuracy (ResNet18, 5-fold CV, 300 passports)")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(f"{HERE}/noise_heatmap.png", dpi=150)
    print(f"\n{metric}\n| train \\ test | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for i, r in enumerate(rows):
        print(f"| {r} | " + " | ".join(f"{v:.3f}" for v in m[i]) + " |")
