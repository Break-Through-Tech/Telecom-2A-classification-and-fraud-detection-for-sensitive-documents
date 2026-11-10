"""Build the model comparison table (REPORT.md) and ROC / confusion-matrix figures from the run JSONs."""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"results"}  # duplicate of an early run
rows = []
for f in sorted(glob.glob(f"{HERE}/*.json")):
    name = os.path.basename(f)[:-5]
    if name in SKIP:
        continue
    r = json.load(open(f))
    rows.append((name, r))
rows.sort(key=lambda x: -x[1]["roc_auc"])

lines = ["| run | acc | precision | recall | F1 | ROC-AUC | face-replace recall | combined recall | inpaint recall |",
         "|---|---|---|---|---|---|---|---|---|"]
for n, r in rows:
    t = r["recall_by_forgery_type"]
    lines.append(f"| {n} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} | "
                 f"{t['fraud3_face_replacement']:.2f} | {t['fraud4_combined']:.2f} | {t['fraud5_inpaint_and_rewrite']:.2f} |")
open(f"{HERE}/results_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ROC curves for every run that saved probabilities
fig, ax = plt.subplots(figsize=(6, 5))
for f in sorted(glob.glob(f"{HERE}/*_probs.npz")):
    d = np.load(f)
    fpr, tpr, _ = roc_curve(d["labels"], d["probs"])
    ax.plot(fpr, tpr, label=os.path.basename(f)[:-10])
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC, 5-fold grouped CV")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{HERE}/roc_curves.png", dpi=150)

# confusion matrix for the best run
best = rows[0][0]
if os.path.exists(f"{HERE}/{best}_probs.npz"):
    d = np.load(f"{HERE}/{best}_probs.npz")
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(d["labels"], d["probs"] >= 0.5, display_labels=["genuine", "fake"], ax=ax)
    ax.set_title(best)
    fig.tight_layout()
    fig.savefig(f"{HERE}/confusion_best.png", dpi=150)
