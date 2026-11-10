"""
Fine-tune a pretrained ResNet18 to classify GRC passports as genuine vs forged.

Data:
  genuine: data/sample/GRC/positive  (pulled from the dataset's positive folder)
  forged : data/GRC/fraud*           (real IDNet forgeries)
idnet_sample/ is not used: its real/ and fake/ folders both contain the same
mix of png and "_fake_" jpg files, so the labels there can't be trusted.

Evaluation is 5-fold CV grouped by photo id, so a document and its forged
copy never land on opposite sides of the split.
"""

import argparse
import glob
import io
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import models, transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import SEED  # noqa: E402

FACE_BOX = (60, 170, 480, 700)  # photo region (left, top, right, bottom) in the native 1136x803 image


def photo_ids(path):
    return re.findall(r"\d{7}", os.path.basename(path))


def load_paths():
    real = glob.glob(f"{ROOT}/data/sample/GRC/positive/*")
    fake = glob.glob(f"{ROOT}/data/GRC/*/*")
    paths = real + fake
    labels = np.array([0] * len(real) + [1] * len(fake))
    # union-find over photo ids so any shared id puts files in the same group
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in paths:
        ids = photo_ids(p)
        for i in ids[1:]:
            parent[find(i)] = find(ids[0])
    groups = np.array([find(photo_ids(p)[0]) for p in paths])
    return paths, labels, groups


class Docs(torch.utils.data.Dataset):
    def __init__(self, paths, labels, train, hw, crop, reencode=0):
        self.paths, self.labels, self.crop, self.reencode = paths, labels, crop, reencode
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        aug = [transforms.ColorJitter(0.2, 0.2, 0.1)] if train else []
        self.tf = transforms.Compose([transforms.Resize(hw), *aug, transforms.ToTensor(), norm])

    def __len__(self):
        return len(self.paths)

    def load(self, p):
        im = Image.open(p).convert("RGB")
        if self.reencode:  # same JPEG round-trip for every image, so file format can't be a shortcut
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=self.reencode)
            im = Image.open(buf).convert("RGB")
        return im.crop(FACE_BOX) if self.crop else im

    def __getitem__(self, i):
        return self.tf(self.load(self.paths[i])), self.labels[i]


def make_model(arch):
    if arch == "effnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        m = getattr(models, arch)(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def predict(model, loader, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.sigmoid(model(x.to(dev)).squeeze(1)).cpu())
    return torch.cat(out).numpy()


def run_fold(tr, te, paths, labels, args, dev):
    torch.manual_seed(SEED)
    model = make_model(args.arch).to(dev)
    pos_w = torch.tensor(float((labels[tr] == 0).sum() / (labels[tr] == 1).sum()), dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    dl = torch.utils.data.DataLoader(Docs([paths[i] for i in tr], labels[tr], True, args.hw, args.crop, args.reencode),
                                     batch_size=args.bs, shuffle=True)
    te_dl = torch.utils.data.DataLoader(Docs([paths[i] for i in te], labels[te], False, args.hw, args.crop, args.reencode), batch_size=args.bs)
    for ep in range(args.epochs):
        model.train()
        tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.float().to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        print(f"  epoch {ep + 1}/{args.epochs} loss {tot / len(tr):.4f}", flush=True)
    return predict(model, te_dl, dev)


def train_final(paths, labels, args, dev):
    """Train on every image and save weights plus the settings predict.py needs."""
    idx = np.arange(len(paths))
    torch.manual_seed(SEED)
    model = make_model(args.arch).to(dev)
    pos_w = torch.tensor(float((labels == 0).sum() / (labels == 1).sum()), dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    dl = torch.utils.data.DataLoader(Docs(paths, labels, True, args.hw, args.crop, args.reencode),
                                     batch_size=args.bs, shuffle=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.float().to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        print(f"  epoch {ep + 1}/{args.epochs} loss {tot / len(idx):.4f}", flush=True)
    torch.save({"state_dict": model.cpu().state_dict(), "arch": args.arch, "hw": args.hw,
                "crop": args.crop, "reencode": args.reencode}, f"{OUT}/{args.name}.pt")
    print(f"saved {OUT}/{args.name}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--arch", default="resnet18", help="resnet18, resnet50, effnet_b0")
    ap.add_argument("--hw", type=int, nargs=2, default=(402, 568), help="input height width")
    ap.add_argument("--crop", action="store_true", help="crop to the photo region first")
    ap.add_argument("--reencode", type=int, default=0,
                    help="JPEG quality to re-encode every image with (0 = off); controls for png/jpg format leakage")
    ap.add_argument("--final", action="store_true", help="skip CV; train on all data and save weights to <name>.pt")
    ap.add_argument("--name", default="results")
    args = ap.parse_args()
    args.hw = tuple(args.hw)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    paths, labels, groups = load_paths()
    print(f"device {dev}; {len(paths)} images ({(labels == 0).sum()} genuine, {(labels == 1).sum()} forged), "
          f"{len(set(groups))} groups")

    if args.final:
        train_final(paths, labels, args, dev)
        return

    probs = np.zeros(len(paths))
    skf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(paths, labels, groups)):
        print(f"fold {k + 1}/{args.folds}: train {len(tr)} test {len(te)}", flush=True)
        probs[te] = run_fold(tr, te, paths, labels, args, dev)
        print(f"  fold auc {roc_auc_score(labels[te], probs[te]):.3f}", flush=True)

    pred = (probs >= 0.5).astype(int)
    res = {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred),
        "f1": f1_score(labels, pred),
        "roc_auc": roc_auc_score(labels, probs),
        "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(labels, pred).tolist(),
    }
    per_type = {}
    for t in sorted({os.path.basename(os.path.dirname(p)) for p, l in zip(paths, labels) if l == 1}):
        idx = [i for i, p in enumerate(paths) if labels[i] == 1 and os.path.basename(os.path.dirname(p)) == t]
        per_type[t] = float((pred[idx] == 1).mean())
    res["recall_by_forgery_type"] = per_type
    np.savez(f"{OUT}/{args.name}_probs.npz", probs=probs, labels=labels, paths=np.array(paths))
    res["args"] = vars(args)
    print(json.dumps(res, indent=2))
    with open(f"{OUT}/{args.name}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
