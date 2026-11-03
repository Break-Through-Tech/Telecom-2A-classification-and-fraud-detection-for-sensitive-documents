"""
Noise comparison on the 300 genuine IDNet passports in idnet_sample/real/passport.

Fakes are the genuine passport with noise added. Train on one noise type, then test on every
noise type (including the generate-fake.py fakes in idnet_sample/fake/passport), so the matrix
shows whether a model trained on one kind of tampering catches the others.
5-fold CV split by passport, so a passport and its noisy copy are never on opposite sides.
Noise is added at native resolution, then the image is resized like in train_cnn.py.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from torchvision import transforms

from train_cnn import OUT, ROOT, SEED, make_model

REAL = f"{ROOT}/idnet_sample/real/passport"
FAKE = f"{ROOT}/idnet_sample/fake/passport"

# severity in the units noted; mild is hard to see, moderate is clearly visible
NOISES = {
    "gaussian_mild": ("gaussian", 5),        # std in 0-255 pixel units
    "gaussian_mod": ("gaussian", 25),
    "poisson_mild": ("poisson", 1500),       # peak photon count, lower = noisier
    "poisson_mod": ("poisson", 60),
    "saltpepper_mild": ("saltpepper", 0.005),  # fraction of pixels flipped
    "saltpepper_mod": ("saltpepper", 0.05),
    "generate_fake": ("file", None),         # fakes made by generate-fake.py (blur + noise + compression)
}
TRAIN_ON = ["gaussian_mod", "poisson_mod", "saltpepper_mod", "generate_fake"]


def add_noise(arr, kind, level, rng):
    x = arr.astype(np.float32)
    if kind == "gaussian":
        x = x + rng.normal(0, level, x.shape)
    elif kind == "poisson":
        x = rng.poisson(x / 255.0 * level) / level * 255.0
    elif kind == "saltpepper":
        m = rng.random(x.shape[:2])
        x[m < level / 2] = 0
        x[(m >= level / 2) & (m < level)] = 255
    return np.clip(x, 0, 255).astype(np.uint8)


class Items(torch.utils.data.Dataset):
    """items: list of (name, noise_key or None, seed). None = clean genuine passport (label 0)."""

    def __init__(self, items, hw, fixed_seed):
        self.items, self.fixed_seed = items, fixed_seed
        self.tf = transforms.Compose([transforms.Resize(hw), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, key, seed = self.items[i]
        if key == "generate_fake":
            im = Image.open(f"{FAKE}/{name}").convert("RGB")
        else:
            im = Image.open(f"{REAL}/{name}").convert("RGB")
            if key is not None:
                kind, level = NOISES[key]
                rng = np.random.default_rng(seed if self.fixed_seed else None)
                im = Image.fromarray(add_noise(np.asarray(im), kind, level, rng))
        return self.tf(im), (0 if key is None else 1)


def loader(items, args, train, fixed_seed):
    return torch.utils.data.DataLoader(Items(items, args.hw, fixed_seed), batch_size=args.bs, shuffle=train,
                                       num_workers=0)


def predict(model, dl, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.sigmoid(model(x.to(dev)).squeeze(1)).cpu())
    return torch.cat(out).numpy()


def train(names, key, args, dev):
    torch.manual_seed(SEED)
    model = make_model(args.arch).to(dev)
    items = [(n, None, 0) for n in names] + [(n, key, 0) for n in names]
    dl = loader(items, args, True, False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
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
        print(f"    epoch {ep + 1}/{args.epochs} loss {tot / len(items):.4f}", flush=True)
    return model


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hw", type=int, nargs=2, default=(402, 568))
    ap.add_argument("--limit", type=int, default=0, help="use only the first N passports (smoke test)")
    ap.add_argument("--train-on", nargs="+", default=TRAIN_ON)
    ap.add_argument("--name", default="noise_experiment")
    args = ap.parse_args()
    args.hw = tuple(args.hw)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    names = sorted(os.listdir(REAL))
    names = [n for n in names if os.path.exists(f"{FAKE}/{n}")]
    if args.limit:
        names = names[:args.limit]
    names = np.array(names)
    print(f"device {dev}; {len(names)} genuine passports", flush=True)

    results = {}
    for tr_key in args.train_on:
        print(f"== train on {tr_key}", flush=True)
        clean_p, fake_p = [], {k: [] for k in NOISES}
        for k, (tr, te) in enumerate(KFold(args.folds, shuffle=True, random_state=SEED).split(names)):
            print(f"  fold {k + 1}/{args.folds}", flush=True)
            model = train(names[tr], tr_key, args, dev)
            test_names = names[te]
            clean_p.append(predict(model, loader([(n, None, 0) for n in test_names], args, False, True), dev))
            for key in NOISES:
                items = [(n, key, 1000 + i) for i, n in enumerate(test_names)]
                fake_p[key].append(predict(model, loader(items, args, False, True), dev))
        clean = np.concatenate(clean_p)
        results[tr_key] = {}
        for key in NOISES:
            fk = np.concatenate(fake_p[key])
            y = np.r_[np.zeros(len(clean)), np.ones(len(fk))]
            results[tr_key][key] = metrics(y, np.r_[clean, fk])
        print(json.dumps({k: round(v["accuracy"], 3) for k, v in results[tr_key].items()}), flush=True)
        with open(f"{OUT}/{args.name}.json", "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
