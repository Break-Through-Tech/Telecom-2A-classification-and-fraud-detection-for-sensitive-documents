"""
Score a passport image with a trained forgery classifier.

    python predict.py path/to/image.png [--model final_r50.pt]

As a library (for the agent milestone):
    from predict import ForgeryClassifier
    clf = ForgeryClassifier("final_r50.pt")
    clf.predict("passport.png")  # {"label": "fake", "fake_probability": 0.97}
"""

import argparse
import io
import os

import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_BOX = (60, 170, 480, 700)


def _build(arch):
    if arch == "effnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        m = getattr(models, arch)(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    return m


class ForgeryClassifier:
    def __init__(self, weights=os.path.join(HERE, "final_r50.pt"), threshold=0.5):
        ck = torch.load(weights, map_location="cpu")
        self.model = _build(ck["arch"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.crop, self.reencode, self.threshold = ck["crop"], ck.get("reencode", 0), threshold
        self.tf = transforms.Compose([
            transforms.Resize(tuple(ck["hw"])), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, path):
        im = Image.open(path).convert("RGB")
        if self.reencode:
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=self.reencode)
            im = Image.open(buf).convert("RGB")
        if self.crop:
            im = im.crop(FACE_BOX)
        with torch.no_grad():
            p = torch.sigmoid(self.model(self.tf(im).unsqueeze(0)).squeeze()).item()
        return {"label": "fake" if p >= self.threshold else "genuine", "fake_probability": round(p, 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=os.path.join(HERE, "final_r50.pt"))
    a = ap.parse_args()
    print(ForgeryClassifier(a.model).predict(a.image))
