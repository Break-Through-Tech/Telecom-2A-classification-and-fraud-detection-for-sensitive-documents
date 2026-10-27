import csv
import glob
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from config import COMPRESSED_DIR, CATEGORIES, SEED

from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location("noise_experiments", "noise-experiments.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)

np.random.seed(SEED)

conditions = {
    "control_no_noise": module.identity,
    "gaussian_blur_whole": module.gaussian_blur,
    "impulse_whole": module.impulse_noise,
    "impulse_local": lambda im: module.apply_local(im, module.impulse_noise),
    "poisson_whole": module.poisson_noise,
    "poisson_local": lambda im: module.apply_local(im, module.poisson_noise),
}

paths = []
for category in CATEGORIES:
    paths.extend(sorted(glob.glob(
        os.path.join(COMPRESSED_DIR, "real", category, "*.*")
    ))[:module.N_PER_CATEGORY])

paths = paths[:30]
if not paths:
    raise SystemExit("No images found.")

rows = []
for path in paths:
    original = cv2.imread(path)
    if original is None:
        continue

    for name, transform in conditions.items():
        transformed = module.webp_roundtrip(transform(original.copy()))
        original_resized = cv2.resize(original, (256, 256))
        transformed_resized = cv2.resize(transformed, (256, 256))

        a = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(transformed_resized, cv2.COLOR_BGR2GRAY)

        rows.append({
            "image": os.path.basename(path),
            "condition": name,
            "ssim": round(structural_similarity(a, b, data_range=255), 5),
            "psnr": round(peak_signal_noise_ratio(a, b, data_range=255), 3),
        })

with open("image_quality_comparison.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "condition", "ssim", "psnr"])
    writer.writeheader()
    writer.writerows(rows)

print("\nAverage image similarity (higher SSIM and PSNR means closer to the original):")
for name in conditions:
    subset = [r for r in rows if r["condition"] == name]
    if subset:
        print(
            f"{name:22s} SSIM={np.mean([r['ssim'] for r in subset]):.4f}  "
            f"PSNR={np.mean([r['psnr'] for r in subset]):.2f} dB"
        )

print(f"\nCompared {len(set(r['image'] for r in rows))} images.")
print("Saved image_quality_comparison.csv")
