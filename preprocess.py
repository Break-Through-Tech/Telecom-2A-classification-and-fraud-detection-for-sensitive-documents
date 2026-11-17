import os
import glob
import sys
from pathlib import Path
import hashlib
 
import cv2
import pandas as pd
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DIR, MANIFEST_PATH, GENERATED_MANIFEST_PATH,
    COMPRESSED_MANIFEST_PATH, USE_COMPRESSED_INPUT, require_empty_outputs,
)


def preprocess_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    # HOG gets its own standardized image during feature extraction.
    # Convert both classes from decoded color identically. This also avoids
    # JPEG's direct grayscale decoder differing from the fake PNG conversion.
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
def preprocess_all():
    require_empty_outputs(PROCESSED_DIR, MANIFEST_PATH)
    input_manifest = COMPRESSED_MANIFEST_PATH if USE_COMPRESSED_INPUT else GENERATED_MANIFEST_PATH
    frame = pd.read_csv(input_manifest)
    if frame.empty:
        raise ValueError("No input images; run the generator first")
    records = []
    for row in frame.to_dict("records"):
        source = Path(row["filepath"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["artifact_sha256"]:
            raise ValueError(f"Input artifact changed since generation: {source}")
        image = preprocess_image(source)
        destination = Path(PROCESSED_DIR) / row["label"] / row["document_type"] / (source.stem + ".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Failed to save {destination}")
        row["filepath"] = str(destination.resolve())
        row["artifact_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        row["image_height"], row["image_width"] = image.shape
        records.append(row)
    output = pd.DataFrame(records)
    destination = Path(MANIFEST_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f"Preprocessed {len(output)} lossless images; manifest: {destination}")
    return output
 
 
if __name__ == "__main__":
    preprocess_all()