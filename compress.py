from PIL import Image
import os
import sys
import hashlib
from pathlib import Path
import pandas as pd
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMPRESSED_DIR, GENERATED_MANIFEST_PATH, COMPRESSED_MANIFEST_PATH, require_empty_outputs

def compress_local_image(input_path, output_path):
    with Image.open(input_path) as image:
        image.convert("RGB").save(output_path, "PNG", optimize=True)


def compress_all():
    require_empty_outputs(COMPRESSED_DIR, COMPRESSED_MANIFEST_PATH)
    frame = pd.read_csv(GENERATED_MANIFEST_PATH)
    if frame.empty:
        raise ValueError("No generated images; run generate-fake-localized.py first")
    records = []
    for row in frame.to_dict("records"):
        source = Path(row["filepath"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["artifact_sha256"]:
            raise ValueError(f"Generated artifact changed: {source}")
        destination = Path(COMPRESSED_DIR) / row["label"] / row["document_type"] / (source.stem + ".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        compress_local_image(source, destination)
        row["filepath"] = str(destination.resolve())
        row["artifact_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        records.append(row)
    output = pd.DataFrame(records)
    destination = Path(COMPRESSED_MANIFEST_PATH)
    temporary = destination.with_suffix(".csv.tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f"Losslessly exported {len(output)} images; manifest: {destination}")
    return output
 
 
if __name__ == "__main__":
    compress_all()