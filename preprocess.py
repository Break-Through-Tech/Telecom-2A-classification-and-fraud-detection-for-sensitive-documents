import os
import glob
import sys
 
import cv2
import pandas as pd
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMPRESSED_DIR, PROCESSED_DIR, MANIFEST_PATH, CATEGORIES, IMG_SIZE
 
 
def preprocess_image(path, size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return resized
 
 
def preprocess_all():
    records = []  # (filepath, label, document_type) for downstream use
 
    for label in ["real", "fake"]:
        for category in CATEGORIES.keys():
            src_dir = os.path.join(COMPRESSED_DIR, label, category)
            dst_dir = os.path.join(PROCESSED_DIR, label, category)
            os.makedirs(dst_dir, exist_ok=True)
 
            for f in glob.glob(os.path.join(src_dir, "*.*")):
                processed = preprocess_image(f)
                if processed is None:
                    continue
                out_path = os.path.join(dst_dir, os.path.basename(f))
                cv2.imwrite(out_path, processed)
                records.append({"filepath": out_path, "label": label, "document_type": category})
 
    df = pd.DataFrame(records)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Preprocessed {len(df)} images -> saved to {os.path.abspath(PROCESSED_DIR)}")
    print(f"Manifest written to {os.path.abspath(MANIFEST_PATH)}")
    return df
 
 
if __name__ == "__main__":
    preprocess_all()