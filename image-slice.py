import os
import glob
import shutil
import random
import sys
import zipfile
import tarfile
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARCHIVE_PATH, RAW_DATA_DIR, SAMPLE_DIR, CATEGORIES, N_PER_CATEGORY, SEED
 
 
def extract_archive(archive_path, dest_dir):
    """Extract a dataset archive, detecting its real format from content, not extension."""
    os.makedirs(dest_dir, exist_ok=True)
 
    if zipfile.is_zipfile(archive_path):
        print(f"'{archive_path}' is a zip archive (regardless of its extension) -- extracting...")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif tarfile.is_tarfile(archive_path):
        print(f"'{archive_path}' is a tar archive -- extracting...")
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(
            f"Couldn't recognize '{archive_path}' as a zip or tar archive. "
            f"Run `file {archive_path}` (macOS/Linux) or check Properties (Windows) "
            f"to confirm what format it actually is."
        )
 
    print(f"Extracted to: {os.path.abspath(dest_dir)}")
 
 
def download_dataset():
    """Download the IDNet dataset via the Kaggle CLI (only used if no local archive exists)."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    exit_code = os.system(
        f"kaggle datasets download -d chitreshkr/idnet-identity-document-analysis "
        f"-p {RAW_DATA_DIR} --unzip"
    )
    if exit_code != 0:
        raise RuntimeError(
            "Kaggle download failed. Make sure the 'kaggle' package is installed "
            "(pip install kaggle) and kaggle.json is set up correctly (see docstring above), "
            f"or place the dataset archive at '{ARCHIVE_PATH}' instead."
        )
 
 
def get_raw_data():
    """Make sure idnet_raw/ is populated, either from a local archive or a fresh download."""
    if os.path.exists(RAW_DATA_DIR) and os.listdir(RAW_DATA_DIR):
        print(f"'{RAW_DATA_DIR}' already exists and is non-empty -- skipping extraction/download.")
        return
 
    if os.path.exists(ARCHIVE_PATH):
        print(f"Found local archive at '{ARCHIVE_PATH}'.")
        extract_archive(ARCHIVE_PATH, RAW_DATA_DIR)
    else:
        print(f"No local archive found at '{ARCHIVE_PATH}' -- downloading via Kaggle API instead.")
        download_dataset()
 
 
def sample_dataset():
    """Randomly sample N_PER_CATEGORY images per document category into SAMPLE_DIR."""
    random.seed(SEED)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
 
    for category, subfolder in CATEGORIES.items():
        search_pattern = os.path.join(RAW_DATA_DIR, "**", subfolder, "**", "*.*")
        all_files = glob.glob(search_pattern, recursive=True)
        all_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
 
        if not all_files:
            print(
                f"WARNING: found 0 images for '{category}' using subfolder name "
                f"'{subfolder}'. Check the actual folder names under "
                f"'{os.path.abspath(RAW_DATA_DIR)}' and update CATEGORIES in config.py."
            )
 
        random.shuffle(all_files)
        sampled_files = all_files[:N_PER_CATEGORY]
 
        out_dir = os.path.join(SAMPLE_DIR, "real", category)
        os.makedirs(out_dir, exist_ok=True)
        for f in sampled_files:
            shutil.copy(f, os.path.join(out_dir, os.path.basename(f)))
 
        print(f"{category}: found {len(all_files)} images, sampled {len(sampled_files)}")
 
    print("\nLocal sample ready at:", os.path.abspath(SAMPLE_DIR))
 
 
if __name__ == "__main__":
    get_raw_data()
    sample_dataset()