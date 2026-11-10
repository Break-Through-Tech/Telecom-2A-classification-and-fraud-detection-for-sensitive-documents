"""
Shared paths and constants for the IDNet fraud detection pipeline.
Import this in every task script so folder names/categories stay consistent.
"""
 
import os
from pathlib import Path
 
# --- Directories (all relative to wherever you run the scripts from) ---
# If you already manually downloaded the dataset (e.g. Kaggle gave you a file
# named "files-archive.file" instead of the usual .zip), point ARCHIVE_PATH
# at it. The extension doesn't matter -- Python checks the file's actual
# format, not its name. Leave it as-is if you want the script to download
# it for you via the Kaggle API instead.
ARCHIVE_PATH = os.path.join("data", "files-archive")
 
# Resolve paths from this config file, not the terminal's working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_PATH = os.path.join(PROJECT_ROOT, "data", "files-archive")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "idnet_raw")

# Original folder and final CSV names, anchored to the repository root.
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "idnet_sample")
COMPRESSED_DIR = os.path.join(PROJECT_ROOT, "idnet_sample_compressed")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "idnet_processed")
SAMPLE_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "sample_manifest.csv")
GENERATED_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "generated_manifest.csv")
COMPRESSED_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "lossless_manifest.csv")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "idnet_processed_manifest.csv")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "idnet_features.csv")
N_FOLDS = 5
# Compression is now optional and lossless; preprocessing defaults to generated PNGs.
USE_COMPRESSED_INPUT = False
 
# --- Sampling settings ---
N_PER_CATEGORY = 200   # 200 images x 3 categories ~= 600 images total
SEED = 42
 
# --- Preprocessing settings ---
IMG_SIZE = (256, 256)  # HOG canvas only; content is fitted without stretching.
TEXTURE_MAX_SIDE = 1024  # No upsampling; set None for native-resolution texture.
 
# --- Document categories -> country/state codes actually present in the archive ---
# This dataset's zip contains per-country/state sub-zips (e.g. GRC.zip, RUS.zip,
# WV.zip) rather than folders named "passports"/"drivers_licenses"/"national_ids".
# Map each document type to the codes available. NOTE: this particular download
# has no passport data -- add a "passport" entry here if you get that separately.
CATEGORIES = {
    "national_id": ["RUS"],
    "drivers_license": ["WV"],
    "passport": ["GRC"]
}

def require_empty_outputs(*paths):
    """Refuse to overwrite outputs; never delete or move the user's files."""
    occupied = []
    for value in paths:
        path = Path(value)
        if path.is_dir() and not path.is_symlink():
            if any(p.is_file() or p.is_symlink() for p in path.rglob("*")):
                occupied.append(str(path))
        elif path.exists() or path.is_symlink():
            occupied.append(str(path))
    if occupied:
        raise FileExistsError(
            "Output files already exist. Move these outputs to a backup before "
            "rerunning this stage:\n  " + "\n  ".join(occupied)
        )