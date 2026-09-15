"""
Shared paths and constants for the IDNet fraud detection pipeline.
Import this in every task script so folder names/categories stay consistent.
"""
 
import os
 
# --- Directories (all relative to wherever you run the scripts from) ---
# If you already manually downloaded the dataset (e.g. Kaggle gave you a file
# named "files-archive.file" instead of the usual .zip), point ARCHIVE_PATH
# at it. The extension doesn't matter -- Python checks the file's actual
# format, not its name. Leave it as-is if you want the script to download
# it for you via the Kaggle API instead.
ARCHIVE_PATH = os.path.join("data", "files-archive.file")
 
RAW_DATA_DIR = "idnet_raw"
SAMPLE_DIR = "idnet_sample"
PROCESSED_DIR = "idnet_processed"
MANIFEST_PATH = "idnet_processed_manifest.csv"
 
# --- Sampling settings ---
N_PER_CATEGORY = 200   # 200 images x 3 categories ~= 600 images total
SEED = 42
 
# --- Preprocessing settings ---
IMG_SIZE = (256, 256)  # (width, height)
 
# --- Document categories -> IDNet subfolder names ---
# NOTE: after you download and unzip IDNet, check the actual folder names
# inside idnet_raw/ and update these values to match.
CATEGORIES = {
    "passport": "passports",
    "drivers_license": "drivers_licenses",
    "national_id": "national_ids",
}