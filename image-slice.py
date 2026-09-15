"""
Task 2: Get a local sample of the IDNet dataset with (close to) zero extra disk usage.
 
This dataset's archive contains one zip PER country/state code (e.g. GRC.zip,
RUS.zip, WV.zip) rather than a flat folder of images, and some of those inner
zips are tens of GB on their own.
 
Instead of extracting an inner zip to disk before sampling from it, this
script opens it as a live, in-memory stream directly from inside the outer
archive (via zipfile's nested `open()`), and reads out only the handful of
chosen images. As long as the outer archive stored the inner zips WITHOUT
additional compression (very common, since compressing an already-compressed
zip gains nothing), this never writes a temporary copy of the inner zip to
disk at all -- only the small sampled images themselves.
 
Run from the project root:
    python scripts/task2_download_and_sample.py
"""
 
import os
import random
import sys
import zipfile
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARCHIVE_PATH, SAMPLE_DIR, CATEGORIES, N_PER_CATEGORY, SEED
 
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
 
 
def check_archive():
    if not os.path.exists(ARCHIVE_PATH):
        raise FileNotFoundError(
            f"Couldn't find the archive at '{ARCHIVE_PATH}'. Update ARCHIVE_PATH in "
            f"config.py to point at your downloaded dataset file."
        )
 
 
def list_available_codes():
    """List the country/state codes (inner zip names) actually present in the archive."""
    with zipfile.ZipFile(ARCHIVE_PATH) as outer_zip:
        return sorted(
            name[:-4] for name in outer_zip.namelist() if name.lower().endswith(".zip")
        )
 
 
def sample_code(code, category, n_images):
    """
    Stream-read one inner zip directly out of the outer archive and sample
    n_images from it -- no temporary extraction to disk.
    """
    inner_zip_name = f"{code}.zip"
 
    with zipfile.ZipFile(ARCHIVE_PATH) as outer_zip:
        info = outer_zip.getinfo(inner_zip_name)
        if info.compress_type != zipfile.ZIP_STORED:
            print(
                f"[{code}] NOTE: this inner zip is compressed (not stored) inside the "
                f"outer archive, so streaming reads may be slower than usual, but will "
                f"still avoid writing a full temp copy to disk."
            )
 
        with outer_zip.open(inner_zip_name) as inner_stream:
            with zipfile.ZipFile(inner_stream) as inner_zip:
                image_names = [
                    n for n in inner_zip.namelist() if n.lower().endswith(IMAGE_EXTENSIONS)
                ]
                random.shuffle(image_names)
                chosen = image_names[:n_images]
 
                out_dir = os.path.join(SAMPLE_DIR, "real", category)
                os.makedirs(out_dir, exist_ok=True)
 
                for name in chosen:
                    data = inner_zip.read(name)
                    # Prefix with the code so filenames from different codes never collide
                    out_name = f"{code}_{os.path.basename(name)}"
                    with open(os.path.join(out_dir, out_name), "wb") as f:
                        f.write(data)
 
                print(f"[{code}] sampled {len(chosen)} of {len(image_names)} available images -> '{category}'")
 
 
def sample_dataset():
    random.seed(SEED)
    available_codes = set(list_available_codes())
    print("Codes found in archive:", sorted(available_codes))
 
    for category, codes in CATEGORIES.items():
        codes_present = [c for c in codes if c in available_codes]
        missing = [c for c in codes if c not in available_codes]
        if missing:
            print(f"WARNING: '{category}' expects codes {missing}, but they're not in the archive.")
        if not codes_present:
            print(f"WARNING: no codes available for '{category}' -- skipping.")
            continue
 
        n_per_code = max(1, N_PER_CATEGORY // len(codes_present))
        for code in codes_present:
            sample_code(code, category, n_per_code)
 
    print("\nLocal sample ready at:", os.path.abspath(SAMPLE_DIR))
 
 
if __name__ == "__main__":
    check_archive()
    sample_dataset()