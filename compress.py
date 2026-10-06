from PIL import Image
import os
import sys
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DIR
 
COMPRESSED_DIR = "idnet_sample_compressed"
QUALITY = 85
SOURCE_EXTENSIONS = (".jpg", ".jpeg", ".png")
 
 
def compress_local_image(input_path: str, output_path: str, quality: int = 85) -> None:
    # 1. Load image directly from local disk
    with Image.open(input_path) as img:
 
        # 2. Pillow Mode Conversion: Safely handle color channels
        #    (converts formats with transparency/palettes like PNG to standard RGB)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
 
        # 3. WebP Encoding & Compression: Save to disk in WebP format
        #    (achieves ~90% size reduction at quality=85 without visible loss)
        img.save(output_path, "WEBP", quality=quality, optimize=True)
 
 
def compress_all(source_dir: str = SAMPLE_DIR, dest_dir: str = COMPRESSED_DIR, quality: int = QUALITY):
    total_before = 0
    total_after = 0
    n_converted = 0
    n_failed = 0
 
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            if not fname.lower().endswith(SOURCE_EXTENSIONS):
                continue
 
            src_path = os.path.join(root, fname)
 
            # Mirror the same folder structure (real/fake, category) under dest_dir
            rel_dir = os.path.relpath(root, source_dir)
            out_dir = os.path.join(dest_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
 
            out_name = os.path.splitext(fname)[0] + ".webp"
            dest_path = os.path.join(out_dir, out_name)
 
            try:
                before_size = os.path.getsize(src_path)
                compress_local_image(src_path, dest_path, quality=quality)
                after_size = os.path.getsize(dest_path)
 
                total_before += before_size
                total_after += after_size
                n_converted += 1
            except Exception as e:
                print(f"FAILED on {src_path}: {e}")
                n_failed += 1
 
    print(f"\nConverted {n_converted} images ({n_failed} failed).")
    if total_before > 0:
        reduction_pct = 100 * (1 - total_after / total_before)
        print(f"Total size before: {total_before / (1024*1024):.1f} MB")
        print(f"Total size after:  {total_after / (1024*1024):.1f} MB")
        print(f"Reduction: {reduction_pct:.1f}%")
    print(f"\nCompressed images written to: {os.path.abspath(dest_dir)}")
 
 
if __name__ == "__main__":
    compress_all()