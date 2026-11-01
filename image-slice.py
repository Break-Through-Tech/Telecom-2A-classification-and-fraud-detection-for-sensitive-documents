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

If no local archive is found at ARCHIVE_PATH, this streams the dataset
directly from Kaggle instead (see stream_sample_from_kaggle below) -- no
download-then-extract step needed.

Run from the project root:
    python scripts/task2_download_and_sample.py
"""

import os
import random
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import requests
from stream_unzip import TruncatedDataError, stream_unzip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARCHIVE_PATH, SAMPLE_DIR, CATEGORIES, N_PER_CATEGORY, SEED

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# IDNet nests each code's images under subfolders by type: typically
# "positive" for genuine documents, plus one or more "fraud*" folders for the
# dataset's OWN pre-labeled forgery examples (face replacement, combined
# tampering, inpainting/rewriting, etc). This sampler previously ignored
# subfolder entirely and grabbed every image matching the extension, which
# silently mixed labeled-fraud documents into what we called "real".
POSITIVE_PATH_MARKER = "positive"
FRAUD_PATH_MARKER = "fraud"


def _filter_genuine_images(image_names):
    """Keep only images under an explicit 'positive' subfolder when one
    exists. If a code has no folder with that name, fall back to excluding
    anything path-tagged as fraud, so a differently-named code doesn't
    silently yield zero images."""
    positive_only = [n for n in image_names if POSITIVE_PATH_MARKER in n.lower()]
    if positive_only:
        return positive_only
    return [n for n in image_names if FRAUD_PATH_MARKER not in n.lower()]


# --- Kaggle streaming fallback, used when ARCHIVE_PATH doesn't exist locally ---
# Kaggle serves this dataset as a single deflate-compressed zip entry
# ("files-archive") which itself decompresses to another zip. That inner zip
# is written in "streamed" (data-descriptor) mode, which means entries only
# become available in the order they appear in the archive -- we can't seek
# to an arbitrary one the way zipfile.ZipFile does for a local file above.
# stream_unzip processes the raw byte stream directly instead. This section
# is adapted from scripts/sample_idnet.py, which already solved the tricky
# low-level parsing quirks (see _strip_empty_placeholder_entries below).
KAGGLE_DATASET = "chitreshkr/idnet-identity-document-analysis"
# Optional safety cap on download volume, in MB. Set to None (the default) to
# disable it entirely. NOTE: this only limits network transfer, not disk
# usage -- unwanted (fraud/non-genuine) entries are drained in memory and
# never written to disk, so there's no disk-space reason to cap this. Since
# entries only arrive in archive order (no seeking), if a code's "fraud*"
# folders are listed before its "positive" folder, reaching enough genuine
# images can require streaming through a lot of fraud data first -- that's
# expected, not a bug, and the per-folder progress log below shows you where
# the stream actually is so you're not waiting blind.
MAX_DOWNLOAD_MB = None

_LOCAL_HEADER_SIG = b"PK\x03\x04"
_FOLLOWERS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x06\x06", b"PK\x01\x02")


def _strip_empty_placeholder_entries(byte_iter):
    """Some archivers emit zero-length STORED entries with the streaming
    (data-descriptor) flag set but no actual data descriptor bytes -- e.g. a
    bare 'GRC.zip' placeholder sitting next to a 'GRC/' folder. stream_unzip
    can't tell where such an entry ends without a size, so splice them out
    before they reach it. Every other entry passes through untouched."""
    buf = bytearray()
    for chunk in byte_iter:
        buf.extend(chunk)
        out = bytearray()
        i = 0
        while True:
            j = buf.find(_LOCAL_HEADER_SIG, i)
            if j == -1:
                keep = min(len(buf), 3)
                cut = max(i, len(buf) - keep)
                out += buf[i:cut]
                i = cut
                break
            out += buf[i:j]
            i = j
            remaining = len(buf) - i
            if remaining < 30:
                break
            header = bytes(buf[i:i + 30])
            (_, _, flags, method, _, _, _, comp_raw, _, fnlen, extralen) = (
                struct.unpack("<IHHHHHIIIHH", header)
            )
            need = 30 + fnlen + extralen
            if remaining < need + 4:
                break
            is_empty_placeholder = False
            if method == 0 and (flags & 0x08):
                extra = bytes(buf[i + 30 + fnlen:i + need])
                true_size = comp_raw
                p = 0
                while p + 4 <= len(extra):
                    tag, sz = struct.unpack("<HH", extra[p:p + 4])
                    if tag == 1 and sz >= 8:
                        true_size = struct.unpack("<Q", extra[p + 4:p + 12])[0]
                    p += 4 + sz
                if true_size == 0 and bytes(buf[i + need:i + need + 4]) in _FOLLOWERS:
                    is_empty_placeholder = True
            if is_empty_placeholder:
                i += need
            else:
                out += buf[i:i + 4]
                i += 4
        yield bytes(out)
        buf = buf[i:]
    if buf:
        yield bytes(buf)


def _get_signed_kaggle_url():
    token_path = Path.home() / ".kaggle" / "access_token"
    if not token_path.exists():
        raise FileNotFoundError(
            f"No Kaggle access token found at '{token_path}'. Generate one from your "
            f"Kaggle account settings and save just the raw token string there (no "
            f"quotes/JSON) to stream the dataset without downloading it locally first."
        )
    token = token_path.read_text().strip()
    resp = requests.get(
        f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}/files-archive",
        headers={"Authorization": f"Bearer {token}"},
        allow_redirects=False,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.headers["Location"]


def _outer_zip_payload_chunks(url, chunk_size=1 << 18):
    """Skip the outer local file header and yield the raw compressed payload
    bytes of the single outer zip entry ("files-archive")."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        it = r.iter_content(chunk_size=chunk_size)
        buf = b""
        header_len = 63  # 30 bytes fixed + filename(13) + extra(20)
        while len(buf) < header_len:
            buf += next(it)
        yield buf[header_len:]
        for chunk in it:
            yield chunk


def _decompressed_chunks(compressed_chunks):
    d = zlib.decompressobj(-15)  # raw deflate, no zlib/gzip header
    for chunk in compressed_chunks:
        out = d.decompress(chunk)
        if out:
            yield out
    tail = d.flush()
    if tail:
        yield tail


def stream_sample_from_kaggle():
    """Stream the dataset directly from Kaggle, with no local archive
    download and no temp files, collecting up to N_PER_CATEGORY genuine
    images per code as they're encountered in archive order."""
    code_to_category = {code: category for category, codes in CATEGORIES.items() for code in codes}
    counts = {code: 0 for code in code_to_category}

    print("Requesting signed download URL from Kaggle...")
    url = _get_signed_kaggle_url()
    downloaded = 0

    def counting_chunks():
        nonlocal downloaded
        for chunk in _outer_zip_payload_chunks(url):
            downloaded += len(chunk)
            yield chunk
            if MAX_DOWNLOAD_MB is not None and downloaded > MAX_DOWNLOAD_MB * 1024 * 1024:
                print(f"Hit the {MAX_DOWNLOAD_MB}MB download safety cap -- stopping early.")
                return

    seen_folders = set()

    byte_stream = _strip_empty_placeholder_entries(_decompressed_chunks(counting_chunks()))
    try:
        for file_name_b, _file_size, unzipped_chunks in stream_unzip(byte_stream):
            name = file_name_b.decode("utf-8", errors="replace")
            folder = name.rsplit("/", 1)[0] if "/" in name else ""
            if folder and folder not in seen_folders:
                seen_folders.add(folder)
                print(f"reached folder '{folder}' ({downloaded / 1e6:.0f}MB downloaded so far)")

            matched_code = next((c for c in code_to_category if name.startswith(f"{c}/")), None)
            is_image = name.lower().endswith(IMAGE_EXTENSIONS)
            is_genuine = POSITIVE_PATH_MARKER in name.lower()
            is_wanted = matched_code and is_image and is_genuine and counts[matched_code] < N_PER_CATEGORY

            if not is_wanted:
                for _ in unzipped_chunks:
                    pass  # still must drain the generator to advance the stream
                continue

            data = b"".join(unzipped_chunks)
            category = code_to_category[matched_code]
            out_dir = os.path.join(SAMPLE_DIR, "real", category)
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{matched_code}_{os.path.basename(name)}"
            with open(os.path.join(out_dir, out_name), "wb") as f:
                f.write(data)
            counts[matched_code] += 1
            print(
                f"[{matched_code}] streamed {counts[matched_code]}/{N_PER_CATEGORY} -> "
                f"'{category}' ({downloaded / 1e6:.0f}MB downloaded so far)"
            )

            if all(counts[c] >= N_PER_CATEGORY for c in code_to_category):
                print("All categories filled -- stopping stream early.")
                break
    except TruncatedDataError:
        # The byte stream ended mid-entry -- either the MAX_DOWNLOAD_MB cap was
        # hit (see the message printed by counting_chunks above) or the
        # connection dropped. Either way, report what was actually collected
        # instead of crashing with no summary; whatever was already written to
        # disk before this point is still valid.
        print(
            "Stream ended before every category was filled (hit the download "
            "cap or a connection drop) -- here's what was collected before that:"
        )

    for code, n in counts.items():
        if n < N_PER_CATEGORY:
            print(f"WARNING: only found {n}/{N_PER_CATEGORY} genuine images for code '{code}' before the stream ended.")

    print("\nStreamed sample ready at:", os.path.abspath(SAMPLE_DIR))


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
                before_filter = len(image_names)
                image_names = _filter_genuine_images(image_names)
                excluded = before_filter - len(image_names)
                if excluded:
                    print(
                        f"[{code}] excluded {excluded} fraud/non-genuine-labeled images "
                        f"out of {before_filter} found -- keeping only genuine ones"
                    )
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

    if not os.path.exists(ARCHIVE_PATH):
        print(f"No local archive at '{ARCHIVE_PATH}' -- streaming directly from Kaggle instead.")
        stream_sample_from_kaggle()
        return

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
    sample_dataset()