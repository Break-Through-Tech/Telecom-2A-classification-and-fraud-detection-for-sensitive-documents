"""
Pull a small local sample from the IDNet dataset on Kaggle without downloading
the full ~49GB archive.

The Kaggle dataset "chitreshkr/idnet-identity-document-analysis" is stored as
a single deflate-compressed zip entry ("files-archive") which itself decompresses
to another zip containing per-country files/folders (e.g. GRC/fraud3_face_replacement/*.png).
Because it's a streamed (data-descriptor) zip, we can't jump to arbitrary
entries via HTTP range requests -- entries only become available in the order
they appear in the archive. This script streams the download, decompresses on
the fly, and saves image files as they're encountered, stopping once it hits
a target image count or a download-size budget (whichever comes first).

Usage:
    python scripts/sample_idnet.py --max-images 300 --max-download-mb 300
"""

import argparse
import struct
import sys
import zlib
from pathlib import Path

import requests
from stream_unzip import stream_unzip

LOCAL_HEADER_SIG = b"PK\x03\x04"
FOLLOWERS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x06\x06", b"PK\x01\x02")


def strip_empty_placeholder_entries(byte_iter):
    """Some archivers emit zero-length STORED entries with the streaming
    (data-descriptor) flag set but no actual data descriptor bytes -- e.g.
    a bare 'GRC.zip' placeholder sitting next to a 'GRC/' folder. Libraries
    that stream-unzip generically refuse these (they can't tell where a
    STORED+streamed entry ends without a size). We've confirmed by hand that
    these specific entries are truly zero bytes with the next entry's local
    header immediately following, so splice them out before they reach the
    unzip library. Every other entry passes through completely untouched.
    """
    buf = bytearray()
    for chunk in byte_iter:
        buf.extend(chunk)
        out = bytearray()
        i = 0
        while True:
            j = buf.find(LOCAL_HEADER_SIG, i)
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
            header = bytes(buf[i : i + 30])
            (_, _, flags, method, _, _, _, comp_raw, _, fnlen, extralen) = (
                struct.unpack("<IHHHHHIIIHH", header)
            )
            need = 30 + fnlen + extralen
            if remaining < need + 4:
                break
            is_empty_placeholder = False
            if method == 0 and (flags & 0x08):
                extra = bytes(buf[i + 30 + fnlen : i + need])
                true_size = comp_raw
                p = 0
                while p + 4 <= len(extra):
                    tag, sz = struct.unpack("<HH", extra[p : p + 4])
                    if tag == 1 and sz >= 8:
                        true_size = struct.unpack("<Q", extra[p + 4 : p + 12])[0]
                    p += 4 + sz
                if true_size == 0 and bytes(buf[i + need : i + need + 4]) in FOLLOWERS:
                    is_empty_placeholder = True
            if is_empty_placeholder:
                i += need
            else:
                out += buf[i : i + 4]
                i += 4
        yield bytes(out)
        buf = buf[i:]
    if buf:
        yield bytes(buf)

DATASET = "chitreshkr/idnet-identity-document-analysis"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def get_signed_url() -> str:
    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    resp = requests.get(
        f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/files-archive",
        headers={"Authorization": f"Bearer {token}"},
        allow_redirects=False,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.headers["Location"]


def outer_zip_payload_chunks(url: str, chunk_size: int = 1 << 18):
    """Skip the outer local file header and yield the raw compressed
    payload bytes of the single outer zip entry ("files-archive")."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        it = r.iter_content(chunk_size=chunk_size)
        buf = b""
        # Outer local file header: 30 bytes fixed + filename(13) + extra(20) = 63 bytes
        header_len = 63
        while len(buf) < header_len:
            buf += next(it)
        yield buf[header_len:]
        for chunk in it:
            yield chunk


def decompressed_chunks(compressed_chunks):
    d = zlib.decompressobj(-15)  # raw deflate, no zlib/gzip header
    for chunk in compressed_chunks:
        out = d.decompress(chunk)
        if out:
            yield out
    tail = d.flush()
    if tail:
        yield tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=300)
    ap.add_argument("--max-download-mb", type=int, default=300)
    ap.add_argument("--out", default="data/sample")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Requesting signed download URL from Kaggle...", file=sys.stderr)
    url = get_signed_url()

    downloaded = 0
    saved = 0
    seen_dirs = {}

    def counting_chunks():
        nonlocal downloaded
        for chunk in outer_zip_payload_chunks(url):
            downloaded += len(chunk)
            yield chunk
            if downloaded > args.max_download_mb * 1024 * 1024:
                return

    try:
        for file_name_b, file_size, unzipped_chunks in stream_unzip(
            strip_empty_placeholder_entries(decompressed_chunks(counting_chunks()))
        ):
            name = file_name_b.decode("utf-8", errors="replace")
            is_image = name.lower().endswith(IMAGE_SUFFIXES)
            is_dir = name.endswith("/")

            if saved >= args.max_images:
                for _ in unzipped_chunks:
                    pass
                continue

            if is_dir or not is_image:
                for _ in unzipped_chunks:
                    pass
                continue

            parent = str(Path(name).parent)
            if seen_dirs.get(parent, 0) >= 25:
                # cap per-folder to spread the budget across categories we do reach
                for _ in unzipped_chunks:
                    pass
                continue

            dest = out_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in unzipped_chunks:
                    f.write(chunk)

            saved += 1
            seen_dirs[parent] = seen_dirs.get(parent, 0) + 1
            if saved % 25 == 0:
                print(
                    f"saved={saved} downloaded={downloaded/1e6:.1f}MB "
                    f"folders={len(seen_dirs)}",
                    file=sys.stderr,
                )
    except StopIteration:
        pass
    except Exception as e:
        print(f"stopped early ({e.__class__.__name__}: {e})", file=sys.stderr)

    print(
        f"\nDone. Saved {saved} images from {len(seen_dirs)} folders, "
        f"~{downloaded/1e6:.1f}MB downloaded, into {out_dir}/",
        file=sys.stderr,
    )
    for d, n in sorted(seen_dirs.items()):
        print(f"  {d}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
