#!/usr/bin/env python3
"""Download YuNet and SFace ONNX models from the OpenCV Zoo.

Usage:
    py tools/download_models.py

Downloads to: controllers/medibot_controller/models/
  - yuNet.onnx  (face detection)
  - sface.onnx  (face recognition)

If a download fails the script prints the exact URL and the target path
so the file can be saved manually.
"""
import os
import sys
import urllib.request
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(ROOT, "controllers", "medibot_controller", "models")

# ---------------------------------------------------------------------------
# Model URLs – OpenCV Zoo GitHub releases
# ---------------------------------------------------------------------------
MODELS = [
    {
        "filename": "yuNet.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "fallback_url": (
            "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
    },
    {
        "filename": "sface.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "fallback_url": (
            "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
    },
]


def download(url: str, dest: str) -> bool:
    """Try to download *url* to *dest*. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MediBot-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
                total += len(chunk)
        return True
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_ok = True

    for spec in MODELS:
        dest = os.path.join(MODELS_DIR, spec["filename"])
        size_kb = os.path.getsize(dest) // 1024 if os.path.isfile(dest) else None

        if size_kb is not None and size_kb > 50:
            print(f"[OK] {spec['filename']} already present – {size_kb} KB")
            continue

        print(f"\n[DOWNLOAD] {spec['filename']} …")
        print(f"  URL: {spec['url']}")

        ok = download(spec["url"], dest)
        if not ok:
            print(f"  Trying fallback URL: {spec['fallback_url']}")
            ok = download(spec["fallback_url"], dest)

        if ok and os.path.isfile(dest):
            size_kb = os.path.getsize(dest) // 1024
            print(f"  [OK] Saved to: {dest}  ({size_kb} KB)")
        else:
            all_ok = False
            print(f"\n  [FAIL] Could not download {spec['filename']}.")
            print(f"  *** Please download manually: ***")
            print(f"  URL : {spec['url']}")
            print(f"  Save to: {dest}")

    # Final report
    print("\n--- Model files in models/ ---")
    for spec in MODELS:
        dest = os.path.join(MODELS_DIR, spec["filename"])
        if os.path.isfile(dest):
            size = os.path.getsize(dest)
            print(f"  {spec['filename']:30s}  {size:>12,} bytes  ({size // 1024} KB)")
        else:
            print(f"  {spec['filename']:30s}  MISSING")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
