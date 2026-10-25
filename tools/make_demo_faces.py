#!/usr/bin/env python3
"""Generate synthetic placeholder face images for demo / testing.

NOTICE: These placeholder faces are synthetic (procedurally generated) and
do not depict any real person. They are for development and testing only.
Do NOT use real photos of people without their explicit written consent.

Usage:
    py tools/make_demo_faces.py

Writes placeholder PNG files to:
    controllers/medibot_controller/faces/<ID>.png
for every family member and visitor listed in patients.json.

If a file already exists (e.g. a real photo supplied by the user) it is
NOT overwritten – the script skips that person and prints a notice.

The script also tries to enrich the placeholder with LFW subset faces
(via scikit-learn) if available and if Python < 3.14 (LFW requires scipy
which does not yet build wheels for 3.14 on Windows).
"""
import json
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FACES_DIR = os.path.join(ROOT, "controllers", "medibot_controller", "faces")
DB_PATH = os.path.join(ROOT, "controllers", "medibot_controller", "patients.json")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# --------------------------------------------------------------------------
# LFW availability check
# --------------------------------------------------------------------------
LFW_AVAILABLE = False
if sys.version_info < (3, 14):
    try:
        from sklearn.datasets import fetch_lfw_people
        LFW_AVAILABLE = True
    except ImportError:
        pass

if not LFW_AVAILABLE:
    print(
        "[make_demo_faces] scikit-learn LFW is not available "
        f"(Python {sys.version_info.major}.{sys.version_info.minor}"
        + (", scipy not yet available for Python 3.14+)" if sys.version_info >= (3, 14) else ").")
    )
    print("[make_demo_faces] Generating synthetic placeholder faces instead.")


# --------------------------------------------------------------------------
# Synthetic face generator
# --------------------------------------------------------------------------
def _hue_to_rgb(h: float) -> tuple:
    """Convert a hue [0, 360) to an BGR tuple (uint8)."""
    h = h % 360
    s, v = 0.55, 0.92
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))  # BGR


def make_placeholder_face(person_id: str, size: int = 256) -> "np.ndarray":
    """Create a simple synthetic face image: coloured oval + eyes + label."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)

    # Deterministic hue per person ID
    hue = (hash(person_id) % 360)
    skin = _hue_to_rgb(hue)
    dark = tuple(max(0, c - 60) for c in skin)

    cx, cy = size // 2, size // 2
    rx, ry = size // 3, int(size * 0.42)

    # Face oval
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, dark, 2)

    # Eyes
    eye_y = cy - ry // 5
    for ex in [cx - rx // 3, cx + rx // 3]:
        cv2.circle(img, (ex, eye_y), rx // 7, (30, 30, 30), -1)
        cv2.circle(img, (ex, eye_y), rx // 12, (255, 255, 255), -1)

    # Nose
    cv2.circle(img, (cx, cy + ry // 8), rx // 12, dark, -1)

    # Mouth
    smile_pts = np.array([
        [cx - rx // 4, cy + ry // 3],
        [cx,           cy + ry // 3 + rx // 8],
        [cx + rx // 4, cy + ry // 3],
    ], np.int32)
    cv2.polylines(img, [smile_pts], False, dark, 2, cv2.LINE_AA)

    # Label
    label = person_id[-12:] if len(person_id) > 12 else person_id
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = size / 512
    cv2.putText(img, label, (4, size - 8), font, scale * 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    return img


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not HAS_CV2:
        print("[make_demo_faces] cv2 is required. Install opencv-python.")
        return 1

    os.makedirs(FACES_DIR, exist_ok=True)

    if not os.path.isfile(DB_PATH):
        print(f"[make_demo_faces] patients.json not found at {DB_PATH}")
        return 1

    with open(DB_PATH, encoding="utf-8") as fh:
        db = json.load(fh)

    # Collect everyone who needs a face file
    tasks = []
    for patient in db.get("patients", []):
        for fam in patient.get("authorised_family", []):
            tasks.append(fam)
    for vis in db.get("unauthorised_visitors", []):
        tasks.append(vis)

    created, skipped = 0, 0
    for person in tasks:
        pid = person.get("id", "UNKNOWN")
        # Support both new face_files list and legacy face_image string
        face_files = person.get("face_files", [])
        if not face_files:
            legacy = person.get("face_image", "")
            if legacy:
                face_files = [legacy]

        for face_rel in face_files:
            fname = os.path.basename(face_rel)
            dest = os.path.join(FACES_DIR, fname)

            if os.path.isfile(dest):
                sz = os.path.getsize(dest)
                print(f"  SKIP  {fname} already exists ({sz} bytes)")
                skipped += 1
                continue

            img = make_placeholder_face(pid)
            cv2.imwrite(dest, img)
            print(f"  GEN   {fname}  (synthetic placeholder – {pid})")
            created += 1

    print(f"\n[make_demo_faces] Done: {created} generated, {skipped} skipped.")
    print("[make_demo_faces] NOTICE: Placeholder faces are purely synthetic.")
    print("[make_demo_faces] Replace them with real consented photos in faces/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
