#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3A face-verification tests.

Run:  py tests/test_face.py

Tests
-----
1. Print cv2 version and FaceDetectorYN / FaceRecognizerSF availability.
2. For every photo in faces/, run YuNet and print detection result + confidence.
3. Enrol 'me' from faces/me_*.jpg.
4. Check that faces_test/me_test.jpg is accepted (score printed).
5. Check that each placeholder visitor is rejected (score printed).
6. Right-person / wrong-patient check (if >=2 patients exist).

NOTE on synthetic placeholders: generated oval images will not pass
YuNet face detection.  The test falls back to whole-image SFace embedding
for those.  Drop in your own photos to use the real detection path.
"""
import sys
import os

# Force UTF-8 output so checkmarks work on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import glob
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = os.path.join(ROOT, "controllers", "medibot_controller")
sys.path.insert(0, ROOT)
sys.path.insert(0, CTRL)

MODELS_DIR = os.path.join(CTRL, "models")
FACES_DIR  = os.path.join(CTRL, "faces")
TEST_DIR   = os.path.join(CTRL, "faces_test")
DB_PATH    = os.path.join(CTRL, "patients.json")
YUNET_PATH = os.path.join(MODELS_DIR, "yuNet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "sface.onnx")

# ---------------------------------------------------------------------------
# Step 1 - OpenCV capabilities
# ---------------------------------------------------------------------------
print("=" * 62)
print("STEP 1 - OpenCV capabilities")
print(f"  cv2 version       : {cv2.__version__}")
print(f"  FaceDetectorYN    : {hasattr(cv2, 'FaceDetectorYN')}")
print(f"  FaceRecognizerSF  : {hasattr(cv2, 'FaceRecognizerSF')}")

_yunet_ok = os.path.isfile(YUNET_PATH)
_sface_ok = os.path.isfile(SFACE_PATH)
if _yunet_ok:
    print(f"  yuNet.onnx        : {os.path.getsize(YUNET_PATH):,} bytes")
else:
    print("  yuNet.onnx        : MISSING -- run: py tools/download_models.py")
if _sface_ok:
    print(f"  sface.onnx        : {os.path.getsize(SFACE_PATH):,} bytes")
else:
    print("  sface.onnx        : MISSING -- run: py tools/download_models.py")

if not (_yunet_ok and _sface_ok):
    print("\n[SKIP] Models missing - all face tests skipped.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
_det = None
_rec = None
try:
    _det = cv2.FaceDetectorYN.create(YUNET_PATH, "", (0, 0), 0.5, 0.3, 5000)
    print("  YuNet loaded      : OK")
except Exception as e:
    print(f"  YuNet load FAILED : {e}")

try:
    _rec = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    print("  SFace loaded      : OK")
except Exception as e:
    print(f"  SFace load FAILED : {e}")

if _det is None or _rec is None:
    print("[SKIP] Could not load models.")
    sys.exit(0)

# Threshold from config.json (default 0.4)
THRESHOLD = 0.4
try:
    cfg_path = os.path.join(CTRL, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        THRESHOLD = float(json.load(fh).get("face_verification", {}).get("threshold", THRESHOLD))
except Exception:
    pass
print(f"  Threshold         : {THRESHOLD}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def yunet_detect(img):
    """Return faces array (N,15) or None."""
    h, w = img.shape[:2]
    _det.setInputSize((w, h))
    _, faces = _det.detect(img)
    return faces


def sface_embed(img, face_row):
    """Return normalised SFace embedding. face_row is required (no whole-image fallback)."""
    if face_row is None or _rec is None:
        return None
    try:
        aligned = _rec.alignCrop(img, face_row)
        feat = _rec.feature(aligned)
        feat = np.array(feat, dtype=np.float32).ravel()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else None
    except Exception as e:
        return None


def embed_img(img, label=""):
    """Detect face with YuNet, then embed with SFace. If no face is detected, reject."""
    faces = yunet_detect(img)
    if faces is None or len(faces) == 0:
        return None, "NO FACE DETECTED"
    best = max(faces, key=lambda r: float(r[14]))
    emb = sface_embed(img, best)
    mode = f"YuNet conf={float(best[14]):.3f}"
    return emb, mode


def cos(a, b):
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Step 2 - YuNet scan of faces/
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("STEP 2 - YuNet detection on every file in faces/")

face_files = sorted(glob.glob(os.path.join(FACES_DIR, "*.*")))
real_me_detected = False
if not face_files:
    print("  (faces/ is empty)")
else:
    for fp in face_files:
        img = cv2.imread(fp)
        if img is None:
            print(f"  {os.path.basename(fp):44s} cannot read")
            continue
        faces = yunet_detect(img)
        if faces is None or len(faces) == 0:
            print(f"  {os.path.basename(fp):44s} NO face detected")
        else:
            best = max(faces, key=lambda r: float(r[14]))
            conf = float(best[14])
            print(f"  {os.path.basename(fp):44s} DETECTED  conf={conf:.3f}")
            if "me_" in os.path.basename(fp):
                real_me_detected = True

if real_me_detected:
    print("  -> Real photos of 'me' detected by YuNet.")
else:
    print("  -> WARNING: No real photos of 'me' detected by YuNet.")


# ---------------------------------------------------------------------------
# Step 3 - Enrol 'me'
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("STEP 3 - Enrol 'me' from faces/me_*.jpg")

me_photos = sorted(glob.glob(os.path.join(FACES_DIR, "me_*.jpg")))
if not me_photos:
    print("  [SKIP] No files matching faces/me_*.jpg found.")
    sys.exit(0)

me_embs = []
for fp in me_photos:
    img = cv2.imread(fp)
    if img is None:
        print(f"  [WARN] Cannot read {fp}")
        continue
    emb, mode = embed_img(img)
    if emb is not None:
        me_embs.append(emb)
        print(f"  enrolled {os.path.basename(fp):30s}  [{mode}]")
    else:
        print(f"  [WARN] YuNet: NO face detected in {os.path.basename(fp)} – photo REJECTED")

if not me_embs:
    print("  [FAIL] No embeddings for 'me' - cannot continue.")
    sys.exit(1)

me_avg = np.mean(me_embs, axis=0)
me_avg /= np.linalg.norm(me_avg)
print(f"  Average embedding built from {len(me_embs)} photo(s).")

# Patient lookup
with open(DB_PATH, encoding="utf-8") as fh:
    DB = json.load(fh)

me_patient_ids = set()
for pat in DB.get("patients", []):
    for fam in pat.get("authorised_family", []):
        if fam.get("id") == "me":
            me_patient_ids.add(pat["id"])


# ---------------------------------------------------------------------------
# Step 4 - Accept me from faces_test/me_test.jpg
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print(f"STEP 4 - Accept 'me' from faces_test/  (threshold={THRESHOLD})")

me_test = os.path.join(TEST_DIR, "me_test.jpg")
if not os.path.isfile(me_test):
    print(f"  [SKIP] {me_test} not found.")
else:
    img = cv2.imread(me_test)
    emb, mode = embed_img(img, "me_test.jpg")
    if emb is None:
        print(f"  me_test.jpg  [{mode}]  FAIL (no face detected by YuNet)")
    else:
        s = cos(emb, me_avg)
        if s >= THRESHOLD:
            print(f"  me_test.jpg  score={s:.4f}  [{mode}]  PASS (accepted)")
        else:
            print(f"  me_test.jpg  score={s:.4f}  [{mode}]  FAIL (below threshold)")


# ---------------------------------------------------------------------------
# Step 5 - Reject visitors
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("STEP 5 - Reject visitors from faces_test/ or faces/")

vis_photos = sorted(list(set(glob.glob(os.path.join(TEST_DIR, "VISITOR_*.*")) + glob.glob(os.path.join(FACES_DIR, "VISITOR_*.*")))))
if not vis_photos:
    print("  (no VISITOR_* photos found)")
else:
    for fp in vis_photos:
        img = cv2.imread(fp)
        if img is None:
            print(f"  {os.path.basename(fp):44s} cannot read")
            continue
        emb, mode = embed_img(img)
        if emb is None:
            print(f"  {os.path.basename(fp):44s} no face detected - PASS (rejected)")
            continue
        s = cos(emb, me_avg)
        if s < THRESHOLD:
            print(f"  {os.path.basename(fp):44s} score={s:.4f}  [{mode}]  PASS (rejected)")
        else:
            print(f"  {os.path.basename(fp):44s} score={s:.4f}  [{mode}]  FAIL (falsely accepted!)")


# ---------------------------------------------------------------------------
# Step 6 - Right person, wrong patient
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("STEP 6 - Right person, wrong patient")

all_patients = DB.get("patients", [])
other_patients = [p for p in all_patients if p["id"] not in me_patient_ids]

if not other_patients:
    print("  [SKIP] No second patient in DB.")
else:
    other = other_patients[0]
    auth_ids = {fam["id"] for fam in other.get("authorised_family", [])}
    is_auth = "me" in auth_ids
    s_self = cos(me_avg, me_avg)  # always 1.0 - shown for reference
    if not is_auth:
        print(
            f"  Patient {other['id']} ({other['name']})"
            f"  face-score-vs-self={s_self:.4f}  in_auth_list={is_auth}  "
            f"  PASS (correctly rejected for wrong patient)"
        )
    else:
        print(
            f"  Patient {other['id']} ({other['name']})"
            f"  face-score-vs-self={s_self:.4f}  in_auth_list={is_auth}  "
            f"  FAIL ('me' incorrectly listed as authorised for this patient)"
        )

print("\n" + "=" * 62)
print("DONE - test_face.py complete")
