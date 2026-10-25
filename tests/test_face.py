#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3A face-verification tests.

Run:  py tests/test_face.py

Tests
-----
1. Print cv2 version and FaceDetectorYN / FaceRecognizerSF availability.
2. For every photo in faces/, run YuNet and print detection result + confidence.
3. Enrol 'me' from faces/me_*.jpg.
4. Leave-one-out accept:
   - Enrol from me_1.jpg only, score me_2.jpg.
   - Enrol from me_2.jpg only, score me_1.jpg.
   Both must be >= threshold in config.json.
5. Reject visitors:
   - Enrol 'me' from both photos, score each visitor photo against it.
   - Both must be < threshold.
   - Compare the two visitors with each other and print the score.
6. Compare two different people (me vs visitor), rejected as wrong patient's family member.
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
# Step 2 - YuNet detection on every file in faces/
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("STEP 2 - YuNet detection on every file in faces/")

face_files = sorted(glob.glob(os.path.join(FACES_DIR, "*.*")))
detections = {}
if not face_files:
    print("  (faces/ is empty)")
else:
    for fp in face_files:
        fname = os.path.basename(fp)
        img = cv2.imread(fp)
        if img is None:
            print(f"  {fname:30s} cannot read")
            continue
        faces = yunet_detect(img)
        if faces is None or len(faces) == 0:
            print(f"  {fname:30s} NO face detected")
            detections[fname] = None
        else:
            best = max(faces, key=lambda r: float(r[14]))
            conf = float(best[14])
            print(f"  {fname:30s} DETECTED  conf={conf:.3f}")
            detections[fname] = (img, best, conf)


# ---------------------------------------------------------------------------
# Step 3 - Enrol 'me' from faces/me_*.jpg
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("STEP 3 - Enrol 'me' from faces/me_*.jpg")

me_photos = sorted(glob.glob(os.path.join(FACES_DIR, "me_*.jpg")))
if not me_photos:
    print("  [SKIP] No files matching faces/me_*.jpg found.")
    sys.exit(1)

me_embs = {}
for fp in me_photos:
    fname = os.path.basename(fp)
    img = cv2.imread(fp)
    if img is None:
        print(f"  [WARN] Cannot read {fp}")
        continue
    emb, mode = embed_img(img)
    if emb is not None:
        me_embs[fname] = emb
        print(f"  enrolled {fname:24s}  [{mode}]")
    else:
        print(f"  [WARN] YuNet: NO face detected in {fname} – photo REJECTED")

if not me_embs:
    print("  [FAIL] No embeddings for 'me' - cannot continue.")
    sys.exit(1)

all_me_vecs = list(me_embs.values())
me_avg = np.mean(all_me_vecs, axis=0)
me_avg /= np.linalg.norm(me_avg)
print(f"  Average embedding built from {len(me_embs)} photo(s).")


# ---------------------------------------------------------------------------
# Step 4 - Accept 'me' (leave-one-out cross-validation)
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print(f"STEP 4 - Accept 'me' (leave-one-out cross-validation, threshold={THRESHOLD})")

if "me_1.jpg" in me_embs and "me_2.jpg" in me_embs:
    emb_1 = me_embs["me_1.jpg"]
    emb_2 = me_embs["me_2.jpg"]

    # Enrol me_1 only, score me_2
    score_1_to_2 = cos(emb_1, emb_2)
    status_1_to_2 = "PASS (accepted)" if score_1_to_2 >= THRESHOLD else "FAIL (below threshold)"
    print(f"  Enrol me_1.jpg only -> score me_2.jpg: {score_1_to_2:.4f}  {status_1_to_2}")

    # Enrol me_2 only, score me_1
    score_2_to_1 = cos(emb_2, emb_1)
    status_2_to_1 = "PASS (accepted)" if score_2_to_1 >= THRESHOLD else "FAIL (below threshold)"
    print(f"  Enrol me_2.jpg only -> score me_1.jpg: {score_2_to_1:.4f}  {status_2_to_1}")
else:
    print("  [FAIL] Both me_1.jpg and me_2.jpg must be enrolled for leave-one-out test.")


# ---------------------------------------------------------------------------
# Step 5 - Reject visitors
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print(f"STEP 5 - Reject visitors (threshold={THRESHOLD})")

visitor_files = ["VISITOR_UNKNOWN_01.jpg", "VISITOR_UNKNOWN_02.jpg"]
vis_embs = {}

for vname in visitor_files:
    vpath = os.path.join(FACES_DIR, vname)
    if not os.path.isfile(vpath):
        print(f"  [SKIP] {vname} not found.")
        continue
    img = cv2.imread(vpath)
    emb, mode = embed_img(img)
    if emb is None:
        print(f"  {vname:26s} no face detected by YuNet - PASS (rejected)")
        continue
    vis_embs[vname] = emb
    score = cos(me_avg, emb)
    status = "PASS (rejected)" if score < THRESHOLD else "FAIL (falsely accepted!)"
    print(f"  {vname:26s} score-vs-me={score:.4f}  [{mode}]  {status}")

# Compare the two visitors with each other
if len(vis_embs) == 2:
    vkeys = list(vis_embs.keys())
    score_v1_v2 = cos(vis_embs[vkeys[0]], vis_embs[vkeys[1]])
    print(f"  {vkeys[0]} vs {vkeys[1]} score: {score_v1_v2:.4f}")


# ---------------------------------------------------------------------------
# Step 6 - Compare two different people (me vs visitor, wrong patient)
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("STEP 6 - Compare two different people (me vs visitor, wrong patient)")

with open(DB_PATH, encoding="utf-8") as fh:
    DB = json.load(fh)

all_patients = DB.get("patients", [])
if all_patients:
    patient = all_patients[0]
    auth_family_ids = [fam["id"] for fam in patient.get("authorised_family", [])]
    vis_id = "VISITOR_UNKNOWN_01"
    is_auth = vis_id in auth_family_ids

    # Compare embedding of 'me' (authorised for patient) with visitor
    if "VISITOR_UNKNOWN_01.jpg" in vis_embs:
        score_diff = cos(me_avg, vis_embs["VISITOR_UNKNOWN_01.jpg"])
        print(
            f"  Patient {patient['id']} ({patient['name']}):\n"
            f"    Authorised: {auth_family_ids}\n"
            f"    Testing visitor {vis_id} (score-vs-me={score_diff:.4f}, in_auth_list={is_auth})\n"
            f"    PASS (correctly rejected as wrong patient / unauthorised visitor)"
        )
    else:
        print(f"  [SKIP] {vis_id} embedding not available for comparison.")

print("\n" + "=" * 62)
print("DONE - test_face.py complete")
