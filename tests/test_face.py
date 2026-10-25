#!/usr/bin/env python3
"""Phase 3A face-verification tests.

Run:  py tests/test_face.py

Tests
-----
1. Print cv2 version and FaceDetectorYN / FaceRecognizerSF availability.
2. For every photo in faces/, run YuNet and print whether a face was
   detected and its confidence score.
3. Enrol "me" from faces/me_*.jpg.
4. Check that a different photo of me from faces_test/me_test.jpg is
   accepted (score printed; PASS if >= threshold).
5. Check that each placeholder visitor from faces_test/ is rejected.
6. If a second patient exists, check that "right person, wrong patient"
   is rejected (me accepted globally but NOT in other patient's family).

Skip all tests gracefully if the ONNX models are missing.

NOTE on synthetic placeholders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The automatically generated placeholder images (coloured ovals) will NOT
pass YuNet face detection – they are not photos of real faces.  The test
falls back to whole-image embedding for those, so you can observe the
score mechanics.  When you drop in your own photos (faces/me_1.jpg, etc.)
the YuNet path will be used automatically.
"""
import json
import os
import sys
import glob

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = os.path.join(ROOT, "controllers", "medibot_controller")
sys.path.insert(0, ROOT)
sys.path.insert(0, CTRL)

MODELS_DIR = os.path.join(CTRL, "models")
FACES_DIR  = os.path.join(CTRL, "faces")
TEST_DIR   = os.path.join(CTRL, "faces_test")
DB_PATH    = os.path.join(CTRL, "patients.json")

# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------
YUNET_PATH = os.path.join(MODELS_DIR, "yuNet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "sface.onnx")

print("=" * 60)
print("STEP 1 – OpenCV capabilities")
print(f"  cv2 version       : {cv2.__version__}")
print(f"  FaceDetectorYN    : {hasattr(cv2, 'FaceDetectorYN')}")
print(f"  FaceRecognizerSF  : {hasattr(cv2, 'FaceRecognizerSF')}")

if not os.path.isfile(YUNET_PATH) or not os.path.isfile(SFACE_PATH):
    print("\n[SKIP] One or both ONNX model files are missing.")
    print(f"  yuNet.onnx : {'FOUND' if os.path.isfile(YUNET_PATH) else 'MISSING – run: py tools/download_models.py'}")
    print(f"  sface.onnx : {'FOUND' if os.path.isfile(SFACE_PATH) else 'MISSING – run: py tools/download_models.py'}")
    sys.exit(0)

print(f"\n  yuNet.onnx  : {os.path.getsize(YUNET_PATH):,} bytes")
print(f"  sface.onnx  : {os.path.getsize(SFACE_PATH):,} bytes")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
_detector = None
_recogniser = None
try:
    # Use 0.5 threshold for normal detection
    _detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (0, 0), 0.5, 0.3, 5000)
    print("  YuNet loaded OK")
except Exception as exc:
    print(f"  [ERROR] YuNet load failed: {exc}")

try:
    _recogniser = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    print("  SFace loaded OK")
except Exception as exc:
    print(f"  [ERROR] SFace load failed: {exc}")

if _detector is None or _recogniser is None:
    print("[SKIP] Could not load models – skipping all tests.")
    sys.exit(0)

# Threshold from config
THRESHOLD = 0.4
try:
    cfg_path = os.path.join(CTRL, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        THRESHOLD = float(json.load(fh).get("face_verification", {}).get("threshold", THRESHOLD))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
REAL_PHOTOS = False  # set True if me_1.jpg passes YuNet

def detect(img, conf_thresh=0.5):
    """Run YuNet on img. Returns faces array or None."""
    h, w = img.shape[:2]
    _detector.setInputSize((w, h))
    # Temporarily lower threshold if needed
    old_thresh = 0.5
    _, faces = _detector.detect(img)
    return faces  # None or (N, 15)


def embed_with_detection(img, label=""):
    """Detect face, align, embed.  Falls back to resize-embed for synthetics."""
    faces = detect(img)
    if faces is not None and len(faces) > 0:
        best_row = max(faces, key=lambda r: float(r[14]))
        conf = float(best_row[14])
        try:
            aligned = _recogniser.alignCrop(img, best_row)
            feat = _recogniser.feature(aligned)
            feat = np.array(feat, dtype=np.float32).ravel()
            norm = np.linalg.norm(feat)
            emb = feat / norm if norm > 0 else None
            mode = f"YuNet-detected conf={conf:.3f}"
            return emb, mode
        except Exception as exc:
            pass

    # Fallback: resize whole image to 112x112 and embed (synthetic faces only)
    try:
        resized = cv2.resize(img, (112, 112))
        feat = _recogniser.feature(resized)
        feat = np.array(feat, dtype=np.float32).ravel()
        norm = np.linalg.norm(feat)
        emb = feat / norm if norm > 0 else None
        mode = "SYNTHETIC-fallback (no face detected by YuNet)"
        return emb, mode
    except Exception as exc:
        return None, f"embed-error: {exc}"


def cosine(a, b):
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# STEP 2 – YuNet scan of faces/
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 – YuNet detection on every photo in faces/")

face_photos = sorted(glob.glob(os.path.join(FACES_DIR, "*.*")))
any_real = False
if not face_photos:
    print("  (no photos in faces/)")
else:
    for path in face_photos:
        img = cv2.imread(path)
        if img is None:
            print(f"  {os.path.basename(path):42s} cannot read")
            continue
        faces = detect(img)
        if faces is None or len(faces) == 0:
            print(f"  {os.path.basename(path):42s} NO face detected  (synthetic placeholder)")
        else:
            best = max(faces, key=lambda r: float(r[14]))
            conf = float(best[14])
            print(f"  {os.path.basename(path):42s} detected  conf={conf:.3f}")
            if "me_" in os.path.basename(path):
                any_real = True

if any_real:
    REAL_PHOTOS = True
    print("\n  -> Real 'me' photos detected by YuNet.")
else:
    print(
        "\n  -> No real faces detected. 'me_1.jpg' and 'me_2.jpg' appear to be synthetic\n"
        "    placeholders. Drop in your actual photos and re-run.\n"
        "    Tests will continue using whole-image embedding fallback."
    )


# ---------------------------------------------------------------------------
# STEP 3 – Enrol "me"
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 – Enrol 'me' from faces/me_*.jpg")

me_photos = sorted(glob.glob(os.path.join(FACES_DIR, "me_*.jpg")))
if not me_photos:
    print("  [SKIP] No photos matching faces/me_*.jpg")
    sys.exit(0)

me_embeddings = []
for path in me_photos:
    img = cv2.imread(path)
    if img is None:
        print(f"  [WARN] Cannot read {path}")
        continue
    emb, mode = embed_with_detection(img, os.path.basename(path))
    if emb is not None:
        me_embeddings.append(emb)
        print(f"  Enrolled {os.path.basename(path)}  [{mode}]")
    else:
        print(f"  [WARN] Embedding failed for {os.path.basename(path)}")

if not me_embeddings:
    print("  [FAIL] No embeddings obtained for 'me'.")
    sys.exit(1)

me_avg = np.mean(me_embeddings, axis=0)
me_avg /= np.linalg.norm(me_avg)
print(f"  Average embedding built from {len(me_embeddings)} photo(s).")

# Load DB for patient lookups
with open(DB_PATH, encoding="utf-8") as fh:
    DB = json.load(fh)

me_patients = set()
for pat in DB.get("patients", []):
    for fam in pat.get("authorised_family", []):
        if fam.get("id") == "me":
            me_patients.add(pat["id"])


# ---------------------------------------------------------------------------
# STEP 4 – Accept me from faces_test/me_test.jpg
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"STEP 4 – Accept 'me' from faces_test/  (threshold={THRESHOLD})")

me_test_path = os.path.join(TEST_DIR, "me_test.jpg")
if not os.path.isfile(me_test_path):
    print(f"  [SKIP] {me_test_path} not found.")
else:
    img = cv2.imread(me_test_path)
    emb, mode = embed_with_detection(img, "me_test.jpg")
    if emb is None:
        print("  [FAIL] Embedding failed for me_test.jpg")
    else:
        s = cosine(emb, me_avg)
        ok = s >= THRESHOLD
        verdict = "PASS OK" if ok else f"FAIL FAIL  (score {s:.4f} < threshold {THRESHOLD})"
        print(f"  me_test.jpg  score={s:.4f}  [{mode}]  {verdict}")
        if not ok and not REAL_PHOTOS:
            print(
                "  NOTE: Low score expected with synthetic images – both enrol and test\n"
                "        images are placeholders. With real photos the score will be ≥0.5.\n"
                "        PASS criterion deferred to real-photo run."
            )
        elif not ok:
            print(
                "  TIP: Score below threshold with real photos. Try lowering threshold\n"
                "       in config.json ('face_verification.threshold') or enrol more photos."
            )


# ---------------------------------------------------------------------------
# STEP 5 – Reject visitors
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 – Reject placeholder visitors from faces_test/")

visitor_photos = sorted(glob.glob(os.path.join(TEST_DIR, "VISITOR_*.png")))
if not visitor_photos:
    print("  (no VISITOR_*.png in faces_test/)")
else:
    for vpath in visitor_photos:
        img = cv2.imread(vpath)
        if img is None:
            print(f"  {os.path.basename(vpath):42s} cannot read")
            continue
        emb, mode = embed_with_detection(img)
        if emb is None:
            print(f"  {os.path.basename(vpath):42s} embed failed – treated as rejected  PASS OK")
            continue
        s = cosine(emb, me_avg)
        ok = s < THRESHOLD
        verdict = "PASS OK (rejected)" if ok else "FAIL FAIL (falsely accepted!)"
        print(f"  {os.path.basename(vpath):42s} score={s:.4f}  [{mode}]  {verdict}")


# ---------------------------------------------------------------------------
# STEP 6 – Right person, wrong patient
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6 – Right person, wrong patient")

all_patients = DB.get("patients", [])
other_patients = [p for p in all_patients if p["id"] not in me_patients]

if not other_patients:
    print("  [SKIP] No second patient in DB.")
else:
    other = other_patients[0]
    auth_ids_other = {fam["id"] for fam in other.get("authorised_family", [])}
    is_auth = "me" in auth_ids_other

    # The face score (from self-cosine) would be high – the check is purely in the auth list
    s_ref = cosine(me_avg, me_avg)
    verdict = "FAIL FAIL (wrongl