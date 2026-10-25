"""Face verification module for MediBot.

Enrolls all face images from ``patients.json`` at startup, computes an
average SFace embedding per person, and provides a ``match(frame)``
function returning ``(person_id, score)``.

The ``face_files`` field in patients.json is a list of relative paths
(relative to the controller directory, e.g. ``"faces/me_1.jpg"``).
Adding more photos of a person is done by appending entries to that list.

Fall-back: if the ONNX models are missing the module prints a clear
warning and ``match()`` always returns ``(None, 0.0)``.
"""
import json
import os
import sys

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()
MODELS_DIR = HERE / "models"
CONFIG_PATH = HERE.parent.parent / "config.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_default_threshold = 0.4
if CONFIG_PATH.is_file():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as _f:
            _cfg = json.load(_f)
        THRESHOLD = float(
            _cfg.get("face_verification", {}).get("threshold", _default_threshold)
        )
    except Exception:
        THRESHOLD = _default_threshold
else:
    THRESHOLD = _default_threshold

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_DETECTOR = None    # cv2.FaceDetectorYN
_RECOGNISER = None  # cv2.FaceRecognizerSF
_ENROLLMENTS: dict[str, np.ndarray] = {}  # person_id -> average embedding


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_models():
    """Load YuNet and SFace ONNX models.  Returns (detector, recogniser)."""
    det_path = str(MODELS_DIR / "yuNet.onnx")
    rec_path = str(MODELS_DIR / "sface.onnx")

    det = None
    if os.path.isfile(det_path):
        if hasattr(cv2, "FaceDetectorYN"):
            try:
                det = cv2.FaceDetectorYN.create(det_path, "", (0, 0), 0.6, 0.3, 5000)
                print(f"[FACE VERIFY] YuNet loaded: {det_path}")
            except Exception as exc:
                print(f"[FACE VERIFY] YuNet create failed: {exc}")
        else:
            print("[FACE VERIFY] cv2.FaceDetectorYN not available in this OpenCV build.")
    else:
        print(f"[FACE VERIFY] YuNet model missing: {det_path}")

    rec = None
    if os.path.isfile(rec_path):
        if hasattr(cv2, "FaceRecognizerSF"):
            try:
                rec = cv2.FaceRecognizerSF.create(rec_path, "")
                print(f"[FACE VERIFY] SFace loaded: {rec_path}")
            except Exception as exc:
                print(f"[FACE VERIFY] SFace create failed: {exc}")
        else:
            print("[FACE VERIFY] cv2.FaceRecognizerSF not available in this OpenCV build.")
    else:
        print(f"[FACE VERIFY] SFace model missing: {rec_path}")

    return det, rec


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def detect_faces_yunet(img: np.ndarray, detector=None):
    """Run YuNet on *img* and return the raw face array (may be empty).

    Returns:
        faces  – numpy array shape (N, 15) or None/empty if no detections.
    """
    d = detector or _DETECTOR
    if d is None or img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    d.setInputSize((w, h))
    _, faces = d.detect(img)
    return faces  # None or (N, 15)


def _best_face_box(faces):
    """Return (x, y, w, h, score) for the highest-confidence detection."""
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda r: float(r[14]))
    x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    score = float(best[14])
    return x, y, w, h, score


def _crop_and_align(img: np.ndarray, face_row: np.ndarray) -> np.ndarray | None:
    """Align and crop face using SFace's alignCrop."""
    if _RECOGNISER is None or face_row is None:
        return None
    try:
        aligned = _RECOGNISER.alignCrop(img, face_row)
        return aligned
    except Exception:
        return None


def _embed(aligned: np.ndarray) -> np.ndarray | None:
    """Return a normalised 128-D SFace feature vector."""
    if _RECOGNISER is None or aligned is None:
        return None
    try:
        feat = _RECOGNISER.feature(aligned)
        feat = np.array(feat, dtype=np.float32).ravel()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------
def enrol_all(patients_json: str | None = None) -> int:
    """Enroll every authorised family member from patients.json.

    Returns the number of people successfully enrolled.
    Prints a YuNet detection result (detected / not detected, confidence)
    for every enrollment photo.  Prints a warning if no face is found in
    a photo.
    """
    global _ENROLLMENTS
    _ENROLLMENTS = {}

    db_path = patients_json or str(HERE / "patients.json")
    if not os.path.isfile(db_path):
        print(f"[ENROLL] patients.json not found: {db_path}")
        return 0

    with open(db_path, encoding="utf-8") as fh:
        db = json.load(fh)

    people = []
    for pat in db.get("patients", []):
        for fam in pat.get("authorised_family", []):
            people.append(fam)
    # Do NOT enrol unauthorised visitors – they should always be rejected.

    total = len(people)
    enrolled = 0

    for person in people:
        pid = person.get("id", "?")
        name = person.get("name", pid)
        face_files = person.get("face_files", [])

        if not face_files:
            # Legacy single-field support
            legacy = person.get("face_image", "")
            if legacy:
                face_files = [legacy]

        embeddings = []
        for rel_path in face_files:
            abs_path = str(HERE / rel_path) if not os.path.isabs(rel_path) else rel_path
            if not os.path.isfile(abs_path):
                print(f"  [WARN] Photo not found: {abs_path} – skipping")
                continue

            img = cv2.imread(abs_path)
            if img is None:
                print(f"  [WARN] Cannot read image: {abs_path}")
                continue

            faces = detect_faces_yunet(img)
            box = _best_face_box(faces)

            if box is None:
                print(
                    f"  [WARN] YuNet: NO FACE DETECTED in '{os.path.basename(abs_path)}' ({abs_path})"
                    f" for {name} ({pid}) – photo REJECTED"
                )
                continue

            x, y, w, h, conf = box
            print(
                f"  [ENROLL] {name} ({pid}) | {os.path.basename(abs_path)}"
                f" | YuNet: detected conf={conf:.3f}  box=({x},{y},{w},{h})"
            )

            face_row = faces[0] if len(faces) == 1 else max(faces, key=lambda r: float(r[14]))
            aligned = _crop_and_align(img, face_row)
            emb = _embed(aligned)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                _ENROLLMENTS[pid] = avg / norm
                enrolled += 1
                print(f"  [ENROLL] {name} ({pid}) – {len(embeddings)} photo(s) enrolled ✓")
        else:
            print(f"  [ENROLL] {name} ({pid}) – NO valid face found in any photo – NOT enrolled")

    print(f"[ENROLL] {enrolled}/{total} people enrolled")
    return enrolled


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def match(frame: np.ndarray) -> tuple[str | None, float]:
    """Match *frame* (BGR ndarray) against enrolled faces.

    Returns:
        (person_id, score)  – person_id is None when no match >= THRESHOLD.
    """
    if not _ENROLLMENTS:
        return None, 0.0

    faces = detect_yunet(frame)
    box = _best_face_box(faces)
    if box is None:
        return None, 0.0

    face_row = faces[0] if len(faces) == 1 else max(faces, key=lambda r: float(r[14]))
    aligned = _crop_and_align(frame, face_row)
    emb = _embed(aligned)
    if emb is None:
        return None, 0.0

    best_id, best_score = None, -1.0
    for pid, stored in _ENROLLMENTS.items():
        s = float(np.dot(emb, stored))
        if s > best_score:
            best_score, best_id = s, pid

    if best_score >= THRESHOLD:
        return best_id, best_score
    return None, best_score


def detect_yunet(frame: np.ndarray):
    """Public helper: run YuNet on *frame* and return raw faces array."""
    return detect_faces_yunet(frame)


def get_enrollments() -> dict:
    """Return a shallow copy of the current enrollment dict."""
    return dict(_ENROLLMENTS)


# ---------------------------------------------------------------------------
# Module initialisation
# ---------------------------------------------------------------------------
print(f"[FACE VERIFY] cv2 {cv2.__version__}  FaceDetectorYN={hasattr(cv2, 'FaceDetectorYN')}  FaceRecognizerSF={hasattr(cv2, 'FaceRecognizerSF')}")
_DETECTOR, _RECOGNISER = _load_models()

if _DETECTOR and _RECOGNISER:
    enrol_all()
else:
    print("[FACE VERIFY] Models unavailable – enrollment skipped. match() will always return (None, 0.0)")
