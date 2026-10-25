'''Face verification module for MediBot.

This module enrolls all face images in `controllers/medibot_controller/faces/` at
startup, computes an average embedding per person using OpenCV's `FaceRecognizerSF`
(and optionally `FaceDetectorYN` for detection), and provides a `match(frame)`
function that returns ``(person_id, score)`` where ``score`` is the cosine
similarity between the supplied face embedding and the stored average.

If the required OpenCV models are missing the module falls back to a stub that
always returns ``(None, 0.0)`` and prints a clear warning.
''' 
import os
import cv2
import numpy as np
import json
from pathlib import Path

# --- Configuration -----------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(HERE, "faces")
MODELS_DIR = os.path.join(HERE, "models")
CONFIG_PATH = os.path.join(HERE, "..", "..", "config.json")

# Load configuration (fallback to defaults if missing)
_default_cfg = {"face_verification": {"threshold": 0.4}}
if os.path.isfile(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _cfg = json.load(f)
            FACE_CFG = _cfg.get("face_verification", _default_cfg["face_verification"])
    except Exception:
        FACE_CFG = _default_cfg["face_verification"]
else:
    FACE_CFG = _default_cfg["face_verification"]
THRESHOLD = float(FACE_CFG.get("threshold", 0.4))

# Global store for enrolments: person_id -> average embedding (np.ndarray)
_ENROLLMENTS = {}

def _load_models():
    detector = None
    recogniser = None
    detector_path = os.path.join(MODELS_DIR, "yuNet.onnx")
    recogniser_path = os.path.join(MODELS_DIR, "sface.onnx")
    if os.path.isfile(detector_path) and hasattr(cv2, "FaceDetectorYN"):
        try:
            detector = cv2.FaceDetectorYN.create(detector_path, "", (0, 0), 0.9, 0.3, 5000)
        except Exception as e:
            print(f"[FACE VERIFY] Failed to create YuNet detector: {e}")
    else:
        print("[FACE VERIFY] YuNet model not found or cv2.FaceDetectorYN unavailable.")
    if os.path.isfile(recogniser_path) and hasattr(cv2, "FaceRecognizerSF"):
        try:
            recogniser = cv2.FaceRecognizerSF.create(recogniser_path, "")
        except Exception as e:
            print(f"[FACE VERIFY] Failed to create SFace recogniser: {e}")
    else:
        print("[FACE VERIFY] SFace model not found or cv2.FaceRecognizerSF unavailable.")
    return detector, recogniser

_DETECTOR, _RECOGNISER = _load_models()

def _detect_face(img):
    """Detect the largest face in *img* using the YuNet detector.
    Returns the cropped face image (BGR) or ``None`` if detection fails.
    """
    if _DETECTOR is None:
        return None
    faces = _DETECTOR.detect(img)
    if faces is None or faces.shape[0] == 0:
        return None
    # faces layout: [x, y, w, h, score, ...]
    best_idx = int(np.argmax(faces[:, 4]))
    x, y, w, h = faces[best_idx, :4].astype(int)
    x = max(x, 0); y = max(y, 0)
    w = max(w, 1); h = max(h, 1)
    return img[y:y+h, x:x+w]

def _embed_face(face_img):
    """Return a normalized 512‑D embedding for *face_img* using SFace.
    ``face_img`` must be a BGR numpy array.
    """
    if _RECOGNISER is None:
        return None
    face_resized = cv2.resize(face_img, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    embedding = _RECOGNISER.feature(face_rgb)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm

def _load_person_images(person_id, face_image_rel):
    base = os.path.splitext(os.path.basename(face_image_rel))[0]
    pattern = f"{base}_*.jpg"
    files = list(Path(FACES_DIR).glob(pattern))
    files += list(Path(FACES_DIR).glob(f"{base}_*.png"))
    return [str(p) for p in files]

def enrol_all():
    """Enroll every person from ``patients.json``.
    Populates ``_ENROLLMENTS`` with averaged embeddings.
    """
    db_path = os.path.join(HERE, "patients.json")
    if not os.path.isfile(db_path):
        print("[FACE VERIFY] patients.json not found – cannot enrol.")
        return
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    people = []
    for p in data.get("patients", []):
        people.extend(p.get("authorised_family", []))
    people.extend(data.get("unauthorised_visitors", []))
    total = len(people)
    enrolled = 0
    for person in people:
        pid = person.get("id")
        rel = person.get("face_image")
        if not pid or not rel:
            continue
        files = _load_person_images(pid, rel)
        embeddings = []
        for path in files:
            img = cv2.imread(path)
            if img is None:
                continue
            face = _detect_face(img) if _DETECTOR else img
            if face is None:
                continue
            emb = _embed_face(face)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            avg = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                _ENROLLMENTS[pid] = avg / norm
                enrolled += 1
                print(f"[ENROLL] {person.get('name','?')} ({pid}) – {len(embeddings)} image(s) enrolled")
    print(f"[ENROLL] {enrolled}/{total} people enrolled")

def match(frame):
    """Match a single *frame* (BGR numpy array) against enrolled faces.
    Returns ``(person_id, score)`` where ``score`` is cosine similarity.
    ``person_id`` is ``None`` if no match exceeds the configured threshold.
    """
    if not _ENROLLMENTS:
        return None, 0.0
    face = _detect_face(frame) if _DETECTOR else frame
    if face is None:
        return None, 0.0
    emb = _embed_face(face)
    if emb is None:
        return None, 0.0
    best_id = None
    best_score = -1.0
    for pid, stored in _ENROLLMENTS.items():
        score = float(np.dot(emb, stored))
        if score > best_score:
            best_score = score
            best_id = pid
    if best_score >= THRESHOLD:
        return best_id, best_score
    return None, best_score

print("[FACE VERIFY] Initialising – loading models and enrolling faces…")
if _DETECTOR and _RECOGNISER:
    enrol_all()
else:
    print("[FACE VERIFY] Required models unavailable – verification will always fail.")
