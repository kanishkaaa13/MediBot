"""Evaluate the synthetic demo faces with the project's YuNet and SFace models."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CTRL=ROOT/"controllers"/"medibot_controller"
FACES=CTRL/"faces"
MODELS=CTRL/"models"
FILES=("family_1.jpg","family_2.jpg","VISITOR_UNKNOWN_01.jpg","VISITOR_UNKNOWN_02.jpg")
det=cv2.FaceDetectorYN.create(str(MODELS/"yuNet.onnx"),"",(512,512),.6,.3,5000)
rec=cv2.FaceRecognizerSF.create(str(MODELS/"sface.onnx"),"")

def embed(name):
    img=cv2.imread(str(FACES/name))
    if img is None: return None,0.0
    det.setInputSize((img.shape[1],img.shape[0])); _,faces=det.detect(img)
    if faces is None or len(faces)==0: return None,0.0
    face=max(faces,key=lambda row:float(row[14])); conf=float(face[14])
    aligned=rec.alignCrop(img,face); feat=np.asarray(rec.feature(aligned),np.float32).ravel()
    return feat/np.linalg.norm(feat),conf

embeddings={}
print(f"OpenCV {cv2.__version__}; YuNet threshold=.6; required detection confidence >=.7")
for filename in FILES:
    emb,conf=embed(filename)
    print(f"{filename}: YuNet confidence={conf:.4f} {'DETECTED' if emb is not None else 'NO FACE'}")
    if emb is not None: embeddings[filename]=emb

if len(embeddings)!=4:
    print("FAIL: not all four synthetic images were detected; no recognition scores available for missing faces.")
    raise SystemExit(1)

f1,f2,v1,v2=(embeddings[name] for name in FILES)
scores={"family leave-one-out 1->2":float(f1@f2),"family leave-one-out 2->1":float(f2@f1),
        "visitor 01 vs family":float(v1@f1),"visitor 02 vs family":float(v2@f1),
        "visitor 01 vs visitor 02":float(v1@v2)}
for label,score in scores.items(): print(f"{label}: SFace cosine={score:.4f}")
accept=all(scores[key]>=.5 for key in ("family leave-one-out 1->2","family leave-one-out 2->1"))
reject=all(scores[key]<=.25 for key in ("visitor 01 vs family","visitor 02 vs family","visitor 01 vs visitor 02"))
db=json.loads((CTRL/"patients.json").read_text(encoding="utf-8"))
patient=next(p for p in db["patients"] if p["ward"]=="WARD-3B")
rahul=next(p for p in patient["authorised_family"] if p["name"]=="Rahul Mehta")
other_patient=next(p for p in db["patients"] if p["id"]!="PAT_W3B_01")
wrong_patient=rahul["id"] not in {x["id"] for x in other_patient["authorised_family"]}
print(f"wrong-patient case ({other_patient['id']}): Rahul Mehta is not authorised: {'PASS' if wrong_patient else 'FAIL'}")
print(f"leave-one-out accept: {'PASS' if accept else 'FAIL'}; reject both visitors: {'PASS' if reject else 'FAIL'}")
if not (accept and reject and wrong_patient): raise SystemExit(1)
