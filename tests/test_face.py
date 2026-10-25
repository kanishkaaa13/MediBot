"""Evaluate the synthetic demo faces with the project's YuNet and SFace models."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CTRL=ROOT/"controllers"/"medibot_controller"
sys.path.insert(0,str(CTRL))
from handover import decide_handover
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
    if conf < .75: return None,conf
    aligned=rec.alignCrop(img,face); feat=np.asarray(rec.feature(aligned),np.float32).ravel()
    return feat/np.linalg.norm(feat),conf

embeddings={}
print(f"OpenCV {cv2.__version__}; YuNet settings unchanged (.6, .3, 5000); required confidence >=.75")
for filename in FILES:
    emb,conf=embed(filename)
    print(f"{filename}: YuNet confidence={conf:.4f} {'DETECTED' if emb is not None else 'NO FACE'}")
    if emb is not None: embeddings[filename]=emb

if len(embeddings)!=4:
    print("FAIL: not all four synthetic images were detected; no recognition scores available for missing faces.")
    raise SystemExit(1)

f1,f2,v1,v2=(embeddings[name] for name in FILES)
rahul_enrolment=f1+f2
rahul_enrolment=rahul_enrolment/np.linalg.norm(rahul_enrolment)
scores={"family leave-one-out 1->2":float(f1@f2),"family leave-one-out 2->1":float(f2@f1),
        "visitor 01 vs Rahul photo 1":float(v1@f1),"visitor 01 vs Rahul photo 2":float(v1@f2),
        "visitor 01 vs Rahul enrolment":float(v1@rahul_enrolment),
        "visitor 02 vs Rahul photo 1":float(v2@f1),"visitor 02 vs Rahul photo 2":float(v2@f2),
        "visitor 02 vs Rahul enrolment":float(v2@rahul_enrolment),
        "visitor 01 vs visitor 02":float(v1@v2)}
for label,score in scores.items(): print(f"{label}: SFace cosine={score:.4f}")
accept=all(scores[key]>=.5 for key in ("family leave-one-out 1->2","family leave-one-out 2->1"))
reject=all(scores[key]<.30 for key in ("visitor 01 vs Rahul enrolment","visitor 02 vs Rahul enrolment"))
db=json.loads((CTRL/"patients.json").read_text(encoding="utf-8"))
patient=next(p for p in db["patients"] if p["ward"]=="WARD-3B")
rahul=next(p for p in patient["authorised_family"] if p["name"]=="Rahul Mehta")
other_patient=next(p for p in db["patients"] if p["id"]!="PAT_W3B_01")
wrong_patient=rahul["id"] not in {x["id"] for x in other_patient["authorised_family"]}
known_people={person["id"]:person for p in db["patients"] for person in p["authorised_family"]}
known_people.update({person["id"]:person for person in db["unauthorised_visitors"]})
known_cards={person["rfid_uid"]:person["id"] for person in known_people.values()}
wrong_result=decide_handover(other_patient,rahul["id"],scores["family leave-one-out 1->2"],
                             rahul["rfid_uid"],known_people,known_cards)
wrong_patient=wrong_patient and not wrong_result[0]
print(f"wrong-patient case ({other_patient['id']}): face score={scores['family leave-one-out 1->2']:.4f}; {wrong_result[1]}: {'PASS' if wrong_patient else 'FAIL'}")
print(f"visitor-to-visitor score (informational only): {scores['visitor 01 vs visitor 02']:.4f}")
print(f"leave-one-out accept: {'PASS' if accept else 'FAIL'}; reject both visitors vs Rahul enrolment (<0.30): {'PASS' if reject else 'FAIL'}")

# Score the real Webots camera captures produced by the isolated handover-pose
# probe. These are 640x480 BGR frames from the actual face_cam device.
for pid in ("VISITOR_UNKNOWN_01", "VISITOR_UNKNOWN_02"):
    path=CTRL/"logs"/f"face_cam_{pid}.png"
    frame=cv2.imread(str(path))
    if frame is None:
        print(f"in-world {pid}: frame missing ({path})")
        continue
    det.setInputSize((frame.shape[1],frame.shape[0])); _,rows=det.detect(frame)
    if rows is None or len(rows)==0:
        print(f"in-world {pid}: YuNet NO FACE; best match=none score=-")
        continue
    face=max(rows,key=lambda row:float(row[14])); conf=float(face[14])
    aligned=rec.alignCrop(frame,face); feature=np.asarray(rec.feature(aligned),np.float32).ravel()
    feature=feature/np.linalg.norm(feature)
    rahul_score=float(feature@rahul_enrolment)
    x,y,w,h=(int(face[i]) for i in range(4))
    print(f"in-world {pid}: YuNet DETECTED conf={conf:.4f} box=({x},{y},{w},{h}); best match=Rahul Mehta score={rahul_score:.4f}")
if not (accept and reject and wrong_patient): raise SystemExit(1)
