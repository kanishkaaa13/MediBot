"""Pure Python tests for patient-specific face/RFID authorization."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_DIR = os.path.join(ROOT, "controllers", "medibot_controller")
sys.path.insert(0, CONTROLLER_DIR)

from handover import (decide_handover, face_verify, unlock_authorized_compartment)

with open(os.path.join(CONTROLLER_DIR, "patients.json"), encoding="utf-8") as f:
    database = json.load(f)
patients = database["patients"]
assert len(patients) >= 4
assert {p["ward"] for p in patients} >= {"WARD-3B", "WARD-2A", "ICU"}
known_people = {person["id"]: person
                for patient in patients for person in patient["authorised_family"]}
known_people.update({p["id"]: p for p in database["unauthorised_visitors"]})
known_cards = {person["rfid_uid"]: person["id"] for person in known_people.values()}
patient = next(p for p in patients if p["ward"] == "WARD-3B")
relative = patient["authorised_family"][0]
other_relative = next(person for p in patients if p["id"] != patient["id"]
                      for person in p["authorised_family"])
unknown_face = database["unauthorised_visitors"][0]


def check(label, condition):
    print(f"{'PASS' if condition else 'FAIL'} {label}")
    if not condition:
        raise AssertionError(label)


class LockSpy:
    def __init__(self):
        self.unlocked = []

    def unlock(self, compartment):
        self.unlocked.append(compartment)


face_id, face_score = face_verify([{"image": b"simulated", "person_id": relative["id"]}])
check("face_verify stub returns selected identity at score 1.0",
      (face_id, face_score) == (relative["id"], 1.0))
check("each patient has two authorised family records",
      all(len(p["authorised_family"]) == 2 for p in patients))
required_fields = {"id", "name", "ward", "bed", "medicine_item", "compartment"}
check("patient records include delivery identity and compartment",
      all(required_fields <= p.keys() and 1 <= p["compartment"] <= 3 for p in patients))
check("all family and visitor RFID UIDs are unique",
      len(known_cards) == len(known_people))
face_files = [os.path.join(ROOT, "worlds", "textures", "faces",
                           os.path.basename(person.get("face_files", [person.get("face_image", "")])[0])) for person in known_people.values()]
check("generated face texture exists for each person", all(os.path.isfile(p) for p in face_files))

success = decide_handover(patient, relative["id"], 1.0, relative["rfid_uid"],
                          known_people, known_cards)
check("authorised face and same person's RFID succeeds", success[0] and success[1] == "authorized")
spy = LockSpy()
unlocked = unlock_authorized_compartment(patient, success, spy)
check("success unlocks exactly the patient's assigned compartment",
      unlocked == patient["compartment"] and spy.unlocked == [patient["compartment"]])

cases = [
    ("unknown face", None, 0.0, relative["rfid_uid"], False, "unknown face"),
    ("face not authorised for this patient", other_relative["id"], 1.0,
     other_relative["rfid_uid"], False, "face not authorised for this patient"),
    ("RFID missing", relative["id"], 1.0, None, False, "RFID missing"),
    ("RFID timeout", relative["id"], 1.0, None, True, "RFID timeout"),
    ("RFID belongs to someone else", relative["id"], 1.0,
     other_relative["rfid_uid"], False, "RFID belongs to someone else"),
]
for label, person_id, score, uid, timeout, expected in cases:
    result = decide_handover(patient, person_id, score, uid, known_people,
                             known_cards, rfid_timed_out=timeout)
    check(f"denial reason: {label}", not result[0] and result[1] == expected)
    spy = LockSpy()
    check(f"denial leaves every compartment locked: {label}",
          unlock_authorized_compartment(patient, result, spy) is None and not spy.unlocked)

unknown_known = decide_handover(patient, unknown_face["id"], 1.0,
                                unknown_face["rfid_uid"], known_people, known_cards)
check("known unauthorised visitor is denied for this patient",
      not unknown_known[0] and unknown_known[1] == "face not authorised for this patient")

print("ALL HANDOVER TESTS PASSED")
