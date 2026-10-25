"""Pure Python face/RFID handover decision logic (independent of Webots)."""


def face_verify(frames):
    """FACE RECOGNITION STUB: return the simulated visitor identity, score 1.0.

    Replace this function with a real model later; callers pass image frames with
    ``person_id`` metadata only while the stub is active.
    """
    if not frames:
        return None, 0.0
    person_id = frames[-1].get("person_id")
    return (person_id, 1.0) if person_id else (None, 0.0)


def decide_handover(patient, face_person_id, face_score, rfid_uid,
                    known_people, known_cards, rfid_timed_out=False, threshold=0.4):
    """Return (allowed, reason, authorized_person_id) for one protocol attempt."""
    if not face_person_id or face_score < threshold:
        return False, "unknown face", None

    authorized = {person["id"] for person in patient["authorised_family"]}
    if face_person_id not in authorized:
        if face_person_id not in known_people:
            return False, "unknown face", None
        return False, "face not authorised for this patient", None

    if not rfid_uid:
        return False, "RFID timeout" if rfid_timed_out else "RFID missing", None

    card_owner = known_cards.get(rfid_uid)
    if card_owner != face_person_id:
        return False, "RFID belongs to someone else", None

    return True, "authorized", face_person_id


def compartment_to_unlock(patient, decision):
    """Only successful decisions return the compartment assigned to the patient."""
    allowed, _reason, _person_id = decision
    return patient["compartment"] if allowed else None


def unlock_authorized_compartment(patient, decision, compartment_manager):
    """Unlock exactly the patient's assigned compartment, or do nothing on denial."""
    compartment = compartment_to_unlock(patient, decision)
    if compartment is None:
        return None
    compartment_manager.unlock(compartment)
    return compartment
