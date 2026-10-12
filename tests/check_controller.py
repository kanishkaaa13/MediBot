#!/usr/bin/env python3
"""Audit medibot_controller.py state machine and device init."""
import sys

with open("controllers/medibot_controller/medibot_controller.py") as f:
    src = f.read()

print("=== STATE MACHINE FLOW CHECK ===")
states = ["IDLE", "NAVIGATE", "APPROACH", "VERIFY", "DELIVER", "RETURN"]
states.append("DENIED")
ok = True
for s in states:
    if f's == "{s}"' in src:
        print(f"  OK   state {s!r} handled")
    else:
        print(f"  FAIL state {s!r} MISSING")
        ok = False

print("\n=== KEY DEVICE REFERENCES ===")
devices = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr",
           "camera", "face_cam", "us_front", "us_left", "us_right",
           "gyro", "accel", "imu", "status_led", "lid_1", "lid_2", "lid_3",
           "led_lock_{n}", "LOCK_LED_{n}"]
for dev in devices:
    found = dev in src
    print(f"  {'OK  ' if found else 'FAIL'} {dev!r}")
    if not found:
        ok = False

print("\n=== LED COLOURS ===")
for colour in ["LED_GREEN", "LED_BLUE", "LED_ORANGE", "LED_RED"]:
    found = colour in src
    print(f"  {'OK  ' if found else 'FAIL'} {colour}")
    if not found:
        ok = False

print("\n=== COMPARTMENTS METHODS ===")
for method in ["lock_all", "unlock", "lock", "is_open", "deliver_cargo", "reset_cargo"]:
    defined = f"def {method}" in src
    called  = f"compartments.{method}" in src
    status  = "OK  " if (defined or called) else "FAIL"
    print(f"  {status} {method}()  defined={defined} called={called}")
    if not (defined or called):
        ok = False

print("\n=== KEY FLOW TRANSITIONS ===")
transitions = [
    ("set_state(\"NAVIGATE\")",   "IDLE -> NAVIGATE on S key"),
    ("set_state(\"APPROACH\")",   "NAVIGATE -> APPROACH when goal reached"),
    ("set_state(\"VERIFY\")",     "APPROACH -> VERIFY when close"),
    ("set_state(\"DELIVER\")",    "VERIFY -> DELIVER after authorization"),
    ("set_state(\"RETURN\")",     "DELIVER -> RETURN after 4s"),
    ("end_mission(",              "RETURN -> end mission at HOME"),
    ("unlock_authorized_compartment(", "VERIFY: unlock only the patient's assigned compartment"),
    ("close_due[n] = now + 4.0", "compartment auto-close timer"),
    ("cargo_due[n] = now + 2.0", "cargo disappearance timer"),
    ("compartments.lock_all()",   "start_mission / reset: lock all"),
    ("compartments.reset_cargo()", "start_mission / reset: reset cargo"),
    ("replan_blocked()",          "obstacle: reroute"),
    ("ekf.predict(",              "localize_step: EKF predict"),
    ("ekf.update(",               "camera_step: EKF update from tag"),
]
for pattern, label in transitions:
    found = pattern in src
    print(f"  {'OK  ' if found else 'FAIL'} {label}")
    if not found:
        ok = False

print("\n=== HANDOVER SAFETY ===")
for check, condition in [
    ("no auto-confirm timer", "AUTO_VERIFY_S" not in src),
    ("bypass keys require dev flag", "self.dev_mode and self.state == \"VERIFY\"" in src),
    ("patient-scoped face/RFID decision", "decide_handover(" in src),
    ("unknown outcomes return DENIED", 's == "DENIED"' in src),
]:
    print(f"  {'OK  ' if condition else 'FAIL'} {check}")
    ok = ok and condition

print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
