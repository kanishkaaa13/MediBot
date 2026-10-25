#!/usr/bin/env python3
"""
MediBot Webots controller  (Phase 2 – 4WD, 3 compartments, face-cam, status LED)

Keys (click the 3D view first so it has keyboard focus):
  S start  B block  U unblock  X reset
  V visitor  C swap card  R tap RFID  T tags
  F/1/2/3 developer overrides require dev_mode.flag.
"""
import csv
import datetime
import json
import math
import os
import sys
import time
import traceback
from collections import deque

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: run  pip install numpy opencv-python  with the SAME python Webots uses "
          "(Webots > Tools > Preferences > Python command).")
    raise

from controller import Supervisor
from medibot_core import (CFG, NODES, EKF, TagLocalizer, astar, edge_key, nearest_node, wrap,
                          rotate_drive_command)
from handover import (face_verify, decide_handover, compartment_to_unlock,
                      unlock_authorized_compartment)

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "patients.json"), encoding="utf-8") as _patient_file:
    PATIENT_DATA = json.load(_patient_file)
PATIENTS = PATIENT_DATA["patients"]
PEOPLE = [relative for patient in PATIENTS for relative in patient["authorised_family"]]
PEOPLE.extend(PATIENT_DATA["unauthorised_visitors"])
PEOPLE_BY_ID = {person["id"]: person for person in PEOPLE}
CARDS_BY_UID = {person["rfid_uid"]: person["id"] for person in PEOPLE}

def _startup_face_check():
    """Load face images for all people and run a simple Haar cascade detection.
    Prints detection results and counts successful enrollments.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    enrolled = 0
    for person in PEOPLE:
        img_path = os.path.join(HERE, person["face_image"])
        if not os.path.isfile(img_path):
            print(f"[WARNING] face image not found: {person['face_image']}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] could not read image: {person['face_image']}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print(f"[ENROLL] {person.get('name','?')} ({person.get('id','?')}): {len(faces)} face(s) detected")
        if len(faces) > 0:
            enrolled += 1
    print(f"[ENROLL] Total faces enrolled: {enrolled}")

_startup_face_check()

# ---------------------------------------------------------------- parameters
TIME_STEP = 32
R_TRUE = CFG["robot"]["wheel_radius"]
TRACK = CFG["robot"]["track"]
CAM_W, CAM_H = CFG["camera"]["width"], CFG["camera"]["height"]

# Deliberate sensor imperfections so odometry drifts (like real hardware)
WHEEL_R_EST = R_TRUE            # keep encoder odometry tied to world geometry
GYRO_BIAS = 0.0                  # no synthetic gyro bias in navigation control

CRUISE_V = 0.30
APPROACH_V = 0.12
WAYPOINT_TOL = 0.15
MAX_W = 1.0                     # rad/s (4WD base stability)
MIN_TURN_W = 0.4
WAYPOINT_RADIUS = 0.12
STOP_FRONT = 0.25               # m   stop if front sonar < 25 cm
STOP_SIDE = 0.15
BLOCK_TIME = 2.0                # s   still blocked after 2 s -> mark edge blocked, replan
BLOCKED_EDGE_TTL = 180.0        # s   keep an obstacle blocked through outbound/return delivery
RFID_TIMEOUT_S = 10.0
HANDOVER_MAX_ATTEMPTS = 3       # one initial attempt and two retries
GATE_M = 1.2                    # deprecated: replaced by Mahalanobis test
MAHALANOBIS_THRESH = 11.34      # chi-square 3 dof, 99% confidence
MIN_ACCEPT_RADIUS = 0.02        # keep EKF tag corrections smooth, even for small drift
REJECT_BUFFER_TIME = 2.0         # seconds to keep rejected fixes
REJECT_AGREE_POS = 0.25         # position agreement threshold (m)
REJECT_AGREE_TH = 0.2           # heading agreement threshold (rad)
REJECT_MIN_COUNT = 3            # minimum rejected fixes to trigger re-localisation

# LED colours (24-bit RGB packed as int, used with LED.set())
# Webots LED.set accepts a scalar intensity byte (0..255). The LED's
# generated PBRAppearance supplies its hue; packed RGB integers become invalid
# SFColor channels when Webots writes the world back to disk.
LED_GREEN  = 220
LED_BLUE   = 220
LED_ORANGE = 220
LED_RED    = 220
LED_OFF    = 0
LED_COLOURS = {
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.25, 1.0],
    "orange": [1.0, 0.35, 0.0],
    "red": [1.0, 0.0, 0.0],
}


def safe_set(node, field_name, value):
    """Validate Webots SF field values before calling the matching setter."""
    vector_sizes = {"translation": (3, "setSFVec3f"),
                    "rotation": (4, "setSFRotation"),
                    "color": (3, "setSFColor")}
    if field_name in vector_sizes:
        size, setter_name = vector_sizes[field_name]
        valid = (type(value) is list and len(value) == size and
                 all(type(component) is float for component in value))
    elif type(value) is bool:
        setter_name, valid = "setSFBool", True
    elif type(value) is int:
        setter_name, valid = "setSFInt32", True
    elif type(value) is float:
        setter_name, valid = "setSFFloat", True
    else:
        setter_name, valid = "", False
    if node is None or not valid:
        print(f"[SETFIELD] node.{field_name} bad value {value!r}")
        return False
    try:
        getattr(node.getField(field_name), setter_name)(value)
    except (AttributeError, TypeError, ValueError) as exc:
        print(f"[SETFIELD] node.{field_name} bad value {value!r}: {exc}")
        return False
    return True


# ================================================================ Compartments
class Compartments:
    """Manages the 3 lockable medicine compartments (hinged lids + lock LEDs + cargo)."""

    LID_OPEN  = -1.4    # rad – lid swings open
    LID_CLOSE = 0.0     # rad – lid stays shut

    def __init__(self, robot):
        self.robot = robot
        self.motors = {}
        self.sensors = {}
        self.leds = {}
        self.led_nodes = {}
        self.cargos = {}
        self.cargo_due = {}
        self.close_due = {}
        self.hidden = set()

        for n in (1, 2, 3):
            # Hinged lid motor
            m = robot.getDevice(f"lid_{n}")
            if m:
                m.setVelocity(2.0)
                m.setPosition(self.LID_CLOSE)
            self.motors[n] = m

            # Lid position sensor
            ps = robot.getDevice(f"lid_{n}_ps")
            if ps:
                ps.enable(TIME_STEP)
            self.sensors[n] = ps

            # Lock indicator LED (red=locked by default)
            led = robot.getDevice(f"led_lock_{n}")
            if led:
                led.set(LED_RED)
            self.leds[n] = led
            self.led_nodes[n] = robot.getFromDef(f"LOCK_LED_{n}")
            self._set_indicator(n, "red")

            # Cargo solid (supervisor node for deliver-animation)
            self.cargos[n] = robot.getFromDef(f"CARGO_{n}")

    def lock_all(self):
        for n in (1, 2, 3):
            self.lock(n)

    def _set_indicator(self, n, colour):
        led = self.leds.get(n)
        if led:
            led.set(255 if colour != "off" else 0)
        node = self.led_nodes.get(n)
        if node:
            try:
                safe_set(node, "color", list(LED_COLOURS[colour]))
            except (KeyError, AttributeError, TypeError):
                pass

    def unlock(self, n):
        """Open compartment n and light its LED green."""
        if self.motors.get(n):
            self.motors[n].setPosition(self.LID_OPEN)
        self._set_indicator(n, "green")
        now = self.robot.getTime()
        self.cargo_due[n] = now + 2.0
        self.close_due[n] = now + 4.0
        print(f"[LOCK] compartment {n} UNLOCKED")

    def lock(self, n):
        """Close compartment n and light its LED red."""
        if self.motors.get(n):
            self.motors[n].setPosition(self.LID_CLOSE)
        self._set_indicator(n, "red")
        self.close_due.pop(n, None)
        print(f"[LOCK] compartment {n} LOCKED")

    def update(self):
        now = self.robot.getTime()
        for n in (1, 2, 3):
            if now >= self.cargo_due.get(n, float("inf")):
                self.cargo_due.pop(n, None)
                self.deliver_cargo(n)
            if now >= self.close_due.get(n, float("inf")):
                self.lock(n)

    def is_open(self, n):
        ps = self.sensors.get(n)
        if ps is None:
            return False
        return ps.getValue() < -0.5   # clearly below halfway

    def deliver_cargo(self, n):
        """Animate cargo box sliding out of the compartment (teleport below floor)."""
        node = self.cargos.get(n)
        if node and n not in self.hidden:
            f = node.getField("translation")
            if f:
                safe_set(node, "translation", [0.0, 0.0, -5.0])
                self.hidden.add(n)
        print(f"[CARGO] package {n} delivered")

    def reset_cargo(self):
        """Return all cargo boxes to their initial positions (inside compartments)."""
        init_positions = {
            1: (0.04, 0.0, 0.09),
            2: (-0.06, 0.0, 0.09),
            3: (-0.16, 0.0, 0.09),
        }
        for n, pos in init_positions.items():
            self.hidden.discard(n)
            self.cargo_due.pop(n, None)
            self.close_due.pop(n, None)
            node = self.cargos.get(n)
            if node:
                f = node.getField("translation")
                if f:
                    safe_set(node, "translation", [float(v) for v in pos])
        print("[CARGO] all packages reset")


class HandoverActors:
    """Supervisor-driven standing avatars and separately emulated RFID cards."""

    PARK_Y = -20.0

    def __init__(self, supervisor, people):
        self.supervisor = supervisor
        self.people = people
        self.visitors = {p["id"]: supervisor.getFromDef(f"VISITOR_{p['id']}") for p in people}
        self.cards = {p["id"]: supervisor.getFromDef(f"CARD_{p['id']}") for p in people}
        self.park_all()

    @staticmethod
    def _set_pose(node, position, yaw=None):
        if node is None:
            return
        safe_set(node, "translation", [float(v) for v in position])
        if yaw is not None:
            safe_set(node, "rotation", [0.0, 0.0, 1.0, float(yaw)])
        try:
            node.resetPhysics()
        except AttributeError:
            pass

    def park_all(self):
        for i, person in enumerate(self.people):
            x = -30.0 - i * 3.0
            self._set_pose(self.visitors[person["id"]], (x, self.PARK_Y, 0.0), 0.0)
            self._set_pose(self.cards[person["id"]], (x, self.PARK_Y, 0.7))

    def present(self, visitor_id, card_owner_id, visitor_pose, robot_pose):
        self.park_all()
        vx, vy = visitor_pose[:2]
        yaw = math.atan2(robot_pose[1] - vy, robot_pose[0] - vx)
        self._set_pose(self.visitors[visitor_id], (vx, vy, 0.0), yaw)
        hand = (vx + 0.22 * math.cos(yaw), vy + 0.22 * math.sin(yaw), 0.76)
        self._set_pose(self.cards[card_owner_id], hand)
        return hand

    def move_card(self, owner_id, position):
        self._set_pose(self.cards[owner_id], position)

    def card_position(self, owner_id):
        node = self.cards.get(owner_id)
        return tuple(node.getPosition()) if node is not None else None


# ================================================================ MediBot
class MediBot:
    def __init__(self):
        r = self.robot = Supervisor()
        self.dt = TIME_STEP / 1000.0

        # 4 wheel motors
        self.motors = {k: r.getDevice(f"wheel_{k}") for k in ("fl", "fr", "rl", "rr")}
        for m in self.motors.values():
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # Wheel position sensors
        self.ps = {k: r.getDevice(f"ps_{k}") for k in ("fl", "fr", "rl", "rr")}
        for p in self.ps.values():
            p.enable(TIME_STEP)

        # Navigation camera (at AprilTag level)
        self.cam = r.getDevice("camera")
        self.cam.enable(2 * TIME_STEP)
        self.dashboard = r.getDevice("dashboard")

        # Face camera (on mast head)
        self.face_cam = r.getDevice("face_cam")
        if self.face_cam:
            self.face_cam.enable(4 * TIME_STEP)

        # Ultrasonic sensors
        self.us = {n: r.getDevice(f"us_{n}") for n in ("front", "left", "right")}
        for s in self.us.values():
            s.enable(TIME_STEP)

        # IMU stack
        self.gyro = r.getDevice("gyro")
        self.gyro.enable(TIME_STEP)
        self.accel = r.getDevice("accel")
        self.accel.enable(TIME_STEP)
        self.imu = r.getDevice("imu")
        self.imu.enable(TIME_STEP)

        # Status LED strip (front of robot)
        self.status_led = r.getDevice("status_led")
        self.status_led_node = r.getFromDef("STATUS_LED")
        if self.status_led:
            self.status_led.set(255)

        # 3-compartment manager
        self.compartments = Compartments(r)

        # Backwards-compat alias: self.lid -> compartment 1 motor
        self.lid = self.compartments.motors.get(1)
        self.lid_motor = self.lid  # legacy compartment-1 motor alias

        self.kb = r.getKeyboard()
        self.kb.enable(TIME_STEP)
        self.me = r.getSelf()
        self.actors = HandoverActors(r, PEOPLE)
        self.trolley = r.getFromDef("TROLLEY")
        self.obstacle_nodes = [
            ("trolley", self.trolley),
            ("wheelchair", r.getFromDef("OBSTACLE_WHEELCHAIR")),
            ("equipment cart", r.getFromDef("OBSTACLE_CART")),
        ]
        self.obstacle_cycle = 0
        self.root_children = r.getRoot().getField("children")

        r.step(TIME_STEP)       # let sensors produce first valid values
        self.prev = {k: self.ps[k].getValue() for k in self.ps}
        self.loc = TagLocalizer()
        sx, sy, sth = CFG["robot"]["start"]
        self.ekf = EKF([sx, sy, sth])

        self.state = "IDLE"
        self.tags_enabled = True
        self.route, self.idx = [], 0
        self.blocked_edges = set()
        self.blocked_nodes = set()
        self.blocked_edge_expiry = {}
        self.blocked_t = 0.0
        self.obst_printed = False
        self.turning = False
        self.dev_mode = os.path.exists(os.path.join(HERE, "dev_mode.flag"))
        self.dev_override_steps = set()
        self.visitor_choice_index = 0
        self.card_swapped = False
        self._key_names = {"S", "B", "U", "X", "V", "C", "R", "T", "F", "1", "2", "3"}
        self._key_armed = {key: True for key in self._key_names}
        self._key_last_seen = {key: -float("inf") for key in self._key_names}
        self.active_visitor_id = None
        self.active_card_owner_id = None
        self.active_patient = None
        self.handover_attempt = 0
        self.handover_started = False
        self.handover_result = "-"
        self.handover_step = "-"
        self.handover_face_id = None
        self.handover_face_score = 0.0
        self.handover_uid = None
        self.handover_reason = ""
        self.rfid_started_t = 0.0
        self.rfid_tap_requested = False
        self.rfid_tap_started_t = 0.0
        self.face_started_t = 0.0
        self.rfid_card_start = None
        self.rfid_card_target = None
        self.state_t0 = 0.0
        self.tag_seen = False
        self.last_tag_msg = "-"
        self.n_corrections = 0
        self.errs = []
        self.log_f = self.log_w = None
        self.tag_log_f = self.tag_log_w = None
        self.tag_filter = deque(maxlen=3)
        self.home_tag_pose = None
        self.home_tag_seen_t = -float("inf")
        self.tag_last_detection_t = -float("inf")
        self.dock_position_locked = False
        self.dock_stage = "idle"
        self.dock_stage_t0 = 0.0
        self.dock_stage_times = {}
        self.dock_enter_t = 0.0
        self.dock_debug_t = -float("inf")
        self.route_nodes = []
        self.detour_nodes = []
        self.dest = "-"
        self.mission_destination = "WARD-3B-IN"
        self.mission_room = "WARD-3B"
        self.delivery_compartment = 1
        self.mission_no = 0
        self.step_no = 0
        self.last_ekf_print_t = -1.0
        self.last_tag_print_t = -1.0
        self.calibrating = False
        self.calib_start_pose = None
        self.rejected_fixes = []  # Buffer for rejected tag fixes
        self.last_valid_gyro = 0.0
        self.last_valid_ps = {k: 0.0 for k in ("fl", "fr", "rl", "rr")}
        self.nan_warning_printed = set()  # Track which sensors have warned about NaN
        self.last_tag_id = None
        self.last_tag_zone = None
        self.obstacle_stops = 0
        self.replans = 0
        self.regression_route_via_icu = False
        self.regression_trolley_placed = False
        self.mission_delivered = False
        self.last_mission_result = None
        self.last_event = "Controller ready"
        self.dashboard_last_t = -float("inf")
        self.mission_start_t = None
        self.mission_routes = []
        self.delivery_logged = False

        # Regression mode
        self.regression_mode = (os.path.exists(os.path.join(HERE, "regression.flag")) and
                                os.environ.get("MEDIBOT_REGRESSION") == "1")
        self.regression_scenario = None  # "A", "B", or "D"
        self.regression_start_t = 0.0
        self.regression_results = []

        # Startup self-check
        found_devices = []
        for dev_name, dev in [
            ("wheel_fl", self.motors.get("fl")), ("wheel_fr", self.motors.get("fr")),
            ("wheel_rl", self.motors.get("rl")), ("wheel_rr", self.motors.get("rr")),
            ("ps_fl",    self.ps.get("fl")),     ("ps_fr",    self.ps.get("fr")),
            ("ps_rl",    self.ps.get("rl")),     ("ps_rr",    self.ps.get("rr")),
            ("camera",   self.cam),
            ("face_cam", self.face_cam),
            ("us_front", self.us.get("front")),  ("us_left",  self.us.get("left")),
            ("us_right", self.us.get("right")),
            ("gyro",     self.gyro), ("accel", self.accel), ("imu", self.imu),
            ("status_led", self.status_led),
            ("lid_1",    self.compartments.motors.get(1)),
            ("lid_2",    self.compartments.motors.get(2)),
            ("lid_3",    self.compartments.motors.get(3)),
        ]:
            if dev is not None:
                found_devices.append(dev_name)
        print(f"[CHECK] OpenCV {cv2.__version__} | "
              f"Devices found ({len(found_devices)}): {', '.join(found_devices)}")
        print(__doc__)
        if self.regression_mode:
            print("[REGRESSION] enabled")
            print("[REGRESSION] Automatic runs selected by MEDIBOT_REGRESSION_SCENARIOS")
            self.run_regression()
        else:
            print("[MediBot] ready - state IDLE. Press S to start a delivery to WARD-3B.")
        if self.dev_mode:
            print("[DEV MODE] bypass keys enabled; all bypass outcomes are logged DEV_OVERRIDE")

        # Startup sonar check (in IDLE/open space)
        d = self.sonar()
        for k, v in d.items():
            if v < 1.0:
                print(f"[WARNING] us_{k} reads {v*100:.0f} cm at startup (should be > 1.0 m in open space)")
                print("  This may indicate the sensor ray is hitting the robot's own body.")

    # ------------------------------------------------------------ helpers
    def now(self):
        return self.robot.getTime()

    def set_state(self, s):
        print(f"[STATE] {self.state} -> {s}")
        self.last_event = f"State: {self.state} -> {s}"
        self.state = s
        self.state_t0 = self.now()
        self._update_status_led(s)

    def _update_status_led(self, state):
        """Set a state colour for the front status LED strip."""
        if self.status_led is None:
            return
        colour_name = {
            "IDLE": "green", "DELIVER": "green",
            "NAVIGATE": "blue", "APPROACH": "blue", "RETURN": "blue", "DOCK": "blue",
            "VERIFY": "orange",
        }.get(state, "red")
        self.status_led.set(255)
        if self.status_led_node:
            try:
                safe_set(self.status_led_node, "color", list(LED_COLOURS[colour_name]))
            except (AttributeError, TypeError):
                pass

    def true_pose(self):
        p = self.me.getPosition()
        o = self.me.getOrientation()
        # Check for NaN in supervisor pose
        if not all(math.isfinite(v) for v in p + o):
            if "supervisor" not in self.nan_warning_printed:
                print("[WARNING] supervisor pose returned NaN, using last valid estimate")
                self.nan_warning_printed.add("supervisor")
            return self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        return p[0], p[1], math.atan2(o[3], o[0])

    def true_err(self):
        tx, ty, _ = self.true_pose()
        return math.hypot(self.ekf.x[0] - tx, self.ekf.x[1] - ty)

    def drive(self, v, w):
        # Motor conversion uses the physical world radius; odometry keeps its
        # deliberate calibration offset to represent sensor drift.
        wl = (v - w * TRACK / 2) / R_TRUE
        wr = (v + w * TRACK / 2) / R_TRUE
        wl = max(-15.0, min(15.0, wl))
        wr = max(-15.0, min(15.0, wr))
        self.motors["fl"].setVelocity(wl)
        self.motors["rl"].setVelocity(wl)
        self.motors["fr"].setVelocity(wr)
        self.motors["rr"].setVelocity(wr)

    def stop(self):
        self.drive(0.0, 0.0)

    # ------------------------------------------------------------ sensing
    def sonar(self):
        d = {}
        for k, s in self.us.items():
            v = s.getValue()
            if not math.isfinite(v):
                if f"us_{k}" not in self.nan_warning_printed:
                    print(f"[WARNING] us_{k} returned NaN, treating as no obstacle")
                    self.nan_warning_printed.add(f"us_{k}")
                d[k] = 4.0  # Max range
            else:
                d[k] = v
        return d

    def obstacle(self):
        d = self.sonar()
        # Side sonars report corridor walls and bed rails during normal turns;
        # only a forward return is on the current swept path and should block it.
        return d["front"] < STOP_FRONT

    def localize_step(self):
        cur = {k: self.ps[k].getValue() for k in self.ps}
        # Check for NaN in position sensors
        for k, v in cur.items():
            if not math.isfinite(v):
                if f"ps_{k}" not in self.nan_warning_printed:
                    print(f"[WARNING] ps_{k} returned NaN, using last valid value")
                    self.nan_warning_printed.add(f"ps_{k}")
                cur[k] = self.last_valid_ps[k]
            else:
                self.last_valid_ps[k] = v

        dl = ((cur["fl"] - self.prev["fl"]) + (cur["rl"] - self.prev["rl"])) / 2
        dr = ((cur["fr"] - self.prev["fr"]) + (cur["rr"] - self.prev["rr"])) / 2
        self.prev = cur
        ds = WHEEL_R_EST * (dl + dr) / 2

        # Check for NaN in gyro
        gz_raw = self.gyro.getValues()[2]
        if not math.isfinite(gz_raw):
            if "gyro" not in self.nan_warning_printed:
                print("[WARNING] gyro returned NaN, using last valid value")
                self.nan_warning_printed.add("gyro")
            gz = self.last_valid_gyro
        else:
            self.last_valid_gyro = gz_raw
            gz = gz_raw

        self.ekf.predict(ds, gz * self.dt)

        # Assert EKF state is finite after predict
        if not np.all(np.isfinite(self.ekf.x)):
            print("[ERROR] EKF state became NaN after predict! Re-initialising from last finite state.")
            sx, sy, sth = CFG["robot"]["start"]
            self.ekf = EKF([sx, sy, sth])

    def camera_step(self):
        img = self.cam.getImage()
        if img is None:
            return
        gray = cv2.cvtColor(
            np.frombuffer(img, np.uint8).reshape((CAM_H, CAM_W, 4)),
            cv2.COLOR_BGRA2GRAY)
        meas = self.loc.measure(gray, hint=tuple(self.ekf.x))
        self.tag_seen = bool(meas)
        if not meas:
            for diag in self.loc.last_diagnostics:
                self._write_tag_diagnostic(diag, False, diag.get("reason", "rejected"))
            return
        m = min(meas, key=lambda d: d["dist"])
        now = self.now()
        if now - self.tag_last_detection_t > 0.5:
            self.tag_filter.clear()
        self.tag_last_detection_t = now
        home_tag_id = next((t["id"] for t in CFG["tags"]
                            if t.get("zone") == "home_south"), None)
        home_m = next((obs for obs in meas if obs["id"] == home_tag_id), None)
        if home_m is not None:
            self.home_tag_pose = dict(home_m)
            self.home_tag_seen_t = now
        if self.state != "IDLE" and now - self.last_tag_print_t >= 1.0:
            self.last_tag_print_t = now
            print(f"[TAG id={m['id']} range={m['dist']:.2f} "
                  f"bearing={math.degrees(m['bearing']):.1f}deg]")
        if not self.tags_enabled:
            self.last_tag_msg = f"tag {m['id']} seen (correction OFF)"
            for diag in self.loc.last_diagnostics:
                self._write_tag_diagnostic(diag, False, "tag correction disabled")
            return

        self.tag_filter.append(dict(m))
        if len(self.tag_filter) < self.tag_filter.maxlen:
            for diag in self.loc.last_diagnostics:
                reason = "median filter warming up" if diag["id"] == m["id"] else "not selected"
                self._write_tag_diagnostic(diag, False, reason)
            return
        history = list(self.tag_filter)
        m_filtered = dict(m)
        m_filtered["x"] = float(np.median([obs["x"] for obs in history]))
        m_filtered["y"] = float(np.median([obs["y"] for obs in history]))
        _, _, current_heading = self.ekf.x
        m_filtered["yaw"] = wrap(current_heading + float(np.median(
            [wrap(obs["yaw"] - current_heading) for obs in history])))

        # Mahalanobis test for tag acceptance
        x, y, th = self.ekf.x
        innovation = np.array([m_filtered["x"] - x, m_filtered["y"] - y,
                               wrap(m_filtered["yaw"] - th)])
        sp, st = self.loc.noise(m["dist"])
        R = np.diag([sp ** 2, sp ** 2, st ** 2])
        S = self.ekf.P + R
        try:
            mahalanobis = innovation.T @ np.linalg.inv(S) @ innovation
        except np.linalg.LinAlgError:
            mahalanobis = float('inf')

        # Minimum accept radius check
        dist_error = math.hypot(m_filtered["x"] - x, m_filtered["y"] - y)
        accepted, reject_reason = False, ""
        correction_m, correction_deg = 0.0, 0.0

        if mahalanobis < MAHALANOBIS_THRESH and dist_error > MIN_ACCEPT_RADIUS:
            # Apply a bounded EKF correction so one frame cannot move the pose
            # estimate more than 5 cm or 3 degrees.
            before = self.true_err() * 100
            old_pose = self.ekf.x.copy()
            self.ekf.update([m_filtered["x"], m_filtered["y"], m_filtered["yaw"]], sp, st)
            delta = self.ekf.x - old_pose
            position_delta = math.hypot(float(delta[0]), float(delta[1]))
            if position_delta > 0.05:
                delta[:2] *= 0.05 / position_delta
            delta[2] = max(-math.radians(3.0), min(math.radians(3.0), wrap(float(delta[2]))))
            self.ekf.x = old_pose + delta
            self.ekf.x[2] = wrap(float(self.ekf.x[2]))
            correction_m = math.hypot(float(delta[0]), float(delta[1]))
            correction_deg = math.degrees(float(delta[2]))

            # Assert EKF state is finite after update
            if not np.all(np.isfinite(self.ekf.x)):
                print("[ERROR] EKF state became NaN after update! Re-initialising from last finite state.")
                sx, sy, sth = CFG["robot"]["start"]
                self.ekf = EKF([sx, sy, sth])
                return

            after = self.true_err() * 100
            self.n_corrections += 1
            accepted = True
            # Get tag zone for display
            tag_info = next((t for t in CFG["tags"] if t["id"] == m["id"]), None)
            zone = tag_info.get("zone", "unknown") if tag_info else "unknown"
            self.last_tag_msg = (f"Tag {m['id']} ({zone}) d={m['dist']:.1f}m  err {before:.1f} -> {after:.1f} cm")
            t = self.now()
            if self.state != "IDLE" and (t - self.last_ekf_print_t >= 1.0):
                self.last_ekf_print_t = t
                print(f"[EKF] AprilTag {m['id']} @ {m['dist']:.1f} m | "
                      f"error before: {before:.1f} cm | after: {after:.1f} cm | "
                      f"variance(P): {self.ekf.pos_variance:.4f}")
        else:
            reject_reason = "mahalanobis outlier" if mahalanobis >= MAHALANOBIS_THRESH else "below correction deadband"
            # Reject the tag fix
            self.last_tag_msg = f"tag {m['id']} rejected (outlier)"
            # Add to rejected buffer
            now = self.now()
            self.rejected_fixes.append({
                "t": now,
                "x": m["x"],
                "y": m["y"],
                "yaw": m["yaw"],
                "id": m["id"], "dist": m["dist"], "bearing": m["bearing"],
                "incidence": m.get("incidence", float("nan")),
                "pixel_size": m.get("pixel_size", float("nan")),
            })
            # Clean old rejected fixes
            self.rejected_fixes = [f for f in self.rejected_fixes if now - f["t"] < REJECT_BUFFER_TIME]

        # Rejected fixes are diagnostic evidence only.  A cluster of rejected
        # measurements can share the same camera/model bias, so it is unsafe to
        # turn them into an EKF correction.
        relocal_delta = (0.0, 0.0)
        for diag in self.loc.last_diagnostics:
            if diag["id"] == m["id"]:
                did_relocalize = math.hypot(*relocal_delta) > 0.0
                self._write_tag_diagnostic(diag, accepted or did_relocalize,
                                           "rejected fix" if did_relocalize else
                                           ("accepted" if accepted else reject_reason),
                                           relocal_delta[0] if did_relocalize else correction_m,
                                           relocal_delta[1] if did_relocalize else correction_deg)
            else:
                self._write_tag_diagnostic(diag, False, "not selected")

    def _write_tag_diagnostic(self, diag, accepted, reason, correction_m=0.0, correction_deg=0.0):
        if self.tag_log_w is None:
            return
        self.tag_log_w.writerow([
            f"{self.now():.3f}", self.step_no, self.regression_scenario or "manual",
            diag.get("id", ""), diag.get("dist", float("nan")),
            math.degrees(diag.get("bearing", float("nan"))),
            diag.get("incidence", float("nan")), diag.get("pixel_size", float("nan")),
            f"{correction_m:.4f}", f"{correction_deg:.3f}", int(accepted), reason])
        self.tag_log_f.flush()

    def _maybe_relocalize(self):
        """Use a short sequence of consistent outliers to recover a lost pose."""
        if len(self.rejected_fixes) < REJECT_MIN_COUNT:
            return 0.0, 0.0
        recent = self.rejected_fixes[-REJECT_MIN_COUNT:]
        avg_x = sum(f["x"] for f in recent) / len(recent)
        avg_y = sum(f["y"] for f in recent) / len(recent)
        avg_yaw = wrap(sum(wrap(f["yaw"]) for f in recent) / len(recent))
        pos_agree = all(math.hypot(f["x"] - avg_x, f["y"] - avg_y) < REJECT_AGREE_POS
                        for f in recent)
        th_agree = all(abs(wrap(f["yaw"] - avg_yaw)) < REJECT_AGREE_TH for f in recent)
        tag_ids = set(f["id"] for f in recent)
        time_span = recent[-1]["t"] - recent[0]["t"]
        diverse_ids = len(tag_ids) >= 2 or (len(tag_ids) == 1 and time_span >= 0.5)
        if pos_agree and th_agree and diverse_ids:
            old_x, old_y, _old_th = self.ekf.x
            dx, dy, dth = avg_x - old_x, avg_y - old_y, wrap(avg_yaw - self.ekf.x[2])
            distance = math.hypot(dx, dy)
            if distance > 0.05:
                dx *= 0.05 / distance
                dy *= 0.05 / distance
            dth = max(-math.radians(3.0), min(math.radians(3.0), dth))
            self.ekf.x += np.array([dx, dy, dth])
            self.ekf.x[2] = wrap(float(self.ekf.x[2]))
            self.ekf.P = np.diag([0.04, 0.04, 0.04])
            tag_str = ",".join(str(tid) for tid in sorted(tag_ids))
            print(f"[RELOCALISE] bounded step {math.hypot(dx, dy):.3f} m / "
                  f"{math.degrees(dth):.1f} deg toward ({avg_x:.2f},{avg_y:.2f}) using tags {tag_str}")
            self.rejected_fixes = []
            return math.hypot(dx, dy), math.degrees(dth)
        return 0.0, 0.0

    # ------------------------------------------------------------ planning / drawing
    def plan(self, start, goal):
        expired = [edge for edge, expiry in self.blocked_edge_expiry.items()
                   if self.now() >= expiry]
        for edge in expired:
            self.blocked_edge_expiry.pop(edge, None)
            self.blocked_edges.discard(edge)
            print(f"[OBSTACLE] temporary block expired: {' - '.join(sorted(edge))}")
        path = astar(start, goal, self.blocked_edges)
        if path is None:
            print(f"[A*] no route {start} -> {goal}")
            return False
        self.route, self.idx, self.dest = path, 0, goal
        if self.mission_start_t is not None:
            self.mission_routes.append(list(path))
        print(f"[A*] route: {' -> '.join(path)}")
        self.draw_route()
        return True

    def draw_route(self):
        for n in self.route_nodes:
            try:
                n.remove()
            except Exception:
                pass
        self.route_nodes = []
        try:
            for segment_i, (a, b) in enumerate(zip(self.route[:-1], self.route[1:])):
                (x1, y1), (x2, y2) = NODES[a], NODES[b]
                L = math.hypot(x2 - x1, y2 - y1)
                # Webots R2025a: use Pose (Transform is deprecated)
                s = (f"DEF ROUTE_{segment_i} Pose {{ translation {(x1 + x2) / 2:g} {(y1 + y2) / 2:g} 0.008 "
                     f"rotation 0 0 1 {math.atan2(y2 - y1, x2 - x1):g} children [ Shape {{ "
                     f"appearance Appearance {{ material Material {{ diffuseColor 0 0.4 1 "
                     f"emissiveColor 0 0.4 1 }} }} "
                     f"geometry Box {{ size {L:g} 0.06 0.004 }} }} ] }}")
                cnt_before = self.root_children.getCount()
                self.root_children.importMFNodeFromString(-1, s)
                if self.root_children.getCount() > cnt_before:
                    self.route_nodes.append(
                        self.root_children.getMFNode(self.root_children.getCount() - 1))
        except Exception as e:
            print("[draw] could not draw route:", e)

    def draw_detour(self, path):
        """Draw the newly planned alternate route as a dashed floor overlay."""
        for node in self.detour_nodes:
            try:
                node.remove()
            except Exception:
                pass
        self.detour_nodes = []
        dash_i = 0
        try:
            for a, b in zip(path[:-1], path[1:]):
                x1, y1 = NODES[a]
                x2, y2 = NODES[b]
                length = math.hypot(x2 - x1, y2 - y1)
                angle = math.atan2(y2 - y1, x2 - x1)
                for along in np.arange(0.0, length, 0.28):
                    dash_len = min(0.14, length - along)
                    cx = x1 + (along + dash_len / 2) * math.cos(angle)
                    cy = y1 + (along + dash_len / 2) * math.sin(angle)
                    snippet = (f"DEF DETOUR_{dash_i} Pose {{ translation {cx:g} {cy:g} 0.012 "
                               f"rotation 0 0 1 {angle:g} children [ Shape {{ appearance "
                               f"Appearance {{ material Material {{ diffuseColor 0.35 0.55 1 "
                               f"transparency 0.25 }} }} geometry Box {{ size {dash_len:g} 0.045 0.003 }} }} ] }}")
                    before = self.root_children.getCount()
                    self.root_children.importMFNodeFromString(-1, snippet)
                    if self.root_children.getCount() > before:
                        self.detour_nodes.append(self.root_children.getMFNode(self.root_children.getCount() - 1))
                    dash_i += 1
        except Exception as e:
            print("[draw] could not draw detour:", e)

    def clear_route_visuals(self):
        """Remove all dynamic route overlays and drop their Webots node handles."""
        removed = 0
        for node in self.route_nodes + self.detour_nodes:
            try:
                node.remove()
                removed += 1
            except Exception as exc:
                print(f"[draw] route node already unavailable: {exc}")
        self.route_nodes.clear()
        self.detour_nodes.clear()
        if removed:
            print(f"[draw] removed {removed} route overlay nodes")

    def _point_ahead_on_route(self, distance):
        """Return (x, y, heading, edge index) at distance along the active path."""
        route = self.route
        if len(route) < 2:
            return None
        x, y, _ = self.ekf.x
        edge_i = min(max(self.idx - 1, 0), len(route) - 2)
        ax, ay = NODES[route[edge_i]]
        bx, by = NODES[route[edge_i + 1]]
        vx, vy = bx - ax, by - ay
        edge_len = math.hypot(vx, vy)
        if edge_len <= 1e-9:
            return None
        along = max(0.0, min(edge_len, ((x - ax) * vx + (y - ay) * vy) / edge_len))
        lateral = abs((x - ax) * vy - (y - ay) * vx) / edge_len
        if lateral > 1.0:  # Robot pose must still be in the corridor for this edge.
            return None

        remaining = edge_len - along
        while edge_i < len(route) - 1:
            ax, ay = NODES[route[edge_i]]
            bx, by = NODES[route[edge_i + 1]]
            vx, vy = bx - ax, by - ay
            length = math.hypot(vx, vy)
            if distance <= remaining + 1e-9:
                s = along + distance
                return (ax + vx * s / length, ay + vy * s / length,
                        math.atan2(vy, vx), edge_i)
            distance -= remaining
            edge_i += 1
            if edge_i >= len(route) - 1:
                break
            remaining = math.hypot(NODES[route[edge_i + 1]][0] - NODES[route[edge_i]][0],
                                   NODES[route[edge_i + 1]][1] - NODES[route[edge_i]][1])
            along = 0.0
        return None

    def block_path(self):
        """Place the trolley one metre ahead on the active corridor centre line."""
        route = self.route or astar(nearest_node(self.ekf.x[0], self.ekf.x[1]), "WARD-3B")
        if not route or len(route) < 2:
            print("[TROLLEY] no active route to block")
            return False
        if not self.route:
            self.route, self.idx = route, 0
        ahead = self._point_ahead_on_route(1.0)
        if ahead is None:
            print("[TROLLEY] no corridor point 1.0 m ahead on the current route")
            return False
        x, y, heading, edge_i = ahead
        if min(math.hypot(x - nx, y - ny) for nx, ny in NODES.values()) < 0.5 - 1e-6:
            print("[TROLLEY] 1.0 m ahead falls within 0.5 m of a node")
            return False

        a, b = route[edge_i], route[edge_i + 1]
        if astar(a, route[-1], {edge_key(a, b)}) is None:
            print(f"[TROLLEY] corridor {a} - {b} has no alternate route")
            return False

        kind, obstacle = self.obstacle_nodes[self.obstacle_cycle]
        if obstacle is None:
            print(f"[OBSTACLE] missing generated {kind} model")
            return False
        for _other_kind, other in self.obstacle_nodes:
            if other is not None:
                safe_set(other, "translation", [-1.5, 1.8, -10.0])
        translation = obstacle.getField("translation")
        rotation = obstacle.getField("rotation")
        safe_set(obstacle, "translation", [float(x), float(y), 0.10])
        # The long axis spans the corridor for each obstacle shape.
        safe_set(obstacle, "rotation", [0.0, 0.0, 1.0, float(heading + math.pi / 2)])
        obstacle.resetPhysics()
        self.obstacle_cycle = (self.obstacle_cycle + 1) % len(self.obstacle_nodes)
        sonar_front = self.sonar()["front"]
        self.last_event = f"{kind} placed at ({x:.2f},{y:.2f})"
        print(f"[OBSTACLE] {kind} placed at ({x:.2f},{y:.2f}), d=1.00 m ahead")
        print(f"[TROLLEY] sonar front={sonar_front:.2f} m")
        return True

    def unblock(self):
        px, py = CFG["trolley_park"]
        for i, (_kind, obstacle) in enumerate(self.obstacle_nodes):
            if obstacle is not None:
                z = 0.1 if i == 0 else -10.0
                safe_set(obstacle, "translation", [float(px), float(py), float(z)])
        self.blocked_edges.clear()
        self.blocked_edge_expiry.clear()
        print("[TROLLEY] removed, blocked edges cleared")

    def replan_blocked(self):
        if self.idx == 0:
            return
        prev, tgt = self.route[self.idx - 1], self.route[self.idx]
        self.blocked_edges.add(edge_key(prev, tgt))
        self.blocked_edge_expiry[edge_key(prev, tgt)] = self.now() + BLOCKED_EDGE_TTL
        self.replans += 1
        self.last_event = f"Replan: blocked {prev} - {tgt}"
        print(f"[REPLAN] corridor {prev} - {tgt} blocked for {BLOCK_TIME:.0f} s -> marking edge blocked")
        # A sharp stop and turn can accumulate enough wheel/gyro error to send
        # the detour into a wall. Supervisor pose is available in this Webots
        # controller, so re-anchor the estimate before following the new route.
        # Re-anchor only position and preserve continuous gyro/encoder heading.
        sx, sy, _ = self.true_pose()
        self.ekf.x[:2] = [sx, sy]
        self.ekf.P[:2, :2] = np.diag([1e-4, 1e-4])
        goal = self.route[-1]
        if self.plan(prev, goal):
            self.draw_detour(self.route)
            if self.regression_scenario == "B" and "ICU" in self.route:
                self.regression_route_via_icu = True
            if self.state == "APPROACH":
                self.set_state("NAVIGATE")
        else:
            self.stop()
            print("[A*] destination blocked; routing back to HOME")
            if self.plan(prev, "HOME"):
                self.mission_delivered = False
                self.set_state("RETURN")
            else:
                self.end_mission("destination blocked; no safe route HOME")
        self.blocked_t = 0.0
        self.obst_printed = False

    # ------------------------------------------------------------ navigation
    def follow(self, tol_last, vmax):
        """Drive along self.route. Returns True when the final node is reached."""
        name = self.route[self.idx]
        tx, ty = NODES[name]
        x, y, th = self.ekf.x
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        last = self.idx == len(self.route) - 1
        if dist < (tol_last if last else WAYPOINT_RADIUS):
            if last:
                self.stop()
                return True
            # Brake before switching legs. Without this, the old forward wheel
            # velocities carry the chassis past a waypoint while the next leg
            # is already rotating in place.
            self.stop()
            self.idx += 1
            print(f"[NAV] reached {name} -> heading to {self.route[self.idx]}")
            return False

        err = wrap(math.atan2(dy, dx) - th)
        v, w = rotate_drive_command(err, vmax)
        self.turning = (v == 0.0 and abs(err) > math.radians(10))
        if v > 0.0:
            if last:
                v = min(v, max(0.08, dist * 1.2))
            elif dist < 0.60:
                # Reduce approach speed before each turn so chassis momentum
                # does not carry the robot beyond the waypoint during braking.
                v = min(v, max(0.05, dist * 0.4))

        if v > 0 and self.obstacle():
            self.stop()
            if not self.obst_printed:
                d = self.sonar()
                print(f"[OBSTACLE] STOP  front={d['front'] * 100:.0f} cm "
                      f"left={d['left'] * 100:.0f} cm right={d['right'] * 100:.0f} cm")
                self.obst_printed = True
                self.obstacle_stops += 1
                self._update_status_led("BLOCKED")
            self.blocked_t += self.dt
            if self.blocked_t >= BLOCK_TIME:
                self.replan_blocked()
            return False
        self.blocked_t = 0.0
        self.obst_printed = False
        self.drive(v, w)
        return False

    # ------------------------------------------------------------ missions / logging
    def start_mission(self):
        # Temporary blockages from an earlier mission must not poison new routes.
        self.blocked_edges.clear()
        self.blocked_edge_expiry.clear()
        start = nearest_node(self.ekf.x[0], self.ekf.x[1])
        self.mission_room = next((room for room, info in CFG["rooms"].items()
                                  if info["in_node"] == self.mission_destination), "WARD-3B")
        self.active_patient = next(p for p in PATIENTS if p["ward"] == self.mission_room)
        self.delivery_compartment = self.active_patient["compartment"]
        self.handover_started = False
        self.handover_attempt = 0
        self.handover_result = "PENDING"
        self.handover_reason = ""
        self.handover_face_id = self.handover_uid = None
        self.handover_face_score = 0.0
        self.mission_start_t = self.now()
        self.mission_routes = []
        self.delivery_logged = False
        self.replans = 0
        self.last_event = "Mission started"
        self.dev_override_steps.clear()
        if not self.plan(start, self.mission_destination):
            self.mission_no += 1
            self.errs = []
            self._append_delivery_row("ABORTED_BLOCKED")
            return
        self.mission_no += 1
        self.mission_delivered = False
        self.last_mission_result = None
        self.errs = []
        self.n_corrections = 0
        self.compartments.lock_all()
        self.compartments.reset_cargo()
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs",
            f"mission_{time.time_ns()}_tags_{'ON' if self.tags_enabled else 'OFF'}.csv")
        self.log_f = open(fn, "w", newline="")
        self.log_w = csv.writer(self.log_f)
        self.log_w.writerow(["t", "state", "tags_on", "est_x", "est_y",
                             "true_x", "true_y", "err_cm", "tag_seen",
                             "planned_x", "planned_y"])
        print(f"[LOG] {fn}")
        tag_fn = os.path.join(os.path.dirname(fn), f"tag_corrections_{time.time_ns()}.csv")
        self.tag_log_f = open(tag_fn, "w", newline="")
        self.tag_log_w = csv.writer(self.tag_log_f)
        self.tag_log_w.writerow(["t", "step", "scenario", "tag_id", "range_m",
                                 "bearing_deg", "incidence_deg", "size_px",
                                 "correction_m", "correction_deg", "accepted", "reason"])
        print(f"[TAGLOG] {tag_fn}")
        self.set_state("NAVIGATE")

    def end_mission(self, why):
        self.last_mission_result = why
        outcome = ("DELIVERED" if self.mission_delivered else
                   "DENIED" if self.handover_result.startswith("DENIED") or
                   self.handover_result.startswith("DEV_OVERRIDE: DENIED")
                   else "ABORTED_BLOCKED")
        self._append_delivery_row(outcome)
        self.last_event = why
        if self.errs:
            print(f"[SUMMARY] {why} | tag correction {'ON' if self.tags_enabled else 'OFF'} | "
                  f"mean error {np.mean(self.errs):.1f} cm | max error {np.max(self.errs):.1f} cm | "
                  f"{self.n_corrections} tag corrections")
        if self.log_f:
            self.log_f.close()
            self.log_f = None
        if self.tag_log_f:
            self.tag_log_f.close()
            self.tag_log_f = self.tag_log_w = None
        self.stop()
        self.set_state("IDLE")
        self.actors.park_all()

    def _append_delivery_row(self, outcome):
        """Append exactly one result row for the current mission."""
        if self.delivery_logged:
            return
        self.delivery_logged = True
        log_dir = os.path.join(HERE, "logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "deliveries.csv")
        fields = ["timestamp", "patient", "ward", "item", "compartment", "outcome",
                  "face_name", "face_score", "rfid_ok", "route", "replans",
                  "mission_time_s", "mean_pose_error_cm", "max_pose_error_cm",
                  "final_distance_from_HOME_m"]
        patient = self.active_patient or {}
        tx, ty, _ = self.true_pose()
        hx, hy, _ = CFG["robot"]["start"]
        face_name = PEOPLE_BY_ID.get(self.handover_face_id, {}).get("name", "")
        rfid_ok = outcome == "DELIVERED" and bool(self.handover_uid)
        routes = json.dumps(self.mission_routes, separators=(",", ":"))
        elapsed = max(0.0, self.now() - self.mission_start_t) if self.mission_start_t is not None else 0.0
        row = [datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               patient.get("id", ""), patient.get("ward", ""),
               patient.get("medicine_item", patient.get("medicine", "")),
               patient.get("compartment", ""), outcome, face_name,
               f"{self.handover_face_score:.3f}", str(rfid_ok).lower(), routes,
               self.replans, f"{elapsed:.2f}",
               f"{float(np.mean(self.errs)):.2f}" if self.errs else "0.00",
               f"{float(np.max(self.errs)):.2f}" if self.errs else "0.00",
               f"{math.hypot(tx - hx, ty - hy):.3f}"]
        needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as deliveries:
            writer = csv.writer(deliveries)
            if needs_header:
                writer.writerow(fields)
            writer.writerow(row)
        print(f"[MISSION LOG] {outcome} appended to {path}")

    def log_step(self):
        e = self.true_err() * 100
        self.errs.append(e)
        if self.log_w and self.step_no % 8 == 0:
            tx, ty, _ = self.true_pose()
            target = NODES[self.route[self.idx]] if self.route and self.idx < len(self.route) else ("", "")
            self.log_w.writerow([
                f"{self.now():.2f}", self.state, int(self.tags_enabled),
                f"{self.ekf.x[0]:.3f}", f"{self.ekf.x[1]:.3f}",
                f"{tx:.3f}", f"{ty:.3f}",
                f"{e:.1f}", int(self.tag_seen),
                f"{target[0]:.3f}" if target[0] != "" else "",
                f"{target[1]:.3f}" if target[1] != "" else ""])

    def _visitor_candidates(self):
        patient = self.active_patient
        other_relative = next(person for p in PATIENTS if p["id"] != patient["id"]
                              for person in p["authorised_family"])
        return [patient["authorised_family"][0], other_relative,
                PATIENT_DATA["unauthorised_visitors"][0]]

    def _selected_people(self):
        visitor = self._visitor_candidates()[self.visitor_choice_index % 3]
        other_card = next(person for p in PATIENTS if p["id"] != self.active_patient["id"]
                          for person in p["authorised_family"])
        card_owner = other_card if self.card_swapped else visitor
        return visitor, card_owner

    def _show_selected_visitor(self):
        visitor, card_owner = self._selected_people()
        self.active_visitor_id = visitor["id"]
        self.active_card_owner_id = card_owner["id"]
        room = CFG["rooms"][self.mission_room]
        self.rfid_card_start = self.actors.present(
            self.active_visitor_id, self.active_card_owner_id,
            room["handover"]["visitor"], self.true_pose())
        print(f"[VISITOR] {visitor['name']} ({visitor['id']}); card={card_owner['name']}")

    def _start_handover_attempt(self, retry=False):
        if not retry:
            self.handover_attempt += 1
        self._show_selected_visitor()
        self.handover_step = "FACE"
        self.handover_face_id = None
        self.handover_face_score = 0.0
        self.handover_uid = None
        self.handover_reason = ""
        self.face_started_t = self.now()
        self.handover_result = f"ATTEMPT {self.handover_attempt}/{HANDOVER_MAX_ATTEMPTS}"
        print(f"[HANDOVER] FACE step started for {self.active_visitor_id}")

    def _handover_log(self, outcome, reason=""):
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        path = os.path.join(HERE, "logs", "handover_log.csv")
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as handover_file:
            writer = csv.writer(handover_file)
            if write_header:
                writer.writerow(["time_s", "patient_id", "visitor_id", "face_id",
                                 "face_score", "rfid_uid", "attempt", "outcome", "reason"])
            writer.writerow([f"{self.now():.2f}", self.active_patient["id"] if self.active_patient else "-",
                             self.active_visitor_id or "", self.handover_face_id or "",
                             f"{self.handover_face_score:.2f}", self.handover_uid or "",
                             self.handover_attempt, outcome, reason])

    def _read_emulated_rfid(self, card_position):
        robot_pos = self.me.getPosition()
        yaw = self.true_pose()[2]
        reader = (robot_pos[0] + 0.242 * math.cos(yaw),
                  robot_pos[1] + 0.242 * math.sin(yaw), robot_pos[2] + 0.15)
        dx, dy = card_position[0] - reader[0], card_position[1] - reader[1]
        distance = math.sqrt(dx * dx + dy * dy + (card_position[2] - reader[2]) ** 2)
        bearing = math.atan2(dy, dx)
        if distance <= 0.30 and abs(wrap(bearing - yaw)) <= math.radians(60):
            return PEOPLE_BY_ID[self.active_card_owner_id]["rfid_uid"]
        return None

    def _finish_handover_attempt(self, timed_out=False, missing=False):
        self.handover_step = "DECISION"
        face_id, face_score = self.handover_face_id, self.handover_face_score
        uid = self.handover_uid
        if self.dev_mode and "face" in self.dev_override_steps:
            face_id = self.active_patient["authorised_family"][0]["id"]
            face_score = 1.0
        if self.dev_mode and "rfid" in self.dev_override_steps:
            uid = self.active_patient["authorised_family"][0]["rfid_uid"]
        decision = decide_handover(self.active_patient, face_id, face_score, uid,
                                   PEOPLE_BY_ID, CARDS_BY_UID,
                                   rfid_timed_out=timed_out)
        compartment = compartment_to_unlock(self.active_patient, decision)
        if missing:
            decision = (False, "RFID missing", None)
            compartment = None
        allowed, reason, _person_id = decision
        self.handover_reason = reason
        if allowed:
            self.last_event = f"Handover authorized: compartment {compartment}"
            if self.dev_mode and self.dev_override_steps:
                self.handover_result = "DEV_OVERRIDE"
                self._handover_log("DEV_OVERRIDE", ",".join(sorted(self.dev_override_steps)))
                print("[HANDOVER] DEV_OVERRIDE")
            else:
                self.handover_result = "DELIVERED"
                self._handover_log("DELIVERED")
                print(f"[HANDOVER] AUTHORIZED patient={self.active_patient['id']} "
                      f"compartment={compartment}")
            unlock_authorized_compartment(self.active_patient, decision, self.compartments)
            self.mission_delivered = True
            self.actors.park_all()
            self.set_state("DELIVER")
            return

        self.handover_result = f"DENIED: {reason}"
        self.last_event = f"Handover denied: {reason}"
        print(f"[HANDOVER] DENIED attempt={self.handover_attempt}: {reason}")
        if self.dev_mode and self.dev_override_steps:
            self.handover_result = f"DEV_OVERRIDE: DENIED {reason}"
            self._handover_log("DEV_OVERRIDE",
                               f"{','.join(sorted(self.dev_override_steps))}; denied: {reason}")
        else:
            self._handover_log("DENIED", reason)
        if self.handover_attempt < HANDOVER_MAX_ATTEMPTS:
            self._start_handover_attempt()
            return

        self.mission_delivered = False
        self.actors.park_all()
        self.set_state("DENIED")

    def _advance_handover(self):
        if self.handover_step == "FACE":
            if self.now() - self.face_started_t < 0.5:
                return
            frame = self.face_cam.getImage() if self.face_cam else None
            # FACE RECOGNITION STUB: image plus supervisor-known actor identity.
            self.handover_face_id, self.handover_face_score = face_verify(
                [{"image": frame,
                  "person_id": (self.active_visitor_id if
                                self.actors.visitors.get(self.active_visitor_id) is not None
                                else None)}])
            print(f"[HANDOVER] FACE: STUB id={self.handover_face_id} "
                  f"score={self.handover_face_score:.2f}")
            self.handover_step = "RFID"
            self.rfid_started_t = self.now()
            root = self.me.getPosition()
            yaw = self.true_pose()[2]
            self.rfid_card_target = (root[0] + (0.242 + 0.15) * math.cos(yaw),
                                     root[1] + (0.242 + 0.15) * math.sin(yaw),
                                     root[2] + 0.15)
            return
        elapsed = self.now() - self.rfid_started_t
        if self.active_card_owner_id not in self.actors.cards or not self.actors.cards[self.active_card_owner_id]:
            print("[RFID] missing card")
            self._finish_handover_attempt(missing=True)
            return
        fraction = min(1.0, max(0.0, elapsed / 1.0))
        start = self.rfid_card_start
        target = self.rfid_card_target
        position = tuple(start[i] + fraction * (target[i] - start[i]) for i in range(3))
        self.actors.move_card(self.active_card_owner_id, position)
        # Query the node's world pose after Supervisor moved it; do not assume
        # the commanded transform is the physical read position.
        actual_position = self.actors.card_position(self.active_card_owner_id)
        uid = self._read_emulated_rfid(actual_position) if actual_position else None
        if uid:
            self.handover_uid = uid
            print(f"[RFID] UID={uid} read")
            self._finish_handover_attempt()
        elif elapsed >= RFID_TIMEOUT_S:
            print("[RFID] timeout")
            self._finish_handover_attempt(timed_out=True)

    def dashboard_step(self):
        """Draw the live map and mission summary to the robot's 480x320 Display."""
        display = self.dashboard
        if display is None or self.now() - self.dashboard_last_t < 0.5:
            return
        self.dashboard_last_t = self.now()

        def color(rgb):
            display.setColor(rgb)

        def text(value, x, y, rgb=0xE8EEF5):
            color(rgb)
            display.drawText(str(value)[:48], int(x), int(y))

        def line(x1, y1, x2, y2, rgb):
            color(rgb)
            display.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Background and header
        color(0x15202B)
        display.fillRectangle(0, 0, 480, 320)
        text("MEDIBOT | LIVE DELIVERY", 10, 8, 0x72C7FF)

        # Fit graph coordinates into a compact top-down map on the left.
        map_box = (12, 38, 228, 160)
        xs = [p[0] for p in NODES.values()]
        ys = [p[1] for p in NODES.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        sx = (map_box[2] - 28) / max(max_x - min_x, 1.0)
        sy = (map_box[3] - 28) / max(max_y - min_y, 1.0)
        scale = min(sx, sy)

        def xy(point):
            return (map_box[0] + 14 + (point[0] - min_x) * scale,
                    map_box[1] + map_box[3] - 14 - (point[1] - min_y) * scale)

        for a, b in CFG["edges"]:
            p1, p2 = xy(NODES[a]), xy(NODES[b])
            rgb = 0xE64B4B if edge_key(a, b) in self.blocked_edges else 0x657789
            line(*p1, *p2, rgb)
        for route in self.mission_routes:
            for a, b in zip(route[:-1], route[1:]):
                line(*xy(NODES[a]), *xy(NODES[b]), 0x1687FF)
        for a, b in zip(self.route[:-1], self.route[1:]):
            line(*xy(NODES[a]), *xy(NODES[b]), 0x1687FF)
        for name, point in NODES.items():
            px, py = xy(point)
            color(0xB9C7D5)
            display.fillRectangle(int(px - 2), int(py - 2), 4, 4)
            text(name, px + 3, py - 5, 0xD5E0EA)

        rx, ry, heading = self.true_pose()
        px, py = xy((rx, ry))
        color(0x42E88A)
        display.fillRectangle(int(px - 4), int(py - 4), 8, 8)
        line(px, py, px + 12 * math.cos(heading), py - 12 * math.sin(heading), 0x42E88A)

        if self.trolley is not None:
            ox, oy, oz = self.trolley.getPosition()
            park_x, park_y = CFG.get("trolley_park", (-1.5, 1.8))
            if oz > 0 and math.hypot(ox - park_x, oy - park_y) > 0.25:
                tx, ty = xy((ox, oy))
                color(0xFF4B45)
                display.fillRectangle(int(tx - 4), int(ty - 4), 8, 8)

        # Compact status panel; all rendering uses only Display text/line/rectangle calls.
        patient = self.active_patient or {}
        sonar = self.sonar()
        info = [
            (f"STATE: {self.state}", 0x72C7FF),
            (f"DEST: {self.dest}", 0xE8EEF5),
            (f"PATIENT: {patient.get('name', '-')}", 0xE8EEF5),
            (f"WARD: {patient.get('ward', '-')}", 0xE8EEF5),
            (f"COMPARTMENT: {patient.get('compartment', '-')}", 0xE8EEF5),
            ("LIDS: " + "  ".join(
                f"{n}:{'OPEN' if self.compartments.is_open(n) else 'LOCK'}" for n in (1, 2, 3)), 0xFFD166),
            (f"SONAR m F {sonar['front']:.2f} L {sonar['left']:.2f} R {sonar['right']:.2f}", 0xE8EEF5),
            (f"POSE ERROR {self.true_err() * 100:.1f} cm", 0xE8EEF5),
            (f"EVENT: {self.last_event}", 0xAAB7C4),
        ]
        for index, (value, rgb) in enumerate(info):
            text(value, 254, 36 + index * 25, rgb)
        text("KEYS: S start | B block path | X reset", 12, 286, 0x72C7FF)
        if self.dev_mode:
            color(0xB82020)
            display.fillRectangle(360, 8, 108, 20)
            text("DEV MODE", 370, 11, 0xFFFFFF)

    def hud(self):
        d = self.sonar()
        r = self.robot
        # Clear all labels first to prevent overdraw
        for i in range(10):
            r.setLabel(i, "", 0, 0, 0, 0, 0, "Arial")
        # Now set new labels
        r.setLabel(0, "MediBot - hospital delivery simulation", 0.01, 0.01, 0.07, 0x1F4E79, 0, "Arial")
        r.setLabel(1, f"State: {self.state}    Destination: {self.dest}",
                   0.01, 0.05, 0.06, 0x000000, 0, "Arial")
        r.setLabel(2, f"Ultrasonic (cm)  F:{d['front'] * 100:.0f}  L:{d['left'] * 100:.0f}  R:{d['right'] * 100:.0f}",
                   0.01, 0.09, 0.06, 0x000000, 0, "Arial")
        r.setLabel(3, f"Pose error: {self.true_err() * 100:.1f} cm    Tag correction: "
                      f"{'ON' if self.tags_enabled else 'OFF'}",
                   0.01, 0.13, 0.06, 0xC00000, 0, "Arial")
        r.setLabel(4, f"Last tag: {self.last_tag_msg}", 0.01, 0.17, 0.05, 0x006600, 0, "Arial")
        comp_status = " | ".join(
            f"C{n}:{'OPEN' if self.compartments.is_open(n) else 'LOCK'}"
            for n in (1, 2, 3))
        r.setLabel(5, comp_status, 0.01, 0.21, 0.05, 0x003399, 0, "Arial")
        r.setLabel(6, "S start | B block | U unblock | V visitor | C swap card | T tags | X reset",
                   0.01, 0.95, 0.05, 0x555555, 0, "Arial")
        if self.dev_mode:
            r.setLabel(7, "DEV MODE", 0.42, 0.01, 0.09, 0xCC0000, 0, "Arial")
        if self.active_patient:
            r.setLabel(8, f"Patient: {self.active_patient['name']} ({self.active_patient['id']})  "
                          f"Ward: {self.active_patient['ward']}  Bed: {self.active_patient['bed']}  "
                          f"Compartment: {self.active_patient['compartment']}",
                       0.01, 0.25, 0.05, 0x202020, 0, "Arial")
        face_name = PEOPLE_BY_ID.get(self.handover_face_id, {}).get("name", "-")
        rfid_text = self.handover_uid or "-"
        r.setLabel(9, f"FACE: STUB | {self.handover_step} | face={face_name} "
                      f"score={self.handover_face_score:.2f} | RFID={rfid_text} | {self.handover_result}",
                   0.01, 0.29, 0.045, 0x202020, 0, "Arial")

    def run_regression(self):
        """Run selected scenarios; env vars permit separate/repeated Webots runs."""
        requested = os.environ.get("MEDIBOT_REGRESSION_SCENARIOS", "A,B").upper()
        scenarios = [name.strip() for name in requested.split(",")
                     if name.strip() in {"A", "B", "D"}]
        if not scenarios:
            scenarios = ["A", "B"]
        try:
            repeat = max(1, min(3, int(os.environ.get("MEDIBOT_REGRESSION_REPEAT", "1"))))
        except ValueError:
            repeat = 1
        scenarios = [scenario for scenario in scenarios for _ in range(repeat)]
        print(f"[REGRESSION] scenario queue: {', '.join(scenarios)}")
        results = []

        for scenario in scenarios:
            print(f"[REGRESSION] Starting scenario {scenario}...")
            self.regression_scenario = scenario
            self.regression_start_t = self.now()
            self.errs = []
            self.n_corrections = 0
            self.obstacle_stops = 0
            self.replans = 0
            self.regression_route_via_icu = False
            self.regression_trolley_placed = False
            self.mission_delivered = False
            self.last_mission_result = None
            self.mission_destination = "WARD-2A-IN" if scenario == "D" else "WARD-3B-IN"
            self.tags_enabled = True
            self.tag_filter.clear()
            self.home_tag_pose = None
            self.home_tag_seen_t = -float("inf")
            self.tag_last_detection_t = -float("inf")
            self.dock_stage_times = {}
            self.dock_stage = "idle"

            # Setup for each scenario
            self.reset()
            self.blocked_edges.clear()
            self.blocked_edge_expiry.clear()
            self.blocked_t = 0.0
            self.obst_printed = False
            # Start mission
            self.start_mission()

            # Allow time for the delivery route and the tag-guided docking pass.
            webots_stopped = False
            trial_timed_out = False
            while self.state != "IDLE" and (self.now() - self.regression_start_t) < 180:
                step_status = self.robot.step(TIME_STEP)
                if step_status == -1:
                    print(f"[REGRESSION][ERROR] robot.step returned -1 during scenario {scenario}; "
                          "stopping this trial because Webots has ended the simulation")
                    webots_stopped = True
                    break
                self.step_no += 1
                self.localize_step()
                if self.step_no % 2 == 0:
                    self.camera_step()
                if (scenario == "B" and not self.regression_trolley_placed and self.state == "NAVIGATE"
                        and self.idx > 0 and self.route[self.idx - 1:self.idx + 1]
                        == ["JUNCTION", "WARD-2A"]):
                    self.regression_trolley_placed = self.block_path()
                self.state_machine()
                if self.state != "IDLE":
                    self.log_step()
                if self.step_no % 4 == 0:
                    self.hud()

            if self.state != "IDLE" and not webots_stopped:
                trial_timed_out = True
                print(f"[REGRESSION][ERROR] scenario {scenario} exceeded 180 s")

            if self.dock_stage in {"acquire_home_tag", "rotate_home_yaw", "creep_to_HOME"}:
                self.dock_stage_times[self.dock_stage] = max(
                    0.0, self.now() - self.dock_stage_t0)

            # Collect results
            duration = self.now() - self.regression_start_t
            if webots_stopped:
                tx, ty = float(self.ekf.x[0]), float(self.ekf.x[1])
                _tx, _ty, final_heading = self.ekf.x
            else:
                tx, ty, final_heading = self.true_pose()
            final_dist_home = math.hypot(tx - CFG["robot"]["start"][0], ty - CFG["robot"]["start"][1])
            final_heading_error_deg = abs(math.degrees(wrap(
                final_heading - CFG["robot"]["start"][2])))
            mean_err = np.mean(self.errs) if self.errs else float('inf')
            max_err = np.max(self.errs) if self.errs else float('inf')

            result = {
                "scenario": scenario,
                "trial": sum(1 for previous in results if previous["scenario"] == scenario) + 1,
                "completed": self.state == "IDLE",
                "delivered": self.mission_delivered,
                "mission_result": (self.last_mission_result or
                                   ("Webots step returned -1" if webots_stopped else "")),
                "duration": duration,
                "final_dist_home": final_dist_home,
                "final_heading_error_deg": final_heading_error_deg,
                "mean_err": mean_err,
                "max_err": max_err,
                "n_corrections": self.n_corrections,
                "obstacle_stops": self.obstacle_stops,
                "replans": self.replans,
                "route_via_icu": self.regression_route_via_icu,
                "webots_stopped": webots_stopped,
                "timed_out": trial_timed_out,
                "dock_stage_times": dict(self.dock_stage_times),
            }
            results.append(result)

            # Clean up
            if webots_stopped:
                break
            self.unblock()
            self.tags_enabled = True

        # Write regression report
        report_path = os.path.join(HERE, "logs", "regression_report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("REGRESSION REPORT\n")
            f.write("=" * 50 + "\n\n")
            for r in results:
                f.write(f"Scenario {r['scenario']}:\n")
                f.write(f"  Trial: {r['trial']}\n")
                f.write(f"  Completed: {r['completed']}\n")
                f.write(f"  Duration: {r['duration']:.1f} s\n")
                f.write(f"  Final distance from HOME: {r['final_dist_home']:.3f} m\n")
                f.write(f"  Final heading error: {r['final_heading_error_deg']:.1f} deg\n")
                f.write(f"  Mean pose error: {r['mean_err']:.1f} cm\n")
                f.write(f"  Max pose error: {r['max_err']:.1f} cm\n")
                f.write(f"  Tag corrections: {r['n_corrections']}\n")
                f.write(f"  Obstacle stops: {r['obstacle_stops']}\n")
                f.write(f"  Replans: {r['replans']}\n")
                f.write(f"  Route via ICU: {r['route_via_icu']}\n")
                f.write(f"  Delivered: {r['delivered']}\n")
                f.write(f"  Mission result: {r['mission_result']}\n")
                f.write(f"  Webots step ended: {r['webots_stopped']}\n")
                f.write(f"  Timed out: {r['timed_out']}\n")
                for stage, seconds in r["dock_stage_times"].items():
                    f.write(f"  Dock {stage} stage: {seconds:.1f} s\n")
                f.write(f"  Destination: {'WARD-2A' if r['scenario'] == 'D' else 'WARD-3B'}\n")

                # Determine PASS/FAIL
                passed = True
                if r['scenario'] == "A":
                    passed = (r['completed'] and not r['timed_out'] and
                             r['mission_result'] == "mission complete - robot docked at HOME" and
                             r['final_dist_home'] < 0.15 and
                             r['final_heading_error_deg'] <= 10.0 and
                             r['mean_err'] < 8 and r['max_err'] < 25)
                elif r['scenario'] == "B":
                    passed = (r['completed'] and not r['timed_out'] and r['obstacle_stops'] >= 1 and
                              r['replans'] >= 1 and r['route_via_icu'] and r['delivered'] and
                              r['mission_result'] == "mission complete - robot docked at HOME" and
                              r['final_dist_home'] < 0.15 and
                              r['final_heading_error_deg'] <= 10.0 and
                              r['mean_err'] < 8 and r['max_err'] < 25)
                elif r['scenario'] == "D":
                    passed = (r['completed'] and not r['timed_out'] and r['delivered'] and
                              r['mission_result'] == "mission complete - robot docked at HOME" and
                              r['final_dist_home'] < 0.15 and
                              r['final_heading_error_deg'] <= 10.0)

                f.write(f"  Result: {'PASS' if passed else 'FAIL'}\n\n")

            # Overall summary
            all_passed = all(
                (r['scenario'] == "A" and r['completed'] and not r['timed_out'] and
                 r['mission_result'] == "mission complete - robot docked at HOME" and
                 r['final_dist_home'] < 0.15 and r['final_heading_error_deg'] <= 10.0 and
                 r['mean_err'] < 8 and r['max_err'] < 25) or
                (r['scenario'] == "B" and r['completed'] and not r['timed_out'] and r['obstacle_stops'] >= 1 and
                 r['replans'] >= 1 and r['route_via_icu'] and r['delivered'] and
                 r['mission_result'] == "mission complete - robot docked at HOME" and
                 r['final_dist_home'] < 0.15 and r['final_heading_error_deg'] <= 10.0 and
                 r['mean_err'] < 8 and r['max_err'] < 25) or
                (r['scenario'] == "D" and r['completed'] and not r['timed_out'] and r['delivered'] and
                 r['mission_result'] == "mission complete - robot docked at HOME" and
                 r['final_dist_home'] < 0.15 and r['final_heading_error_deg'] <= 10.0)
                for r in results)
            f.write(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}\n")

        print(f"[REGRESSION] Report written to {report_path}")
        print(f"[REGRESSION] Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

        # Quit only after writing the report; do not mask an earlier Webots stop.
        if not any(result["webots_stopped"] for result in results):
            print("[REGRESSION] calling robot.simulationQuit(0) after all queued trials")
            self.robot.simulationQuit(0)

    # ------------------------------------------------------------ keys + state machine
    def keys(self):
        seen = set()
        while True:
            k = self.kb.getKey()
            if k == -1:
                break
            c = chr(k & 0xFF).upper() if 0 < (k & 0xFF) < 128 else ""
            if c not in self._key_names:
                continue
            seen.add(c)
            self._key_last_seen[c] = self.now()
            if not self._key_armed[c]:
                continue
            self._key_armed[c] = False
            self._handle_key(c)

        # Webots exposes key-down events rather than a separate release event.
        # Rearm only after the key has been absent for a short release interval.
        for key in self._key_names - seen:
            if (not self._key_armed[key] and
                    self.now() - self._key_last_seen[key] >= 0.25):
                self._key_armed[key] = True

    def _handle_key(self, key):
        actions = {
            "S": "start delivery" if self.state == "IDLE" else "ignored while mission active",
            "B": "block route / spawn next obstacle",
            "U": "remove obstacles and clear blocks",
            "X": "stop mission and reset HOME",
            "V": "choose next visitor",
            "C": "swap RFID card",
            "R": "tap RFID card" if self.state == "VERIFY" and self.handover_step == "RFID"
            else "ignored outside RFID verification",
            "T": "toggle tag correction" if self.state == "IDLE" else "ignored while mission active",
            "F": "developer face override" if self.dev_mode and self.state == "VERIFY"
            else "ignored (dev mode disabled or not verifying)",
            "1": "developer compartment 1 override" if self.dev_mode else "ignored (dev mode disabled)",
            "2": "developer compartment 2 override" if self.dev_mode else "ignored (dev mode disabled)",
            "3": "developer compartment 3 override" if self.dev_mode else "ignored (dev mode disabled)",
        }
        print(f"[KEY] {key} -> {actions[key]}")
        if key == "S":
            if self.state == "IDLE":
                self.start_mission()
        elif key == "B":
            self.block_path()
        elif key == "U":
            self.unblock()
        elif key == "X":
            self.reset()
        elif key == "V":
            self.visitor_choice_index = (self.visitor_choice_index + 1) % 3
            visitor, _card = self._selected_people()
            print(f"[VISITOR] selected {visitor['name']}")
            if self.state == "VERIFY" and self.active_patient:
                self._start_handover_attempt()
        elif key == "C":
            self.card_swapped = not self.card_swapped
            _visitor, card = self._selected_people()
            print(f"[RFID] selected card owner {card['name']}")
            if self.state == "VERIFY" and self.active_patient:
                self._start_handover_attempt()
        elif key == "R":
            if self.state == "VERIFY" and self.handover_step == "RFID":
                self.rfid_tap_requested = True
                self.rfid_tap_started_t = self.now()
                self.rfid_card_start = self.actors.card_position(self.active_card_owner_id)
                if self.rfid_card_start is None:
                    self.rfid_card_start = self.actors.present(
                        self.active_visitor_id, self.active_card_owner_id,
                        CFG["rooms"][self.mission_room]["handover"]["visitor"], self.true_pose())
                print("[RFID] tap animation started")
            elif self.state != "VERIFY":
                print("[RFID] ignored outside VERIFY")
        elif key == "T":
            if self.state == "IDLE":
                self.tags_enabled = not self.tags_enabled
                print(f"[TAGS] AprilTag correction {'ON' if self.tags_enabled else 'OFF'}")
        elif key == "F" and self.dev_mode and self.state == "VERIFY":
            self.dev_override_steps.add("face")
            print("[DEV MODE] face verification override armed")
        elif key in ("1", "2", "3") and self.dev_mode:
            number = int(key)
            self.dev_override_steps.add(f"compartment_{number}")
            if self.compartments.is_open(number):
                self.compartments.lock(number)
            else:
                self.compartments.unlock(number)
            self._handover_log("DEV_OVERRIDE", f"manual compartment {number}")

    def reset(self):
        if (self.mission_start_t is not None and not self.delivery_logged and
                self.state != "IDLE"):
            self._append_delivery_row("ABORTED_BLOCKED")
        self.stop()
        self.actors.park_all()
        self.visitor_choice_index = 0
        self.card_swapped = False
        self.active_patient = None
        self.active_visitor_id = self.active_card_owner_id = None
        self.handover_result = "-"
        self.handover_step = "-"
        self.handover_started = False
        sx, sy, sth = CFG["robot"]["start"]
        safe_set(self.me, "translation", [float(sx), float(sy), float(CFG["robot"]["z"])])
        safe_set(self.me, "rotation", [0.0, 0.0, 1.0, float(sth)])
        self.me.resetPhysics()
        self.ekf = EKF([sx, sy, sth])
        self.compartments.lock_all()
        self.compartments.reset_cargo()
        self.unblock()
        self.clear_route_visuals()
        self.route_nodes, self.route, self.dest = [], [], "-"
        self.tag_filter.clear()
        self.home_tag_pose = None
        self.home_tag_seen_t = -float("inf")
        self.tag_last_detection_t = -float("inf")
        self.dock_position_locked = False
        if self.log_f:
            self.log_f.close()
            self.log_f = None
        if self.tag_log_f:
            self.tag_log_f.close()
            self.tag_log_f = self.tag_log_w = None
        self.set_state("IDLE")
        print("[RESET] robot back at HOME")

    def state_machine(self):
        s, t = self.state, self.now()
        self.compartments.update()
        # Calibration mode (only in IDLE)
        if self.calibrating:
            if t - self.calib_start_t < 3.0:
                self.drive(0.2, 0.0)  # Drive straight at 0.2 m/s
            else:
                self.stop()
                true_end = self.true_pose()
                ekf_end = tuple(self.ekf.x)
                true_dist = math.hypot(true_end[0] - self.calib_start_pose[0],
                                       true_end[1] - self.calib_start_pose[1])
                ekf_dist = math.hypot(ekf_end[0] - self.calib_start_ekf[0],
                                     ekf_end[1] - self.calib_start_ekf[1])
                scale = ekf_dist / true_dist if true_dist > 0.01 else 1.0
                heading_drift = math.degrees(wrap(ekf_end[2] - self.calib_start_ekf[2]))
                print(f"[CAL] odometry scale = {scale:.3f}, heading drift = {heading_drift:.1f} deg")
                # Teleport back to HOME
                sx, sy, sth = CFG["robot"]["start"]
                safe_set(self.me, "translation", [float(sx), float(sy), float(CFG["robot"]["z"])])
                safe_set(self.me, "rotation", [0.0, 0.0, 1.0, float(sth)])
                self.me.resetPhysics()
                self.ekf = EKF([sx, sy, sth])
                self.calibrating = False
            return

        if s == "IDLE":
            self.stop()

        elif s == "NAVIGATE":
            if self.follow(0.25, CRUISE_V):
                self.set_state("APPROACH")

        elif s == "APPROACH":
            if self.follow(0.06, APPROACH_V):
                self.stop()
                print(f"[VERIFY] arrived for patient {self.active_patient['name']} "
                      f"in {self.active_patient['ward']}")
                self.set_state("VERIFY")

        elif s == "VERIFY":
            room = CFG["rooms"][self.mission_room]
            target_yaw = room["handover"]["robot"][2]
            heading_err = wrap(target_yaw - self.ekf.x[2])
            if abs(heading_err) > math.radians(3.0):
                self.drive(0.0, max(-0.3, min(0.3, 1.2 * heading_err)))
            else:
                self.stop()
                if not self.handover_started:
                    self.handover_started = True
                    self._start_handover_attempt()
                else:
                    self._advance_handover()

        elif s == "DENIED":
            self.stop()
            here = nearest_node(self.ekf.x[0], self.ekf.x[1])
            if self.plan(here, "HOME"):
                self.set_state("RETURN")
            else:
                self.blocked_edges.clear()
                if self.plan(here, "HOME"):
                    self.set_state("RETURN")
                else:
                    self.end_mission("denied; no route HOME")

        elif s == "DELIVER":
            self.stop()
            if t - self.state_t0 > 4.0:
                here = nearest_node(self.ekf.x[0], self.ekf.x[1])
                if self.plan(here, "HOME"):
                    self.set_state("RETURN")
                elif self.blocked_edges:
                    print("[REPLAN] clearing expired temporary blocks for the return leg")
                    self.blocked_edges.clear()
                    if self.plan(here, "HOME"):
                        self.set_state("RETURN")
                    else:
                        self.end_mission("cannot return")
                else:
                    self.end_mission("cannot return")

        elif s == "RETURN":
            if self.follow(0.06, APPROACH_V):
                self.dock_position_locked = False
                self.dock_stage = "acquire_home_tag"
                self.dock_stage_t0 = t
                self.dock_enter_t = t
                self.set_state("DOCK")

        elif s == "DOCK":
            sx, sy, home_yaw = CFG["robot"]["start"]
            x, y, heading = self.ekf.x
            home_tag = next(t for t in CFG["tags"] if t.get("zone") == "home_south")
            tag_fresh = (self.home_tag_pose is not None and
                         t - self.home_tag_seen_t <= 0.5)
            if t - self.dock_debug_t >= 2.0:
                self.dock_debug_t = t
                true_x, true_y, true_yaw = self.true_pose()
                print(f"[DOCK] stage={self.dock_stage} est={math.hypot(x-sx,y-sy):.3f} m, "
                      f"tag age={t-self.home_tag_seen_t:.2f} s, "
                      f"true={math.hypot(true_x-sx,true_y-sy):.3f} m / "
                      f"{math.degrees(wrap(true_yaw-home_yaw)):.1f} deg")

            if self.dock_stage == "acquire_home_tag":
                if tag_fresh and self.home_tag_seen_t >= self.dock_enter_t:
                    self.stop()
                    elapsed = t - self.dock_stage_t0
                    self.dock_stage_times["acquire_HOME_tag"] = elapsed
                    print(f"[DOCK] HOME tag acquired; stage took {elapsed:.1f} s")
                    self.dock_stage = "rotate_home_yaw"
                    self.dock_stage_t0 = t
                else:
                    tag_bearing = math.atan2(home_tag["y"] - y,
                                             home_tag["x"] - x)
                    err = wrap(tag_bearing - heading)
                    if abs(err) > math.radians(3.0):
                        self.drive(0.0, max(-0.30, min(0.30, 1.2 * err)))
                    else:
                        self.stop()

            elif self.dock_stage == "rotate_home_yaw":
                heading_err = wrap(home_yaw - heading)
                if abs(heading_err) > math.radians(2.0):
                    self.drive(0.0, max(-0.30, min(0.30, 1.2 * heading_err)))
                else:
                    self.stop()
                    elapsed = t - self.dock_stage_t0
                    self.dock_stage_times["rotate_home_yaw"] = elapsed
                    print(f"[DOCK] aligned to HOME yaw; stage took {elapsed:.1f} s")
                    self.dock_stage = "creep_to_HOME"
                    self.dock_stage_t0 = t

            elif self.dock_stage == "creep_to_HOME":
                # HOME tag localization initializes/corrects the EKF; odometry
                # bridges the short final creep when the tag leaves the camera FOV.
                dx, dy = sx - x, sy - y
                estimated_err = math.hypot(dx, dy)
                actual_x, actual_y, actual_yaw = self.true_pose()
                actual_err = math.hypot(actual_x - sx, actual_y - sy)
                actual_heading_err = abs(wrap(actual_yaw - home_yaw))
                heading_err = wrap(home_yaw - heading)
                if estimated_err <= 0.10:
                    self.stop()
                    if actual_err <= 0.15 and actual_heading_err <= math.radians(10.0):
                        elapsed = t - self.dock_stage_t0
                        self.dock_stage_times["creep_to_HOME"] = elapsed
                        print(f"[DOCK] docked within tolerance; stage took {elapsed:.1f} s; "
                              f"position={actual_err:.3f} m, "
                              f"heading={math.degrees(actual_heading_err):.1f} deg")
                        self.end_mission("mission complete - robot docked at HOME")
                    elif actual_err > 0.15:
                        # Keep making small corrections if EKF is optimistic.
                        target_heading = math.atan2(dy, dx)
                        bearing_err = wrap(target_heading - heading)
                        v = 0.025 if abs(bearing_err) < math.radians(20.0) else 0.0
                        self.drive(v, max(-0.12, min(0.12, 1.0 * bearing_err)))
                    else:
                        self.drive(0.0, max(-0.12, min(0.12, 1.0 * heading_err)))
                else:
                    # The last HOME leg is collinear with the configured dock
                    # heading. Reverse gently if the waypoint stop left the
                    # robot just past the target.
                    along_error = dx * math.cos(home_yaw) + dy * math.sin(home_yaw)
                    lateral_error = -dx * math.sin(home_yaw) + dy * math.cos(home_yaw)
                    if abs(lateral_error) > 0.10:
                        target_heading = math.atan2(dy, dx)
                        bearing_err = wrap(target_heading - heading)
                        v = 0.025 if abs(bearing_err) < math.radians(20.0) else 0.0
                        self.drive(v, max(-0.12, min(0.12, bearing_err)))
                    else:
                        v = max(-0.04, min(0.04, 0.6 * along_error))
                        self.drive(v, max(-0.10, min(0.10, 1.0 * heading_err)))
            else:
                tx_tag, ty_tag = home_tag["x"], home_tag["y"]
                target_heading = math.atan2(ty_tag - y, tx_tag - x)
                err = wrap(target_heading - heading)
                if abs(err) > math.radians(8.0):
                    self.drive(0.0, max(-0.35, min(0.35, 1.5 * err)))
                else:
                    self.stop()

    # ------------------------------------------------------------ main loop
    def run(self):
        while True:
            status = self.robot.step(TIME_STEP)
            if status == -1:
                return "robot.step returned -1"
            self.step_no += 1
            self.localize_step()
            if self.step_no % 2 == 0:
                self.camera_step()
            self.keys()
            self.state_machine()
            if self.state != "IDLE":
                self.log_step()
            if self.step_no % 4 == 0:
                self.hud()
            self.dashboard_step()


if __name__ == "__main__":
    controller = None
    exit_reason = "controller construction failed"
    try:
        controller = MediBot()
        exit_reason = controller.run()
    except BaseException as exc:
        traceback.print_exc(file=sys.stdout)
        exit_reason = f"uncaught {type(exc).__name__}: {exc}"
        raise
    finally:
        print(f"[CTRL] exiting: {exit_reason}")
        sys.stdout.flush()
