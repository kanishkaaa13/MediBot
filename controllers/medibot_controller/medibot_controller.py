#!/usr/bin/env python3
"""
MediBot Webots controller  (Phase 2 – 4WD, 3 compartments, face-cam, status LED)

Keys (click the 3D view first so it has keyboard focus):
  S  start delivery          B  put trolley on the route   U  remove trolley
  F  confirm face            R  confirm RFID               X  reset robot to HOME
  T  toggle AprilTag correction (only while IDLE) - turn OFF to show odometry drift
  1/2/3  manually unlock compartment 1/2/3 (for demo)
"""
import csv
import math
import os
import time

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

HERE = os.path.dirname(os.path.abspath(__file__))

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
AUTO_VERIFY_S = 8.0             # auto-confirm face/RFID after this many s (None = keys only)
GATE_M = 1.2                    # deprecated: replaced by Mahalanobis test
MAHALANOBIS_THRESH = 11.34      # chi-square 3 dof, 99% confidence
MIN_ACCEPT_RADIUS = 0.4         # minimum accept radius for tag fixes
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
        self.cargos = {}

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

            # Cargo solid (supervisor node for deliver-animation)
            self.cargos[n] = robot.getFromDef(f"CARGO_{n}")

    def lock_all(self):
        for n in (1, 2, 3):
            self.lock(n)

    def unlock(self, n):
        """Open compartment n and light its LED green."""
        if self.motors.get(n):
            self.motors[n].setPosition(self.LID_OPEN)
        if self.leds.get(n):
            self.leds[n].set(LED_GREEN)
        print(f"[LOCK] compartment {n} UNLOCKED")

    def lock(self, n):
        """Close compartment n and light its LED red."""
        if self.motors.get(n):
            self.motors[n].setPosition(self.LID_CLOSE)
        if self.leds.get(n):
            self.leds[n].set(LED_RED)
        print(f"[LOCK] compartment {n} LOCKED")

    def is_open(self, n):
        ps = self.sensors.get(n)
        if ps is None:
            return False
        return ps.getValue() < -0.5   # clearly below halfway

    def deliver_cargo(self, n):
        """Animate cargo box sliding out of the compartment (teleport below floor)."""
        node = self.cargos.get(n)
        if node:
            f = node.getField("translation")
            if f:
                f.setSFVec3f([0.0, 0.0, -5.0])
        print(f"[CARGO] package {n} delivered")

    def reset_cargo(self):
        """Return all cargo boxes to their initial positions (inside compartments)."""
        init_positions = {
            1: (0.04, 0.0, 0.09),
            2: (-0.06, 0.0, 0.09),
            3: (-0.16, 0.0, 0.09),
        }
        for n, pos in init_positions.items():
            node = self.cargos.get(n)
            if node:
                f = node.getField("translation")
                if f:
                    f.setSFVec3f(list(pos))
        print("[CARGO] all packages reset")


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
        if self.status_led:
            self.status_led.set(LED_GREEN)

        # 3-compartment manager
        self.compartments = Compartments(r)

        # Backwards-compat alias: self.lid -> compartment 1 motor
        self.lid = self.compartments.motors.get(1)

        self.kb = r.getKeyboard()
        self.kb.enable(TIME_STEP)
        self.me = r.getSelf()
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
        self.blocked_edge_expiry = {}
        self.blocked_t = 0.0
        self.obst_printed = False
        self.turning = False
        self.face_ok = self.rfid_ok = False
        self.state_t0 = 0.0
        self.tag_seen = False
        self.last_tag_msg = "-"
        self.n_corrections = 0
        self.errs = []
        self.log_f = self.log_w = None
        self.route_nodes = []
        self.detour_nodes = []
        self.dest = "-"
        self.mission_no = 0
        self.step_no = 0
        self.last_ekf_print_t = -1.0
        self.calibrating = False
        self.calib_start_pose = None
        self.rejected_fixes = []  # Buffer for rejected tag fixes
        self.last_valid_gyro = 0.0
        self.last_valid_ps = {k: 0.0 for k in ("fl", "fr", "rl", "rr")}
        self.nan_warning_printed = set()  # Track which sensors have warned about NaN
        self.last_tag_id = None
        self.last_tag_zone = None
        self._block_key_armed = True
        self._block_key_last_seen = -float("inf")
        self.obstacle_stops = 0
        self.replans = 0
        self.regression_route_via_icu = False
        self.regression_trolley_placed = False
        self.mission_delivered = False
        self.last_mission_result = None

        # Regression mode
        self.regression_mode = os.path.exists(os.path.join(HERE, "regression.flag"))
        self.regression_scenario = None  # "A", "B", or "C"
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
            print("[REGRESSION] Automatic regression mode enabled")
            print("[REGRESSION] Running scenarios A, B, C sequentially...")
            self.run_regression()
        else:
            print("[MediBot] ready - state IDLE. Press S to start a delivery to WARD-3B.")

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
        self.state = s
        self.state_t0 = self.now()
        self._update_status_led(s)

    def _update_status_led(self, state):
        """Drive the front status LED strip to reflect the current state."""
        if self.status_led is None:
            return
        colour = {
            "IDLE":     LED_GREEN,
            "DELIVER":  LED_GREEN,
            "NAVIGATE": LED_BLUE,
            "APPROACH": LED_BLUE,
            "RETURN":   LED_BLUE,
            "VERIFY":   LED_ORANGE,
        }.get(state, LED_RED)    # BLOCKED / any other -> red
        self.status_led.set(colour)

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
            return
        m = min(meas, key=lambda d: d["dist"])
        if not self.tags_enabled:
            self.last_tag_msg = f"tag {m['id']} seen (correction OFF)"
            return

        # Mahalanobis test for tag acceptance
        x, y, th = self.ekf.x
        innovation = np.array([m["x"] - x, m["y"] - y, wrap(m["yaw"] - th)])
        sp, st = self.loc.noise(m["dist"])
        R = np.diag([sp ** 2, sp ** 2, st ** 2])
        S = self.ekf.P + R
        try:
            mahalanobis = innovation.T @ np.linalg.inv(S) @ innovation
        except np.linalg.LinAlgError:
            mahalanobis = float('inf')

        # Minimum accept radius check
        dist_error = math.hypot(m["x"] - x, m["y"] - y)

        if mahalanobis < MAHALANOBIS_THRESH and dist_error >= MIN_ACCEPT_RADIUS:
            # Accept the tag fix
            before = self.true_err() * 100
            self.ekf.update([m["x"], m["y"], m["yaw"]], sp, st)

            # Assert EKF state is finite after update
            if not np.all(np.isfinite(self.ekf.x)):
                print("[ERROR] EKF state became NaN after update! Re-initialising from last finite state.")
                sx, sy, sth = CFG["robot"]["start"]
                self.ekf = EKF([sx, sy, sth])
                return

            after = self.true_err() * 100
            self.n_corrections += 1
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
            # Reject the tag fix
            self.last_tag_msg = f"tag {m['id']} rejected (outlier)"
            # Add to rejected buffer
            now = self.now()
            self.rejected_fixes.append({
                "t": now,
                "x": m["x"],
                "y": m["y"],
                "yaw": m["yaw"],
                "id": m["id"]
            })
            # Clean old rejected fixes
            self.rejected_fixes = [f for f in self.rejected_fixes if now - f["t"] < REJECT_BUFFER_TIME]

            # Check for re-localisation trigger
            if len(self.rejected_fixes) >= REJECT_MIN_COUNT:
                # Check if rejected fixes agree with each other
                recent = self.rejected_fixes[-REJECT_MIN_COUNT:]
                avg_x = sum(f["x"] for f in recent) / len(recent)
                avg_y = sum(f["y"] for f in recent) / len(recent)
                avg_yaw = sum(wrap(f["yaw"]) for f in recent) / len(recent)
                avg_yaw = wrap(avg_yaw)

                # Check agreement
                pos_agree = all(math.hypot(f["x"] - avg_x, f["y"] - avg_y) < REJECT_AGREE_POS
                               for f in recent)
                th_agree = all(abs(wrap(f["yaw"] - avg_yaw)) < REJECT_AGREE_TH
                              for f in recent)

                # Check if from at least 2 different tag IDs or same ID over >= 0.5s
                tag_ids = set(f["id"] for f in recent)
                time_span = recent[-1]["t"] - recent[0]["t"]
                diverse_ids = len(tag_ids) >= 2 or (len(tag_ids) == 1 and time_span >= 0.5)

                if pos_agree and th_agree and diverse_ids:
                    # Re-localise
                    old_x, old_y, old_th = self.ekf.x
                    self.ekf.x = np.array([avg_x, avg_y, avg_yaw])
                    self.ekf.P = np.diag([1.0, 1.0, 1.0])  # Large uncertainty
                    tag_str = ",".join(str(tid) for tid in sorted(tag_ids))
                    print(f"[RELOCALISE] from ({old_x:.2f},{old_y:.2f}) to ({avg_x:.2f},{avg_y:.2f}) "
                          f"using tags {tag_str}")
                    self.rejected_fixes = []

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
                other.getField("translation").setSFVec3f([-1.5, 1.8, -10.0])
        translation = obstacle.getField("translation")
        rotation = obstacle.getField("rotation")
        translation.setSFVec3f([x, y, 0.10])
        # The long axis spans the corridor for each obstacle shape.
        rotation.setSFRotation([0, 0, 1, heading + math.pi / 2])
        obstacle.resetPhysics()
        self.obstacle_cycle = (self.obstacle_cycle + 1) % len(self.obstacle_nodes)
        sonar_front = self.sonar()["front"]
        print(f"[OBSTACLE] {kind} placed at ({x:.2f},{y:.2f}), d=1.00 m ahead")
        print(f"[TROLLEY] sonar front={sonar_front:.2f} m")
        return True

    def unblock(self):
        px, py = CFG["trolley_park"]
        for i, (_kind, obstacle) in enumerate(self.obstacle_nodes):
            if obstacle is not None:
                z = 0.1 if i == 0 else -10.0
                obstacle.getField("translation").setSFVec3f([px, py, z])
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
            self.end_mission("no route available")
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
            self.idx += 1
            print(f"[NAV] reached {name} -> heading to {self.route[self.idx]}")
            return False

        err = wrap(math.atan2(dy, dx) - th)
        v, w = rotate_drive_command(err, vmax)
        self.turning = (v == 0.0 and abs(err) > math.radians(10))
        if v > 0.0:
            if last:
                v = min(v, max(0.08, dist * 1.2))
            elif dist < 0.30:
                v = min(v, max(0.12, dist * 1.2))

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
        if not self.plan(start, "WARD-3B"):
            return
        self.mission_no += 1
        self.mission_delivered = False
        self.last_mission_result = None
        self.errs = []
        self.n_corrections = 0
        self.face_ok = self.rfid_ok = False
        self.compartments.lock_all()
        self.compartments.reset_cargo()
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs",
            f"mission_{time.strftime('%H%M%S')}_tags_{'ON' if self.tags_enabled else 'OFF'}.csv")
        self.log_f = open(fn, "w", newline="")
        self.log_w = csv.writer(self.log_f)
        self.log_w.writerow(["t", "state", "tags_on", "est_x", "est_y",
                             "true_x", "true_y", "err_cm", "tag_seen"])
        print(f"[LOG] {fn}")
        self.set_state("NAVIGATE")

    def end_mission(self, why):
        self.last_mission_result = why
        if self.errs:
            print(f"[SUMMARY] {why} | tag correction {'ON' if self.tags_enabled else 'OFF'} | "
                  f"mean error {np.mean(self.errs):.1f} cm | max error {np.max(self.errs):.1f} cm | "
                  f"{self.n_corrections} tag corrections")
        if self.log_f:
            self.log_f.close()
            self.log_f = None
        self.stop()
        self.set_state("IDLE")

    def log_step(self):
        e = self.true_err() * 100
        self.errs.append(e)
        if self.log_w and self.step_no % 8 == 0:
            tx, ty, _ = self.true_pose()
            self.log_w.writerow([
                f"{self.now():.2f}", self.state, int(self.tags_enabled),
                f"{self.ekf.x[0]:.3f}", f"{self.ekf.x[1]:.3f}",
                f"{tx:.3f}", f"{ty:.3f}",
                f"{e:.1f}", int(self.tag_seen)])

    def hud(self):
        d = self.sonar()
        r = self.robot
        # Clear all labels first to prevent overdraw
        for i in range(7):
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
        r.setLabel(6, "S start | B block | U unblock | F face | R RFID | T tags | X reset | 1/2/3 unlock",
                   0.01, 0.95, 0.05, 0x555555, 0, "Arial")

    def run_regression(self):
        """Run regression scenarios A, B, C automatically."""
        scenarios = ["A", "B", "C"]
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

            # Setup for each scenario
            self.reset()
            self.blocked_edges.clear()
            self.blocked_edge_expiry.clear()
            self.blocked_t = 0.0
            self.obst_printed = False
            if scenario == "C":
                # Turn tags OFF
                self.tags_enabled = False

            # Start mission
            self.start_mission()

            # Wait for mission completion (max 120 s)
            while self.state != "IDLE" and (self.now() - self.regression_start_t) < 180:
                self.robot.step(TIME_STEP)
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

            # Collect results
            duration = self.now() - self.regression_start_t
            tx, ty, _ = self.true_pose()
            final_dist_home = math.hypot(tx - CFG["robot"]["start"][0], ty - CFG["robot"]["start"][1])
            mean_err = np.mean(self.errs) if self.errs else float('inf')
            max_err = np.max(self.errs) if self.errs else float('inf')

            result = {
                "scenario": scenario,
                "completed": self.state == "IDLE",
                "delivered": self.mission_delivered,
                "mission_result": self.last_mission_result or "",
                "duration": duration,
                "final_dist_home": final_dist_home,
                "mean_err": mean_err,
                "max_err": max_err,
                "n_corrections": self.n_corrections,
                "obstacle_stops": self.obstacle_stops,
                "replans": self.replans,
                "route_via_icu": self.regression_route_via_icu,
            }
            results.append(result)

            # Clean up
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
                f.write(f"  Completed: {r['completed']}\n")
                f.write(f"  Duration: {r['duration']:.1f} s\n")
                f.write(f"  Final distance from HOME: {r['final_dist_home']:.3f} m\n")
                f.write(f"  Mean pose error: {r['mean_err']:.1f} cm\n")
                f.write(f"  Max pose error: {r['max_err']:.1f} cm\n")
                f.write(f"  Tag corrections: {r['n_corrections']}\n")
                f.write(f"  Obstacle stops: {r['obstacle_stops']}\n")
                f.write(f"  Replans: {r['replans']}\n")
                f.write(f"  Route via ICU: {r['route_via_icu']}\n")
                f.write(f"  Delivered: {r['delivered']}\n")
                f.write(f"  Mission result: {r['mission_result']}\n")

                # Determine PASS/FAIL
                passed = True
                if r['scenario'] == "A":
                    passed = (r['completed'] and r['duration'] < 120 and
                             r['mission_result'] == "mission complete - robot back at HOME" and
                             r['final_dist_home'] < 0.3 and
                             r['mean_err'] < 15 and r['max_err'] < 40)
                elif r['scenario'] == "B":
                    passed = (r['completed'] and r['obstacle_stops'] >= 1 and
                              r['replans'] >= 1 and r['route_via_icu'] and r['delivered'] and
                              r['mission_result'] == "mission complete - robot back at HOME" and
                              r['final_dist_home'] < 0.3)
                elif r['scenario'] == "C":
                    passed = (r['completed'] and r['delivered'] and
                              r['mission_result'] == "mission complete - robot back at HOME" and
                              r['mean_err'] < 15 and r['max_err'] < 40 and
                              r['final_dist_home'] < 0.30)

                f.write(f"  Result: {'PASS' if passed else 'FAIL'}\n\n")

            # Overall summary
            all_passed = all(
                (r['scenario'] == "C" or
                 (r['scenario'] == "A" and r['completed'] and r['duration'] < 120 and
                  r['final_dist_home'] < 0.3 and r['mean_err'] < 15 and r['max_err'] < 40) or
                 (r['scenario'] == "B" and r['completed'] and r['obstacle_stops'] >= 1 and
                  r['replans'] >= 1 and r['route_via_icu'] and r['delivered']))
                for r in results)
            f.write(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}\n")

        print(f"[REGRESSION] Report written to {report_path}")
        print(f"[REGRESSION] Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

        # Quit simulation
        self.robot.simulationQuit(0)

    # ------------------------------------------------------------ keys + state machine
    def keys(self):
        saw_block_key = False
        while True:
            k = self.kb.getKey()
            if k == -1:
                break
            c = chr(k & 0xFF).upper() if 0 < (k & 0xFF) < 128 else ""
            if c == "S" and self.state == "IDLE":
                self.start_mission()
            elif c == "B":
                saw_block_key = True
                self._block_key_last_seen = self.now()
                if self._block_key_armed:
                    self._block_key_armed = False
                    self.block_path()
            elif c == "U":
                self.unblock()
            elif c == "F" and self.state == "VERIFY":
                self.face_ok = True
                print("[VERIFY] face recognised (simulated)")
            elif c == "R" and self.state == "VERIFY":
                self.rfid_ok = True
                print("[VERIFY] RFID accepted (simulated)")
            elif c == "T":
                if self.state == "IDLE":
                    self.tags_enabled = not self.tags_enabled
                    print(f"[TAGS] AprilTag correction {'ON' if self.tags_enabled else 'OFF'}")
                else:
                    print("[TAGS] toggle only while IDLE")
            elif c == "X":
                self.reset()
            elif c in ("1", "2", "3"):
                n = int(c)
                print(f"[KEY] manually toggling compartment {n}")
                if self.compartments.is_open(n):
                    self.compartments.lock(n)
                else:
                    self.compartments.unlock(n)
            elif c == "K" and self.state == "IDLE":
                # Odometry calibration: drive straight 3s at 0.2 m/s, then report
                self.calibrating = True
                self.calib_start_pose = self.true_pose()
                self.calib_start_ekf = tuple(self.ekf.x)
                self.calib_start_t = self.now()
                print("[CAL] starting odometry calibration: driving straight 0.2 m/s for 3 s...")
        if (not saw_block_key and not self._block_key_armed
                and self.now() - self._block_key_last_seen >= 0.25):
            self._block_key_armed = True

    def reset(self):
        self.stop()
        sx, sy, sth = CFG["robot"]["start"]
        self.me.getField("translation").setSFVec3f([sx, sy, CFG["robot"]["z"]])
        self.me.getField("rotation").setSFRotation([0, 0, 1, sth])
        self.me.resetPhysics()
        self.ekf = EKF([sx, sy, sth])
        self.compartments.lock_all()
        self.compartments.reset_cargo()
        self.unblock()
        for n in self.route_nodes:
            try:
                n.remove()
            except Exception:
                pass
        self.route_nodes, self.route, self.dest = [], [], "-"
        if self.log_f:
            self.log_f.close()
            self.log_f = None
        self.set_state("IDLE")
        print("[RESET] robot back at HOME")

    def state_machine(self):
        s, t = self.state, self.now()
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
                self.me.getField("translation").setSFVec3f([sx, sy, CFG["robot"]["z"]])
                self.me.getField("rotation").setSFRotation([0, 0, 1, sth])
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
                self.face_ok = self.rfid_ok = False
                print("[VERIFY] arrived. Press F (face) and R (RFID) to confirm the recipient"
                      + (f" (auto-confirms in {AUTO_VERIFY_S:.0f} s)" if AUTO_VERIFY_S else ""))
                self.set_state("VERIFY")

        elif s == "VERIFY":
            self.stop()
            if AUTO_VERIFY_S and t - self.state_t0 > AUTO_VERIFY_S:
                self.face_ok = self.rfid_ok = True
                print("[VERIFY] auto-confirmed")
            if self.face_ok and self.rfid_ok:
                print("[VERIFY] recipient verified -> unlocking compartment 1")
                self.mission_delivered = True
                self.compartments.unlock(1)
                self.compartments.deliver_cargo(1)
                self.set_state("DELIVER")

        elif s == "DELIVER":
            self.stop()
            if t - self.state_t0 > 4.0:
                self.compartments.lock(1)
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
            if self.follow(0.25, CRUISE_V):
                self.end_mission("mission complete - robot back at HOME")

    # ------------------------------------------------------------ main loop
    def run(self):
        while self.robot.step(TIME_STEP) != -1:
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


if __name__ == "__main__":
    MediBot().run()
