#!/usr/bin/env python3
"""
MediBot Webots controller.

Keys (click the 3D view first so it has keyboard focus):
  S start delivery   B put trolley on the route   U remove trolley
  F confirm face     R confirm RFID               X reset robot to HOME
  T toggle AprilTag correction (only while IDLE)  - turn OFF to show odometry drift
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
from medibot_core import (CFG, NODES, EKF, TagLocalizer, astar, edge_key, nearest_node, wrap)

# ---------------------------------------------------------------- parameters
TIME_STEP = 32
R_TRUE = CFG["robot"]["wheel_radius"]
TRACK = CFG["robot"]["track"]
CAM_W, CAM_H = CFG["camera"]["width"], CFG["camera"]["height"]

# Deliberate sensor imperfections so odometry drifts (like real hardware)
WHEEL_R_EST = R_TRUE * 1.03     # 3 % wheel-radius calibration error
GYRO_BIAS = 0.004               # rad/s gyro bias

CRUISE_V = 0.30
APPROACH_V = 0.12
WAYPOINT_TOL = 0.15
MAX_W = 1.2
MIN_TURN_W = 0.4
STOP_FRONT = 0.30               # m   stop if front sonar < 30 cm
STOP_SIDE = 0.15
BLOCK_TIME = 2.0                # s   still blocked after 2 s -> mark edge blocked, replan
AUTO_VERIFY_S = 8.0             # auto-confirm face/RFID after this many s (None = keys only)
GATE_M = 1.2                    # reject tag fixes that disagree with the estimate by more than this


class MediBot:
    def __init__(self):
        r = self.robot = Supervisor()
        self.dt = TIME_STEP / 1000.0
        self.motors = {k: r.getDevice(f"wheel_{k}") for k in ("fl", "fr", "rl", "rr")}
        for m in self.motors.values():
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.ps = {k: r.getDevice(f"ps_{k}") for k in ("fl", "fr", "rl", "rr")}
        for p in self.ps.values():
            p.enable(TIME_STEP)
        self.cam = r.getDevice("camera")
        self.cam.enable(2 * TIME_STEP)
        self.us = {n: r.getDevice(f"us_{n}") for n in ("front", "left", "right")}
        for s in self.us.values():
            s.enable(TIME_STEP)
        self.gyro = r.getDevice("gyro")
        self.gyro.enable(TIME_STEP)
        self.accel = r.getDevice("accel")
        self.accel.enable(TIME_STEP)
        self.imu = r.getDevice("imu")
        self.imu.enable(TIME_STEP)
        self.lid = r.getDevice("lid_motor")
        self.lid.setVelocity(2.0)
        self.lid.setPosition(0.0)
        self.kb = r.getKeyboard()
        self.kb.enable(TIME_STEP)
        self.me = r.getSelf()
        self.trolley = r.getFromDef("TROLLEY")
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
        self.dest = "-"
        self.mission_no = 0
        self.step_no = 0
        print(__doc__)
        print("[MediBot] ready - state IDLE. Press S to start a delivery to WARD-3B.")

    # ------------------------------------------------------------ helpers
    def now(self):
        return self.robot.getTime()

    def set_state(self, s):
        print(f"[STATE] {self.state} -> {s}")
        self.state = s
        self.state_t0 = self.now()

    def true_pose(self):
        p = self.me.getPosition()
        o = self.me.getOrientation()
        return p[0], p[1], math.atan2(o[3], o[0])

    def true_err(self):
        tx, ty, _ = self.true_pose()
        return math.hypot(self.ekf.x[0] - tx, self.ekf.x[1] - ty)

    def drive(self, v, w):
        wl = (v - w * TRACK / 2) / WHEEL_R_EST
        wr = (v + w * TRACK / 2) / WHEEL_R_EST
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
        return {k: s.getValue() for k, s in self.us.items()}

    def obstacle(self):
        d = self.sonar()
        return d["front"] < STOP_FRONT or d["left"] < STOP_SIDE or d["right"] < STOP_SIDE

    def localize_step(self):
        cur = {k: self.ps[k].getValue() for k in self.ps}
        dl = ((cur["fl"] - self.prev["fl"]) + (cur["rl"] - self.prev["rl"])) / 2
        dr = ((cur["fr"] - self.prev["fr"]) + (cur["rr"] - self.prev["rr"])) / 2
        self.prev = cur
        ds = WHEEL_R_EST * (dl + dr) / 2
        gz = self.gyro.getValues()[2] + GYRO_BIAS
        self.ekf.predict(ds, gz * self.dt)

    def camera_step(self):
        img = self.cam.getImage()
        if img is None:
            return
        gray = cv2.cvtColor(np.frombuffer(img, np.uint8).reshape((CAM_H, CAM_W, 4)), cv2.COLOR_BGRA2GRAY)
        meas = self.loc.measure(gray, hint=tuple(self.ekf.x))
        self.tag_seen = bool(meas)
        if not meas:
            return
        m = min(meas, key=lambda d: d["dist"])
        if not self.tags_enabled:
            self.last_tag_msg = f"tag {m['id']} seen (correction OFF)"
            return
        x, y, _ = self.ekf.x
        if math.hypot(m["x"] - x, m["y"] - y) > GATE_M:
            self.last_tag_msg = f"tag {m['id']} rejected (outlier)"
            return
        before = self.true_err() * 100
        sp, st = self.loc.noise(m["dist"])
        self.ekf.update([m["x"], m["y"], m["yaw"]], sp, st)
        after = self.true_err() * 100
        self.n_corrections += 1
        self.last_tag_msg = (f"tag {m['id']} d={m['dist']:.1f}m  err {before:.1f} -> {after:.1f} cm  "
                             f"var(P)={self.ekf.pos_variance:.4f}")
        print(f"[EKF] AprilTag {m['id']} @ {m['dist']:.1f} m | error before: {before:.1f} cm | "
              f"after: {after:.1f} cm | variance(P): {self.ekf.pos_variance:.4f}")

    # ------------------------------------------------------------ planning / drawing
    def plan(self, start, goal):
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
            for a, b in zip(self.route[:-1], self.route[1:]):
                (x1, y1), (x2, y2) = NODES[a], NODES[b]
                L = math.hypot(x2 - x1, y2 - y1)
                s = (f"Transform {{ translation {(x1 + x2) / 2:g} {(y1 + y2) / 2:g} 0.008 "
                     f"rotation 0 0 1 {math.atan2(y2 - y1, x2 - x1):g} children [ Shape {{ "
                     f"appearance Appearance {{ material Material {{ diffuseColor 0 0.4 1 emissiveColor 0 0.4 1 }} }} "
                     f"geometry Box {{ size {L:g} 0.06 0.004 }} }} ] }}")
                self.root_children.importMFNodeFromString(-1, s)
                self.route_nodes.append(self.root_children.getMFNode(self.root_children.getCount() - 1))
        except Exception as e:
            print("[draw] could not draw route:", e)

    def block_path(self):
        """Put the trolley on the middle of a corridor that has an alternative route."""
        route = self.route if self.state in ("NAVIGATE", "APPROACH", "RETURN") else astar("HOME", "WARD-3B")
        start = max(self.idx, 0) if self.state != "IDLE" else 0
        goal = route[-1]
        for i in range(start, len(route) - 1):
            a, b = route[i], route[i + 1]
            if astar(a, goal, {edge_key(a, b)}) is not None:
                mx = (NODES[a][0] + NODES[b][0]) / 2
                my = (NODES[a][1] + NODES[b][1]) / 2
                self.trolley.getField("translation").setSFVec3f([mx, my, 0.35])
                self.trolley.resetPhysics()
                print(f"[TROLLEY] placed on corridor {a} - {b}")
                return
        print("[TROLLEY] no corridor with an alternative route to block")

    def unblock(self):
        px, py = CFG["trolley_park"]
        self.trolley.getField("translation").setSFVec3f([px, py, 0.35])
        self.blocked_edges.clear()
        print("[TROLLEY] removed, blocked edges cleared")

    def replan_blocked(self):
        if self.idx == 0:
            return
        prev, tgt = self.route[self.idx - 1], self.route[self.idx]
        self.blocked_edges.add(edge_key(prev, tgt))
        print(f"[REPLAN] corridor {prev} - {tgt} blocked for {BLOCK_TIME:.0f} s -> marking edge blocked")
        goal = self.route[-1]
        if self.plan(prev, goal):
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
        if dist < (tol_last if last else WAYPOINT_TOL):
            if last:
                self.stop()
                return True
            self.idx += 1
            print(f"[NAV] reached {name} -> heading to {self.route[self.idx]}")
            return False

        err = wrap(math.atan2(dy, dx) - th)
        if abs(err) > 0.5:
            self.turning = True
        elif abs(err) < 0.12:
            self.turning = False
        if self.turning:
            v = 0.0
            w = math.copysign(max(MIN_TURN_W, min(MAX_W, 2.0 * abs(err))), err)
        else:
            v = vmax * max(0.35, 1.0 - abs(err) / 0.5)
            if last:
                v = min(v, max(0.08, dist * 1.2))
            w = max(-1.0, min(1.0, 2.5 * err))

        if v > 0 and self.obstacle():
            self.stop()
            if not self.obst_printed:
                d = self.sonar()
                print(f"[OBSTACLE] STOP  front={d['front'] * 100:.0f} cm left={d['left'] * 100:.0f} cm "
                      f"right={d['right'] * 100:.0f} cm")
                self.obst_printed = True
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
        start = nearest_node(self.ekf.x[0], self.ekf.x[1])
        if not self.plan(start, "WARD-3B"):
            return
        self.mission_no += 1
        self.errs = []
        self.n_corrections = 0
        self.face_ok = self.rfid_ok = False
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs",
                          f"mission_{time.strftime('%H%M%S')}_tags_{'ON' if self.tags_enabled else 'OFF'}.csv")
        self.log_f = open(fn, "w", newline="")
        self.log_w = csv.writer(self.log_f)
        self.log_w.writerow(["t", "state", "tags_on", "est_x", "est_y", "true_x", "true_y", "err_cm", "tag_seen"])
        print(f"[LOG] {fn}")
        self.set_state("NAVIGATE")

    def end_mission(self, why):
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
            self.log_w.writerow([f"{self.now():.2f}", self.state, int(self.tags_enabled),
                                 f"{self.ekf.x[0]:.3f}", f"{self.ekf.x[1]:.3f}", f"{tx:.3f}", f"{ty:.3f}",
                                 f"{e:.1f}", int(self.tag_seen)])

    def hud(self):
        d = self.sonar()
        r = self.robot
        r.setLabel(0, "MediBot - hospital delivery simulation", 0.01, 0.01, 0.07, 0x1F4E79, 0, "Arial")
        r.setLabel(1, f"State: {self.state}    Destination: {self.dest}", 0.01, 0.05, 0.06, 0x000000, 0, "Arial")
        r.setLabel(2, f"Ultrasonic (cm)  F:{d['front'] * 100:.0f}  L:{d['left'] * 100:.0f}  R:{d['right'] * 100:.0f}",
                   0.01, 0.09, 0.06, 0x000000, 0, "Arial")
        r.setLabel(3, f"Pose error: {self.true_err() * 100:.1f} cm    Tag correction: "
                      f"{'ON' if self.tags_enabled else 'OFF'}", 0.01, 0.13, 0.06, 0xC00000, 0, "Arial")
        r.setLabel(4, f"Last tag: {self.last_tag_msg}", 0.01, 0.17, 0.05, 0x006600, 0, "Arial")
        r.setLabel(5, "S start | B block | U unblock | F face | R RFID | T tags on/off | X reset",
                   0.01, 0.95, 0.05, 0x555555, 0, "Arial")

    # ------------------------------------------------------------ keys + state machine
    def keys(self):
        while True:
            k = self.kb.getKey()
            if k == -1:
                break
            c = chr(k & 0xFF).upper() if 0 < (k & 0xFF) < 128 else ""
            if c == "S" and self.state == "IDLE":
                self.start_mission()
            elif c == "B":
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

    def reset(self):
        self.stop()
        sx, sy, sth = CFG["robot"]["start"]
        self.me.getField("translation").setSFVec3f([sx, sy, CFG["robot"]["z"]])
        self.me.getField("rotation").setSFRotation([0, 0, 1, sth])
        self.me.resetPhysics()
        self.ekf = EKF([sx, sy, sth])
        self.lid.setPosition(0.0)
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
                print("[VERIFY] recipient verified -> unlocking compartment")
                self.lid.setPosition(-1.4)
                self.set_state("DELIVER")
        elif s == "DELIVER":
            self.stop()
            if t - self.state_t0 > 4.0:
                self.lid.setPosition(0.0)
                here = nearest_node(self.ekf.x[0], self.ekf.x[1])
                if self.plan(here, "HOME"):
                    self.set_state("RETURN")
                else:
                    self.end_mission("cannot return")
        elif s == "RETURN":
            if self.follow(0.08, CRUISE_V):
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
