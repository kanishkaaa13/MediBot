"""
medibot_core.py - pure-Python logic (no Webots import) so it can be unit-tested.

  * graph + A* path planning
  * EKF (x, y, theta)
  * AprilTag (tag36h11) detection and robot-pose-from-tag computation
"""
import heapq
import json
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.json")) as _f:
    CFG = json.load(_f)

NODES = {k: tuple(v) for k, v in CFG["nodes"].items()}
EDGES = [tuple(e) for e in CFG["edges"]]
ADJ = {}
for _a, _b in EDGES:
    ADJ.setdefault(_a, []).append(_b)
    ADJ.setdefault(_b, []).append(_a)

# ICU is a restricted zone: driving through it costs a little extra, so equal-length
# routes prefer the Ward 2A corridor and the ICU corridor is the fallback.
ICU_PENALTY = 0.25


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def edge_key(a, b):
    return frozenset((a, b))


def edge_cost(a, b):
    c = math.dist(NODES[a], NODES[b])
    if "ICU" in (a, b):
        c += ICU_PENALTY
    return c


def astar(start, goal, blocked=frozenset()):
    """A* on the waypoint graph. `blocked` is a set of edge_key(a, b). Returns node list or None."""
    def h(n):
        return math.dist(NODES[n], NODES[goal])

    openq = [(h(start), 0.0, start)]
    came, best, closed = {}, {start: 0.0}, set()
    while openq:
        _, g, n = heapq.heappop(openq)
        if n in closed:
            continue
        if n == goal:
            path = [n]
            while n in came:
                n = came[n]
                path.append(n)
            return path[::-1]
        closed.add(n)
        for m in ADJ[n]:
            if edge_key(n, m) in blocked:
                continue
            ng = g + edge_cost(n, m)
            if ng < best.get(m, 1e9):
                best[m] = ng
                came[m] = n
                heapq.heappush(openq, (ng + h(m), ng, m))
    return None


def nearest_node(x, y):
    return min(NODES, key=lambda n: math.hypot(NODES[n][0] - x, NODES[n][1] - y))


class EKF:
    """State [x, y, theta]. Predict: encoders (distance) + gyro (heading). Update: AprilTag pose."""

    def __init__(self, x0):
        self.x = np.array(x0, float)
        self.P = np.diag([1e-4, 1e-4, 1e-4])

    def predict(self, ds, dth):
        x, y, th = self.x
        tm = th + 0.5 * dth
        c, s = math.cos(tm), math.sin(tm)
        self.x = np.array([x + ds * c, y + ds * s, wrap(th + dth)])
        F = np.eye(3)
        F[0, 2] = -ds * s
        F[1, 2] = ds * c
        sig_s = 0.30 * abs(ds) + 1e-5       # distance noise (inflated: covers slip + wheel-radius bias)
        sig_t = 0.006 + 0.02 * abs(dth)     # heading noise (gyro bias / scrub)
        G = np.array([[c, -ds * s], [s, ds * c], [0, 1]])
        Q = G @ np.diag([sig_s ** 2, sig_t ** 2]) @ G.T
        self.P = F @ self.P @ F.T + Q

    def update(self, z, sig_p, sig_th):
        R = np.diag([sig_p ** 2, sig_p ** 2, sig_th ** 2])
        y = np.array(z, float) - self.x
        y[2] = wrap(y[2])
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        I = np.eye(3)
        self.P = (I - K) @ self.P @ (I - K).T + K @ R @ K.T   # Joseph form

    @property
    def pos_variance(self):
        return float(self.P[0, 0] + self.P[1, 1])


class TagLocalizer:
    """Detects tag36h11 AprilTags in a grayscale camera image and returns robot pose estimates."""

    def __init__(self, cfg=CFG):
        cam = cfg["camera"]
        self.w, self.h = cam["width"], cam["height"]
        self.cam_x = cam["x"]
        fx = self.w / (2 * math.tan(cam["fov"] / 2))
        self.K = np.array([[fx, 0, self.w / 2], [0, fx, self.h / 2], [0, 0, 1]], np.float64)
        self.tags = {t["id"]: t for t in cfg["tags"]}
        s = cfg["tag"]["size"] / 2
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):
            det = cv2.aruco.ArucoDetector(dictionary, params)
            self._detect = det.detectMarkers
        else:
            self._detect = lambda g: cv2.aruco.detectMarkers(g, dictionary, parameters=params)

    def _candidates(self, corners, tag):
        """Both IPPE solutions for one tag (near head-on views are ambiguous) as robot-pose dicts."""
        n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self.obj, corners, self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        cands = []
        for rvec, tvec, err in zip(rvecs, tvecs, errs):
            Rm, _ = cv2.Rodrigues(rvec)
            n = Rm[:, 2]              # tag normal (out of the face) in camera frame (x right, y down, z fwd)
            t = tvec.reshape(3)
            if n[2] > -0.25:          # facing away from the camera: impossible solution
                continue
            f = t[2] + self.cam_x     # tag position in robot frame: forward
            l = -t[0]                 # ... and left
            phi = math.atan2(-n[0], n[2])           # tag normal angle in robot frame
            yaw = wrap(tag["yaw"] - phi)
            cs, sn = math.cos(yaw), math.sin(yaw)
            cands.append({
                "x": tag["x"] - (f * cs - l * sn),
                "y": tag["y"] - (f * sn + l * cs),
                "yaw": yaw,
                "dist": float(np.linalg.norm(t)),
                "reproj": float(np.ravel(err)[0]),
            })
        return cands

    def measure(self, gray, hint=None, max_dist=3.3):
        """Return list of {id, x, y, yaw, dist}: one robot-pose estimate per visible known tag.

        `hint` = current (x, y, yaw) estimate. A square tag seen almost head-on has two valid
        solutions, so we keep the one that agrees best with the hint (the other is the mirror
        tilt). Without a hint the lowest reprojection error wins.
        """
        corners, ids, _ = self._detect(gray)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.flatten()):
            tag = self.tags.get(int(i))
            if tag is None:
                continue
            cands = self._candidates(c.reshape(4, 2).astype(np.float32), tag)
            cands = [k for k in cands if k["dist"] <= max_dist]
            if not cands:
                continue
            if hint is None:
                best = min(cands, key=lambda k: k["reproj"])
            else:
                best = min(cands, key=lambda k: math.hypot(k["x"] - hint[0], k["y"] - hint[1])
                           + 0.5 * abs(wrap(k["yaw"] - hint[2])))
            best["id"] = int(i)
            out.append(best)
        return out

    @staticmethod
    def noise(dist):
        """(sigma_position, sigma_heading) of a tag measurement - grows with distance."""
        return 0.02 + 0.03 * dist, 0.03 + 0.04 * dist
