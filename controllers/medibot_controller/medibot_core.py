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
HEADING_TOL = math.radians(10)
HEADING_KP = 2.5
MAX_ANGULAR_SPEED = 1.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotate_drive_command(error, vmax):
    """Rotate first, then drive with bounded proportional heading correction."""
    error = wrap(error)
    angular = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, HEADING_KP * error))
    if abs(error) > HEADING_TOL:
        return 0.0, angular
    return vmax, angular


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
        fy = fx  # Webots camera pixels are square; fieldOfView is horizontal.
        self.K = np.array([[fx, 0, self.w / 2], [0, fy, self.h / 2], [0, 0, 1]], np.float64)
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
        """Both IPPE solutions for one tag (near head-on views are ambiguous) as robot-pose dicts.

        IPPE_SQUARE can return n[2] > 0 (normal toward camera rather than away) for nearly
        square projections with slight asymmetry. In that case we fall back to the solution
        with the most negative n[2] so the hint-based disambiguator can still pick the right one.
        """
        n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self.obj, corners, self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        all_raw = []
        cands = []
        for rvec, tvec, err in zip(rvecs, tvecs, errs):
            Rm, _ = cv2.Rodrigues(rvec)
            n = Rm[:, 2]              # tag normal in camera frame (x right, y down, z fwd)
            t = tvec.reshape(3)
            f = t[2] + self.cam_x     # tag position in robot frame: forward
            l = -t[0]                 # ... and left
            # Vertical tag normal is horizontal in world coordinates. In camera
            # coordinates n=(right, down, forward); solve its relative yaw.
            phi = math.atan2(-n[0], -n[2])
            yaw = wrap(tag["yaw"] - phi)
            cs, sn = math.cos(yaw), math.sin(yaw)
            entry = {
                "x": tag["x"] - (f * cs - l * sn),
                "y": tag["y"] - (f * sn + l * cs),
                "yaw": yaw,
                "dist": float(np.linalg.norm(t)),
                "bearing": math.atan2(l, f),
                "reproj": float(np.ravel(err)[0]),
                "_nz": float(n[2]),   # tag normal z; should be < -0.25 for valid (facing camera)
            }
            all_raw.append(entry)
            if n[2] <= -0.25:         # normal points toward camera: valid geometry
                cands.append(entry)

        # Fallback: if IPPE_SQUARE degenerated (e.g. near-square perturbed view), keep the
        # solution with the most negative n[2]. The hint will still pick the better one.
        if not cands and all_raw:
            cands = [min(all_raw, key=lambda e: -e["_nz"])]

        # Remove internal key before returning
        for c in cands:
            c.pop("_nz", None)
        return cands

    def measure(self, gray, hint=None, max_dist=4.0, max_reproj=8.0):
        """Return list of {id, x, y, yaw, dist, incidence}: robot-pose estimates from visible tags.

        `hint` = current (x, y, yaw) estimate used to:
          - skip tags that are geometrically too far away (fast reject),
          - compute the incidence angle (angle between camera ray and tag normal),
          - disambiguate the two IPPE pose solutions (pick the one closest to hint).

        Filters applied:
          * incidence > 55 deg  → reject (grazing view gives wildly wrong pose)
          * reprojection error  > max_reproj pixels → reject (bad corner detection)

        The incidence angle is computed from the HINT position (current EKF estimate), not
        from the tag-derived candidate position.  Using the candidate for incidence would be
        circular: a grazing view produces a wrong candidate, whose direction to the tag then
        appears head-on, defeating the filter.
        """
        # Run the detector ONCE for the whole image (not per-tag).
        corners_all, ids_all, _ = self._detect(gray)
        if ids_all is None:
            return []

        results = []
        for tag in self.tags.values():
            tx, ty, tyaw = tag["x"], tag["y"], tag["yaw"]

            # --- 1. Distance pre-filter (uses hint if available) ---
            ref_x = hint[0] if hint else tx
            ref_y = hint[1] if hint else ty
            if math.hypot(ref_x - tx, ref_y - ty) > max_dist:
                continue

            # --- 2. Find this tag's corners in the detector output ---
            corners_for_tag = None
            for i, cid in enumerate(ids_all):
                if int(cid) == tag["id"]:
                    corners_for_tag = corners_all[i]
                    break
            if corners_for_tag is None:
                continue

            pix = corners_for_tag.reshape(4, 2)
            edge_px = min(float(np.linalg.norm(pix[(i + 1) % 4] - pix[i])) for i in range(4))
            if edge_px < 20.0:
                continue

            # --- 3. Solve for pose (both IPPE solutions) ---
            tag_cands = self._candidates(corners_for_tag, tag)

            # --- 4. Reprojection filter ---
            tag_cands = [c for c in tag_cands if c["reproj"] <= max_reproj]
            if not tag_cands:
                continue

            # --- 5. Hint-based disambiguation: pick candidate closest to hint ---
            if hint and len(tag_cands) > 1:
                tag_cands = [min(tag_cands,
                                 key=lambda c: math.hypot(c["x"] - hint[0], c["y"] - hint[1]))]

            cand = tag_cands[0]

            # --- 6. Incidence angle from HINT position (not from candidate) ---
            # Use the hint if available; otherwise fall back to candidate (first call).
            obs_x = hint[0] if hint else cand["x"]
            obs_y = hint[1] if hint else cand["y"]
            dx = tx - obs_x
            dy = ty - obs_y
            dist_hint = math.hypot(dx, dy)
            if dist_hint < 0.01:
                inc = 0.0
            else:
                # cam_to_tag: unit vector from observer toward tag
                cam_to_tag = (dx / dist_hint, dy / dist_hint)
                # tag_normal: direction the tag face points (away from the wall)
                tag_normal = (math.cos(tyaw), math.sin(tyaw))
                # For a head-on view (tag facing robot), cam_to_tag ≈ −tag_normal → dot ≈ −1 → inc ≈ 0°
                dot = cam_to_tag[0] * tag_normal[0] + cam_to_tag[1] * tag_normal[1]
                inc = math.degrees(math.acos(max(-1.0, min(1.0, -dot))))

            if inc > 60.0:
                continue

            # Discard isolated pose jumps before the EKF blend. A later stable
            # observation can still relocalize through the controller buffer.
            if hint and math.hypot(cand["x"] - hint[0], cand["y"] - hint[1]) > 0.5:
                continue

            cand["id"] = tag["id"]
            cand["incidence"] = inc
            cand["pixel_size"] = edge_px
            results.append(cand)

        return results

    @staticmethod
    def noise(dist):
        """(sigma_position, sigma_heading) of a tag measurement - grows with distance."""
        return 0.02 + 0.03 * dist, 0.03 + 0.04 * dist
