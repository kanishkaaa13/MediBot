#!/usr/bin/env python3
"""Self-test (no Webots needed): python tests/test_core.py

1. A* routes (normal + with a blocked corridor)
2. Renders each AprilTag board as the robot camera would see it, runs the real detector +
   pose code, and checks the robot pose is recovered.
3. EKF: odometry with drift, corrected by tag fixes.
"""
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "controllers", "medibot_controller"))

import generate_world as gw          # noqa: E402
from medibot_core import CFG, EKF, TagLocalizer, astar, edge_key, wrap   # noqa: E402

fails = 0


def check(name, ok, info=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {info}" if info else ""))
    fails += 0 if ok else 1


# ---------------------------------------------------------------- 1. A*
r = astar("HOME", "WARD-3B")
check("A* default route via WARD-2A", r == ["HOME", "PHARMACY", "JUNCTION", "WARD-2A", "WARD-3B"], str(r))
r = astar("JUNCTION", "WARD-3B", {edge_key("JUNCTION", "WARD-2A")})
check("A* reroute via ICU when corridor blocked", r == ["JUNCTION", "ICU", "WARD-3B"], str(r))
check("A* None when HOME cut off", astar("HOME", "WARD-3B", {edge_key("HOME", "PHARMACY")}) is None)

# ---------------------------------------------------------------- 2. camera synthetic render
cam = CFG["camera"]
W, H = cam["width"], cam["height"]
fx = W / (2 * math.tan(cam["fov"] / 2))
loc = TagLocalizer()
tex_cache = {}


def project(px, py, pz, rx, ry, rth, cam_world_z):
    """World point -> pixel for a camera on robot pose (rx, ry, rth)."""
    dx, dy = px - rx, py - ry
    f = dx * math.cos(rth) + dy * math.sin(rth) - cam["x"]
    l = -dx * math.sin(rth) + dy * math.cos(rth)
    u = pz - cam_world_z
    if f <= 0.05:
        return None
    return (W / 2 - fx * l / f, H / 2 - fx * u / f)


def render(rx, ry, rth, tag):
    """Render the tag texture of `tag` as seen from the robot (plain white board + grey floor)."""
    img = np.full((H, W), 120, np.uint8)
    tex = gw.make_tag_image(tag["id"])
    hw = gw.BOARD_W / 2
    Y = tag["yaw"]
    right = (-math.sin(Y), math.cos(Y))                 # viewer's right on the face (see generator)
    cz = gw.WHEEL_R + cam["z"]
    pts = []
    for sr, su in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:  # BL, BR, TR, TL as seen by the viewer
        px = tag["x"] + sr * hw * right[0]
        py = tag["y"] + sr * hw * right[1]
        pz = tag["z"] + su * hw
        p = project(px, py, pz, rx, ry, rth, cz)
        if p is None:
            return None
        pts.append(p)
    src = np.float32([[0, 599], [599, 599], [599, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(src, np.float32(pts))
    cv2.warpPerspective(tex, M, (W, H), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
    return img


# Dynamically select test tags and poses from CFG["tags"] across corridors and endpoints
tests = []
for tag in CFG["tags"][::2]:  # pick front face of each board across the hospital
    d = 1.5
    rx = tag["x"] + d * math.cos(tag["yaw"])
    ry = tag["y"] + d * math.sin(tag["yaw"])
    rth = wrap(tag["yaw"] + math.pi)
    tests.append(((rx, ry, rth), tag["id"]))

# Also test a perturbed pose (with lateral and heading offset)
t0 = CFG["tags"][0]
tests.append(((t0["x"] + 1.2 * math.cos(t0["yaw"]) - 0.05 * math.sin(t0["yaw"]),
               t0["y"] + 1.2 * math.sin(t0["yaw"]) + 0.05 * math.cos(t0["yaw"]),
               wrap(t0["yaw"] + math.pi + 0.06)), t0["id"]))
for (rx, ry, rth), tid in tests:
    tag = loc.tags[tid]
    img = render(rx, ry, rth, tag)
    if img is None:
        check(f"tag {tid} render", False)
        continue
    m = [d for d in loc.measure(img, hint=(rx + 0.15, ry - 0.1, rth + 0.06)) if d["id"] == tid]
    if not m:
        check(f"tag {tid} from ({rx},{ry}) detected", False)
        continue
    m = m[0]
    pe = math.hypot(m["x"] - rx, m["y"] - ry)
    he = abs(wrap(m["yaw"] - rth))
    check(f"tag {tid} pose recovered", pe < 0.08 and he < 0.1, f"pos err {pe * 100:.1f} cm, yaw err {math.degrees(he):.1f} deg, d={m['dist']:.2f} m")

# ---------------------------------------------------------------- 3. EKF
rng = np.random.default_rng(1)
ekf = EKF([0, 0, 0])
true = np.array([0.0, 0.0, 0.0])
before = after = None
for k in range(400):                       # 0.3 m/s * 12.8 s straight, gyro bias 0.004 rad/s, 3 % odometry
    ds = 0.3 * 0.032
    true += [ds * math.cos(true[2]), ds * math.sin(true[2]), 0]
    ekf.predict(ds * 1.03, 0.004 * 0.032)
before = np.hypot(*(ekf.x[:2] - true[:2])) * 100
for _ in range(5):                         # tag stays in view for several frames
    ekf.update([true[0] + rng.normal(0, .02), true[1] + rng.normal(0, .02), true[2]], 0.07, 0.1)
after = np.hypot(*(ekf.x[:2] - true[:2])) * 100
check("EKF tag update reduces error", after < 0.4 * before, f"{before:.1f} cm -> {after:.1f} cm")

print("\nALL TESTS PASSED" if fails == 0 else f"\n{fails} TEST(S) FAILED")
sys.exit(1 if fails else 0)
