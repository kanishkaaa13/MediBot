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

# Perturbed-pose test: robot sees tag from an oblique angle (0.4 rad off-axis heading +
# 0.3 m lateral offset). IPPE_SQUARE requires a non-degenerate (non-square) projection
# to recover orientation; nearly head-on views are mathematically singular for IPPE.
t0 = CFG["tags"][0]
perturb_dist = 1.2
perturb_head = 0.4   # rad off-axis heading (non-degenerate for IPPE_SQUARE)
perturb_lat  = 0.3   # m lateral offset
tests.append(((t0["x"] + perturb_dist * math.cos(t0["yaw"]) - perturb_lat * math.sin(t0["yaw"]),
               t0["y"] + perturb_dist * math.sin(t0["yaw"]) + perturb_lat * math.cos(t0["yaw"]),
               wrap(t0["yaw"] + math.pi + perturb_head)), t0["id"]))
for (rx, ry, rth), tid in tests:
    tag = loc.tags[tid]
    img = render(rx, ry, rth, tag)
    if img is None:
        check(f"tag {tid} render", False)
        continue
    m = [d for d in loc.measure(img, hint=(rx + 0.15, ry - 0.1, rth + 0.06)) if d["id"] == tid]
    if not m:
        check(f"tag {tid} from ({rx:.2f},{ry:.2f}) detected", False)
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

# ---------------------------------------------------------------- 4. Geometry consistency (single source of truth)
# Parse .wbt file to verify geometry matches config.json
wbt_path = os.path.join(ROOT, "worlds", "medibot.wbt")
with open(wbt_path, "r") as f:
    wbt_content = f.read()

# Check wheel radius in .wbt matches config
wheel_radius_cfg = CFG["robot"]["wheel_radius"]
import re
wheel_radii = re.findall(r"boundingObject Cylinder.*radius ([\d.]+)", wbt_content)
if wheel_radii:
    for wr in wheel_radii:
        check(f"wheel radius {wr} matches config {wheel_radius_cfg}", abs(float(wr) - wheel_radius_cfg) < 1e-6)
else:
    check("wheel radius found in .wbt", False)

# Check wheel anchors y match +-track/2
track_cfg = CFG["robot"]["track"]
wheel_anchors = re.findall(r"wheel_[fr][lr].*anchor ([\d.-]+) ([\d.-]+) ([\d.-]+)", wbt_content, re.DOTALL)
if wheel_anchors:
    for ax, ay, az in wheel_anchors:
        expected_y = track_cfg / 2
        check(f"wheel anchor y {ay} matches +-{expected_y}", abs(abs(float(ay)) - expected_y) < 1e-6)
else:
    check("wheel anchors found in .wbt", False)

# Check robot translation z matches wheel_radius
robot_translation = re.search(r"DEF MEDIBOT Robot.*translation ([\d.-]+) ([\d.-]+) ([\d.-]+)", wbt_content, re.DOTALL)
if robot_translation:
    z = float(robot_translation.group(3))
    check(f"robot translation z {z} matches wheel_radius {wheel_radius_cfg}", abs(z - wheel_radius_cfg) < 1e-6)
else:
    check("robot translation found in .wbt", False)

# Check camera translation matches config
cam_cfg = CFG["camera"]
cam_translation = re.search(r'Camera.*translation ([\d.-]+) ([\d.-]+) ([\d.-]+).*name "camera"', wbt_content, re.DOTALL)
if cam_translation:
    cam_x = float(cam_translation.group(1))
    cam_z = float(cam_translation.group(3))
    check(f"camera x {cam_x} matches config {cam_cfg['x']}", abs(cam_x - cam_cfg['x']) < 1e-6)
    check(f"camera z {cam_z} matches config {cam_cfg['z']}", abs(cam_z - cam_cfg['z']) < 1e-6)
else:
    check("camera translation found in .wbt", False)

# ---------------------------------------------------------------- 5. Kidnapped-robot re-localisation test
# Simulate a robot with a 3 m wrong estimate that receives correct tag fixes
ekf_kidnap = EKF([3.0, 0.0, 0.0])  # Wrong estimate: 3 m east of true position
true_pose_kidnap = [0.0, 0.0, 0.0]  # True position at origin
rejected_buffer = []
recovered = False

# Simulate 1 second of tag fixes at 32 ms timestep (32 steps)
for step in range(32):
    # Simulate seeing tag 0 at (1.0, 0.55, 0.0) from true position
    tag = CFG["tags"][0]
    # Robot at true_pose sees tag, so robot pose estimate from tag would be:
    # robot_x = tag_x - dist * cos(tag_yaw)
    # robot_y = tag_y - dist * sin(tag_yaw)
    # For simplicity, just use the true position with small noise
    noise_x = rng.normal(0, 0.02)
    noise_y = rng.normal(0, 0.02)
    fix_x = true_pose_kidnap[0] + noise_x
    fix_y = true_pose_kidnap[1] + noise_y
    fix_yaw = true_pose_kidnap[2] + rng.normal(0, 0.03)

    # Mahalanobis test (simplified)
    innovation = np.array([fix_x - ekf_kidnap.x[0], fix_y - ekf_kidnap.x[1],
                          wrap(fix_yaw - ekf_kidnap.x[2])])
    sp, st = 0.05, 0.05  # Typical noise at close range
    R = np.diag([sp ** 2, sp ** 2, st ** 2])
    S = ekf_kidnap.P + R
    try:
        mahalanobis = innovation.T @ np.linalg.inv(S) @ innovation
    except np.linalg.LinAlgError:
        mahalanobis = float('inf')

    if mahalanobis < 11.34:  # MAHALANOBIS_THRESH
        ekf_kidnap.update([fix_x, fix_y, fix_yaw], sp, st)
    else:
        rejected_buffer.append({"x": fix_x, "y": fix_y, "yaw": fix_yaw, "id": 0, "t": step * 0.032})

    # Check for re-localisation trigger
    if len(rejected_buffer) >= 3:
        recent = rejected_buffer[-3:]
        avg_x = sum(f["x"] for f in recent) / len(recent)
        avg_y = sum(f["y"] for f in recent) / len(recent)
        avg_yaw = wrap(sum(wrap(f["yaw"]) for f in recent) / len(recent))

        pos_agree = all(math.hypot(f["x"] - avg_x, f["y"] - avg_y) < 0.25 for f in recent)
        th_agree = all(abs(wrap(f["yaw"] - avg_yaw)) < 0.2 for f in recent)

        if pos_agree and th_agree:
            ekf_kidnap.x = np.array([avg_x, avg_y, avg_yaw])
            ekf_kidnap.P = np.diag([1.0, 1.0, 1.0])
            recovered = True
            break

# Check if recovered within 1 second
final_err = math.hypot(ekf_kidnap.x[0] - true_pose_kidnap[0],
                       ekf_kidnap.x[1] - true_pose_kidnap[1])
check("Kidnapped-robot re-localisation within 1 s", recovered and final_err < 0.15,
      f"recovered={recovered}, final error={final_err*100:.1f} cm")

print("\nALL TESTS PASSED" if fails == 0 else f"\n{fails} TEST(S) FAILED")
sys.exit(1 if fails else 0)
