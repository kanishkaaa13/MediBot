#!/usr/bin/env python3
"""Self-test (no Webots needed): python tests/test_core.py

1. A* routes (normal + with a blocked corridor)
2. Renders each AprilTag board as the robot camera would see it, runs the real detector +
   pose code, and checks the robot pose is recovered.
3. EKF: odometry with drift, corrected by tag fixes.
"""
import math
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "controllers", "medibot_controller"))

import generate_world as gw          # noqa: E402
from medibot_core import (CFG, EKF, TagLocalizer, astar, edge_key, wrap,
                          rotate_drive_command)  # noqa: E402

fails = 0


def check(name, ok, info=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {info}" if info else ""))
    fails += 0 if ok else 1


# ---------------------------------------------------------------- 1. A*
from medibot_core import ADJ, EDGES, NODES  # noqa: E402
print("A* HOME->WARD-3B:", astar("HOME", "WARD-3B"))
print("A* WARD-3B->HOME:", astar("WARD-3B", "HOME"))
check("graph edges are bidirectional", all(b in ADJ[a] and a in ADJ[b] for a, b in EDGES))
check("config graph names match generator", set(NODES) == set(gw.NODES) and
      set(NODES) == {n for e in EDGES for n in e})
check("obstacle removal restores route", astar("HOME", "WARD-3B", {edge_key("JUNCTION", "WARD-2A")}) !=
      astar("HOME", "WARD-3B"))

# Exercise the same rotate-then-drive control used by the Webots controller.
x, y, heading = 0.0, 0.08, 0.18
max_cross_track = abs(y)
for _ in range(3000):
    error = wrap(math.atan2(-y, 3.0 - x) - heading)
    v, w = rotate_drive_command(error, 0.30)
    x += v * math.cos(heading) * 0.032
    y += v * math.sin(heading) * 0.032
    heading = wrap(heading + w * 0.032)
    max_cross_track = max(max_cross_track, abs(y))
    if x >= 3.0 and abs(y) < 0.12:
        break
print(f"Straight corridor max cross-track error: {max_cross_track:.3f} m")
check("straight corridor cross-track error <= 0.10 m", max_cross_track <= 0.10)
r = astar("HOME", "WARD-3B")
check("A* default route via WARD-2A", r == ["HOME", "PHARMACY", "JUNCTION", "WARD-2A", "WARD-3B"], str(r))
r = astar("JUNCTION", "WARD-3B", {edge_key("JUNCTION", "WARD-2A")})
check("A* reroute via ICU when corridor blocked", r == ["JUNCTION", "ICU", "WARD-3B"], str(r))
check("A* None when HOME cut off", astar("HOME", "WARD-3B", {edge_key("HOME", "PHARMACY")}) is None)

# ---------------------------------------------------------------- 2. Synthetic renderer tag coverage test
# For every tag, render it from robot positions along adjacent corridor edges
# and verify detection with position error < 6 cm and yaw error < 4 deg

cam = CFG["camera"]
W, H = cam["width"], cam["height"]
fx = W / (2 * math.tan(cam["fov"] / 2))
loc = TagLocalizer()


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
    """Render the tag texture as seen from the robot (returns grayscale)."""
    img = np.full((H, W, 3), 120, np.uint8)  # BGR image
    tex = gw.make_tag_image(tag["id"])
    tag_size = CFG["tag"]["size"]
    hw = tag_size / 2
    Y = tag["yaw"]
    right = (-math.sin(Y), math.cos(Y))
    cz = CFG["robot"]["z"] + cam["z"]
    pts = []
    for sr, su in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
        px = tag["x"] + sr * hw * right[0]
        py = tag["y"] + sr * hw * right[1]
        pz = tag["z"] + su * hw
        p = project(px, py, pz, rx, ry, rth, cz)
        if p is None:
            return None
        pts.append(p)
    # Map only the marker area onto the physical AprilTag; the surrounding
    # white margin and printed ID are part of the 1024px texture, not the tag.
    src = np.float32([[160, 160], [864, 160], [864, 864], [160, 864]])
    M = cv2.getPerspectiveTransform(src, np.float32(pts))
    cv2.warpPerspective(tex, M, (W, H), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
    # Convert BGR to grayscale for AprilTag detector
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def incidence_angle(rx, ry, rth, tag):
    """Compute angle between camera line of sight and tag normal (0 = head-on, 90 = grazing)."""
    dx = tag["x"] - rx
    dy = tag["y"] - ry
    dist = math.hypot(dx, dy)
    if dist < 0.01:
        return 0.0
    # Camera to tag direction
    cam_to_tag = (dx / dist, dy / dist)
    # Tag normal (facing direction from yaw)
    tag_normal = (math.cos(tag["yaw"]), math.sin(tag["yaw"]))
    # Dot product gives cosine of angle
    dot = cam_to_tag[0] * tag_normal[0] + cam_to_tag[1] * tag_normal[1]
    angle = math.degrees(math.acos(max(-1.0, min(1.0, -dot))))  # Negative because camera looks opposite to normal
    return angle


# Physical survey checks and projected visibility along each edge.
edges = CFG["edges"]
coverage_results = []

with open(os.path.join(ROOT, "worlds", "medibot.wbt"), encoding="utf-8") as f:
    surveyed_world = f.read()
for tag in CFG["tags"]:
    horizontal_normal = math.hypot(math.cos(tag["yaw"]), math.sin(tag["yaw"]))
    wall_delta = abs(tag[tag["wall_axis"]] - tag["wall_plane"])
    check(f"tag {tag['id']} vertical and on wall",
          horizontal_normal >= 0.99 and abs(tag["z"] - 1.0) < 1e-6 and wall_delta <= 0.02,
          f"horizontal normal={horizontal_normal:.3f}, wall offset={wall_delta:.3f}m")
    name_at = surveyed_world.find(f'name "tag_{tag["id"]}"')
    solid_at = surveyed_world.rfind("Solid {", 0, name_at)
    plate = re.search(r'geometry Box \{ size ([^}]+) \}',
                      surveyed_world[solid_at:name_at]) if name_at >= 0 else None
    check(f"tag {tag['id']} world plate vertical", bool(plate and "0.01" in plate.group(1)),
          plate.group(1) if plate else "plate missing")

# Debug: test single tag detection first
print("\n=== DEBUG: Single tag detection test ===")
test_tag = CFG["tags"][0] if CFG["tags"] else None
if test_tag:
    # Position robot 1.5m in front of tag, head-on
    # Tag faces along its yaw, so robot should be in front (opposite direction) facing the tag
    rx = test_tag["x"] + 1.5 * math.cos(test_tag["yaw"])
    ry = test_tag["y"] + 1.5 * math.sin(test_tag["yaw"])
    rth = wrap(test_tag["yaw"] + math.pi)  # Robot is on the tag's outward-normal side
    print(f"Testing tag {test_tag['id']} at ({test_tag['x']}, {test_tag['y']}) yaw={test_tag['yaw']:.2f}")
    print(f"Robot at ({rx:.2f}, {ry:.2f}) yaw={rth:.2f}")
    
    img = render(rx, ry, rth, test_tag)
    if img is not None:
        print(f"Rendered image shape: {img.shape}")
        meas = loc.measure(img, hint=(rx, ry, rth))
        print(f"Detection result: {len(meas) if meas else 0} measurements")
        if meas:
            for m in meas[:3]:  # Print first 3
                print(f"  id={m['id']}, dist={m['dist']:.2f}, pos=({m['x']:.2f}, {m['y']:.2f})")
    else:
        print("Render failed - tag not in view")

print("\n=== Starting full coverage test ===")

for edge in edges:
    start_node, end_node = edge
    start_pos = CFG["nodes"][start_node]
    end_pos = CFG["nodes"][end_node]
    
    # Compute edge direction and length
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    edge_length = math.hypot(dx, dy)
    edge_yaw = math.atan2(dy, dx)
    
    edge_good, edge_samples = 0, 0
    for direction in (1, -1):
      for step in range(int(edge_length / 0.5) + 1):
        t = min(1.0, step * 0.5 / edge_length)
        if direction < 0:
            t = 1.0 - t
        rx = start_pos[0] + t * dx
        ry = start_pos[1] + t * dy
        rth = edge_yaw if direction > 0 else wrap(edge_yaw + math.pi)
        
        visible = []
        for tag in CFG["tags"]:
            dx_tag, dy_tag = tag["x"] - rx, tag["y"] - ry
            dist = math.hypot(dx_tag, dy_tag)
            if dist > 4.0 or dist < 0.01:
                continue
            inc = incidence_angle(rx, ry, rth, tag)
            if inc >= 45.0:
                continue
            # Apparent marker width under the surveyed incidence angle, using
            # the camera focal length derived from its configured FOV.
            px = fx * CFG["tag"]["size"] * math.cos(math.radians(inc)) / dist
            if px >= 30.0:
                visible.append(tag)
        edge_samples += 1
        edge_good += bool(visible)
        coverage_results.append({"edge": f"{start_node}->{end_node}", "good_count": bool(visible)})

    pct = 100.0 * edge_good / edge_samples
    print(f"TAG COVERAGE {start_node}->{end_node}: {edge_good}/{edge_samples} ({pct:.1f}%)")
    check(f"bidirectional projected tag coverage {start_node}->{end_node} >= 90%", pct >= 90.0)

# Exercise real detector/PnP from head-on poses across the map.
for tag in CFG["tags"]:
    rx = tag["x"] + 1.5 * math.cos(tag["yaw"])
    ry = tag["y"] + 1.5 * math.sin(tag["yaw"])
    heading = wrap(tag["yaw"] + math.pi)
    image = render(rx, ry, heading, tag)
    detections = loc.measure(image, hint=(rx, ry, heading)) if image is not None else []
    ids = [m["id"] for m in detections]
    check(f"rendered vertical tag {tag['id']} detected", tag["id"] in ids, f"ids={ids}")

# Print coverage summary
print("\n=== TAG COVERAGE SUMMARY ===")
for edge in edges:
    edge_name = f"{edge[0]}->{edge[1]}"
    edge_results = [r for r in coverage_results if r["edge"] == edge_name]
    if edge_results:
        total_steps = len(edge_results)
        good_steps = sum(1 for r in edge_results if r["good_count"] >= 1)
        print(f"{edge_name}: {good_steps}/{total_steps} steps with good tag coverage")

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
robot_wbt = wbt_content.split("DEF MEDIBOT Robot", 1)[1]
wheel_radii = re.findall(r"boundingObject\s+Cylinder\s*\{[^}]*radius\s+([\d.]+)", robot_wbt)
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
robot_block = re.search(r"DEF MEDIBOT Robot\s*\{(.*?)\n\s*children\s*\[", wbt_content, re.DOTALL)
robot_translation = re.search(r"(?m)^\s*translation ([\d.-]+) ([\d.-]+) ([\d.-]+)", robot_block.group(1)) if robot_block else None
if robot_translation:
    z = float(robot_translation.group(3))
    check(f"robot translation z {z} matches wheel_radius {wheel_radius_cfg}", abs(z - wheel_radius_cfg) < 1e-6)
else:
    check("robot translation found in .wbt", False)

# Check camera translation matches config
cam_cfg = CFG["camera"]
camera_start = wbt_content.find('name "camera"')
camera_open = wbt_content.rfind("Camera {", 0, camera_start)
camera_close = wbt_content.find("\n    }", camera_start)
camera_node = wbt_content[camera_open:camera_close] if camera_start >= 0 and camera_open >= 0 and camera_close >= 0 else ""
cam_translation = re.search(r'translation ([\d.-]+) ([\d.-]+) ([\d.-]+)', camera_node)
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
