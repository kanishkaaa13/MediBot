#!/usr/bin/env python3
"""Quick validation of config.json and .wbt structure."""
import json, sys, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- config.json checks ----
cfg_path = os.path.join(ROOT, "controllers", "medibot_controller", "config.json")
cfg = json.load(open(cfg_path))

required_keys = ["nodes", "edges", "tags", "tag", "camera", "face_camera", "robot", "trolley_park"]
robot_keys = ["wheel_radius", "track", "start", "z", "footprint"]

all_ok = True
for k in required_keys:
    if k in cfg:
        print(f"OK   config[{k!r}]")
    else:
        print(f"FAIL config missing {k!r}")
        all_ok = False

for k in robot_keys:
    if k in cfg["robot"]:
        print(f"OK   config.robot[{k!r}] = {cfg['robot'][k]}")
    else:
        print(f"FAIL config.robot missing {k!r}")
        all_ok = False

print(f"\nNodes : {list(cfg['nodes'].keys())}")
print(f"Edges : {cfg['edges']}")
print(f"Tags  : {len(cfg['tags'])} entries")
print(f"Camera: w={cfg['camera']['width']} h={cfg['camera']['height']} fov={cfg['camera']['fov']}")
print(f"Face  : w={cfg['face_camera']['width']} h={cfg['face_camera']['height']}")
print(f"WHEEL_R={cfg['robot']['wheel_radius']}  TRACK={cfg['robot']['track']}")
print(f"Start : {cfg['robot']['start']}")

# ---- .wbt file checks ----
wbt_path = os.path.join(ROOT, "worlds", "medibot.wbt")
with open(wbt_path) as f:
    wbt = f.read()

print("\n---- .wbt checks ----")
checks = {
    "#VRML_SIM R2025a": "R2025a header",
    "coordinateSystem \"ENU\"": "ENU coordinate system",
    "DEF MEDIBOT Robot": "MEDIBOT robot node",
    "DEF TROLLEY": "TROLLEY solid",
    "controller \"medibot_controller\"": "controller assigned",
    "supervisor TRUE": "supervisor mode",
    "name \"camera\"": "navigation camera",
    "name \"dashboard\"": "robot dashboard Display",
    "width 480\n      height 320": "480x320 dashboard resolution",
    "name \"face_cam\"": "face camera",
    "name \"status_led\"": "status LED",
    "name \"lid_1\"": "lid_1 motor",
    "name \"lid_2\"": "lid_2 motor",
    "name \"lid_3\"": "lid_3 motor",
    "name \"wheel_fl\"": "wheel_fl motor",
    "name \"wheel_fr\"": "wheel_fr motor",
    "name \"wheel_rl\"": "wheel_rl motor",
    "name \"wheel_rr\"": "wheel_rr motor",
    "name \"us_front\"": "us_front sonar",
    "name \"us_left\"": "us_left sonar",
    "name \"us_right\"": "us_right sonar",
    "name \"gyro\"": "gyro",
    "name \"accel\"": "accelerometer",
    "name \"imu\"": "IMU",
    "name \"led_lock_1\"": "led_lock_1",
    "name \"led_lock_2\"": "led_lock_2",
    "name \"led_lock_3\"": "led_lock_3",
    "DEF CARGO_1": "CARGO_1 solid",
    "DEF CARGO_2": "CARGO_2 solid",
    "DEF CARGO_3": "CARGO_3 solid",
    "PointLight": "ceiling lights",
    "tag_00.png": "at least one AprilTag texture",
}
for pattern, label in checks.items():
    found = pattern in wbt
    status = "OK  " if found else "FAIL"
    print(f"{status} {label!r:35s}  [{pattern!r}]")
    if not found:
        all_ok = False

# Check tag board count
tag_boards = len(re.findall(r'tag_board_', wbt))
print(f"\nTag boards in .wbt: {tag_boards} (expected 12)")

# Check textures exist
print("\n---- Texture files ----")
tex_dir = os.path.join(ROOT, "worlds", "textures")
expected_textures = (
    [f"tag_{i:02d}.png" for i in range(0, 24, 2)] +
    ["sign_pharmacy.png", "sign_icu.png", "sign_ward_2a.png", "sign_ward_3b.png",
     "sign_nurse_station.png", "sign_junction_dir.png", "sign_reception.png",
     "sign_home_charging.png", "medibot_logo.png"]
)
for t in expected_textures:
    path = os.path.join(tex_dir, t)
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    status = "OK  " if exists else "MISS"
    print(f"{status} {t} ({size} bytes)")
    if not exists:
        all_ok = False

print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
