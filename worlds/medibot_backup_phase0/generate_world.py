#!/usr/bin/env python3
"""
MediBot - Webots world generator.

Run from the project root:   python generate_world.py

Creates
  worlds/medibot.wbt                         hospital world + robot + trolley + AprilTag boards
  worlds/textures/tag_XX.png                 real AprilTag (tag36h11) images
  controllers/medibot_controller/config.json shared graph / tag table used by the controller

Everything the controller needs to know about the world comes from config.json,
so edit the constants below, re-run this script, and the world + controller stay in sync.
"""
import json
import math
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WORLD_DIR = os.path.join(ROOT, "worlds")
TEX_DIR = os.path.join(WORLD_DIR, "textures")
CTRL_DIR = os.path.join(ROOT, "controllers", "medibot_controller")

# ----------------------------------------------------------------------------
# Waypoint graph (metres, 1 Webots unit = 1 m) - same graph as the Wokwi version
# ----------------------------------------------------------------------------
NODES = {
    "HOME": (0.0, 0.0),
    "PHARMACY": (2.0, 0.0),
    "JUNCTION": (4.0, 0.0),
    "WARD-2A": (4.0, 3.0),
    "ICU": (7.0, 0.0),
    "WARD-3B": (7.0, 3.0),
}
EDGES = [
    ("HOME", "PHARMACY"),
    ("PHARMACY", "JUNCTION"),
    ("JUNCTION", "WARD-2A"),
    ("JUNCTION", "ICU"),
    ("WARD-2A", "WARD-3B"),
    ("ICU", "WARD-3B"),
]

# AprilTag boards: (x, y, yaw of FRONT-face normal).  Every board is double sided:
#   front face id = 2*i      (normal points along yaw)
#   back  face id = 2*i + 1  (normal points along yaw + pi)
# Boards stand 0.55 m beside the corridor centre line and face along the corridor so
# the robot sees one face when driving each way.
PI = math.pi
BOARDS = [
    (1.0, 0.55, 0.0),        # HOME -> PHARMACY leg
    (3.0, -0.55, 0.0),       # PHARMACY -> JUNCTION leg
    (5.5, -0.55, 0.0),       # JUNCTION -> ICU leg
    (5.5, 3.55, 0.0),        # WARD-2A -> WARD-3B leg
    (3.45, 1.5, PI / 2),     # JUNCTION -> WARD-2A leg
    (7.55, 1.5, PI / 2),     # ICU -> WARD-3B leg
    (8.0, 3.0, PI),          # behind WARD-3B (seen while approaching it)
    (-1.2, 0.0, 0.0),        # behind HOME (seen while returning)
]
TAG_SIZE = 0.20      # black square side, metres
TAG_PAD = 0.05       # white margin per side, metres
TAG_Z = 0.25         # tag centre height above floor
BOARD_W = TAG_SIZE + 2 * TAG_PAD   # 0.30
BOARD_H = 0.50

# Robot geometry
WHEEL_R = 0.05
WHEEL_X = 0.10
WHEEL_Y = 0.115
CHASSIS = (0.30, 0.18, 0.08)
CAM = {"width": 640, "height": 480, "fov": 1.2, "x": 0.14, "z": 0.18}

TROLLEY_PARK = (-1.0, 3.4)


# ----------------------------------------------------------------------------
# AprilTag texture
# ----------------------------------------------------------------------------
def make_tag_image(tag_id):
    """Return a 600x600 uint8 image: 400 px tag36h11 marker + 100 px white margin."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 80, borderBits=1)
    else:  # OpenCV < 4.7
        marker = cv2.aruco.drawMarker(dictionary, tag_id, 80, borderBits=1)
    marker = cv2.resize(marker, (400, 400), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)


# ----------------------------------------------------------------------------
# .wbt text helpers
# ----------------------------------------------------------------------------
def box_solid(name, x, y, z, sx, sy, sz, rgb, collide=True, yaw=0.0, def_name=None):
    d = f"DEF {def_name} " if def_name else ""
    s = (f"{d}Solid {{\n"
         f"  translation {x:g} {y:g} {z:g}\n"
         f"  rotation 0 0 1 {yaw:g}\n"
         f"  name \"{name}\"\n"
         f"  children [\n"
         f"    Shape {{\n"
         f"      appearance PBRAppearance {{ baseColor {rgb[0]:g} {rgb[1]:g} {rgb[2]:g} roughness 1 metalness 0 }}\n"
         f"      geometry Box {{ size {sx:g} {sy:g} {sz:g} }}\n"
         f"    }}\n"
         f"  ]\n")
    if collide:
        s += f"  boundingObject Box {{ size {sx:g} {sy:g} {sz:g} }}\n"
    return s + "}\n"


def tag_face(tex_name):
    h = BOARD_W / 2
    return (f"Shape {{\n"
            f"  appearance Appearance {{ texture ImageTexture {{ url [ \"textures/{tex_name}\" ] filtering 0 }} }}\n"
            f"  geometry IndexedFaceSet {{\n"
            f"    coord Coordinate {{ point [ 0.006 {-h:g} {-h:g}, 0.006 {h:g} {-h:g}, 0.006 {h:g} {h:g}, 0.006 {-h:g} {h:g} ] }}\n"
            f"    texCoord TextureCoordinate {{ point [ 0 0, 1 0, 1 1, 0 1 ] }}\n"
            f"    coordIndex [ 0 1 2 3 -1 ]\n"
            f"    texCoordIndex [ 0 1 2 3 -1 ]\n"
            f"  }}\n"
            f"}}\n")


def tag_board(i, x, y, yaw):
    front, back = f"tag_{2 * i:02d}.png", f"tag_{2 * i + 1:02d}.png"
    return (f"Solid {{\n"
            f"  translation {x:g} {y:g} {TAG_Z:g}\n"
            f"  rotation 0 0 1 {yaw:.5f}\n"
            f"  name \"tag_board_{i}\"\n"
            f"  children [\n"
            f"    Shape {{\n"
            f"      appearance PBRAppearance {{ baseColor 1 1 1 roughness 1 metalness 0 }}\n"
            f"      geometry Box {{ size 0.01 {BOARD_W:g} {BOARD_H:g} }}\n"
            f"    }}\n"
            f"    {tag_face(front)}"
            f"    Transform {{\n"
            f"      rotation 0 0 1 3.14159265\n"
            f"      children [\n"
            f"        {tag_face(back)}"
            f"      ]\n"
            f"    }}\n"
            f"  ]\n"
            f"  boundingObject Box {{ size 0.01 {BOARD_W:g} {BOARD_H:g} }}\n"
            f"}}\n")


def wheel(tag, x, y):
    return (f"HingeJoint {{\n"
            f"  jointParameters HingeJointParameters {{ axis 0 1 0 anchor {x:g} {y:g} 0 }}\n"
            f"  device [\n"
            f"    RotationalMotor {{ name \"wheel_{tag}\" maxVelocity 20 }}\n"
            f"    PositionSensor {{ name \"ps_{tag}\" }}\n"
            f"  ]\n"
            f"  endPoint Solid {{\n"
            f"    translation {x:g} {y:g} 0\n"
            f"    rotation 1 0 0 1.5708\n"
            f"    children [\n"
            f"      Shape {{\n"
            f"        appearance PBRAppearance {{ baseColor 0.1 0.1 0.1 roughness 1 metalness 0 }}\n"
            f"        geometry Cylinder {{ height 0.03 radius {WHEEL_R:g} }}\n"
            f"      }}\n"
            f"    ]\n"
            f"    name \"wheel_{tag}_solid\"\n"
            f"    boundingObject Cylinder {{ height 0.03 radius {WHEEL_R:g} }}\n"
            f"    physics Physics {{ density -1 mass 0.15 }}\n"
            f"  }}\n"
            f"}}\n")


def sonar(name, x, y, yaw):
    return (f"DistanceSensor {{\n"
            f"  translation {x:g} {y:g} 0.02\n"
            f"  rotation 0 0 1 {yaw:g}\n"
            f"  name \"{name}\"\n"
            f"  type \"sonar\"\n"
            f"  lookupTable [ 0 0 0, 4 4 0 ]\n"
            f"  numberOfRays 3\n"
            f"  aperture 0.3\n"
            f"}}\n")


def robot_node():
    cx, cy, cz = CHASSIS
    wheels = "".join([
        wheel("fl", WHEEL_X, WHEEL_Y), wheel("fr", WHEEL_X, -WHEEL_Y),
        wheel("rl", -WHEEL_X, WHEEL_Y), wheel("rr", -WHEEL_X, -WHEEL_Y),
    ])
    return f"""DEF MEDIBOT Robot {{
  translation 0 0 {WHEEL_R:g}
  rotation 0 0 1 0
  name "MediBot"
  controller "medibot_controller"
  supervisor TRUE
  children [
    Transform {{
      translation 0 0 0.02
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.92 0.94 0.97 roughness 0.6 metalness 0 }}
          geometry Box {{ size {cx:g} {cy:g} {cz:g} }}
        }}
      ]
    }}
    Transform {{
      translation 0.13 0 0.12
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.15 0.35 0.75 roughness 0.6 metalness 0 }}
          geometry Box {{ size 0.02 0.02 0.12 }}
        }}
      ]
    }}
    Camera {{
      translation {CAM['x']:g} 0 {CAM['z']:g}
      name "camera"
      width {CAM['width']}
      height {CAM['height']}
      fieldOfView {CAM['fov']:g}
      near 0.05
    }}
    {sonar("us_front", 0.16, 0.0, 0.0)}
    {sonar("us_left", 0.16, 0.07, 0.7854)}
    {sonar("us_right", 0.16, -0.07, -0.7854)}
    InertialUnit {{ translation 0 0 0.03 name "imu" }}
    Gyro {{ translation 0 0 0.03 name "gyro" }}
    Accelerometer {{ translation 0 0 0.03 name "accel" }}
    HingeJoint {{
      jointParameters HingeJointParameters {{ axis 0 1 0 anchor -0.14 0 0.066 }}
      device [
        RotationalMotor {{ name "lid_motor" minPosition -1.6 maxPosition 0 maxVelocity 3 }}
        PositionSensor {{ name "lid_ps" }}
      ]
      endPoint Solid {{
        translation -0.14 0 0.066
        children [
          Transform {{
            translation 0.14 0 0.004
            children [
              Shape {{
                appearance PBRAppearance {{ baseColor 0.2 0.7 0.35 roughness 0.6 metalness 0 }}
                geometry Box {{ size 0.28 0.16 0.008 }}
              }}
            ]
          }}
        ]
        name "lid"
        boundingObject Transform {{
          translation 0.14 0 0.004
          children [ Box {{ size 0.28 0.16 0.008 }} ]
        }}
        physics Physics {{ density -1 mass 0.05 }}
      }}
    }}
    {wheels}
  ]
  boundingObject Transform {{
    translation 0 0 0.02
    children [ Box {{ size {cx:g} {cy:g} {cz:g} }} ]
  }}
  physics Physics {{ density -1 mass 3 }}
}}
"""


def look_at(pos, target):
    """Webots Viewpoint orientation (axis, angle) looking from pos at target (ENU, z up)."""
    f = np.array(target, float) - np.array(pos, float)
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1.0], f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    R = np.column_stack([f, left, up])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / np.linalg.norm(axis)
    return axis, angle


def build_world():
    parts = []
    parts.append("#VRML_SIM R2023b utf8\n")
    axis, ang = look_at((3.5, -6.5, 7.5), (3.5, 1.5, 0.0))
    parts.append(f"""WorldInfo {{
  title "MediBot hospital delivery"
  basicTimeStep 32
  coordinateSystem "ENU"
}}
Viewpoint {{
  orientation {axis[0]:.5f} {axis[1]:.5f} {axis[2]:.5f} {ang:.5f}
  position 3.5 -6.5 7.5
}}
Background {{ skyColor [ 0.78 0.86 0.95 ] }}
DirectionalLight {{ ambientIntensity 0.75 direction -0.3 0.4 -1 intensity 0.8 }}
""")

    # floor: x -1.5..8.5, y -1..4.5
    parts.append(box_solid("floor", 3.5, 1.75, -0.05, 10.0, 5.5, 0.1, (0.82, 0.84, 0.86)))

    # coloured zone patches (visual only)
    zones = {
        "HOME": (0.3, 0.8, 0.4), "PHARMACY": (0.3, 0.5, 0.9), "JUNCTION": (0.65, 0.65, 0.7),
        "WARD-2A": (0.2, 0.7, 0.7), "ICU": (0.9, 0.3, 0.3), "WARD-3B": (0.2, 0.7, 0.7),
    }
    for n, (x, y) in NODES.items():
        parts.append(box_solid("zone_" + n, x, y, 0.002, 0.9, 0.9, 0.004, zones[n], collide=False))

    # perimeter walls
    wall = (0.93, 0.93, 0.9)
    parts.append(box_solid("wall_s", 3.5, -1.05, 0.3, 10.2, 0.1, 0.6, wall))
    parts.append(box_solid("wall_n", 3.5, 4.55, 0.3, 10.2, 0.1, 0.6, wall))
    parts.append(box_solid("wall_w", -1.55, 1.75, 0.3, 0.1, 5.7, 0.6, wall))
    parts.append(box_solid("wall_e", 8.55, 1.75, 0.3, 0.1, 5.7, 0.6, wall))
    # ward partition wall along y = 3.95 with door gaps at x = 4 and x = 7
    for i, (x0, x1) in enumerate([(-1.5, 3.4), (4.6, 6.4), (7.6, 8.5)]):
        parts.append(box_solid(f"ward_wall_{i}", (x0 + x1) / 2, 3.95, 0.3, x1 - x0, 0.1, 0.6, wall))
    # nurse station block in the middle of the corridor ring
    parts.append(box_solid("nurse_station", 5.5, 1.5, 0.4, 1.6, 1.0, 0.8, (0.55, 0.7, 0.85)))
    # decoration: beds in the wards
    for i, (x, y) in enumerate([(2.0, 4.25), (3.0, 4.25), (5.0, 4.25), (6.0, 4.25), (7.9, 4.25), (0.5, 4.25)]):
        parts.append(box_solid(f"bed_{i}", x, y, 0.2, 0.9, 0.45, 0.4, (0.95, 0.95, 1.0)))

    # AprilTag boards
    tags = []
    for i, (x, y, yaw) in enumerate(BOARDS):
        parts.append(tag_board(i, x, y, yaw))
        tags.append({"id": 2 * i, "x": x, "y": y, "yaw": yaw, "z": TAG_Z, "board": i})
        yb = (yaw + PI + PI) % (2 * PI) - PI
        tags.append({"id": 2 * i + 1, "x": x, "y": y, "yaw": yb, "z": TAG_Z, "board": i})

    # trolley obstacle (parked; the controller teleports it onto the route with key B)
    parts.append(box_solid("trolley", TROLLEY_PARK[0], TROLLEY_PARK[1], 0.35, 0.5, 0.45, 0.7,
                           (0.95, 0.55, 0.1), def_name="TROLLEY"))
    parts.append(robot_node())
    return "".join(parts), tags


def main():
    os.makedirs(TEX_DIR, exist_ok=True)
    os.makedirs(CTRL_DIR, exist_ok=True)
    world, tags = build_world()
    for t in tags:
        cv2.imwrite(os.path.join(TEX_DIR, f"tag_{t['id']:02d}.png"), make_tag_image(t["id"]))
    with open(os.path.join(WORLD_DIR, "medibot.wbt"), "w") as f:
        f.write(world)
    cfg = {
        "nodes": {k: list(v) for k, v in NODES.items()},
        "edges": [list(e) for e in EDGES],
        "tags": tags,
        "tag": {"size": TAG_SIZE},
        "camera": CAM,
        "robot": {"wheel_radius": WHEEL_R, "track": 2 * WHEEL_Y, "start": [0.0, 0.0, 0.0], "z": WHEEL_R},
        "trolley_park": list(TROLLEY_PARK),
    }
    with open(os.path.join(CTRL_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote worlds/medibot.wbt, {len(tags)} tag textures, controllers/medibot_controller/config.json")
    print("Open worlds/medibot.wbt in Webots and press Run.")


if __name__ == "__main__":
    main()
