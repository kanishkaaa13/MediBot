#!/usr/bin/env python3
"""
MediBot - Webots hospital world generator (Phase 2).

Generates:
  worlds/medibot.wbt                         Hospital world + Phase 2 MediBot robot (mast, 3 compartments, 2 cameras)
  worlds/textures/tag_XX.png                 Real AprilTag (tag36h11) images
  worlds/textures/sign_*.png                 High-res hospital signs generated with Pillow
  worlds/textures/medibot_logo.png           MediBot brand badge
  controllers/medibot_controller/config.json Single source of truth for graph, tags, robot

Run from the project root:   py generate_world.py
"""
import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
WORLD_DIR = os.path.join(ROOT, "worlds")
TEX_DIR = os.path.join(WORLD_DIR, "textures")
CTRL_DIR = os.path.join(ROOT, "controllers", "medibot_controller")

# ----------------------------------------------------------------------------
# Waypoint graph (metres, 1 Webots unit = 1 m) - KEPT FIXED
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

PI = math.pi
TAG_SIZE = 0.20      # black square side, metres
TAG_PAD = 0.05       # white margin per side, metres
TAG_Z = 0.25         # tag centre height above floor (aligned with navigation camera)
BOARD_W = TAG_SIZE + 2 * TAG_PAD   # 0.30 m
BOARD_H = 0.50

# Robot geometry (Phase 2 MediBot 4WD platform)
# Footprint: 0.44 m long x 0.31 m wide with wheels (<= 0.5 x 0.4 m)
WHEEL_R = 0.06
WHEEL_X = 0.14
WHEEL_Y = 0.14
TRACK = 2 * WHEEL_Y       # 0.28 m
CHASSIS = (0.44, 0.24, 0.12)

# Navigation Camera: mounted on lower mast at tag height (WHEEL_R + CAM['z'] = 0.06 + 0.19 = 0.25 m = TAG_Z)
CAM = {"width": 640, "height": 480, "fov": 1.2, "x": 0.22, "z": 0.19}

# Face Camera: mounted on mast head at ~0.95 m height looking forward & slightly down
FACE_CAM = {"width": 640, "height": 480, "fov": 0.9, "x": 0.19, "z": 0.88}

# Safe trolley parking spot in reception alcove (away from transit lines)
TROLLEY_PARK = (-1.5, 1.8)


# ----------------------------------------------------------------------------
# Automatic Wall-Mounted AprilTag Placement
# ----------------------------------------------------------------------------
def generate_wall_tags():
    """Generate wall-mounted tag plates based on graph edges and room geometry.

    Returns list of (x, y, z, yaw, zone_name, tag_id) where:
    - x, y, z: tag centre position
    - yaw: direction of plate's outward normal (facing corridor)
    - zone_name: descriptive location name
    - tag_id: assigned ID
    """
    tags = []
    tag_id = 0

    # Wall positions (based on world geometry)
    # South corridor wall: y = -0.70
    # North corridor wall: y = 3.70
    # West wall: x = -2.60
    # East wall: x = 9.20
    # Door gaps: Pharmacy (x=1.4 to 2.6), ICU (x=6.4 to 7.6), Ward 2A (x=3.4 to 4.6), Ward 3B (x=6.4 to 7.6)

    def add_tag(x, y, yaw, zone):
        nonlocal tag_id
        tags.append((x, y, TAG_Z, yaw, zone, tag_id))
        tag_id += 1

    # Zone 0-9: South corridor (HOME -> PHARMACY -> JUNCTION -> ICU)
    # South wall (y = -0.70), tags facing north (yaw = 0)
    # Door gaps: Pharmacy (1.4-2.6), ICU (6.4-7.6)
    south_wall_y = -0.70
    corridor_spacing = 1.5

    # HOME -> PHARMACY edge (x: 0 to 2)
    for x in [0.5, 1.0]:  # Before pharmacy door
        add_tag(x, south_wall_y, 0.0, "south_corridor")

    # PHARMACY -> JUNCTION edge (x: 2 to 4)
    # Skip door gap (1.4-2.6), add tags after
    for x in [2.8, 3.3]:
        add_tag(x, south_wall_y, 0.0, "south_corridor")

    # JUNCTION -> ICU edge (x: 4 to 7)
    # Skip ICU door gap (6.4-7.6)
    for x in [4.5, 5.0, 5.5, 6.0]:
        add_tag(x, south_wall_y, 0.0, "south_corridor")

    # Zone 10-19: North corridor (JUNCTION -> WARD-2A -> WARD-3B)
    # North wall (y = 3.70), tags facing south (yaw = PI)
    # Door gaps: Ward 2A (3.4-4.6), Ward 3B (6.4-7.6)
    north_wall_y = 3.70

    # JUNCTION -> WARD-2A edge (x: 4 to 4, actually this is vertical)
    # This is the vertical corridor at x=4, y: 0 to 3
    # West wall of this corridor (x=3.88, facing east, yaw=0)
    # East wall of this corridor (x=4.12, facing west, yaw=PI)
    west_wall_x = 3.88  # Just west of centre line
    east_wall_x = 4.12  # Just east of centre line

    # Vertical corridor JUNCTION -> WARD-2A (y: 0 to 3)
    # Alternate sides, skip door gap at Ward 2A (y=3.4-4.6 for north corridor)
    for y in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        # West wall facing east
        add_tag(west_wall_x, y, 0.0, "west_corridor")

    # Zone 20-29: East corridor (ICU -> WARD-3B)
    # Vertical corridor at x=7, y: 0 to 3
    # West wall (x=6.88, facing east, yaw=0)
    # East wall (x=7.12, facing west, yaw=PI)
    west_wall_x_icu = 6.88
    east_wall_x_icu = 7.12

    for y in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        # West wall facing east
        add_tag(west_wall_x_icu, y, 0.0, "east_corridor")

    # Zone 30-39: End walls and room entrances
    # HOME end wall (x = -2.60, facing east, yaw = 0)
    add_tag(-2.55, 0.0, 0.0, "home_end")

    # ICU end wall (x = 9.20, facing west, yaw = PI)
    add_tag(9.15, 0.0, PI, "icu_end")

    # WARD-3B end wall (x = 9.20, facing west, yaw = PI)
    add_tag(9.15, 3.0, PI, "ward3b_end")

    # Room door frames (tags facing into corridor)
    # Pharmacy door (x=2, y=-0.70, facing south, yaw=-PI/2)
    add_tag(2.0, -0.65, -PI/2, "pharmacy_door")

    # ICU door (x=7, y=-0.70, facing south, yaw=-PI/2)
    add_tag(7.0, -0.65, -PI/2, "icu_door")

    # Ward 2A door (x=4, y=3.70, facing north, yaw=PI/2)
    add_tag(4.0, 3.65, PI/2, "ward2a_door")

    # Ward 3B door (x=7, y=3.70, facing north, yaw=PI/2)
    add_tag(7.0, 3.65, PI/2, "ward3b_door")

    # Zone 40-49: Inside rooms (tags facing entry)
    # Pharmacy (room centre ~1.9, -1.6), tag facing door (facing north, yaw=0)
    add_tag(1.9, -1.55, 0.0, "pharmacy_room")

    # ICU (room centre ~7.1, -1.9), tag facing door (facing north, yaw=0)
    add_tag(7.1, -1.85, 0.0, "icu_room")

    # Ward 2A (room centre ~3.8, 4.5), tag facing door (facing south, yaw=PI)
    add_tag(3.8, 4.45, PI, "ward2a_room")

    # Ward 3B (room centre ~7.0, 4.5), tag facing door (facing south, yaw=PI)
    add_tag(7.0, 4.45, PI, "ward3b_room")

    return tags


def tag_plate(tag_id, x, y, z, yaw):
    """Thin wall-mounted tag plate (0.01 m thick) with AprilTag texture."""
    return (f"Solid {{\n"
            f"  translation {x:g} {y:g} {z:g}\n"
            f"  rotation 0 0 1 {yaw:g}\n"
            f"  children [\n"
            f"    Shape {{\n"
            f"      appearance PBRAppearance {{\n"
            f"        baseColor 1 1 1\n"
            f"        roughness 0.5\n"
            f"        metalness 0\n"
            f"        texture ImageTexture {{ url \"../worlds/textures/tag_{tag_id:02d}.png\" }}\n"
            f"      }}\n"
            f"      geometry Box {{ size {TAG_SIZE} {TAG_SIZE} 0.01 }}\n"
            f"    }}\n"
            f"  ]\n"
            f"  boundingObject Box {{ size {TAG_SIZE} {TAG_SIZE} 0.01 }}\n"
            f"  name \"tag_{tag_id}\"\n"
            f"}}\n")


def generate_tag_map(tags):
    """Generate tag_map.md with tag details."""
    md = "# AprilTag Map\n\n"
    md += "| ID | Location | Wall | Room | Position (x, y, z) | Yaw |\n"
    md += "|----|----------|------|------|-------------------|-----|\n"

    for x, y, z, yaw, zone, tag_id in tags:
        wall = "unknown"
        if "south" in zone:
            wall = "south"
        elif "north" in zone:
            wall = "north"
        elif "west" in zone:
            wall = "west"
        elif "east" in zone:
            wall = "east"
        elif "door" in zone:
            wall = "door_frame"
        elif "room" in zone:
            wall = "room_wall"
        elif "end" in zone:
            wall = "end_wall"

        room = "corridor"
        if "pharmacy" in zone:
            room = "pharmacy"
        elif "icu" in zone:
            room = "icu"
        elif "ward2a" in zone:
            room = "ward_2a"
        elif "ward3b" in zone:
            room = "ward_3b"
        elif "home" in zone:
            room = "home"

        md += f"| {tag_id} | {zone} | {wall} | {room} | ({x:.2f}, {y:.2f}, {z:.2f}) | {yaw:.2f} |\n"

    return md


# ----------------------------------------------------------------------------
# Signage and Texture Generation (Pillow + OpenCV)
# ----------------------------------------------------------------------------
def make_tag_image(tag_id):
    """Return a 600x600 uint8 image: 400 px tag36h11 marker + 100 px white margin with ID text."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 80, borderBits=1)
    else:  # OpenCV < 4.7
        marker = cv2.aruco.drawMarker(dictionary, tag_id, 80, borderBits=1)
    marker = cv2.resize(marker, (400, 400), interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)

    # Add ID text below the tag (in the white margin)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Try to load a font for the ID text
    font = None
    for font_path in [
        "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/calibrib.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 40)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    # Draw ID text centered below the tag
    text = str(tag_id)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (600 - tw) / 2
    y = 520  # Below the tag (tag is at y=100 to 500)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def make_sign_textures(tex_dir):
    """Generate high-resolution PNG textures for hospital signs and robot logo using Pillow."""
    signs = [
        ("sign_ward_2a.png", "WARD 2A", (16, 128, 128)),
        ("sign_ward_3b.png", "WARD 3B", (16, 128, 128)),
        ("sign_icu.png", "ICU", (180, 24, 32)),
        ("sign_pharmacy.png", "PHARMACY", (24, 72, 160)),
        ("sign_nurse_station.png", "NURSE STATION", (18, 96, 136)),
        ("sign_home_charging.png", "HOME / CHARGING", (20, 130, 60)),
        ("sign_junction_dir.png", "<- PHARMACY  |  ICU ->", (28, 64, 120)),
        ("sign_reception.png", "RECEPTION", (50, 70, 100)),
    ]
    for fn, text, bg in signs:
        img = Image.new("RGB", (1024, 256), bg)
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 1013, 245], outline=(255, 255, 255), width=8)
        draw.rectangle([20, 20, 1003, 235], outline=(255, 255, 255), width=2)
        font = None
        for font_path in [
            "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/calibrib.ttf",
        ]:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 110)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (1024 - tw) / 2
        y = (256 - th) / 2 - bbox[1]
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        img.save(os.path.join(tex_dir, fn))

    # MediBot Robot Brand Logo with red cross
    logo_img = Image.new("RGB", (512, 256), (255, 255, 255))
    ldraw = ImageDraw.Draw(logo_img)
    ldraw.rectangle([8, 8, 503, 247], outline=(20, 80, 160), width=6)
    cx, cy = 90, 128
    cw, cl = 22, 60
    ldraw.rectangle([cx - cw, cy - cl, cx + cw, cy + cl], fill=(220, 20, 30))
    ldraw.rectangle([cx - cl, cy - cw, cx + cl, cy + cw], fill=(220, 20, 30))
    lfont = None
    for fp in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"]:
        if os.path.exists(fp):
            lfont = ImageFont.truetype(fp, 68)
            break
    if lfont is None:
        lfont = ImageFont.load_default()
    ldraw.text((185, 92), "MediBot", fill=(20, 70, 150), font=lfont)
    logo_img.save(os.path.join(tex_dir, "medibot_logo.png"))


# ----------------------------------------------------------------------------
# .wbt text helpers
# ----------------------------------------------------------------------------
def box_solid(name, x, y, z, sx, sy, sz, rgb, collide=True, yaw=0.0, def_name=None, roughness=1.0, metalness=0.0):
    d = f"DEF {def_name} " if def_name else ""
    s = (f"{d}Solid {{\n"
         f"  translation {x:g} {y:g} {z:g}\n"
         f"  rotation 0 0 1 {yaw:g}\n"
         f"  name \"{name}\"\n"
         f"  children [\n"
         f"    Shape {{\n"
         f"      appearance PBRAppearance {{\n"
         f"        baseColor {rgb[0]:g} {rgb[1]:g} {rgb[2]:g}\n"
         f"        roughness {roughness:g}\n"
         f"        metalness {metalness:g}\n"
         f"      }}\n"
         f"      geometry Box {{ size {sx:g} {sy:g} {sz:g} }}\n"
         f"    }}\n"
         f"  ]\n")
    if collide:
        s += f"  boundingObject Box {{ size {sx:g} {sy:g} {sz:g} }}\n"
    return s + "}\n"


def cylinder_solid(name, x, y, z, r, h, rgb, collide=False, yaw=0.0):
    s = (f"Solid {{\n"
         f"  translation {x:g} {y:g} {z:g}\n"
         f"  rotation 0 0 1 {yaw:g}\n"
         f"  name \"{name}\"\n"
         f"  children [\n"
         f"    Shape {{\n"
         f"      appearance PBRAppearance {{ baseColor {rgb[0]:g} {rgb[1]:g} {rgb[2]:g} roughness 0.5 metalness 0.2 }}\n"
         f"      geometry Cylinder {{ radius {r:g} height {h:g} }}\n"
         f"    }}\n"
         f"  ]\n")
    if collide:
        s += f"  boundingObject Cylinder {{ radius {r:g} height {h:g} }}\n"
    return s + "}\n"


def point_light(name, x, y, z, intensity=0.6, radius=4.5):
    return (f"PointLight {{\n"
            f"  attenuation 0 0 1\n"
            f"  castShadows FALSE\n"
            f"  color 1.0 0.98 0.94\n"
            f"  intensity {intensity:g}\n"
            f"  location {x:g} {y:g} {z:g}\n"
            f"  radius {radius:g}\n"
            f"}}\n")


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


def wall_sign(name, tex_name, x, y, z, w, h, yaw):
    hw, hh = w / 2, h / 2
    return (f"Solid {{\n"
            f"  translation {x:g} {y:g} {z:g}\n"
            f"  rotation 0 0 1 {yaw:.5f}\n"
            f"  name \"{name}\"\n"
            f"  children [\n"
            f"    Shape {{\n"
            f"      appearance Appearance {{ texture ImageTexture {{ url [ \"textures/{tex_name}\" ] filtering 2 }} }}\n"
            f"      geometry IndexedFaceSet {{\n"
            f"        coord Coordinate {{ point [ 0.005 {-hw:g} {-hh:g}, 0.005 {hw:g} {-hh:g}, 0.005 {hw:g} {hh:g}, 0.005 {-hw:g} {hh:g} ] }}\n"
            f"        texCoord TextureCoordinate {{ point [ 0 0, 1 0, 1 1, 0 1 ] }}\n"
            f"        coordIndex [ 0 1 2 3 -1 ]\n"
            f"        texCoordIndex [ 0 1 2 3 -1 ]\n"
            f"      }}\n"
            f"    }}\n"
            f"    Shape {{\n"
            f"      appearance PBRAppearance {{ baseColor 0.2 0.2 0.2 roughness 0.5 metalness 0.5 }}\n"
            f"      geometry Box {{ size 0.008 {w + 0.04:g} {h + 0.04:g} }}\n"
            f"    }}\n"
            f"  ]\n"
            f"}}\n")


def floor_sign(name, tex_name, x, y, w, length, yaw=0.0):
    hw, hl = w / 2, length / 2
    return (f"Solid {{\n"
            f"  translation {x:g} {y:g} 0.002\n"
            f"  rotation 0 0 1 {yaw:.5f}\n"
            f"  name \"{name}\"\n"
            f"  children [\n"
            f"    Shape {{\n"
            f"      appearance Appearance {{ texture ImageTexture {{ url [ \"textures/{tex_name}\" ] filtering 2 }} }}\n"
            f"      geometry IndexedFaceSet {{\n"
            f"        coord Coordinate {{ point [ {-hw:g} {-hl:g} 0, {hw:g} {-hl:g} 0, {hw:g} {hl:g} 0, {-hw:g} {hl:g} 0 ] }}\n"
            f"        texCoord TextureCoordinate {{ point [ 0 0, 1 0, 1 1, 0 1 ] }}\n"
            f"        coordIndex [ 0 1 2 3 -1 ]\n"
            f"        texCoordIndex [ 0 1 2 3 -1 ]\n"
            f"      }}\n"
            f"    }}\n"
            f"  ]\n"
            f"}}\n")


# ----------------------------------------------------------------------------
# Procedural Hospital Prop Builders
# ----------------------------------------------------------------------------
def hospital_bed(name, x, y, yaw, has_monitor=False, is_icu=False):
    parts = []
    bed_col = (0.2, 0.45, 0.7) if is_icu else (0.85, 0.88, 0.92)
    parts.append(box_solid(f"{name}_frame", x, y, 0.15, 1.8, 0.85, 0.25, (0.25, 0.28, 0.3), collide=True, yaw=yaw))
    parts.append(box_solid(f"{name}_mattress", x, y, 0.32, 1.7, 0.8, 0.12, (0.95, 0.95, 0.98), collide=False, yaw=yaw))
    hx = x - 0.85 * math.cos(yaw)
    hy = y - 0.85 * math.sin(yaw)
    parts.append(box_solid(f"{name}_headboard", hx, hy, 0.38, 0.08, 0.85, 0.45, bed_col, collide=False, yaw=yaw))
    px = x - 0.65 * math.cos(yaw)
    py = y - 0.65 * math.sin(yaw)
    parts.append(box_solid(f"{name}_pillow", px, py, 0.40, 0.35, 0.55, 0.08, (1, 1, 1), collide=False, yaw=yaw))
    bx = x + 0.55 * math.sin(yaw)
    by = y - 0.55 * math.cos(yaw)
    parts.append(box_solid(f"{name}_cabinet", bx, by, 0.25, 0.4, 0.4, 0.5, (0.85, 0.9, 0.95), collide=True, yaw=yaw))
    if has_monitor:
        mx = x - 0.55 * math.sin(yaw) - 0.4 * math.cos(yaw)
        my = y + 0.55 * math.cos(yaw) - 0.4 * math.sin(yaw)
        parts.append(cylinder_solid(f"{name}_iv_stand", mx, my, 0.6, 0.02, 1.2, (0.8, 0.8, 0.85)))
        parts.append(box_solid(f"{name}_monitor", mx, my, 1.0, 0.1, 0.25, 0.2, (0.1, 0.15, 0.2), yaw=yaw))
        parts.append(box_solid(f"{name}_screen", mx + 0.05 * math.cos(yaw), my + 0.05 * math.sin(yaw), 1.0,
                               0.01, 0.22, 0.16, (0.1, 0.9, 0.3), collide=False, yaw=yaw, roughness=0.1))
    return "".join(parts)


def wheelchair(name, x, y, yaw):
    parts = []
    parts.append(box_solid(f"{name}_seat", x, y, 0.32, 0.45, 0.45, 0.06, (0.15, 0.2, 0.3), collide=True, yaw=yaw))
    parts.append(box_solid(f"{name}_back", x - 0.2 * math.cos(yaw), y - 0.2 * math.sin(yaw), 0.55,
                           0.06, 0.45, 0.42, (0.15, 0.2, 0.3), collide=False, yaw=yaw))
    w1_x = x + 0.26 * math.sin(yaw)
    w1_y = y - 0.26 * math.cos(yaw)
    w2_x = x - 0.26 * math.sin(yaw)
    w2_y = y + 0.26 * math.cos(yaw)
    parts.append(cylinder_solid(f"{name}_w1", w1_x, w1_y, 0.3, 0.28, 0.03, (0.2, 0.2, 0.2)))
    parts.append(cylinder_solid(f"{name}_w2", w2_x, w2_y, 0.3, 0.28, 0.03, (0.2, 0.2, 0.2)))
    return "".join(parts)


def hospital_bench(name, x, y, yaw):
    return box_solid(name, x, y, 0.25, 1.4, 0.45, 0.45, (0.2, 0.4, 0.65), collide=True, yaw=yaw)


def nurse_station_island(x, y):
    parts = []
    parts.append(box_solid("nurse_counter", x, y, 0.42, 1.5, 0.9, 0.84, (0.92, 0.95, 0.98), collide=True))
    parts.append(box_solid("nurse_counter_top", x, y, 0.85, 1.54, 0.94, 0.04, (0.2, 0.35, 0.5), collide=False))
    parts.append(box_solid("nurse_pc_1", x - 0.35, y, 0.97, 0.05, 0.35, 0.22, (0.15, 0.15, 0.18), collide=False))
    parts.append(box_solid("nurse_pc_2", x + 0.35, y, 0.97, 0.05, 0.35, 0.22, (0.15, 0.15, 0.18), collide=False))
    parts.append(box_solid("nurse_screen_1", x - 0.35, y - 0.03, 0.97, 0.01, 0.32, 0.19, (0.2, 0.7, 1.0), collide=False))
    parts.append(box_solid("nurse_screen_2", x + 0.35, y - 0.03, 0.97, 0.01, 0.32, 0.19, (0.2, 0.7, 1.0), collide=False))
    return "".join(parts)


def charging_dock(x, y):
    parts = []
    parts.append(box_solid("charging_tower", x, y, 0.45, 0.25, 0.5, 0.9, (0.18, 0.22, 0.28), collide=True))
    parts.append(box_solid("charging_led", x + 0.13, y, 0.75, 0.02, 0.25, 0.06, (0.1, 0.95, 0.3), collide=False))
    parts.append(box_solid("charging_pads", x + 0.18, y, 0.003, 0.2, 0.3, 0.006, (0.85, 0.75, 0.2), collide=False))
    return "".join(parts)


# ----------------------------------------------------------------------------
# Phase 2 Robot Platform Assembly (Mast, 3 Compartments, Wheels)
# ----------------------------------------------------------------------------
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
            f"        appearance PBRAppearance {{ baseColor 0.15 0.15 0.15 roughness 0.8 metalness 0.2 }}\n"
            f"        geometry Cylinder {{ height 0.032 radius {WHEEL_R:g} }}\n"
            f"      }}\n"
            f"      Shape {{\n"
            f"        appearance PBRAppearance {{ baseColor 0.85 0.88 0.92 roughness 0.4 metalness 0.7 }}\n"
            f"        geometry Cylinder {{ height 0.034 radius {WHEEL_R * 0.5:g} }}\n"
            f"      }}\n"
            f"    ]\n"
            f"    name \"wheel_{tag}_solid\"\n"
            f"    boundingObject Cylinder {{ height 0.032 radius {WHEEL_R:g} }}\n"
            f"    physics Physics {{ density -1 mass 0.25 }}\n"
            f"  }}\n"
            f"}}\n")


def sonar(name, x, y, yaw):
    return (f"DistanceSensor {{\n"
            f"  translation {x:g} {y:g} 0.04\n"
            f"  rotation 0 0 1 {yaw:g}\n"
            f"  name \"{name}\"\n"
            f"  type \"sonar\"\n"
            f"  lookupTable [ 0 0 0, 4 4 0 ]\n"
            f"  numberOfRays 3\n"
            f"  aperture 0.3\n"
            f"}}\n")


def compartment(n, x, anchor_x, cargo_col):
    """One lockable medicine compartment: numbered hinged lid, lock LED, and cargo box."""
    lid_w = 0.19
    lid_l = 0.08
    return f"""
    # Compartment {n} tray rim
    Pose {{
      translation {x:g} 0 0.122
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.2 0.24 0.3 roughness 0.5 }}
          geometry Box {{ size {lid_l + 0.01:g} {lid_w + 0.01:g} 0.006 }}
        }}
      ]
    }}
    # Lock Indicator LED {n} (red when locked, green when unlocked)
    LED {{
      translation {x:g} 0.115 0.135
      name "led_lock_{n}"
      gradual TRUE
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 1 0 0 roughness 0.2 emissiveColor 1 0 0 }}
          geometry Sphere {{ radius 0.008 }}
        }}
      ]
    }}
    # Medicine Cargo Box {n}
    DEF CARGO_{n} Solid {{
      translation {x:g} 0 0.09
      name "cargo_{n}"
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor {cargo_col[0]:g} {cargo_col[1]:g} {cargo_col[2]:g} roughness 0.4 }}
          geometry Box {{ size 0.045 0.07 0.035 }}
        }}
      ]
    }}
    # Hinged Compartment Lid {n}
    HingeJoint {{
      jointParameters HingeJointParameters {{ axis 0 1 0 anchor {anchor_x:g} 0 0.13 }}
      device [
        RotationalMotor {{ name "lid_{n}" minPosition -1.6 maxPosition 0 maxVelocity 3 }}
        PositionSensor {{ name "lid_{n}_ps" }}
      ]
      endPoint Solid {{
        translation {anchor_x:g} 0 0.13
        name "lid_{n}_solid"
        children [
          Pose {{
            translation {x - anchor_x:g} 0 0.004
            children [
              Shape {{
                appearance PBRAppearance {{ baseColor 0.18 0.55 0.85 roughness 0.4 metalness 0.1 }}
                geometry Box {{ size {lid_l:g} {lid_w:g} 0.008 }}
              }}
              # Embossed Compartment Number Tag
              Pose {{
                translation 0 0 0.005
                children [
                  Shape {{
                    appearance PBRAppearance {{ baseColor 1 1 1 roughness 0.3 }}
                    geometry Box {{ size 0.03 0.03 0.002 }}
                  }}
                ]
              }}
            ]
          }}
        ]
        boundingObject Pose {{
          translation {x - anchor_x:g} 0 0.004
          children [ Box {{ size {lid_l:g} {lid_w:g} 0.008 }} ]
        }}
        physics Physics {{ density -1 mass 0.03 }}
      }}
    }}
"""


def robot_node():
    cx, cy, cz = CHASSIS
    wheels = "".join([
        wheel("fl", WHEEL_X, WHEEL_Y), wheel("fr", WHEEL_X, -WHEEL_Y),
        wheel("rl", -WHEEL_X, WHEEL_Y), wheel("rr", -WHEEL_X, -WHEEL_Y),
    ])
    # 3 lockable medicine compartments
    comp1 = compartment(1, 0.04, 0.08, (0.85, 0.15, 0.15))   # Compartment 1 (red cargo box)
    comp2 = compartment(2, -0.06, -0.02, (0.15, 0.45, 0.85)) # Compartment 2 (blue cargo box)
    comp3 = compartment(3, -0.16, -0.12, (0.15, 0.75, 0.35)) # Compartment 3 (green cargo box)

    return f"""DEF MEDIBOT Robot {{
  translation 0 0 {WHEEL_R:g}
  rotation 0 0 1 0
  name "MediBot"
  controller "medibot_controller"
  supervisor TRUE
  children [
    # Main Body Chassis
    Pose {{
      translation 0 0 0.06
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.95 0.96 0.98 roughness 0.5 metalness 0.1 }}
          geometry Box {{ size {cx:g} {cy:g} {cz:g} }}
        }}
      ]
    }}
    # Hospital Blue Side Skirts
    Pose {{
      translation 0 0.122 0.06
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.12 0.40 0.80 roughness 0.4 }}
          geometry Box {{ size 0.42 0.008 0.08 }}
        }}
      ]
    }}
    Pose {{
      translation 0 -0.122 0.06
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.12 0.40 0.80 roughness 0.4 }}
          geometry Box {{ size 0.42 0.008 0.08 }}
        }}
      ]
    }}
    # Front Bumper
    Pose {{
      translation 0.222 0 0.06
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.2 0.25 0.3 roughness 0.6 }}
          geometry Box {{ size 0.025 0.24 0.08 }}
        }}
      ]
    }}
    # Front MediBot Logo Badge
    Pose {{
      translation 0.235 0 0.085
      children [
        Shape {{
          appearance Appearance {{ texture ImageTexture {{ url [ "textures/medibot_logo.png" ] }} }}
          geometry IndexedFaceSet {{
            coord Coordinate {{ point [ 0.002 -0.06 -0.025, 0.002 0.06 -0.025, 0.002 0.06 0.025, 0.002 -0.06 0.025 ] }}
            texCoord TextureCoordinate {{ point [ 0 0, 1 0, 1 1, 0 1 ] }}
            coordIndex [ 0 1 2 3 -1 ]
            texCoordIndex [ 0 1 2 3 -1 ]
          }}
        }}
      ]
    }}
    # Front Status LED Strip
    LED {{
      translation 0.236 0 0.038
      name "status_led"
      gradual TRUE
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0 1 0 roughness 0.2 emissiveColor 0 1 0 }}
          geometry Box {{ size 0.008 0.14 0.018 }}
        }}
      ]
    }}
    # Tall Mast Column (~0.85m tall)
    Pose {{
      translation 0.14 0 0.48
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.94 0.95 0.97 roughness 0.5 metalness 0.2 }}
          geometry Box {{ size 0.045 0.055 0.72 }}
        }}
      ]
    }}
    # Mast Blue Racing Stripe
    Pose {{
      translation 0.164 0 0.48
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.12 0.40 0.80 roughness 0.4 }}
          geometry Box {{ size 0.005 0.035 0.70 }}
        }}
      ]
    }}
    # RFID Badge Reader Module
    Pose {{
      translation 0.166 0 0.40
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.18 0.22 0.28 roughness 0.3 }}
          geometry Box {{ size 0.012 0.07 0.09 }}
        }}
        Shape {{
          appearance PBRAppearance {{ baseColor 0.2 0.75 0.95 roughness 0.3 emissiveColor 0.1 0.4 0.6 }}
          geometry Box {{ size 0.015 0.045 0.045 }}
        }}
      ]
    }}
    # Mast Head Holding Face Camera
    Pose {{
      translation 0.14 0 0.88
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.12 0.35 0.75 roughness 0.4 metalness 0.3 }}
          geometry Box {{ size 0.09 0.13 0.08 }}
        }}
      ]
    }}
    # Face Camera on Head (pointing forward and slightly down at patient/nurse faces)
    Camera {{
      translation 0.19 0 0.88
      rotation 0 1 0 0.18
      name "face_cam"
      width {FACE_CAM['width']}
      height {FACE_CAM['height']}
      fieldOfView {FACE_CAM['fov']:g}
      near 0.05
    }}
    # Navigation Camera at AprilTag Level
    Camera {{
      translation {CAM['x']:g} 0 {CAM['z']:g}
      name "camera"
      width {CAM['width']}
      height {CAM['height']}
      fieldOfView {CAM['fov']:g}
      near 0.05
    }}
    # Front Sonars
    {sonar("us_front", 0.236, 0.0, 0.0)}
    {sonar("us_left", 0.226, 0.08, 0.7854)}
    {sonar("us_right", 0.226, -0.08, -0.7854)}
    # IMU, Gyro, Accelerometer
    InertialUnit {{ translation 0 0 0.04 name "imu" }}
    Gyro {{ translation 0 0 0.04 name "gyro" }}
    Accelerometer {{ translation 0 0 0.04 name "accel" }}
    # 3 Lockable Medicine Compartments
    {comp1}
    {comp2}
    {comp3}
    # 4 Wheels
    {wheels}
  ]
  # Low BoundingObject and Stable Physics Center of Mass (Cannot tip over)
  boundingObject Pose {{
    translation 0 0 0.06
    children [ Box {{ size {cx:g} {cy:g} {cz:g} }} ]
  }}
  physics Physics {{
    density -1
    mass 8
    centerOfMass [ 0 0 -0.01 ]
  }}
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


# ----------------------------------------------------------------------------
# Complete World Construction
# ----------------------------------------------------------------------------
def build_world():
    parts = []
    # R2025a header
    parts.append("#VRML_SIM R2025a utf8\n\n")

    # Viewpoint: top-down overview of the entire hospital
    # Follow-robot tip: in Webots 3D view, right-click robot -> Follow Object
    axis, ang = look_at((3.5, -5.2, 8.8), (3.5, 1.2, 0.0))
    parts.append(f"""WorldInfo {{
  title "MediBot hospital delivery"
  basicTimeStep 32
  coordinateSystem "ENU"
}}
Viewpoint {{
  orientation {axis[0]:.5f} {axis[1]:.5f} {axis[2]:.5f} {ang:.5f}
  position 3.5 -5.2 8.8
}}
Background {{ skyColor [ 0.78 0.86 0.95 ] }}
DirectionalLight {{ ambientIntensity 0.75 direction -0.3 0.4 -1 intensity 0.8 }}
""")

    # Ceiling PointLights for warm hospital ambience
    lights = [
        ("light_home", 0.0, 0.0, 2.3, 0.7),
        ("light_pharmacy", 2.0, -1.8, 2.3, 0.7),
        ("light_junction", 4.0, 0.0, 2.3, 0.7),
        ("light_waiting", 4.5, -2.0, 2.3, 0.7),
        ("light_icu", 7.0, -1.8, 2.3, 0.7),
        ("light_corridor_w", 4.0, 1.5, 2.3, 0.7),
        ("light_nurse", 5.5, 1.5, 2.3, 0.7),
        ("light_corridor_e", 7.0, 1.5, 2.3, 0.7),
        ("light_ward2a", 3.8, 4.4, 2.3, 0.7),
        ("light_ward3b", 7.0, 4.4, 2.3, 0.7),
    ]
    for ln, lx, ly, lz, li in lights:
        parts.append(point_light(ln, lx, ly, lz, intensity=li))

    # Hospital Floor: 12.0m x 8.4m, top at z=0.0
    parts.append(box_solid("floor", 3.3, 1.2, -0.05, 12.0, 8.4, 0.1, (0.86, 0.88, 0.90)))

    # Transit guide lines on corridor centre lines (blue line)
    parts.append(box_solid("line_south", 3.5, 0.0, 0.001, 7.5, 0.04, 0.002, (0.1, 0.45, 0.9), collide=False))
    parts.append(box_solid("line_west", 4.0, 1.5, 0.001, 0.04, 3.0, 0.002, (0.1, 0.45, 0.9), collide=False))
    parts.append(box_solid("line_north", 5.5, 3.0, 0.001, 3.0, 0.04, 0.002, (0.1, 0.45, 0.9), collide=False))
    parts.append(box_solid("line_east", 7.0, 1.5, 0.001, 0.04, 3.0, 0.002, (0.1, 0.45, 0.9), collide=False))

    # Floor Zone Signs at Waypoint Nodes
    parts.append(floor_sign("zone_HOME", "sign_home_charging.png", 0.0, 0.0, 1.1, 0.85))
    parts.append(floor_sign("zone_PHARMACY", "sign_pharmacy.png", 2.0, 0.0, 1.1, 0.85))
    parts.append(floor_sign("zone_JUNCTION", "sign_junction_dir.png", 4.0, 0.0, 1.2, 0.85))
    parts.append(floor_sign("zone_WARD2A", "sign_ward_2a.png", 4.0, 3.0, 1.1, 0.85))
    parts.append(floor_sign("zone_ICU", "sign_icu.png", 7.0, 0.0, 1.1, 0.85))
    parts.append(floor_sign("zone_WARD3B", "sign_ward_3b.png", 7.0, 3.0, 1.1, 0.85))

    # Wall Signs above room entrances and key locations (height z=1.05m)
    parts.append(wall_sign("header_pharmacy", "sign_pharmacy.png", 2.0, -0.72, 1.05, 0.9, 0.25, PI / 2))
    parts.append(wall_sign("header_icu", "sign_icu.png", 7.0, -0.72, 1.05, 0.9, 0.25, PI / 2))
    parts.append(wall_sign("header_ward2a", "sign_ward_2a.png", 4.0, 3.72, 1.05, 0.9, 0.25, -PI / 2))
    parts.append(wall_sign("header_ward3b", "sign_ward_3b.png", 7.0, 3.72, 1.05, 0.9, 0.25, -PI / 2))
    parts.append(wall_sign("sign_nurse_front", "sign_nurse_station.png", 5.5, 1.02, 0.72, 0.9, 0.24, -PI / 2))
    parts.append(wall_sign("sign_dir_junction", "sign_junction_dir.png", 4.0, -2.75, 1.15, 1.1, 0.28, PI / 2))
    parts.append(wall_sign("sign_reception_wall", "sign_reception.png", -1.2, 2.65, 1.15, 0.9, 0.25, -PI / 2))

    # Perimeter and Room Walls (height 0.8m)
    wall_col = (0.92, 0.92, 0.90)
    wh = 0.8
    wz = wh / 2

    # Outer perimeter boundary walls
    parts.append(box_solid("wall_south_outer", 3.3, -2.80, wz, 12.0, 0.12, wh, wall_col))
    parts.append(box_solid("wall_north_outer", 3.3, 5.20, wz, 12.0, 0.12, wh, wall_col))
    parts.append(box_solid("wall_west_outer", -2.60, 1.2, wz, 0.12, 8.4, wh, wall_col))
    parts.append(box_solid("wall_east_outer", 9.20, 1.2, wz, 0.12, 8.4, wh, wall_col))

    # South Corridor Wall along y = -0.70 with door gaps at Pharmacy (x=2) and ICU (x=7)
    # Open waiting area alcove: x = 3.3 to 5.7 gives 2.8m clearance south of JUNCTION
    parts.append(box_solid("wall_s_alcove", -1.55, -0.70, wz, 2.1, 0.12, wh, wall_col))
    parts.append(box_solid("wall_s_pharm_w", 0.95, -0.70, wz, 0.9, 0.12, wh, wall_col))
    # [Door opening 1.2m at Pharmacy: x = 1.4 to 2.6]
    parts.append(box_solid("wall_s_pharm_e", 2.95, -0.70, wz, 0.7, 0.12, wh, wall_col))
    # [Open waiting area alcove: x = 3.3 to 5.7]
    parts.append(box_solid("wall_s_icu_w", 6.05, -0.70, wz, 0.7, 0.12, wh, wall_col))
    # [Door opening 1.2m at ICU: x = 6.4 to 7.6]
    parts.append(box_solid("wall_s_icu_e", 8.40, -0.70, wz, 1.6, 0.12, wh, wall_col))

    # Room partition dividers (South side)
    parts.append(box_solid("div_pharm_w", 0.50, -1.75, wz, 0.12, 2.1, wh, wall_col))
    parts.append(box_solid("div_pharm_e", 3.30, -1.75, wz, 0.12, 2.1, wh, wall_col))
    parts.append(box_solid("div_icu_w", 5.70, -1.75, wz, 0.12, 2.1, wh, wall_col))

    # North Corridor Wall along y = 3.70 with door gaps at Ward 2A (x=4) and Ward 3B (x=7)
    parts.append(box_solid("wall_n_ward2a_w", 2.80, 3.70, wz, 1.2, 0.12, wh, wall_col))
    # [Door opening 1.2m at Ward 2A: x = 3.4 to 4.6]
    parts.append(box_solid("wall_n_middle", 5.50, 3.70, wz, 1.8, 0.12, wh, wall_col))
    # [Door opening 1.2m at Ward 3B: x = 6.4 to 7.6]
    parts.append(box_solid("wall_n_ward3b_e", 8.40, 3.70, wz, 1.6, 0.12, wh, wall_col))

    # Room partition dividers (North side)
    parts.append(box_solid("div_ward2a_w", 2.20, 4.45, wz, 0.12, 1.5, wh, wall_col))
    parts.append(box_solid("div_wards_mid", 5.20, 4.45, wz, 0.12, 1.5, wh, wall_col))

    # West alcove divider (Home / Reception wall)
    parts.append(box_solid("wall_w_home_n", -1.55, 0.70, wz, 2.1, 0.12, wh, wall_col))

    # Room Furniture and Models
    # 1. HOME: Charging Station Dock
    parts.append(charging_dock(-2.35, 0.0))

    # 2. RECEPTION: Desk near Home
    parts.append(box_solid("reception_desk", -1.0, 2.0, 0.42, 1.4, 0.6, 0.84, (0.85, 0.9, 0.95), collide=True))
    parts.append(box_solid("reception_pc", -1.0, 2.0, 0.95, 0.05, 0.35, 0.22, (0.15, 0.15, 0.18), collide=False))
    parts.append(wheelchair("wheelchair_reception", -0.5, 0.75, PI / 2))

    # 3. PHARMACY: Counter, medicine shelves, fridge
    parts.append(box_solid("pharm_counter", 1.9, -1.6, 0.42, 1.5, 0.5, 0.84, (0.9, 0.95, 0.98), collide=True))
    parts.append(box_solid("pharm_shelves", 1.9, -2.65, 0.60, 2.2, 0.28, 1.2, (0.8, 0.85, 0.9), collide=True))
    parts.append(box_solid("pharm_fridge", 0.85, -2.40, 0.55, 0.5, 0.5, 1.1, (0.95, 0.95, 0.98), collide=True))

    # 4. WAITING AREA: Hospital Benches
    parts.append(hospital_bench("waiting_bench_1", 3.9, -2.35, 0.0))
    parts.append(hospital_bench("waiting_bench_2", 5.1, -2.35, 0.0))

    # 5. ICU: 2 ICU Intensive Care Beds with Monitors
    parts.append(hospital_bed("icu_bed_1", 6.4, -1.9, -PI / 2, has_monitor=True, is_icu=True))
    parts.append(hospital_bed("icu_bed_2", 7.8, -1.9, -PI / 2, has_monitor=True, is_icu=True))

    # 6. NURSE STATION: Central Island Desk & Monitors
    parts.append(nurse_station_island(5.5, 1.5))

    # 7. WARD 2A: 4 Hospital Beds + Bedside Tables + Curtains
    for idx, bx in enumerate([2.7, 3.4, 4.1, 4.8]):
        parts.append(hospital_bed(f"ward2a_bed_{idx}", bx, 4.5, -PI / 2))
        if idx > 0:
            parts.append(box_solid(f"ward2a_curtain_{idx}", bx - 0.35, 4.5, 0.75, 0.03, 1.2, 1.4,
                                   (0.35, 0.7, 0.7), collide=False))

    # 8. WARD 3B: 4 Hospital Beds + Bedside Tables + Curtains
    for idx, bx in enumerate([5.8, 6.6, 7.4, 8.2]):
        parts.append(hospital_bed(f"ward3b_bed_{idx}", bx, 4.5, -PI / 2))
        if idx > 0:
            parts.append(box_solid(f"ward3b_curtain_{idx}", bx - 0.4, 4.5, 0.75, 0.03, 1.2, 1.4,
                                   (0.35, 0.7, 0.7), collide=False))

    # 9. Wall-mounted AprilTag plates (thin plates on walls)
    wall_tags = generate_wall_tags()
    tags = []
    for x, y, z, yaw, zone, tag_id in wall_tags:
        parts.append(tag_plate(tag_id, x, y, z, yaw))
        tags.append({"id": tag_id, "x": x, "y": y, "yaw": yaw, "z": z, "zone": zone})

    # 10. Trolley Obstacle (DEF TROLLEY) - with physics for collision detection
    parts.append(box_solid("trolley", TROLLEY_PARK[0], TROLLEY_PARK[1], 0.35, 0.5, 0.45, 0.7,
                           (0.95, 0.55, 0.1), def_name="TROLLEY"))
    # Add physics to trolley (manually since box_solid doesn't add it by default)
    parts[-1] = parts[-1].rstrip("}\n") + "  physics Physics { density -1 mass 15 }\n}\n"

    # 11. Robot (DEF MEDIBOT)
    parts.append(robot_node())

    return "".join(parts), tags


def check_clearance(parts):
    """Check that no objects are within 0.55 m of any graph edge centre line or node."""
    violations = []
    clearance = 0.55

    # Extract object positions from parts
    import re
    for part in parts:
        # Look for Solid with translation
        match = re.search(r'Solid\s*\{\s*translation\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', part)
        if match:
            ox, oy, oz = float(match.group(1)), float(match.group(2)), float(match.group(3))
            # Skip floor and very low objects
            if oz < 0.1:
                continue
            # Skip the robot itself (starts at 0,0)
            if abs(ox) < 0.1 and abs(oy) < 0.1:
                continue
            # Skip trolley (it's meant to be moved)
            if "trolley" in part.lower():
                continue

            # Check distance to all nodes
            for node_name, (nx, ny) in NODES.items():
                dist = math.hypot(ox - nx, oy - ny)
                if dist < clearance:
                    violations.append(f"Object at ({ox:.2f}, {oy:.2f}) is {dist:.2f} m from node {node_name}")

            # Check distance to all edge centre lines
            for a, b in EDGES:
                ax, ay = NODES[a]
                bx, by = NODES[b]
                # Point-to-line-segment distance
                ab_len = math.hypot(bx - ax, by - ay)
                if ab_len == 0:
                    continue
                # Project point onto line
                t = max(0, min(1, ((ox - ax) * (bx - ax) + (oy - ay) * (by - ay)) / (ab_len ** 2)))
                proj_x = ax + t * (bx - ax)
                proj_y = ay + t * (by - ay)
                dist = math.hypot(ox - proj_x, oy - proj_y)
                if dist < clearance:
                    violations.append(f"Object at ({ox:.2f}, {oy:.2f}) is {dist:.2f} m from edge {a}-{b}")

    if violations:
        print("CLEARANCE VIOLATIONS DETECTED:")
        for v in violations:
            print(f"  {v}")
        raise ValueError(f"{len(violations)} clearance violations found. Objects must be at least {clearance} m from graph edges/nodes.")
    else:
        print("CLEARANCE CHECK PASSED: All objects are >= 0.55 m from graph edges/nodes.")


def main():
    os.makedirs(TEX_DIR, exist_ok=True)
    os.makedirs(CTRL_DIR, exist_ok=True)

    # 1. Generate sign textures and robot logo with Pillow
    make_sign_textures(TEX_DIR)

    # 2. Build world and tags
    world, tags = build_world()

    # 3. Check clearance violations
    check_clearance(world.split("\n"))

    # 4. Generate tag_map.md
    wall_tags = generate_wall_tags()
    tag_map_md = generate_tag_map(wall_tags)
    with open(os.path.join(ROOT, "tag_map.md"), "w", encoding="utf-8") as f:
        f.write(tag_map_md)

    # 5. Generate AprilTag markers
    for t in tags:
        cv2.imwrite(os.path.join(TEX_DIR, f"tag_{t['id']:02d}.png"), make_tag_image(t["id"]))

    # 4. Write worlds/medibot.wbt
    with open(os.path.join(WORLD_DIR, "medibot.wbt"), "w", encoding="utf-8") as f:
        f.write(world)

    # 6. Write controllers/medibot_controller/config.json
    cfg = {
        "nodes": {k: list(v) for k, v in NODES.items()},
        "edges": EDGES,
        "tags": [{"id": t["id"], "x": t["x"], "y": t["y"], "yaw": t["yaw"], "z": t["z"], "zone": t["zone"]} for t in tags],
        "tag": {"size": TAG_SIZE},
        "camera": CAM,
        "face_camera": FACE_CAM,
        "robot": {
            "wheel_radius": WHEEL_R,
            "track": 2 * WHEEL_Y,
            "start": [0.0, 0.0, 0.0],
            "z": WHEEL_R,
            "footprint": list(CHASSIS),
        },
        "trolley_park": list(TROLLEY_PARK),
    }
    with open(os.path.join(CTRL_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Wrote worlds/medibot.wbt, {len(tags)} tag textures, 8 sign textures + logo, controllers/medibot_controller/config.json")
    print("Open worlds/medibot.wbt in Webots and press Run.")


if __name__ == "__main__":
    main()
