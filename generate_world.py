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
import os
import sys
import math
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
rng = np.random.default_rng(42)

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


def _indent(text, spaces=4):
    """Re-indent every line of a multi-line VRML snippet by `spaces` spaces.

    When a helper function returns a multi-line VRML block and that block is
    interpolated into an f-string template, Python only indents the first line
    (because the f-string prefix is only prepended to the first token).  All
    subsequent lines keep their original indentation.  Calling _indent() on the
    whole block before embedding it ensures every line sits at the correct depth
    in the output .wbt file, so Webots can parse the brace structure correctly.
    """
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line
                     for line in text.splitlines())


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
    
    Tags are placed on cross walls (perpendicular to corridor) to face head-on
    to approaching robots, avoiding grazing-angle pose errors.
    """
    tags = []
    tag_id = 0

    def add_tag(x, y, yaw, zone):
        nonlocal tag_id
        tags.append((x, y, TAG_Z, yaw, zone, tag_id))
        tag_id += 1

    # Zone 0-9: HOME -> PHARMACY (eastbound corridor, y=1.5, x: 0 to 2)
    # Tag on far end wall (x=2) facing west
    add_tag(2.0, 1.5, PI, "pharmacy_end")
    # Tag on near end wall (x=0) facing east
    add_tag(0.0, 1.5, 0.0, "home_end")

    # Zone 10-19: PHARMACY -> JUNCTION (eastbound, y=1.5, x: 2 to 4)
    add_tag(4.0, 1.5, PI, "junction_south")
    add_tag(2.0, 1.5, 0.0, "pharmacy_start")

    # Zone 20-29: JUNCTION -> ICU (eastbound, y=1.5, x: 4 to 7)
    add_tag(7.0, 1.5, PI, "icu_end")
    add_tag(4.0, 1.5, 0.0, "junction_south_start")

    # Zone 30-39: JUNCTION -> WARD-2A (northbound, x=4.0, y: 0 to 3.7)
    # Tag on far end wall (y=3.7) facing south
    add_tag(4.0, 3.7, PI, "ward2a_end")
    # Tag on near end wall (y=0) facing north
    add_tag(4.0, 0.0, 0.0, "junction_north")

    # Zone 40-49: ICU -> WARD-3B (northbound, x=7.0, y: 0 to 3.7)
    add_tag(7.0, 3.7, PI, "ward3b_end")
    add_tag(7.0, 0.0, 0.0, "icu_north")

    # Zone 50-59: WARD-2A -> WARD-3B (eastbound, y=3.7, x: 4 to 7)
    add_tag(7.0, 3.7, PI, "ward3b_north")
    add_tag(4.0, 3.7, 0.0, "ward2a_north")

    # Zone 60-69: Room door frames (head-on tags facing corridor)
    add_tag(2.0, -0.65, -PI/2, "pharmacy_door")
    add_tag(7.0, -0.65, -PI/2, "icu_door")
    add_tag(4.0, 3.65, PI/2, "ward2a_door")
    add_tag(7.0, 3.65, PI/2, "ward3b_door")

    # Zone 70-79: Inside rooms (tags facing entry)
    add_tag(1.9, -1.55, 0.0, "pharmacy_room")
    add_tag(7.1, -1.85, 0.0, "icu_room")
    add_tag(3.8, 4.45, PI, "ward2a_room")
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
            f"        baseColorMap ImageTexture {{ url [ \"textures/tag_{tag_id:02d}.png\" ] filtering 0 }}\n"
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
# Procedural Texture Generation (Pillow)
# ----------------------------------------------------------------------------
def make_floor_texture_corridor():
    """Light blue-grey vinyl floor with faint tile lines."""
    img = Image.new("RGB", (512, 512), (200, 205, 210))
    draw = ImageDraw.Draw(img)

    # Faint tile lines (1m tiles = 128px at 512px for 4m coverage)
    tile_color = (190, 195, 200)
    for i in range(0, 513, 128):
        draw.line([(i, 0), (i, 512)], fill=tile_color, width=1)
        draw.line([(0, i), (512, i)], fill=tile_color, width=1)

    return img


def make_floor_texture_ward():
    """Warm light grey ward floor."""
    img = Image.new("RGB", (512, 512), (220, 220, 215))
    draw = ImageDraw.Draw(img)

    # Subtle pattern
    tile_color = (210, 210, 205)
    for i in range(0, 513, 64):
        draw.line([(i, 0), (i, 512)], fill=tile_color, width=1)
        draw.line([(0, i), (512, i)], fill=tile_color, width=1)

    return img


def make_floor_texture_pharmacy():
    """Pharmacy floor - slightly different pattern."""
    img = Image.new("RGB", (512, 512), (215, 225, 230))
    draw = ImageDraw.Draw(img)

    # Checker pattern
    tile_color = (205, 215, 220)
    for x in range(0, 512, 128):
        for y in range(0, 512, 128):
            if ((x // 128) + (y // 128)) % 2 == 0:
                draw.rectangle([x, y, x + 128, y + 128], fill=tile_color)

    return img


def make_wall_texture():
    """White plaster wall with grey skirting band."""
    img = Image.new("RGB", (512, 512), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Grey skirting band at bottom (0.1m = 64px at 512px for 0.8m height scaled)
    draw.rectangle([0, 448, 512, 512], fill=(180, 180, 180))

    # Subtle plaster texture
    for _ in range(500):
        x = rng.integers(0, 512)
        y = rng.integers(0, 448)
        img.putpixel((x, y), (240, 240, 240))

    return img


def make_monitor_texture():
    """Monitor screen with green heartbeat line."""
    img = Image.new("RGB", (256, 256), (20, 20, 30))
    draw = ImageDraw.Draw(img)

    # Green heartbeat line
    points = []
    for x in range(20, 236, 5):
        y = 128 + int(20 * math.sin(x * 0.1))
        points.append((x, y))
    draw.line(points, fill=(0, 255, 0), width=2)

    # Grid lines
    for i in range(0, 256, 32):
        draw.line([(i, 0), (i, 256)], fill=(0, 50, 0), width=1)
        draw.line([(0, i), (256, i)], fill=(0, 50, 0), width=1)

    return img


def make_curtain_texture():
    """Teal curtain fold texture."""
    img = Image.new("RGB", (256, 256), (70, 130, 130))
    draw = ImageDraw.Draw(img)

    # Fold lines
    for x in range(0, 256, 32):
        shade = 60 + (x % 32) * 2
        draw.line([(x, 0), (x, 256)], fill=(shade, shade + 60, shade + 60), width=32)

    return img


def make_floor_marker(color, letter):
    """Floor marker texture (green A or blue B)."""
    img = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(img)

    # White circle
    draw.ellipse([28, 28, 228, 228], fill=(255, 255, 255))

    # Letter
    font = None
    for font_path in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 150)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (256 - tw) / 2
    y = (256 - th) / 2 - bbox[1]
    draw.text((x, y), letter, fill=color, font=font)

    return img


# ----------------------------------------------------------------------------
# Signage and Texture Generation (Pillow + OpenCV)
# ----------------------------------------------------------------------------
def make_tag_image(tag_id):
    """Return a 1024x1024 tag36h11 texture with a proportional white margin."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 80, borderBits=1)
    else:  # OpenCV < 4.7
        marker = cv2.aruco.drawMarker(dictionary, tag_id, 80, borderBits=1)
    marker = cv2.resize(marker, (704, 704), interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(marker, 160, 160, 160, 160, cv2.BORDER_CONSTANT, value=255)

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
                font = ImageFont.truetype(font_path, 64)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    # Draw ID text centered below the tag
    text = str(tag_id)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (1024 - tw) / 2
    y = 840  # Below the tag (tag is at y=160 to 864)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def make_sign_textures(tex_dir):
    """Generate high-resolution PNG textures for hospital signs, floors, walls, and markers."""
    # Floor textures
    make_floor_texture_corridor().save(os.path.join(tex_dir, "floor_corridor.png"))
    make_floor_texture_ward().save(os.path.join(tex_dir, "floor_ward.png"))
    make_floor_texture_pharmacy().save(os.path.join(tex_dir, "floor_pharmacy.png"))

    # Wall texture
    make_wall_texture().save(os.path.join(tex_dir, "wall_plaster.png"))

    # Monitor texture
    make_monitor_texture().save(os.path.join(tex_dir, "monitor_screen.png"))

    # Curtain texture
    make_curtain_texture().save(os.path.join(tex_dir, "curtain_teal.png"))

    # Floor markers
    make_floor_marker((0, 180, 0), "A").save(os.path.join(tex_dir, "marker_a.png"))
    make_floor_marker((0, 100, 200), "B").save(os.path.join(tex_dir, "marker_b.png"))

    # Door signs — power-of-two dimensions (1024x256) to avoid Webots rescale warning
    signs = [
        ("sign_ward_2a.png", "WARD 2A"),
        ("sign_ward_3b.png", "WARD 3B"),
        ("sign_icu.png", "ICU"),
        ("sign_pharmacy.png", "PHARMACY"),
        ("sign_nurse_station.png", "NURSE STATION"),
        ("sign_home_charging.png", "HOME / CHARGING"),
        ("sign_medicine_delivery.png", "MEDICINE DELIVERY"),
        ("sign_junction_dir.png", "JUNCTION"),
        ("sign_reception.png", "RECEPTION"),
    ]
    SW, SH = 1024, 256   # power-of-two sign dimensions
    for fn, text in signs:
        img = Image.new("RGB", (SW, SH), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([4, 4, SW - 4, SH - 4], outline=(50, 50, 50), width=6)
        draw.rectangle([10, 10, SW - 10, SH - 10], outline=(200, 200, 200), width=3)

        font = None
        for font_path in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"]:
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
        x = (SW - tw) / 2
        y = (SH - th) / 2 - bbox[1]
        draw.text((x, y), text, fill=(30, 30, 30), font=font)
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


def box_solid_textured(name, x, y, z, sx, sy, sz, appearance, yaw=0.0):
    """Box with custom appearance string (for textured walls/floors).

    `appearance` must be a VRML *value* (e.g. 'USE WallAppearance' or a full
    'PBRAppearance { ... }' block).  The field name 'appearance' is added by
    this function — do NOT include it in the `appearance` argument.
    """
    return (f"Solid {{\n"
            f"  translation {x:g} {y:g} {z:g}\n"
            f"  rotation 0 0 1 {yaw:g}\n"
            f"  name \"{name}\"\n"
            f"  children [\n"
            f"    Shape {{\n"
            f"      appearance {appearance}\n"
            f"      geometry Box {{ size {sx:g} {sy:g} {sz:g} }}\n"
            f"    }}\n"
            f"  ]\n"
            f"  boundingObject Box {{ size {sx:g} {sy:g} {sz:g} }}\n"
            f"  locked TRUE\n"
            f"}}\n")


def floor_marker(name, x, y, texture):
    """Floor marker (A or B) - decorative only, no collision."""
    return (f"Solid {{\n"
            f"  translation {x:g} {y:g} 0.005\n"
            f"  children [\n"
            f"    Shape {{\n"
            f"      appearance PBRAppearance {{\n"
            f"        baseColor 1 1 1\n"
            f"        roughness 0.9\n"
            f"        metalness 0\n"
            f"        baseColorMap ImageTexture {{ url [ \"textures/{texture}\" ] }}\n"
            f"      }}\n"
            f"      geometry Box {{ size 0.5 0.5 0.01 }}\n"
            f"    }}\n"
            f"  ]\n"
            f"  name \"{name}\"\n"
            f"}}\n")


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

    # Build wheel and compartment snippets, then re-indent every line so they
    # sit at the correct depth (4 spaces = one level inside "children [").
    # Without _indent() the f-string only indents the first line of each
    # multi-line block, leaving every subsequent line at column 0, which makes
    # Webots see those nodes as top-level world children and reject the file.
    wheels = _indent(
        "".join([
            wheel("fl",  WHEEL_X,  WHEEL_Y),
            wheel("fr",  WHEEL_X, -WHEEL_Y),
            wheel("rl", -WHEEL_X,  WHEEL_Y),
            wheel("rr", -WHEEL_X, -WHEEL_Y),
        ]),
        4,
    )
    # 3 lockable medicine compartments
    comp1 = _indent(compartment(1,  0.04,  0.08, (0.85, 0.15, 0.15)), 4)
    comp2 = _indent(compartment(2, -0.06, -0.02, (0.15, 0.45, 0.85)), 4)
    comp3 = _indent(compartment(3, -0.16, -0.12, (0.15, 0.75, 0.35)), 4)

    # Sonar snippets (single-line helpers, still benefit from consistent indenting)
    s_front = _indent(sonar("us_front", 0.236,  0.0,    0.0   ), 4)
    s_left  = _indent(sonar("us_left",  0.226,  0.08,   0.7854), 4)
    s_right = _indent(sonar("us_right", 0.226, -0.08,  -0.7854), 4)

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
{s_front}
{s_left}
{s_right}
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

    # Viewpoint: steep overhead angle to see rooms clearly
    axis, ang = look_at((3.5, -8.0, 12.0), (3.5, 1.5, 0.0))
    parts.append(f"""WorldInfo {{
  title "MediBot hospital delivery"
  basicTimeStep 32
  coordinateSystem "ENU"
}}
Viewpoint {{
  orientation {axis[0]:.5f} {axis[1]:.5f} {axis[2]:.5f} {ang:.5f}
  position 3.5 -8.0 12.0
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

    # Perimeter and Room Walls (height 1.4m with texture)
    wh = 1.4
    wz = wh / 2

    # DEF/USE for texture reuse (performance optimization).
    # These strings are VRML *values* — the field name 'appearance' is
    # prepended by box_solid_textured().  The first wall/floor carries the
    # DEF so subsequent nodes can USE it.
    wall_appearance_def = (
        'DEF WallAppearance PBRAppearance {\n'
        '        baseColor 0.96 0.96 0.96\n'
        '        roughness 0.8\n'
        '        metalness 0\n'
        '        baseColorMap ImageTexture { url [ "textures/wall_plaster.png" ] }\n'
        '      }'
    )
    floor_appearance_def = (
        'DEF FloorAppearance PBRAppearance {\n'
        '        baseColor 0.78 0.8 0.82\n'
        '        roughness 0.6\n'
        '        metalness 0\n'
        '        baseColorMap ImageTexture { url [ "textures/floor_corridor.png" ] }\n'
        '      }'
    )
    wall_appearance  = "USE WallAppearance"
    floor_appearance = "USE FloorAppearance"

    # Outer perimeter boundary walls (first one carries the DEF, rest USE it)
    parts.append(box_solid_textured("wall_south_outer", 3.3, -2.80, wz, 12.0, 0.12, wh, wall_appearance_def))
    parts.append(box_solid_textured("wall_north_outer", 3.3, 5.20, wz, 12.0, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_west_outer", -2.60, 1.2, wz, 0.12, 8.4, wh, wall_appearance))
    parts.append(box_solid_textured("wall_east_outer", 9.20, 1.2, wz, 0.12, 8.4, wh, wall_appearance))

    # South Corridor Wall along y = -0.70 with door gaps at Pharmacy (x=2) and ICU (x=7)
    parts.append(box_solid_textured("wall_s_alcove", -1.55, -0.70, wz, 2.1, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_s_pharm_w", 0.95, -0.70, wz, 0.9, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_s_pharm_e", 2.95, -0.70, wz, 0.7, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_s_icu_w", 6.05, -0.70, wz, 0.7, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_s_icu_e", 8.40, -0.70, wz, 1.6, 0.12, wh, wall_appearance))

    # Room partition dividers (South side)
    parts.append(box_solid_textured("div_pharm_w", 0.50, -1.75, wz, 0.12, 2.1, wh, wall_appearance))
    parts.append(box_solid_textured("div_pharm_e", 3.30, -1.75, wz, 0.12, 2.1, wh, wall_appearance))
    parts.append(box_solid_textured("div_icu_w", 5.70, -1.75, wz, 0.12, 2.1, wh, wall_appearance))

    # North Corridor Wall along y = 3.70 with door gaps at Ward 2A (x=4) and Ward 3B (x=7)
    parts.append(box_solid_textured("wall_n_ward2a_w", 2.80, 3.70, wz, 1.2, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_n_middle", 5.50, 3.70, wz, 1.8, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_n_ward3b_e", 8.40, 3.70, wz, 1.6, 0.12, wh, wall_appearance))

    # Room partition dividers (North side)
    parts.append(box_solid_textured("div_ward2a_w", 2.20, 4.45, wz, 0.12, 1.5, wh, wall_appearance))
    parts.append(box_solid_textured("div_wards_mid", 5.20, 4.45, wz, 0.12, 1.5, wh, wall_appearance))

    # West alcove divider (Home / Reception wall)
    parts.append(box_solid_textured("wall_w_home_n", -1.55, 0.70, wz, 2.1, 0.12, wh, wall_appearance))

    # Floor with corridor texture
    parts.append(box_solid_textured("floor_main", 3.3, 1.5, 0.01, 12.0, 8.0, 0.02, floor_appearance_def))

    # Floor markers (A at HOME, B at WARD-3B)
    parts.append(floor_marker("marker_a", 0.0, 0.0, "marker_a.png"))
    parts.append(floor_marker("marker_b", 7.0, 3.0, "marker_b.png"))

    # Door frames (white) and glass panels for room entrances
    # Pharmacy door frame
    parts.append(box_solid("door_frame_pharm_l", 1.35, -0.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_pharm_r", 2.65, -0.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_pharm_t", 2.0, -0.64, 1.35, 1.3, 0.05, 0.1, (0.95, 0.95, 0.95)))
    # Glass panel (semi-transparent)
    parts.append(box_solid("door_glass_pharm", 2.0, -0.70, 0.7, 1.2, 0.05, 1.2, (0.8, 0.9, 1.0, 0.3)))

    # ICU door frame
    parts.append(box_solid("door_frame_icu_l", 6.35, -0.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_icu_r", 7.65, -0.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_icu_t", 7.0, -0.64, 1.35, 1.3, 0.05, 0.1, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_glass_icu", 7.0, -0.70, 0.7, 1.2, 0.05, 1.2, (0.8, 0.9, 1.0, 0.3)))

    # Ward 2A door frame
    parts.append(box_solid("door_frame_ward2a_l", 3.35, 3.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_ward2a_r", 4.65, 3.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_ward2a_t", 4.0, 3.76, 1.35, 1.3, 0.05, 0.1, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_glass_ward2a", 4.0, 3.70, 0.7, 1.2, 0.05, 1.2, (0.8, 0.9, 1.0, 0.3)))

    # Ward 3B door frame
    parts.append(box_solid("door_frame_ward3b_l", 6.35, 3.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_ward3b_r", 7.65, 3.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_ward3b_t", 7.0, 3.76, 1.35, 1.3, 0.05, 0.1, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_glass_ward3b", 7.0, 3.70, 0.7, 1.2, 0.05, 1.2, (0.8, 0.9, 1.0, 0.3)))

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
    tx, ty = TROLLEY_PARK
    parts.append(
        f"DEF TROLLEY Solid {{\n"
        f"  translation {tx:g} {ty:g} 0.35\n"
        f"  rotation 0 0 1 0\n"
        f"  name \"trolley\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{\n"
        f"        baseColor 0.95 0.55 0.1\n"
        f"        roughness 1\n"
        f"        metalness 0\n"
        f"      }}\n"
        f"      geometry Box {{ size 0.5 0.45 0.7 }}\n"
        f"    }}\n"
        f"  ]\n"
        f"  boundingObject Box {{ size 0.5 0.45 0.7 }}\n"
        f"  physics Physics {{ density -1 mass 15 }}\n"
        f"}}\n"
    )

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

    # Auto-validate the generated file so a broken world is caught immediately.
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "validate_world.py")],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("\ngenerate_world.py: world file failed validation – do NOT open in Webots.")
        sys.exit(1)


if __name__ == "__main__":
    main()
