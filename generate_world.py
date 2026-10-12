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
PATIENTS_PATH = os.path.join(CTRL_DIR, "patients.json")

# ----------------------------------------------------------------------------
# Waypoint graph (metres, 1 Webots unit = 1 m); room nodes extend corridor map.
# ----------------------------------------------------------------------------
PI = math.pi
DOOR_WIDTH = 1.25
NODES = {
    "HOME": (0.0, 0.0),
    "PHARMACY": (2.0, 0.0),
    "JUNCTION": (4.0, 0.0),
    "WARD-2A": (4.0, 3.0),
    "ICU": (7.0, 0.0),
    "WARD-3B": (7.0, 3.0),
    "WARD-2A-DOOR": (4.0, 3.7),
    "WARD-2A-IN": (4.0, 5.2),
    "WARD-3B-DOOR": (7.0, 3.7),
    "WARD-3B-IN": (7.0, 5.2),
    "ICU-DOOR": (7.0, -0.7),
    "ICU-IN": (7.0, -2.2),
}
EDGES = [
    ("HOME", "PHARMACY"),
    ("PHARMACY", "JUNCTION"),
    ("JUNCTION", "WARD-2A"),
    ("JUNCTION", "ICU"),
    ("WARD-2A", "WARD-3B"),
    ("ICU", "WARD-3B"),
    ("WARD-2A", "WARD-2A-DOOR"),
    ("WARD-2A-DOOR", "WARD-2A-IN"),
    ("WARD-3B", "WARD-3B-DOOR"),
    ("WARD-3B-DOOR", "WARD-3B-IN"),
    ("ICU", "ICU-DOOR"),
    ("ICU-DOOR", "ICU-IN"),
]

ROOMS = {
    "WARD-2A": {
        "door_node": "WARD-2A-DOOR", "in_node": "WARD-2A-IN",
        "door_width": DOOR_WIDTH, "aisle_width": 1.2, "handover_area": [1.2, 1.2],
        "bounds": [0.46, 5.14, 3.76, 9.44], "compartment": 2,
        "handover": {"robot": [4.0, 5.2, -PI / 2],
                     "visitor": [4.0, 4.2, PI / 2]},
    },
    "WARD-3B": {
        "door_node": "WARD-3B-DOOR", "in_node": "WARD-3B-IN",
        "door_width": DOOR_WIDTH, "aisle_width": 1.2, "handover_area": [1.2, 1.2],
        "bounds": [5.26, 10.74, 3.76, 9.44], "compartment": 1,
        "handover": {"robot": [7.0, 5.2, -PI / 2],
                     "visitor": [7.0, 4.2, PI / 2]},
    },
    "ICU": {
        "door_node": "ICU-DOOR", "in_node": "ICU-IN",
        "door_width": DOOR_WIDTH, "aisle_width": 1.2, "handover_area": [1.2, 1.2],
        "bounds": [4.86, 10.74, -5.54, -0.76], "compartment": 3,
        "handover": {"robot": [7.0, -2.2, PI / 2],
                     "visitor": [7.0, -1.2, -PI / 2]},
    },
}
ROOM_BEDS = {
    "WARD-2A": ((1.5, 6.3), (2.5, 6.3), (1.5, 8.3), (2.5, 8.3)),
    "WARD-3B": ((8.2, 6.3), (9.2, 6.3), (8.2, 8.3), (9.2, 8.3)),
    "ICU": ((6.0, -4.0), (8.0, -4.0)),
}

TAG_SIZE = 0.25      # black AprilTag square side, metres
TAG_PAD = TAG_SIZE * (160 / 704)  # 160px margin on a 704px marker
TAG_Z = 1.00         # centre height of each vertical wall marker
BOARD_W = TAG_SIZE + 2 * TAG_PAD  # 1024px board, 704px marker area
BOARD_H = 0.50

# Robot geometry (Phase 2 MediBot 4WD platform)
# Footprint: 0.44 m long x 0.31 m wide with wheels (<= 0.5 x 0.4 m)
WHEEL_R = 0.06
WHEEL_X = 0.14
WHEEL_Y = 0.14
TRACK = 2 * WHEEL_Y       # 0.28 m
CHASSIS = (0.44, 0.24, 0.12)

# Navigation Camera: forward and level, with optical centre at the wall tag height.
CAM = {"width": 960, "height": 720, "fov": 1.6, "x": 0.22, "z": TAG_Z - WHEEL_R}

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
    """Return unique surveyed vertical tag poses on corridor and room walls.

    Pose fields are (x, y, z, yaw, zone, id, wall_axis, wall_plane). ``yaw``
    points along the wall-normal from the wall into the visible space.
    """
    tags = []
    tag_id = 0

    def add_tag(x, y, yaw, zone, wall_axis, wall_plane):
        nonlocal tag_id
        tags.append((x, y, TAG_Z, yaw, zone, tag_id, wall_axis, wall_plane))
        tag_id += 1
    # Long south corridor wall; the face toward the corridor is y=-0.64.
    # Long north wall tags are offset toward the corridor and alternate with these.
    for x, zone in [(0.0, "home_south"), (3.0, "pharmacy_south_door_east"),
                    (5.9, "junction_south"), (8.4, "icu_south")]:
        add_tag(x, -0.63, PI / 2, zone, "y", -0.64)
    for x, zone in [(2.5, "home_north"), (5.5, "ward2a_north_approach"),
                    (8.5, "ward3b_north_approach")]:
        add_tag(x, 3.63, -PI / 2, zone, "y", 3.64)

    # Door-adjacent markers on the corridor side, clear of the door openings.
    for x, zone in [(1.0, "pharmacy_door_west"),
                    (6.0, "icu_door_west"), (8.0, "icu_door_east")]:
        add_tag(x, -0.63, PI / 2, zone, "y", -0.64)
    for x, zone in [(3.15, "ward2a_door_west"), (4.9, "ward2a_door_east"),
                    (6.0, "ward3b_door_west"), (8.0, "ward3b_door_east")]:
        add_tag(x, 3.63, -PI / 2, zone, "y", 3.64)

    # Head-on tags at corridor ends, flush to the west/east perimeter walls.
    add_tag(-2.53, 0.0, 0.0, "home_end", "x", -2.54)
    add_tag(9.13, 0.0, PI, "east_corridor_end", "x", 9.14)

    # Room-facing approach tags at the door jambs, plus head-on tags on the
    # opposite walls.  The room-facing tags support localization on entry.
    add_tag(1.9, -2.73, PI / 2, "pharmacy_room", "y", -2.74)
    add_tag(8.1, -0.77, -PI / 2, "icu_door_facing", "y", -0.76)
    add_tag(3.1, 3.77, PI / 2, "ward2a_door_facing", "y", 3.76)
    add_tag(8.1, 3.77, PI / 2, "ward3b_door_facing", "y", 3.76)
    add_tag(7.0, -5.53, PI / 2, "icu_room", "y", -5.54)
    add_tag(3.8, 9.43, -PI / 2, "ward2a_room", "y", 9.44)
    add_tag(7.0, 9.43, -PI / 2, "ward3b_room", "y", 9.44)
    # Room-side approach tags cover the short final aisle from either heading.
    # These sit on the shared room divider walls, with their normals facing inward.
    add_tag(5.13, 4.7, PI, "ward2a_entry_side", "x", 5.14)
    add_tag(5.27, 4.7, 0.0, "ward3b_entry_side", "x", 5.26)

    # Door-jamb tags are head-on from the graph's door nodes.
    add_tag(3.39, 3.30, 0.0, "ward2a_door_jamb", "x", 3.40)
    add_tag(6.39, 3.30, 0.0, "ward3b_door_jamb", "x", 6.40)
    add_tag(6.39, -0.70, 0.0, "icu_door_jamb", "x", 6.40)

    return tags


def tag_plate(tag_id, x, y, z, yaw):
    """Vertical, thin tag plate with its outward normal at surveyed ``yaw``."""
    rotation = yaw - PI / 2  # local +Y face becomes the surveyed horizontal normal
    return (f"Solid {{\n"
            f"  translation {x:g} {y:g} {z:g}\n"
            f"  rotation 0 0 1 {rotation:g}\n"
            f"  children [\n"
            f"    Shape {{\n"
            f"      appearance PBRAppearance {{\n"
            f"        baseColor 1 1 1\n"
            f"        roughness 0.5\n"
            f"        metalness 0\n"
            f"        baseColorMap ImageTexture {{ url [ \"textures/tag_{tag_id:02d}.png\" ] filtering 0 }}\n"
            f"      }}\n"
            # The PNG includes its white border around the black marker, so
            # the physical plate must match BOARD_W, not just the black tag.
            f"      geometry Box {{ size {BOARD_W:g} 0.01 {BOARD_W:g} }}\n"
            f"    }}\n"
            f"  ]\n"
            f"  boundingObject Box {{ size {BOARD_W:g} 0.01 {BOARD_W:g} }}\n"
            f"  name \"tag_{tag_id}\"\n"
            f"}}\n")


def generate_tag_map(tags):
    """Generate tag_map.md with tag details."""
    md = "# AprilTag Map\n\n"
    md += "| ID | Location | Wall | Room | Position (x, y, z) | Yaw |\n"
    md += "|----|----------|------|------|-------------------|-----|\n"

    for x, y, z, yaw, zone, tag_id, wall_axis, wall_plane in tags:
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
    tw = bbox[2] - bbox[0]
    x = (1024 - tw) / 2
    y = 884  # In the lower white margin (marker ends at y=864)
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
def box_solid(name, x, y, z, sx, sy, sz, rgb, collide=True, yaw=0.0, def_name=None, roughness=1.0, metalness=0.0, transparency=0.0):
    d = f"DEF {def_name} " if def_name else ""
    # SFColor is exactly three normalized channels; alpha belongs in
    # transparency, never as a fourth baseColor component.
    rgb = tuple(max(0.0, min(1.0, float(c))) for c in rgb[:3])
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
         f"        transparency {max(0.0, min(1.0, transparency)):g}\n"
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
    rgb = tuple(max(0.0, min(1.0, float(c))) for c in rgb[:3])
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


def tag_face(tex_name, width=BOARD_W, height=None):
    """Verified +X-facing textured quad; optional face-sized dimensions."""
    height = width if height is None else height
    hw, hh = width / 2, height / 2
    return (f"Shape {{\n"
            f"  appearance Appearance {{ texture ImageTexture {{ url [ \"textures/{tex_name}\" ] filtering 0 }} }}\n"
            f"  geometry IndexedFaceSet {{\n"
            f"    coord Coordinate {{ point [ 0.006 {-hw:g} {-hh:g}, 0.006 {hw:g} {-hh:g}, 0.006 {hw:g} {hh:g}, 0.006 {-hw:g} {hh:g} ] }}\n"
            f"    texCoord TextureCoordinate {{ point [ 0 0, 1 0, 1 1, 0 1 ] }}\n"
            f"    coordIndex [ 0 1 2 3 -1 ]\n"
            f"    texCoordIndex [ 0 1 2 3 -1 ]\n"
            f"  }}\n"
            f"}}\n")


def make_face_image(person_id):
    """Generate a distinct, non-photographic face image for the simulation stub."""
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(person_id))
    skin = (196 + seed % 45, 135 + (seed // 5) % 65, 105 + (seed // 11) % 55)
    hair = (30 + seed % 90, 22 + (seed // 7) % 55, 18 + (seed // 13) % 45)
    portrait = Image.new("RGB", (256, 320), (226, 235, 241))
    d = ImageDraw.Draw(portrait)
    d.ellipse((26, 12, 230, 268), fill=hair)
    d.ellipse((47, 32, 209, 255), fill=skin)
    d.rectangle((50, 42, 206, 116), fill=hair)
    d.ellipse((72, 128, 100, 146), fill=(28, 35, 43))
    d.ellipse((156, 128, 184, 146), fill=(28, 35, 43))
    d.line((128, 144, 116, 186, 135, 188), fill=(115, 72, 61), width=6)
    d.arc((88, 174, 170, 230), start=10, end=170, fill=(115, 44, 48), width=6)
    d.rectangle((86, 257, 170, 319), fill=(36 + seed % 60, 90, 138))
    portrait = portrait.resize((384, 480), Image.Resampling.LANCZOS)
    image = Image.new("RGB", (512, 512), (239, 243, 246))
    image.paste(portrait, ((512 - portrait.width) // 2, (512 - portrait.height) // 2))
    return image


def make_compartment_image(number):
    """Create a power-of-two compartment number badge for a textured quad."""
    image = Image.new("RGB", (512, 512), (232, 240, 247))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((52, 52, 460, 460), radius=58, fill=(25, 67, 104))
    try:
        font = ImageFont.truetype("arial.ttf", 250)
    except OSError:
        font = ImageFont.load_default()
    draw.text((256, 256), str(number), fill=(255, 255, 255), font=font, anchor="mm")
    return image


def visitor_nodes(people):
    """Generate off-stage standing visitors and their individually identified RFID cards."""
    nodes = []
    for i, person in enumerate(people):
        pid = person["id"]
        x, y = -30.0 - i * 3.0, -20.0
        face_path = person["face_image"].replace("\\", "/")
        nodes.append(f'''DEF VISITOR_{pid} Solid {{
  translation {x:g} {y:g} 0
  name "visitor_{pid}"
  children [
    Shape {{ appearance PBRAppearance {{ baseColor 0.16 0.34 0.55 roughness 0.8 }} geometry Cylinder {{ radius 0.15 height 0.68 }} }}
    Pose {{ translation 0 0 0.96 children [ Shape {{ appearance PBRAppearance {{ baseColor 0.82 0.72 0.62 roughness 0.8 }} geometry Sphere {{ radius 0.14 }} }} ] }}
    Pose {{ translation 0.145 0 0.96 children [ {tag_face(face_path, 0.19, 0.25)} ] }}
  ]
}}
DEF CARD_{pid} Solid {{
  translation {x:g} {y:g} 0.7
  name "card_{pid}"
  children [ Shape {{ appearance PBRAppearance {{ baseColor 0.95 0.72 0.12 roughness 0.45 }} geometry Box {{ size 0.09 0.06 0.008 }} }} ]
  boundingObject Box {{ size 0.09 0.06 0.008 }}
  physics Physics {{ density -1 mass 0.02 }}
}}
''')
    return "".join(nodes)


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
def hospital_bed(name, x, y, yaw, has_monitor=False, is_icu=False, lateral_side=-1):
    parts = []
    bed_col = (0.2, 0.45, 0.7) if is_icu else (0.85, 0.88, 0.92)
    parts.append(box_solid(f"{name}_frame", x, y, 0.15, 1.8, 0.85, 0.25, (0.25, 0.28, 0.3), collide=True, yaw=yaw))
    parts.append(box_solid(f"{name}_mattress", x, y, 0.32, 1.7, 0.8, 0.12, (0.95, 0.95, 0.98), collide=False, yaw=yaw))
    bx, by = x + 0.35 * math.cos(yaw), y + 0.35 * math.sin(yaw)
    parts.append(box_solid(f"{name}_blue_blanket", bx, by, 0.395, 0.90, 0.78, 0.035,
                           (0.20, 0.43, 0.82), collide=False, yaw=yaw, roughness=0.9))
    hx = x - 0.85 * math.cos(yaw)
    hy = y - 0.85 * math.sin(yaw)
    parts.append(box_solid(f"{name}_headboard", hx, hy, 0.38, 0.08, 0.85, 0.45, bed_col, collide=False, yaw=yaw))
    px = x - 0.65 * math.cos(yaw)
    py = y - 0.65 * math.sin(yaw)
    parts.append(box_solid(f"{name}_pillow", px, py, 0.40, 0.35, 0.55, 0.08, (1, 1, 1), collide=False, yaw=yaw))
    side_x = -0.55 * math.sin(yaw) * lateral_side
    side_y = 0.55 * math.cos(yaw) * lateral_side
    bx = x + side_x
    by = y + side_y
    parts.append(box_solid(f"{name}_cabinet", bx, by, 0.25, 0.4, 0.4, 0.5, (0.85, 0.9, 0.95), collide=True, yaw=yaw))
    parts.append(box_solid(f"{name}_bed_number", x + 0.45 * math.cos(yaw),
                           y + 0.45 * math.sin(yaw), 0.43, 0.30, 0.16, 0.012,
                           (0.95, 0.98, 1.0), collide=False, yaw=yaw))
    if has_monitor:
        mx = x + side_x - 0.4 * math.cos(yaw)
        my = y + side_y - 0.4 * math.sin(yaw)
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
    # Ten angled counter modules form a shallow curved reception desk.
    for i, angle_deg in enumerate(range(-100, 101, 20)):
        angle = math.radians(angle_deg)
        px, py = x + 0.58 * math.sin(angle), y + 0.58 * math.cos(angle)
        yaw = -angle
        parts.append(box_solid(f"nurse_counter_{i}", px, py, 0.43, 0.36, 0.25, 0.86,
                               (0.2, 0.35, 0.5), collide=True, yaw=yaw))
    for i, px in enumerate((x - 0.32, x + 0.32)):
        parts.append(box_solid(f"nurse_pc_{i}", px, y + 0.48, 0.98, 0.06, 0.28, 0.22,
                               (0.12, 0.14, 0.18), collide=False))
        parts.append(box_solid(f"nurse_screen_{i}", px, y + 0.45, 0.98, 0.015, 0.23, 0.16,
                               (0.12, 0.62, 0.88), collide=False))
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
    DEF LOCK_LED_{n} LED {{
      color [ 1 0 0 ]
      translation {x:g} 0.115 0.135
      name "led_lock_{n}"
      gradual TRUE
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 1 1 1 roughness 0.2 emissiveColor 0.15 0.15 0.15 }}
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
        RotationalMotor {{ name "lid_{n}" minPosition -1.6 maxPosition 0 maxVelocity 3 maxTorque 2 }}
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
                  Shape {{
                    appearance PBRAppearance {{ baseColor 0.05 0.08 0.12 roughness 0.4 }}
                    geometry Box {{ size 0.002 0.03 0.03 }}
                  }}
                  Pose {{
                    rotation 0 1 0 -1.5708
                    children [ {tag_face(f"compartment_{n}.png", 0.026)} ]
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
    DEF STATUS_LED LED {{
      color [ 0 1 0 ]
      translation 0.236 0 0.038
      name "status_led"
      gradual TRUE
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 1 1 1 roughness 0.2 emissiveColor 0.15 0.15 0.15 }}
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
    # Live top-down navigation and handover dashboard
    Display {{
      translation 0.205 0 0.72
      rotation 0 1 0 1.5708
      name "dashboard"
      width 480
      height 320
    }}
    # Visible emulated RFID reader module on the front bumper
    Pose {{
      translation 0.242 0 0.15
      children [
        Shape {{ appearance PBRAppearance {{ baseColor 0.10 0.12 0.16 roughness 0.45 metalness 0.15 }} geometry Box {{ size 0.055 0.095 0.07 }} }}
        Shape {{ appearance PBRAppearance {{ baseColor 0.12 0.72 0.92 emissiveColor 0.05 0.25 0.35 }} geometry Box {{ size 0.012 0.055 0.04 }} }}
      ]
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
    with open(PATIENTS_PATH, encoding="utf-8") as patient_file:
        patient_data = json.load(patient_file)
    people = [relative for patient in patient_data["patients"]
              for relative in patient["authorised_family"]]
    people.extend(patient_data["unauthorised_visitors"])
    # R2025a header
    parts.append("#VRML_SIM R2025a utf8\n\n")

    # One named default overhead viewpoint; Follow Object can be selected in Webots.
    axis, ang = look_at((3.5, -8.0, 12.0), (3.5, 1.5, 0.0))
    parts.append(f"""WorldInfo {{
  title "MediBot hospital delivery"
  basicTimeStep 32
  coordinateSystem "ENU"
}}
DEF VIEW_OVERVIEW Viewpoint {{
  description "Overview - hospital floor"
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
    parts.append(box_solid("floor", 4.1, 1.95, -0.05, 13.6, 15.1, 0.1, (0.86, 0.88, 0.90)))

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
    # Outer perimeter boundary walls (first one carries the DEF, rest USE it)
    outer_x = 10.8
    south_y, north_y = -5.6, 9.5
    parts.append(box_solid_textured("wall_south_outer", 4.1, south_y, wz, 14.0, 0.12, wh, wall_appearance_def))
    parts.append(box_solid_textured("wall_north_outer", 4.1, north_y, wz, 14.0, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_west_outer", -2.60, (south_y + north_y) / 2, wz,
                                   0.12, north_y - south_y, wh, wall_appearance))
    parts.append(box_solid_textured("wall_east_outer", outer_x, (south_y + north_y) / 2, wz,
                                   0.12, north_y - south_y, wh, wall_appearance))

    # South Corridor Wall along y = -0.70 with door gaps at Pharmacy (x=2) and ICU (x=7)
    parts.append(box_solid_textured("wall_s_alcove", -1.55, -0.70, wz, 2.1, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_s_pharm_w", 0.95, -0.70, wz, 0.9, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_s_pharm_e", 2.95, -0.70, wz, 0.7, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_s_icu_w", 6.05, -0.70, wz, 0.7, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_s_icu_e", 8.40, -0.70, wz, 1.6, 0.12, wh, wall_appearance))

    # Room partition dividers (South side)
    parts.append(box_solid_textured("div_pharm_w", 0.50, -1.75, wz, 0.12, 2.1, wh, wall_appearance))
    parts.append(box_solid_textured("div_pharm_e", 3.30, -1.75, wz, 0.12, 2.1, wh, wall_appearance))
    parts.append(box_solid_textured("div_icu_w", 4.80, -3.15, wz, 0.12, 4.9, wh, wall_appearance))

    # North Corridor Wall along y = 3.70 with door gaps at Ward 2A (x=4) and Ward 3B (x=7)
    parts.append(box_solid_textured("wall_n_ward2a_w", 2.80, 3.70, wz, 1.2, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_n_middle", 5.50, 3.70, wz, 1.8, 0.12, wh, wall_appearance))
    parts.append(box_solid_textured("wall_n_ward3b_e", 8.40, 3.70, wz, 1.6, 0.12, wh, wall_appearance))

    # Room partition dividers (North side)
    parts.append(box_solid_textured("div_ward2a_w", 0.40, 6.60, wz, 0.12, 5.8, wh, wall_appearance))
    parts.append(box_solid_textured("div_wards_mid", 5.20, 6.60, wz, 0.12, 5.8, wh, wall_appearance))

    # West alcove divider (Home / Reception wall)
    parts.append(box_solid_textured("wall_w_home_n", -1.55, 0.70, wz, 2.1, 0.12, wh, wall_appearance))

    # Floor with corridor texture
    parts.append(box_solid_textured("floor_main", 4.1, 1.95, 0.01, 13.6, 15.1, 0.02, floor_appearance_def))

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

    # ICU entrance has no leaf; the clear opening is the configured width.
    frame_offset = (DOOR_WIDTH + 0.05) / 2
    parts.append(box_solid("door_frame_icu_l", 7.0 - frame_offset, -0.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_icu_r", 7.0 + frame_offset, -0.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_icu_t", 7.0, -0.64, 1.35, DOOR_WIDTH + 0.05, 0.05, 0.1, (0.95, 0.95, 0.95)))

    # Ward 2A door frame
    parts.append(box_solid("door_frame_ward2a_l", 4.0 - frame_offset, 3.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_ward2a_r", 4.0 + frame_offset, 3.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_ward2a_t", 4.0, 3.76, 1.35, DOOR_WIDTH + 0.05, 0.05, 0.1, (0.95, 0.95, 0.95)))

    # Ward 3B door frame
    parts.append(box_solid("door_frame_ward3b_l", 7.0 - frame_offset, 3.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_ward3b_r", 7.0 + frame_offset, 3.70, 0.7, 0.05, 0.12, 1.4, (0.95, 0.95, 0.95)))
    parts.append(box_solid("door_frame_ward3b_t", 7.0, 3.76, 1.35, DOOR_WIDTH + 0.05, 0.05, 0.1, (0.95, 0.95, 0.95)))

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

    # 4. WAITING AREA: Four individual chairs, planters and wall storage.
    for idx, wx in enumerate((3.55, 4.25, 4.95, 5.65)):
        parts.append(hospital_bench(f"waiting_chair_{idx + 1}", wx, -2.25, 0.0))
    for idx, (px, py) in enumerate(((3.35, -1.25), (5.95, -1.25), (8.75, 1.0))):
        parts.append(cylinder_solid(f"planter_{idx}", px, py, 0.18, 0.22, 0.34, (0.72, 0.75, 0.76), collide=True))
        parts.append(cylinder_solid(f"plant_trunk_{idx}", px, py, 0.75, 0.035, 0.8, (0.30, 0.22, 0.12)))
        for leaf_i, (ox, oy) in enumerate(((0.0, 0.0), (0.16, 0.02), (-0.13, 0.12), (0.04, -0.16))):
            parts.append(cylinder_solid(f"plant_leaf_{idx}_{leaf_i}", px + ox, py + oy, 1.0, 0.10, 0.34,
                                        (0.18, 0.48, 0.28), yaw=leaf_i * PI / 4))
    parts.append(box_solid("waiting_wall_cabinet", 4.6, -2.72, 1.05, 1.8, 0.20, 0.48,
                           (0.88, 0.91, 0.94), collide=False))

    # 5. ICU: 2 ICU Intensive Care Beds with Monitors
    parts.append(hospital_bed("icu_bed_1", 6.0, -4.0, PI / 2,
                              has_monitor=True, is_icu=True, lateral_side=1))
    parts.append(hospital_bed("icu_bed_2", 8.0, -4.0, -PI / 2,
                              has_monitor=True, is_icu=True, lateral_side=1))

    # 6. NURSE STATION: Central Island Desk & Monitors
    parts.append(nurse_station_island(5.5, 1.5))
    parts.append(wheelchair("nurse_station_chair", 5.5, 0.75, PI / 2))
    parts.append(cylinder_solid("nurse_clock_rim", 5.5, 2.75, 1.12, 0.22, 0.06, (0.18, 0.22, 0.26)))

    # 7. WARD 2A: 4 Hospital Beds + Bedside Tables + Curtains
    for idx, (bx, by) in enumerate(ROOM_BEDS["WARD-2A"]):
        parts.append(hospital_bed(f"ward2a_bed_{idx + 1}", bx, by, -PI / 2, lateral_side=-1))
    for idx, by in enumerate((6.5, 8.3)):
        parts.append(box_solid(f"ward2a_curtain_{idx + 1}", 2.0, by, 0.75, 0.03, 1.2, 1.4,
                               (0.35, 0.7, 0.7), collide=False))
    parts.append(cylinder_solid("ward2a_curtain_rail", 2.0, 7.4, 1.48, 0.025, 2.6,
                                (0.78, 0.82, 0.86), yaw=PI / 2))

    # 8. WARD 3B: 4 Hospital Beds + Bedside Tables + Curtains
    for idx, (bx, by) in enumerate(ROOM_BEDS["WARD-3B"]):
        parts.append(hospital_bed(f"ward3b_bed_{idx + 1}", bx, by, -PI / 2, lateral_side=1))
    for idx, by in enumerate((6.5, 8.3)):
        parts.append(box_solid(f"ward3b_curtain_{idx + 1}", 8.7, by, 0.75, 0.03, 1.2, 1.4,
                               (0.35, 0.7, 0.7), collide=False))
    parts.append(cylinder_solid("ward3b_curtain_rail", 8.7, 7.4, 1.48, 0.025, 2.8,
                                (0.78, 0.82, 0.86), yaw=PI / 2))

    # High-set outer-wall glazing and contrasting lower privacy panels.
    for name, wx in (("ward2a", 3.75), ("ward3b", 8.8)):
        parts.append(box_solid(f"{name}_outer_window", wx, 9.32, 0.92, 1.75, 0.035, 0.72,
                               (0.48, 0.72, 0.84), collide=False, transparency=0.28, roughness=0.2))
        parts.append(box_solid(f"{name}_window_sill", wx, 9.28, 0.53, 1.85, 0.12, 0.07,
                               (0.88, 0.90, 0.92), collide=False))

    # Medicine delivery counter and pharmacy stock in coded, colour-coded cartons.
    parts.append(box_solid("medicine_delivery_counter", 0.0, -1.65, 0.48, 1.7, 0.58, 0.92,
                           (0.90, 0.93, 0.96), collide=True))
    parts.append(box_solid("medicine_delivery_monitor", -0.45, -1.62, 1.05, 0.06, 0.32, 0.24,
                           (0.12, 0.14, 0.18), collide=False))
    stock_colors = ((0.88, 0.25, 0.28), (0.22, 0.55, 0.84), (0.88, 0.68, 0.18),
                    (0.32, 0.70, 0.42), (0.72, 0.38, 0.72), (0.92, 0.92, 0.90))
    for i, color in enumerate(stock_colors):
        sx, sy = 1.05 + (i % 3) * 0.62, -2.48 - (i // 3) * 0.25
        parts.append(box_solid(f"pharmacy_medicine_box_{i + 1}", sx, sy, 0.42, 0.20, 0.16, 0.20,
                               color, collide=False))

    # 9. Wall-mounted AprilTag plates (thin plates on walls)
    wall_tags = generate_wall_tags()
    tags = []
    for x, y, z, yaw, zone, tag_id, wall_axis, wall_plane in wall_tags:
        parts.append(tag_plate(tag_id, x, y, z, yaw))
        tags.append({"id": tag_id, "x": x, "y": y, "yaw": yaw, "z": z,
                     "zone": zone, "wall_axis": wall_axis, "wall_plane": wall_plane})

    # 10. Trolley Obstacle (DEF TROLLEY) - with physics for collision detection
    tx, ty = TROLLEY_PARK
    parts.append(
        f"DEF TROLLEY Solid {{\n"
        f"  translation {tx:g} {ty:g} 0.1\n"
        f"  rotation 0 0 1 0\n"
        f"  name \"trolley\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{\n"
        f"        baseColor 0.82 0.08 0.06\n"
        f"        roughness 1\n"
        f"        metalness 0\n"
        f"      }}\n"
        f"      geometry Box {{ size 0.6 0.5 0.2 }}\n"
        f"    }}\n"
        f"  ]\n"
        f"  boundingObject Box {{ size 0.6 0.5 0.2 }}\n"
        f"  physics Physics {{ density -1 mass 15 }}\n"
        f"}}\n"
    )

    # Alternate movable obstacle models stay below the floor until selected by B.
    parts.append("""DEF OBSTACLE_WHEELCHAIR Solid {
  translation -1.5 1.8 -10
  name "wheelchair_obstacle"
  children [
    Shape { appearance PBRAppearance { baseColor 0.12 0.32 0.72 roughness 0.75 }
      geometry Box { size 0.6 0.5 0.45 } }
    Shape { appearance PBRAppearance { baseColor 0.08 0.08 0.1 roughness 0.8 }
      geometry Cylinder { radius 0.24 height 0.035 } }
  ]
  boundingObject Box { size 0.6 0.5 0.45 }
  physics Physics { density -1 mass 18 }
}
DEF OBSTACLE_CART Solid {
  translation -1.5 1.8 -10
  name "equipment_cart_obstacle"
  children [ Shape { appearance PBRAppearance { baseColor 0.72 0.76 0.8 roughness 0.55 metalness 0.35 }
    geometry Box { size 0.65 0.55 0.55 } } ]
  boundingObject Box { size 0.65 0.55 0.55 }
  physics Physics { density -1 mass 20 }
}
""")

    # Park every generated visitor and RFID card off-stage until a handover.
    parts.append(visitor_nodes(people))

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
    with open(PATIENTS_PATH, encoding="utf-8") as patient_file:
        patient_data = json.load(patient_file)
    people = [relative for patient in patient_data["patients"]
              for relative in patient["authorised_family"]]
    people.extend(patient_data["unauthorised_visitors"])
    face_dir = os.path.join(TEX_DIR, "faces")
    os.makedirs(face_dir, exist_ok=True)
    for person in people:
        make_face_image(person["id"]).save(os.path.join(face_dir, f"{person['id']}.png"))
    for number in (1, 2, 3):
        make_compartment_image(number).save(os.path.join(TEX_DIR, f"compartment_{number}.png"))
    for number in (1, 2, 3):
        make_compartment_image(number).save(os.path.join(TEX_DIR, f"compartment_{number}.png"))

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
        "tags": [{"id": t["id"], "x": t["x"], "y": t["y"], "yaw": t["yaw"], "z": t["z"],
                  "zone": t["zone"], "wall_axis": t["wall_axis"], "wall_plane": t["wall_plane"]}
                 for t in tags],
        "rooms": ROOMS,
        "tag": {"size": TAG_SIZE},
        "camera": CAM,
        "face_camera": FACE_CAM,
        "robot": {
            "wheel_radius": WHEEL_R,
            "track": 2 * WHEEL_Y,
            "start": [0.0, 0.0, 0.0],
            "z": WHEEL_R,
            "footprint": list(CHASSIS),
            "clearance_footprint": [0.46, 2 * (WHEEL_Y + 0.016)],
            "mast_footprint": [0.045, 0.055],
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
