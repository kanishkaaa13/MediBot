"""Fail when generated Webots SFColor fields contain channels outside [0, 1]."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "worlds" / "medibot.wbt"
text = WORLD.read_text(encoding="utf-8")
color_fields = ("baseColor", "diffuseColor", "emissiveColor", "color")
pattern = re.compile(
    r"\b(" + "|".join(color_fields) + r")\s+"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)
matches = list(pattern.finditer(text))
bad = [(m.group(1), tuple(float(m.group(i)) for i in (2, 3, 4)),
        text.count("\n", 0, m.start()) + 1)
       for m in matches
       if any(not 0.0 <= float(m.group(i)) <= 1.0 for i in (2, 3, 4))]
for field, values, line in bad:
    print(f"FAIL SFColor {field} at line {line}: {values}")
assert not bad, f"{len(bad)} out-of-range SFColor value(s) in {WORLD}"
print(f"PASS {len(matches)} SFColor fields in {WORLD.name} are within [0, 1]")
