#!/usr/bin/env python3
"""
validate_world.py  –  Grammar-aware + node-type-aware validator for medibot.wbt

Checks
------
1.  File exists and is non-empty
2.  Correct R2025a header
3.  Brace balance == 0
4.  Final parse depth == 0
5.  No 'repeat TRUE'  (removed in R2025a)
6.  No bare 'Transform {'  (use Pose in R2025a)
7.  No '../worlds/' texture paths
8.  All referenced textures exist on disk
9.  No duplicate DEF names
10. Grammar: DEF/USE never appears where a field name is expected
11. Node-type field check: 'texture' is not a valid field in PBRAppearance
    (must be 'baseColorMap'); flags any unknown field that matches a known
    wrong name for that node type.
"""
import os, re, sys

ROOT   = os.path.dirname(os.path.abspath(__file__))
WBT    = os.path.join(ROOT, "worlds", "medibot.wbt")
TEXDIR = os.path.join(ROOT, "worlds", "textures")
HEADER = "#VRML_SIM R2025a utf8"

# Fields that are WRONG for a given node type → suggested replacement
WRONG_FIELDS = {
    "PBRAppearance": {
        "texture": "baseColorMap",
    },
    "ImageTexture": {
        "repeat": "(remove — not valid in R2025a)",
    },
}

errors = []

def err(msg):  errors.append(msg);  print(f"  ERROR   {msg}")
def ok(msg):                         print(f"  ok      {msg}")

# ── Load ─────────────────────────────────────────────────────────────────────
print(f"\n[validate_world]  {WBT}\n")
if not os.path.isfile(WBT):
    err("worlds/medibot.wbt not found"); sys.exit(1)

size = os.path.getsize(WBT)
ok(f"file exists ({size:,} bytes)") if size >= 1000 else err(f"file too small ({size} bytes)")

with open(WBT, encoding="utf-8") as fh:
    raw = fh.read(); lines = raw.splitlines()

# ── 2. Header ────────────────────────────────────────────────────────────────
(ok if lines[0].strip() == HEADER else err)(
    f"header: {lines[0].strip()!r}" if lines[0].strip() != HEADER else "header correct")

# ── 3&4. Brace balance ───────────────────────────────────────────────────────
o, c = raw.count("{"), raw.count("}")
(ok if o == c else err)(f"brace balance {o-c:+d}  ({{ {o}  }} {c})" if o != c else f"brace balance OK  ({{ {o}  }} {c})")
depth = sum(l.count("{") - l.count("}") for l in lines)
(ok if depth == 0 else err)(f"parse depth at EOF: {depth}" if depth else "parse depth returns to 0")

# ── 5. repeat TRUE ───────────────────────────────────────────────────────────
hits = [i+1 for i,l in enumerate(lines) if re.search(r'\brepeat\s+TRUE\b', l)]
[err(f"L{n}: 'repeat TRUE' invalid in R2025a") for n in hits] or ok("no 'repeat TRUE' fields")

# ── 6. bare Transform ────────────────────────────────────────────────────────
hits = [i+1 for i,l in enumerate(lines) if re.match(r'\s*Transform\s*\{', l)]
[err(f"L{n}: bare Transform — use Pose in R2025a") for n in hits] or ok("no bare Transform nodes")

# ── 7. ../worlds/ paths ──────────────────────────────────────────────────────
hits = [i+1 for i,l in enumerate(lines) if '../worlds/' in l]
[err(f"L{n}: '../worlds/' in texture URL") for n in hits] or ok("no '../worlds/' paths")

# ── 8. Texture files on disk ─────────────────────────────────────────────────
refs = re.findall(r'"textures/([^"]+)"', raw)
miss = sorted({t for t in refs if not os.path.isfile(os.path.join(TEXDIR, t))})
[err(f"missing texture: textures/{t}") for t in miss] or ok(f"all {len(set(refs))} textures present")

# ── 9. Duplicate DEFs ────────────────────────────────────────────────────────
names = re.findall(r'\bDEF\s+(\w+)', raw)
seen, dupes = set(), set()
for n in names:
    (dupes if n in seen else seen).add(n); seen.add(n)
[err(f"duplicate DEF: {d}") for d in sorted(dupes)] or ok(f"no duplicate DEFs ({len(seen)} unique)")

# ── 10 & 11. Tokeniser: grammar + node-type field check ──────────────────────
TOKEN = re.compile(
    r'(?P<comment>#[^\n]*)'
    r'|(?P<string>"[^"]*")'
    r'|(?P<lbrace>\{)'
    r'|(?P<rbrace>\})'
    r'|(?P<lbracket>\[)'
    r'|(?P<rbracket>\])'
    r'|(?P<word>[A-Za-z_][\w.]*)'
    r'|(?P<number>[-+]?\d[\d.eE+\-]*)'
    r'|(?P<comma>,)'
    r'|(?P<ws>\s+)',
    re.MULTILINE,
)

WANT_FIELD = "F"   # inside node body: next non-trivial token must be a field name or }
WANT_VALUE = "V"   # after field name: next is a value
FREE       = "R"   # inside [...] or top-level

def linecol(pos):
    ln = raw[:pos].count('\n') + 1
    col = pos - raw.rfind('\n', 0, pos)
    return ln, col

grammar_errors = []
# Stack of (state, node_type)
stack = [(FREE, "")]

for m in TOKEN.finditer(raw):
    kind, val, pos = m.lastgroup, m.group(), m.start()
    if kind in ('ws', 'comment'): continue
    state, node_type = stack[-1]

    if kind == 'lbrace':
        stack.append((WANT_FIELD, node_type))   # keep node_type context

    elif kind == 'rbrace':
        if stack: stack.pop()
        # consumed a node value — back to WANT_FIELD in parent
        if stack and stack[-1][0] == WANT_VALUE:
            stack[-1] = (WANT_FIELD, stack[-1][1])

    elif kind == 'lbracket':
        stack.append((FREE, node_type))

    elif kind == 'rbracket':
        if stack: stack.pop()
        if stack and stack[-1][0] == WANT_VALUE:
            stack[-1] = (WANT_FIELD, stack[-1][1])

    elif kind == 'word':
        if state == WANT_FIELD:
            if val in ('DEF', 'USE'):
                ln, col = linecol(pos)
                grammar_errors.append(
                    f"L{ln}:{col}: '{val}' where field name expected "
                    f"(missing field name before {val})")
                stack[-1] = (WANT_VALUE, node_type)
            elif val[0].islower() or val == '_':
                # Valid field name — check against known wrong fields
                if node_type in WRONG_FIELDS and val in WRONG_FIELDS[node_type]:
                    ln, col = linecol(pos)
                    sugg = WRONG_FIELDS[node_type][val]
                    grammar_errors.append(
                        f"L{ln}:{col}: field '{val}' not valid in {node_type} "
                        f"(use '{sugg}')")
                stack[-1] = (WANT_VALUE, node_type)
            # else: upper-case = unexpected NodeType in field pos (e.g. USE target)

        elif state == WANT_VALUE:
            if val in ('DEF', 'USE', 'IS'): pass   # valid node/reference prefixes
            elif val[0].isupper():
                # Node type — update context for the { that follows
                # The upcoming lbrace will push a new frame; we need to pass node_type
                # We do this by replacing the top-of-stack with the new type BEFORE { arrives
                # Actually we handle it by peeking: the word before { is the NodeType
                # Store it so lbrace can pick it up
                # Replace the current frame's context
                stack[-1] = (WANT_VALUE, val)
            else:
                # Literal value (TRUE/FALSE/enum) — back to WANT_FIELD
                stack[-1] = (WANT_FIELD, node_type)

        # FREE: anything valid

    elif kind in ('number', 'string', 'comma'):
        pass   # literal values; field transition handled by next lower-case word

if grammar_errors:
    for e in grammar_errors: err(f"grammar: {e}")
else:
    ok("grammar: no DEF/USE in field-name position; no wrong field names in known node types")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"VALIDATION FAILED  –  {len(errors)} error(s)")
    sys.exit(1)
else:
    print(f"VALIDATION PASSED  –  0 errors")
    sys.exit(0)
