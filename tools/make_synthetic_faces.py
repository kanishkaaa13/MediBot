#!/usr/bin/env python3
"""Generate structurally varied synthetic portraits and evaluate them locally.

Image construction uses Pillow and NumPy only. The existing OpenCV YuNet/SFace
models are used only by the optional seed-search/evaluation command.
"""
from __future__ import annotations

import argparse
import io
import itertools
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
FACES = ROOT / "controllers" / "medibot_controller" / "faces"
MODELS = ROOT / "controllers" / "medibot_controller" / "models"
SIZE = 512
SCALE = 2
FILES = ("family_1.jpg", "family_2.jpg", "VISITOR_UNKNOWN_01.jpg", "VISITOR_UNKNOWN_02.jpg")


def _params(seed: int) -> dict:
    """One stable, identity-defining parameter set for a seed."""
    r = np.random.default_rng(int(seed))
    return {
        "seed": int(seed), "ratio": float(r.uniform(.70, .95)),
        "height": float(r.uniform(278, 322)), "width": float(r.uniform(196, 230)),
        "jaw": str(r.choice(("pointed", "square", "round"))),
        "forehead": float(r.uniform(.26, .37)), "cheek": float(r.uniform(.82, 1.16)),
        "eye_size": float(r.uniform(13, 21)), "eye_spacing": float(r.uniform(.31, .47)),
        "eye_y": float(r.uniform(.39, .48)), "eye_tilt": float(r.uniform(-12, 12)),
        "lid": str(r.choice(("almond", "hooded", "round", "narrow"))),
        "brow_thick": float(r.uniform(2.8, 7.2)), "brow_arch": float(r.uniform(-8, 10)),
        "brow_gap": float(r.uniform(3, 13)),
        "nose_len": float(r.uniform(.15, .27)), "nose_width": float(r.uniform(14, 28)),
        "nostril": float(r.uniform(2, 5.5)), "mouth_width": float(r.uniform(34, 68)),
        "lip_thick": float(r.uniform(2, 6)), "mouth_y": float(r.uniform(.70, .79)),
        "corner": float(r.uniform(-7, 7)), "ear_size": float(r.uniform(.085, .15)),
        "ear_y": float(r.uniform(.43, .57)),
        "hair_style": str(r.choice(("crop", "side_part", "long", "receding", "volume"))),
        "hairline": float(r.uniform(-.12, .16)), "beard": str(r.choice(("none", "stubble", "beard", "moustache"))),
        "glasses": str(r.choice(("none", "round", "rectangular", "none"))),
        "detail": str(r.choice(("none", "freckles", "wrinkles", "freckles", "none"))),
        "age": float(r.uniform(0, 1)), "skin": r.uniform([119, 75, 54], [232, 190, 160]),
        "hair": r.uniform([16, 12, 10], [100, 76, 56]),
        "iris": tuple(int(x) for x in r.choice(((61, 91, 104), (91, 65, 47), (54, 91, 67), (105, 81, 58)))),
        "bg1": tuple(int(x) for x in r.integers(218, 249, 3)),
        "bg2": tuple(int(x) for x in r.integers(224, 255, 3)),
        "light_angle": float(r.uniform(-math.pi, math.pi)),
        "pose": float(r.uniform(-5, 5)),
    }


def _paint(seed: int, variant: bool = False) -> Image.Image:
    p = _params(seed)
    rng = np.random.default_rng(int(seed) + (991 if variant else 0))
    S = SIZE * SCALE
    cx = SIZE * .5
    top, bottom = 60.0 + p["forehead"] * 32, 443.0
    fh = bottom - top
    fw = p["height"] * p["ratio"]
    # Ratio sets the actual head proportions, while width remains in a useful
    # detector-friendly range. A soft horizontal background gradient differs
    # by identity but remains plain and unobtrusive.
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    mix = np.clip(.34 + .28 * (xx / SIZE) + .16 * np.sin(yy / SIZE * np.pi), 0, 1)[..., None]
    bg = np.asarray(p["bg1"])[None, None, :] * (1 - mix) + np.asarray(p["bg2"])[None, None, :] * mix
    canvas = Image.fromarray(np.uint8(np.clip(bg, 0, 255))).resize((S, S), Image.Resampling.BICUBIC).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")

    def box(b): return tuple(round(v * SCALE) for v in b)
    def poly(points, fill, outline=None, width=1):
        pts = [(round(x * SCALE), round(y * SCALE)) for x, y in points]
        draw.polygon(pts, fill=fill)
        if outline: draw.line(pts + pts[:1], fill=outline, width=max(1, round(width * SCALE)), joint="curve")

    # Neck is behind the head so its upper edge disappears naturally under
    # the jaw instead of drawing a dark ellipse across the chin.
    draw.rounded_rectangle(box((cx-fw*.18, 395, cx+fw*.18, 496)), radius=22*SCALE,
                           fill=tuple(int(v*.88) for v in p["skin"])+(255,))

    # Individual jaw silhouette: temples, cheeks, jaw and chin all have
    # independent curvature governed by seed-derived structural traits.
    cheek_y = top + fh * .56
    jaw_y = top + fh * .79
    chin_y = bottom
    temple = fw * .48
    cheek = fw * .50 * p["cheek"]
    jaw = fw * ({"pointed": .22, "square": .42, "round": .34}[p["jaw"]])
    if p["jaw"] == "pointed":
        chin_shape = [(cx - jaw, jaw_y), (cx - jaw*.56, chin_y-12), (cx, chin_y+5), (cx+jaw*.56, chin_y-12), (cx+jaw, jaw_y)]
    elif p["jaw"] == "square":
        chin_shape = [(cx-jaw, jaw_y), (cx-jaw*.96, chin_y-31), (cx-jaw*.66, chin_y), (cx, chin_y+2), (cx+jaw*.66, chin_y), (cx+jaw*.96, chin_y-31), (cx+jaw, jaw_y)]
    else:
        chin_shape = [(cx-jaw, jaw_y), (cx-jaw*.83, chin_y-26), (cx-jaw*.48, chin_y+1), (cx, chin_y+6), (cx+jaw*.48, chin_y+1), (cx+jaw*.83, chin_y-26), (cx+jaw, jaw_y)]
    face_pts = [(cx-temple, top), (cx-fw*.52, top+fh*.20), (cx-fw*.54, cheek_y),
                (cx-cheek, cheek_y+20), (cx-jaw, jaw_y)] + chin_shape[1:-1] + [
                (cx+jaw, jaw_y), (cx+cheek, cheek_y+20), (cx+fw*.54, cheek_y),
                (cx+fw*.52, top+fh*.20), (cx+temple, top)]
    # Radial and directional skin shading, softly blurred into a photographic
    # diffuse-light field clipped to the custom face silhouette.
    skin = np.asarray(p["skin"], dtype=float)
    lightx = math.cos(p["light_angle"]) * .33
    lighty = math.sin(p["light_angle"]) * .26
    nx, ny = (xx-cx)/(fw*.62), (yy-(top+fh*.48))/(fh*.66)
    radial = np.maximum(0, 1 - .22*(nx*nx+ny*ny))
    shade = np.clip(.76 + .23*radial + lightx*nx + lighty*ny, .57, 1.14)
    grain = rng.normal(0, 1.1, (SIZE, SIZE, 1))
    skin_pixels = np.uint8(np.clip(skin[None,None,:]*shade[...,None]+grain, 0, 255))
    skin_im = Image.fromarray(skin_pixels).resize((S,S), Image.Resampling.BICUBIC).convert("RGBA")
    mask = Image.new("L", (S,S), 0)
    ImageDraw.Draw(mask).polygon([(round(x*SCALE), round(y*SCALE)) for x,y in face_pts], fill=255)
    # Cast shadow offset follows illumination direction.
    shadow = Image.new("RGBA", (S,S), (0,0,0,0))
    ImageDraw.Draw(shadow).polygon([(round((x-lightx*9)*SCALE),round((y+5)*SCALE)) for x,y in face_pts], fill=(24,18,15,45))
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(12*SCALE)))
    skin_im.putalpha(mask)
    canvas = Image.alpha_composite(canvas, skin_im)
    draw = ImageDraw.Draw(canvas, "RGBA")
    base = tuple(int(v) for v in skin)
    hair = tuple(int(v) for v in p["hair"])
    hair_rgba = hair + (255,)
    # Ears have seed-specific size and height, with inner folds.
    earw = fw * p["ear_size"]
    ey = top + fh*p["ear_y"]
    for side in (-1,1):
        ex = cx + side*(fw*.49)
        draw.ellipse(box((ex-earw, ey-earw*1.42, ex+earw, ey+earw*1.42)), fill=base+(255,), outline=tuple(max(0,v-35) for v in base)+(120,), width=2*SCALE)
        draw.arc(box((ex-earw*.5,ey-earw*.8,ex+earw*.6,ey+earw*.8)), 30, 310, fill=(100,55,48,115), width=2*SCALE)
    # Ears sit over the cheek edge; the neck remains behind the head.

    # Hair styles have distinct silhouettes and hairlines.
    hairline = top + p["hairline"]*35
    if p["hair_style"] == "crop":
        draw.pieslice(box((cx-fw*.59, hairline-77, cx+fw*.59, top+fh*.48)), 180, 360, fill=hair_rgba)
        draw.rectangle(box((cx-fw*.59, hairline-3, cx+fw*.59, hairline+13)), fill=hair_rgba)
    elif p["hair_style"] == "side_part":
        draw.pieslice(box((cx-fw*.62, hairline-78, cx+fw*.62, top+fh*.43)), 180, 360, fill=hair_rgba)
        poly([(cx-fw*.52,hairline+3),(cx-fw*.10,hairline-25),(cx+fw*.18,hairline-3),(cx+fw*.52,hairline+4)],hair_rgba)
        draw.line([(round((cx-fw*.05)*SCALE),round((hairline-40)*SCALE)),(round((cx+fw*.02)*SCALE),round((hairline+5)*SCALE))],fill=tuple(min(255,int(c*1.5)) for c in hair)+(170,),width=2*SCALE)
    elif p["hair_style"] == "long":
        draw.rounded_rectangle(box((cx-fw*.62,top+20,cx-fw*.42,bottom+38)),radius=28*SCALE,fill=hair_rgba)
        draw.rounded_rectangle(box((cx+fw*.42,top+20,cx+fw*.62,bottom+38)),radius=28*SCALE,fill=hair_rgba)
        draw.pieslice(box((cx-fw*.61,hairline-75,cx+fw*.61,top+fh*.42)),180,360,fill=hair_rgba)
    elif p["hair_style"] == "receding":
        draw.pieslice(box((cx-fw*.59,top-35,cx+fw*.59,top+fh*.32)),190,350,fill=hair_rgba)
        if p["age"] > .55:
            draw.arc(box((cx-fw*.46,top+fh*.27,cx-fw*.05,top+fh*.49)),190,280,fill=(85,54,43,105),width=2*SCALE)
            draw.arc(box((cx+fw*.05,top+fh*.27,cx+fw*.46,top+fh*.49)),260,350,fill=(85,54,43,105),width=2*SCALE)
    else:  # curly/voluminous: irregular curls form a broad silhouette.
        draw.pieslice(box((cx-fw*.67,top-42,cx+fw*.67,top+fh*.39)),180,360,fill=hair_rgba)
        for i in range(20):
            x=cx+float(rng.uniform(-fw*.63,fw*.63)); y=top+float(rng.uniform(-38,38))
            radius=float(rng.uniform(10,23))
            draw.ellipse(box((x-radius,y-radius,x+radius,y+radius)),fill=hair_rgba)

    # Brow and under-brow shadows; eye centers move independently by identity.
    eye_y = top + fh*p["eye_y"]
    eye_dx = fw*p["eye_spacing"]
    er = p["eye_size"]
    for side in (-1,1):
        ex = cx + side*eye_dx
        tilt = side*p["eye_tilt"]
        halfw = er*1.48
        eh = er*({"almond":.54,"hooded":.48,"round":.78,"narrow":.37}[p["lid"]])
        # Brow shadow softly sits above each lid.
        brow_y = eye_y-er-p["brow_gap"]
        shadow=Image.new("RGBA",(S,S),(0,0,0,0)); shd=ImageDraw.Draw(shadow)
        shd.ellipse(box((ex-halfw-4,brow_y-13,ex+halfw+4,eye_y-eh*.35)),fill=(35,23,20,52))
        canvas=Image.alpha_composite(canvas,shadow.filter(ImageFilter.GaussianBlur(6*SCALE)))
        draw=ImageDraw.Draw(canvas,"RGBA")
        eyepts=[(ex-halfw,eye_y+tilt*.025),(ex-halfw*.55,eye_y-eh*.68),(ex,eye_y-eh*.78),(ex+halfw*.55,eye_y-eh*.68),(ex+halfw,eye_y-tilt*.025),(ex+halfw*.5,eye_y+eh*.55),(ex,eye_y+eh*.68),(ex-halfw*.5,eye_y+eh*.55)]
        poly(eyepts,(246,239,226,255),outline=(65,43,37,180),width=1.3)
        ir=er*.50
        draw.ellipse(box((ex-ir,eye_y-ir,ex+ir,eye_y+ir)),fill=p["iris"]+(255,))
        draw.ellipse(box((ex-ir*.43,eye_y-ir*.6,ex+ir*.43,eye_y+ir*.6)),fill=(23,24,25,255))
        draw.ellipse(box((ex-ir*.30-2,eye_y-ir*.48-3,ex-ir*.30+3,eye_y-ir*.48+2)),fill=(255,255,255,245))
        draw.ellipse(box((ex+ir*.32,eye_y+ir*.25,ex+ir*.32+2,eye_y+ir*.25+2)),fill=(255,255,255,170))
        draw.line([(round(x*SCALE),round(y*SCALE)) for x,y in eyepts[:5]],fill=(53,37,33,230),width=2*SCALE)
        arch=p["brow_arch"]
        browpts=[(ex-halfw-2,brow_y+3),(ex-halfw*.35,brow_y-arch),(ex+halfw*.25,brow_y-arch*.5),(ex+halfw+2,brow_y+4)]
        draw.line([(round(x*SCALE),round(y*SCALE)) for x,y in browpts],fill=hair_rgba,width=max(2,round(p["brow_thick"]*SCALE)),joint="curve")
        # Glasses frames are distinct round/rectangular shapes.
        if p["glasses"] != "none":
            gb=(ex-halfw-5,eye_y-eh-5,ex+halfw+5,eye_y+eh+5)
            draw.ellipse(box(gb),outline=(37,42,45,220),width=2*SCALE) if p["glasses"]=="round" else draw.rounded_rectangle(box(gb),radius=5*SCALE,outline=(37,42,45,220),width=2*SCALE)
    if p["glasses"] != "none":
        draw.line(box((cx-eye_dx+er*1.5,eye_y,cx+eye_dx-er*1.5,eye_y)),fill=(37,42,45,215),width=2*SCALE)

    # Nose length, width, nostril size and bridge shadow vary independently.
    nose_start=eye_y+er*.72
    nose_end=nose_start+fh*p["nose_len"]
    nw=p["nose_width"]
    ns=Image.new("RGBA",(S,S),(0,0,0,0)); nd=ImageDraw.Draw(ns)
    nd.polygon([(round((cx-3)*SCALE),round(nose_start*SCALE)),(round((cx+nw*.18)*SCALE),round((nose_end-10)*SCALE)),(round((cx+nw*.64)*SCALE),round(nose_end*SCALE)),(round((cx-nw*.62)*SCALE),round(nose_end*SCALE))],fill=(49,32,28,57))
    canvas=Image.alpha_composite(canvas,ns.filter(ImageFilter.GaussianBlur(7*SCALE)))
    draw=ImageDraw.Draw(canvas,"RGBA")
    draw.line([(round((cx+3)*SCALE),round((nose_start+8)*SCALE)),(round((cx+1)*SCALE),round((nose_end-5)*SCALE))],fill=(255,230,208,90),width=3*SCALE)
    draw.arc(box((cx-nw,nose_end-13,cx+nw,nose_end+14)),8,172,fill=tuple(int(v*.55) for v in base)+(210,),width=2*SCALE)
    nost=p["nostril"]
    for side in (-1,1): draw.ellipse(box((cx+side*nw*.38-nost,nose_end-1-nost/2,cx+side*nw*.38+nost,nose_end-1+nost/2)),fill=(57,37,33,125))

    # Mouth geometry and expression have separate parameters.
    mouth_y=top+fh*p["mouth_y"]+(1.7 if variant else 0)
    mw=p["mouth_width"]
    corner=p["corner"]+(1.5 if variant else 0)
    draw.line([(round((cx-mw/2)*SCALE),round((mouth_y+corner)*SCALE)),(round(cx*SCALE),round((mouth_y-2)*SCALE)),(round((cx+mw/2)*SCALE),round((mouth_y-corner)*SCALE))],fill=(102,52,53,235),width=max(2,round(p["lip_thick"]*SCALE)))
    draw.arc(box((cx-mw/2,mouth_y-4,cx+mw/2,mouth_y+13+corner)),5,175,fill=(137,67,66,240),width=max(2,round(p["lip_thick"]*SCALE)))
    draw.line([(round((cx-mw*.32)*SCALE),round((mouth_y+8)*SCALE)),(round(cx*SCALE),round((mouth_y+10+corner*.35)*SCALE)),(round((cx+mw*.32)*SCALE),round((mouth_y+8)*SCALE))],fill=(233,179,166,190),width=2*SCALE)

    # Age marks/freckles and beard shapes add non-colour identity cues.
    if p["age"]>.62 or p["detail"]=="wrinkles":
        for side in (-1,1):
            draw.arc(box((cx+side*eye_dx-er*1.5,eye_y+er*.6,cx+side*eye_dx+er*1.5,eye_y+er*2.0)),5,160,fill=(100,65,52,95),width=1*SCALE)
    if p["detail"]=="freckles":
        for side in (-1,1):
            for _ in range(13):
                fx=cx+side*float(rng.uniform(fw*.15,fw*.38)); fy=eye_y+float(rng.uniform(17,52)); rr=float(rng.uniform(.6,1.5))
                draw.ellipse(box((fx-rr,fy-rr,fx+rr,fy+rr)),fill=(100,63,46,95))
    if p["beard"] in ("beard","stubble"):
        opacity=32 if p["beard"]=="stubble" else 100
        beard_pts=[(cx-fw*.38,mouth_y+14),(cx-fw*.34,jaw_y-20),(cx-fw*.19,chin_y-6),(cx,chin_y+1),(cx+fw*.19,chin_y-6),(cx+fw*.34,jaw_y-20),(cx+fw*.38,mouth_y+14)]
        poly(beard_pts, hair+(opacity,))
        if p["beard"]=="stubble":
            for _ in range(180):
                bx=float(rng.uniform(cx-fw*.34,cx+fw*.34)); by=float(rng.uniform(mouth_y+15,chin_y-5))
                draw.ellipse(box((bx,by,bx+1.0,by+1.0)),fill=hair+(65,))
    elif p["beard"]=="moustache":
        for side in (-1,1):
            left,right=sorted((cx+side*3,cx+side*mw*.28))
            draw.ellipse(box((left,mouth_y-7,right,mouth_y+2)),fill=hair+(210,))

    # Chin bounce/shadow and final light rotation. Variant preserves all
    # structural traits and only nudges pose, illumination and mouth corners.
    chin_shadow=Image.new("RGBA",(S,S),(0,0,0,0)); cd=ImageDraw.Draw(chin_shadow)
    cd.ellipse(box((cx-fw*.27,chin_y-22,cx+fw*.27,chin_y+1)),fill=(32,22,18,42))
    chin_shadow.putalpha(Image.fromarray(np.minimum(np.asarray(chin_shadow.getchannel("A")),np.asarray(mask))))
    canvas=Image.alpha_composite(canvas,chin_shadow.filter(ImageFilter.GaussianBlur(5*SCALE)))
    out=canvas.convert("RGB").resize((SIZE,SIZE),Image.Resampling.LANCZOS)
    angle=(p["pose"] + (4.0 if variant else 0.0))
    fill=tuple(int(v) for v in p["bg1"])
    out=out.rotate(angle,resample=Image.Resampling.BICUBIC,fillcolor=fill)
    if variant:
        a=np.asarray(out,dtype=np.float32)
        # Distinct but mild second-photo lighting shift.
        ramp=np.linspace(.985,1.025,SIZE,dtype=np.float32)[None,:,None]
        out=Image.fromarray(np.uint8(np.clip(a*ramp+2.0,0,255)))
    out=out.filter(ImageFilter.GaussianBlur(.42))
    a=np.asarray(out,dtype=np.float32)
    a=np.uint8(np.clip(a+rng.normal(0,.85,a.shape),0,255))
    return Image.fromarray(a,"RGB")


def make_face(seed: int, *, variant: bool = False) -> Image.Image:
    """Render an identity determined by seed; variant changes pose/light/expression only."""
    return _paint(seed, variant=variant)


def make_variant(seed: int) -> Image.Image:
    return make_face(seed, variant=True)


def _save_preview():
    sheet=Image.new("RGB",(3*320,360),(245,245,242)); d=ImageDraw.Draw(sheet)
    for i,name in enumerate((FILES[0],FILES[2],FILES[3])):
        path=FACES/name
        im=Image.open(path).convert("RGB") if path.is_file() else make_face(400+i)
        im.thumbnail((300,300),Image.Resampling.LANCZOS); sheet.paste(im,(i*320+10,10)); d.text((i*320+10,320),name,fill=(35,35,35))
    sheet.save(ROOT/"faces_preview.png")


def _model_embed(image, detector, recognizer):
    jpeg=io.BytesIO(); image.save(jpeg,format="JPEG",quality=95)
    bgr=__import__("cv2").imdecode(np.frombuffer(jpeg.getvalue(),dtype=np.uint8),__import__("cv2").IMREAD_COLOR)
    detector.setInputSize((bgr.shape[1],bgr.shape[0])); _, rows=detector.detect(bgr)
    if rows is None or not len(rows): return None,0.0
    row=max(rows,key=lambda f:float(f[14])); conf=float(row[14])
    if conf<.75:return None,conf
    aligned=recognizer.alignCrop(bgr,row); feat=np.asarray(recognizer.feature(aligned),np.float32).ravel()
    norm=float(np.linalg.norm(feat))
    return (feat/norm if norm else None),conf


def _evaluate(max_seeds=500):
    import cv2
    det_path=MODELS/"yuNet.onnx"; rec_path=MODELS/"sface.onnx"
    if not det_path.is_file() or not rec_path.is_file(): raise RuntimeError("Existing YuNet/SFace models are required; no downloads are attempted.")
    # Match the project's detector configuration. Only the required confidence
    # acceptance is stricter; neither detector nor recognition thresholds move.
    # These are the same settings used by face_verify._load_models().
    detector=cv2.FaceDetectorYN.create(str(det_path),"",(SIZE,SIZE),0.6,0.3,5000)
    recognizer=cv2.FaceRecognizerSF.create(str(rec_path),"")
    candidates=[]
    for seed in range(max_seeds):
        image=make_face(seed)
        emb,conf=_model_embed(image,detector,recognizer)
        if emb is not None: candidates.append({"seed":seed,"image":image,"emb":emb,"conf":conf})
        if seed%50==49: print(f"searched {seed+1}/{max_seeds} seeds; YuNet-qualified identities={len(candidates)}")
    if not candidates:
        print(f"No candidates passed YuNet confidence >= 0.75 in {max_seeds} seeds.")
        return False
    best=None
    # For each possible family identity evaluate its structurally stable variant.
    for family in candidates:
        family_variant=make_variant(family["seed"])
        fam2,conf2=_model_embed(family_variant,detector,recognizer)
        if fam2 is None: continue
        fscore=float(family["emb"]@fam2)
        if fscore<.5: continue
        remaining=[c for c in candidates if c["seed"]!=family["seed"]]
        visitors=[]
        for c in remaining:
            sf=float(c["emb"]@family["emb"])
            if sf<=.2: visitors.append((c,sf))
        for (v1,s1),(v2,s2) in itertools.combinations(visitors,2):
            s12=float(v1["emb"]@v2["emb"])
            scores={"family_1_vs_family_2":fscore,"visitor_1_vs_family":s1,
                    "visitor_2_vs_family":s2,"visitor_1_vs_visitor_2":s12}
            quality=min(.2-s1,.2-s2,.2-s12,fscore-.5)
            if best is None or quality>best[0]: best=(quality,family,v1,v2,family_variant,conf2,scores)
            if s12<=.2:
                chosen=(family,v1,v2)
                _write_set(chosen,family_variant)
                print("WINNING SEEDS:",family["seed"],v1["seed"],v2["seed"])
                print("YuNet confidences:",family["conf"],conf2,v1["conf"],v2["conf"])
                print("PAIRWISE SFACE SCORES:",scores)
                return True
    if best:
        _,family,v1,v2,family_variant,conf2,scores=best
        _write_set((family,v1,v2),family_variant)
        print("NO ACCEPTABLE SET. Best candidate seeds:",family["seed"],v1["seed"],v2["seed"])
        print("Best pairwise SFace scores:",scores)
        print("Targets were family >= 0.5, all visitor pairs <= 0.2; none met every target. Settings unchanged.")
    else:
        print("NO ACCEPTABLE SET: no candidate family variant reached SFace 0.5 with YuNet confidence >= 0.75.")
    return False


def _write_set(chosen, family_variant):
    FACES.mkdir(parents=True,exist_ok=True)
    images=(chosen[0]["image"],family_variant,chosen[1]["image"],chosen[2]["image"])
    for image,name in zip(images,FILES): image.save(FACES/name,quality=96)
    _save_preview()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--render-only",action="store_true"); parser.add_argument("--seeds",type=int,default=500)
    args=parser.parse_args(); FACES.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        imgs=(make_face(1),make_variant(1),make_face(2),make_face(3))
        for im,name in zip(imgs,FILES): im.save(FACES/name,quality=96)
        _save_preview(); return 0
    return 0 if _evaluate(max(300,args.seeds)) else 2


if __name__=="__main__": raise SystemExit(main())
